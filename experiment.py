import pandas as pd
import mlflow
import numpy as np

from hyperparameter import Hyperparameter
from frameworks import Searcher
from utils import get_commit_hash
from data.loader import Parser, get_datasets
from metrics import Metric

from multiprocessing import Pool
from itertools import product
from collections import defaultdict
from functools import partial


class Experiment:
    def __init__(self, estimator,
                       hyperparams: dict[str, Hyperparameter],
                       searchers: list[Searcher],
                       datasets: list[Parser],
                       metric: Metric):

        self.estimator = estimator
        self.hyperparams = hyperparams
        self.searchers = {str(x): x for x in searchers}
        self.datasets = {x.name: x for x in get_datasets(*datasets)}
        self.metric = metric

    def run(self, n_jobs: int = 1,
                  non_deterministic_trials: int = 1,
                  mlflow_uri: str | None = None):

        assert non_deterministic_trials > 0, 'Something very strange'
        assert n_jobs >= -1, 'Something very strange'
        self.non_deterministic_trials = non_deterministic_trials
        self.n_jobs = n_jobs

        if mlflow_uri is None:
            frame = self.start_pool()
            return pd.DataFrame(frame).applymap(self.apply)

        assert len(self.datasets) == 1, 'Mlflow supports one dataset'
        self.setup_mlflow(mlflow_uri)
        dataset_name = next(iter(self.datasets))
        with mlflow.start_run(experiment_id=self.id, run_name=dataset_name):
            self.log_params()
            frame = self.start_pool()
            self.log_final_metric(frame, dataset_name)
        return pd.DataFrame(frame).applymap(self.apply)

    def start_pool(self):
        trials = self.get_trials()
        frame = defaultdict(lambda: defaultdict(list))

        with Pool(self.n_jobs) as pool:
            result = pool.starmap(self.objective, trials)
            for dataset, searcher, value in result:
                frame[searcher][dataset].append(value)
        return frame
    
    def get_trials(self):
        names = []
        for i, (name, searcher) in enumerate(self.searchers.items(), start=1):
            value = 1 if searcher.is_deterministic else self.non_deterministic_trials
            names.extend((name, f'{searcher.framework_name}/searcher{i}/trial{j}')
                         for j in range(1, value + 1))
        return list((x, *y) for x, y in product(list(self.datasets), names))
    
    @staticmethod
    def apply(x: list):
        if len(x) == 1:
            return x.pop()
        return {'min': np.min(x), 'max': np.max(x), 'mean': np.mean(x)}

    def objective(self, dname: str, sname: str, experiment_name: str):
        searcher, dataset = self.searchers[sname], self.datasets[dname]
        value = searcher.tune(self.estimator, self.hyperparams, dataset, self.metric,
                              experiment_name=experiment_name)
        return dname, sname, value

    def log_params(self):
        for i, (name, searcher) in enumerate(self.searchers.items(), start=1):
            mlflow.log_param(f'Searcher/{i}', name)
            mlflow.log_param(f'Framework/{searcher.framework_name}-version', searcher.framework_version())
        for name, param in self.hyperparams.items():
            mlflow.log_param(f'Hyperparam/{name}', str(param))
        mlflow.log_param('Experiment/metric', self.metric.log_params())
        mlflow.log_param('Experiment/n_jobs', self.n_jobs)
        mlflow.log_param('Experiment/non_deterministic_trials', self.non_deterministic_trials)
        mlflow.log_param('Utils/frameworks-compare-hash-commit', get_commit_hash())
    
    def log_final_metric(self, frame, dataset_name):
        mlflow.log_param('Utils/final-metric', 'mean')
        frameworks_metric = defaultdict(list)
        for name, searcher in self.searchers.items():
            values = frame[name][dataset_name]
            frameworks_metric[searcher.framework_name].append(np.mean(values))
        for name, values in frameworks_metric.items():
            mlflow.log_metric(name, np.max(values))

    def setup_mlflow(self, mlflow_uri):
        mlflow.set_tracking_uri(mlflow_uri)
        if isinstance(self.estimator, partial):
            experiment_name = self.estimator.func.__name__.lower()
        else:
            experiment_name = self.estimator.__name__.lower()
        self.id = mlflow.set_experiment(experiment_name).experiment_id
