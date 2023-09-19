import pandas as pd
import mlflow
from mlflow.entities import Metric as M
import numpy as np

from hyperparameter import Hyperparameter
from frameworks import Searcher, Point
from utils import get_commit_hash
from data.loader import Parser, get_datasets
from metrics import DATASET_TO_METRIC
from metrics import Metric

from multiprocessing import Pool
from itertools import product
from collections import defaultdict
from functools import partial

from time import time

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings


class Experiment:
    def __init__(self, estimator,
                       hyperparams: dict[str, Hyperparameter],
                       searchers: list[Searcher],
                       parsers: list[Parser],
                       metric: Metric = None):

        self.estimator = estimator
        self.hyperparams = hyperparams
        self.searchers = {str(x): x for x in searchers}
        
        datasets = get_datasets(*parsers)
        
        self.datasets = {x.name: x for x in datasets}
        self.metrics = {x.name: DATASET_TO_METRIC[y] if (metric is None) else metric
                        for x, y in zip(datasets, parsers)}

    @ignore_warnings(category=ConvergenceWarning)
    def run(self, n_jobs: int = 1,
                  non_deterministic_trials: int = 1,
                  is_mlflow_log: bool = False) -> pd.DataFrame:

        assert non_deterministic_trials > 0, 'Something very strange'
        assert n_jobs >= -1, 'Something very strange'
        self.non_deterministic_trials = non_deterministic_trials
        self.n_jobs = n_jobs

        if not is_mlflow_log:
            self.mlflow_logging = False
            frame, time = self.start_pool()
            return pd.DataFrame(frame).applymap(self.apply)

        assert len(self.datasets) == 1, 'Mlflow supports one dataset'
        self.mlflow_logging = True
        dataset_name = next(iter(self.datasets))

        self.client, self.run_id = self.setup_mlflow('http://192.168.2.126:8892/', dataset_name)
        self.log_params()
        frame, time = self.start_pool()
        self.log_final_metric(frame, dataset_name)
        self.client.set_terminated(self.run_id)
        
        return pd.DataFrame(frame).applymap(self.apply), time

    def start_pool(self):
        trials = self.get_trials()
        frame = defaultdict(lambda: defaultdict(list))

        with Pool(self.n_jobs) as pool:
            start = time()
            result = pool.starmap(self.objective, trials)
            time_value = time() - start

            for dataset, searcher, value in result:
                frame[searcher][dataset].append(value)

        return frame, time_value
    
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
    
    def log_metrics(self, experiment_name, points: list[Point]):
        metrics = []
        for i, p in enumerate(points, start=1):
            m = M(f'trials/{experiment_name}/metric', p.value, int(p.timepoint), i)
            metrics.append(m)
            for key, value in p.params.items():
                m = M(f'trials/{experiment_name}/{key}', value, int(p.timepoint), i)
                metrics.append(m)
        self.client.log_batch(self.run_id, metrics)


    def objective(self, dname: str, sname: str, experiment_name: str):
        searcher, dataset, metric = self.searchers[sname], self.datasets[dname], self.metrics[dname]
        points = searcher.tune(self.estimator, self.hyperparams, dataset, metric)
        if self.mlflow_logging:
            self.log_metrics(experiment_name, points)
        return dname, sname, max(x.value for x in points)

    def log_params(self):
        for i, (name, searcher) in enumerate(self.searchers.items(), start=1):
            self.client.log_param(self.run_id, f'Searcher/{i}', name)
            self.client.log_param(self.run_id, f'Framework/{searcher.framework_name}-version', searcher.framework_version())
        for name, param in self.hyperparams.items():
            self.client.log_param(self.run_id, f'Hyperparam/{name}', str(param))
        for value in self.metrics.values():
            self.client.log_param(self.run_id, 'Experiment/metric', value.log_params())
        self.client.log_param(self.run_id, 'Experiment/n_jobs', self.n_jobs)
        self.client.log_param(self.run_id, 'Experiment/non_deterministic_trials', self.non_deterministic_trials)
        self.client.log_param(self.run_id, 'Utils/frameworks-compare-hash-commit', get_commit_hash())
    
    def log_final_metric(self, frame, dataset_name):
        self.client.log_param(self.run_id, 'Utils/final-metric', 'mean')
        frameworks_metric = defaultdict(list)
        for name, searcher in self.searchers.items():
            values = frame[name][dataset_name]
            frameworks_metric[searcher.framework_name].append(np.mean(values))
        metrics = []
        for name, values in frameworks_metric.items():
            m = M(name, max(values), 1, 0)
            metrics.append(m)

        self.client.log_batch(self.run_id, metrics)

    def setup_mlflow(self, mlflow_uri, dataset_name):
        client = mlflow.MlflowClient(mlflow_uri)
        if isinstance(self.estimator, partial):
            experiment_name = self.estimator.func.__name__.lower()
        else:
            experiment_name = self.estimator.__name__.lower()
        
        if (experiment := client.get_experiment_by_name(experiment_name)) is not None:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = client.create_experiment(experiment_name)
        
        run = client.create_run(experiment_id, run_name=dataset_name, start_time=int(time() * 1000))
        return client, run.info.run_id
