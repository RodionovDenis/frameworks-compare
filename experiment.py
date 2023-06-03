import pandas as pd
import mlflow
import os

from hyperparameter import Hyperparameter
from frameworks import Searcher

from data.loader import Parser
from metrics import Metric
from multiprocessing import Pool
from itertools import product


class Experiment:
    def __init__(self, estimator,
                       hyperparams: dict[str, Hyperparameter],
                       frameworks: list[Searcher],
                       datasets: list[Parser],
                       metric: Metric):

        self.estimator = estimator
        self.frameworks = {x.name: x for x in frameworks}
        self.datasets = {x.name: x for x in datasets}
        self.hyperparams = hyperparams
        self.metric = metric
        self.default = 'Default'

    def setup_mlflow(self, mlflow_uri):
        mlflow.set_tracking_uri(mlflow_uri)
        experiment_name = self.estimator.__name__
        self.id = mlflow.set_experiment(experiment_name).experiment_id

    def log_params(self):
        for name, param in self.hyperparams.items():
            mlflow.log_param(name, str(param))
        first = next(iter(self.frameworks.values()))
        mlflow.log_param('max_iter', first.max_iter)
        mlflow.log_param('metric', self.metric.name)

    def run(self, n_jobs=1, mlflow_uri="http://127.0.0.1:5000"):

        columns = [self.default] + list(self.frameworks)
        indexes = list(self.datasets)
        frame = pd.DataFrame(index=indexes, columns=columns)

        products = list(product(columns, indexes))

        self.setup_mlflow(mlflow_uri)
        with mlflow.start_run(experiment_id=self.id, run_name='/'.join(indexes)):
            self.log_params()
            with Pool(n_jobs) as pool:
                result = pool.map(self.objective, products)
            for (column, row), value in result:
                frame[column][row] = value
            frame.to_csv('result.csv')
            mlflow.log_artifact('result.csv', 'result.csv')
            os.remove('result.csv')

        return frame
    
    def objective(self, args):
        framework, dataset = self.frameworks.get(args[0]), self.datasets.get(args[1])
        if args[0] == self.default:
            value = self.metric(self.estimator(), dataset)
        else:
            value = framework.tune(self.estimator, self.hyperparams, dataset, self.metric)
        return args, value

    def setup_mlflow(self, mlflow_uri):
        mlflow.set_tracking_uri(mlflow_uri)
        experiment_name = self.estimator.__name__.lower()
        self.id = mlflow.set_experiment(experiment_name).experiment_id
