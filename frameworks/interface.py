import mlflow
import numpy as np

from abc import ABC, abstractclassmethod

from metrics import Metric
from data.loader import Dataset
from hyperparameter import Hyperparameter


class Searcher(ABC):
    def __init__(self, max_iter: int):
        self.max_iter = max_iter
        self.current_step = 0

    def tune(self, estimator,
                   hyperparams: dict[str, Hyperparameter],
                   dataset: Dataset,
                   metric: Metric):
        self.estimator, self.hyperparams, self.dataset, self.metric = estimator, hyperparams, dataset, metric
        return self.find_best_value()

    @abstractclassmethod
    def find_best_value(self):
        pass

    def calculate_metric_with_log(self, arguments):
        model = self.estimator(**arguments)
        value = self.metric(model, self.dataset)
        self.current_step += 1
        mlflow.log_metric(f'{self.name}/{self.dataset.name}', np.abs(value), step=self.current_step)
        return value


def get_frameworks(*args, max_iter):
    frameworks = []
    for framework in args:
        frameworks.append(framework(max_iter))
    return frameworks
