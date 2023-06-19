import mlflow
import numpy as np

from abc import ABC, abstractclassmethod, abstractproperty

from metrics import Metric
from data.loader import Dataset
from hyperparameter import Hyperparameter, Categorial


class Searcher(ABC):
    def __init__(self, max_iter: int, *, name: str, is_deterministic: bool):
        self.max_iter = max_iter
        self.name = name
        self.is_deterministic = is_deterministic
    
    @abstractclassmethod
    def find_best_value(self):
        pass
    
    @abstractclassmethod
    def searcher_params(self) -> dict:
        pass

    def log_searcher_params(self, **kwargs):
        common_params = {
            'max_iter': self.max_iter,
            'is_deterministic': self.is_deterministic
        }
        if (val := self.searcher_params()) is not None: 
            common_params.update(val)
        mlflow.log_param(f'Searcher/{self.name}', common_params)

    def tune(self, estimator,
                   hyperparams: dict[str, Hyperparameter],
                   dataset: Dataset,
                   metric: Metric,
                   suffix_for_log: int | None = None):

        self.estimator, self.hyperparams, self.dataset, self.metric = estimator, hyperparams, dataset, metric
        self.is_regression = self.dataset.type == 'regression'
        self.current_step = 0
        self.log_name = f"{self.name}{suffix_for_log if suffix_for_log else ''}"

        return self.find_best_value()

    def calculate_metric(self, arguments):
        model = self.estimator(**arguments)
        value = self.metric(model, self.dataset)
        self.current_step += 1
        if mlflow.active_run() is not None:
            self.log_hyperparameters_point(arguments, step=self.current_step)
            mlflow.log_metric(self.log_name, np.abs(value), step=self.current_step)
        return value
    
    def log_hyperparameters_point(self, arguments, step):
        for name, value in arguments.items():
            x = self.hyperparams[name]
            if isinstance(x, Categorial):
                value = x.values.index(value)
            mlflow.log_metric(f'{self.log_name}_{name}', value, step=step)
