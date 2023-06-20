import mlflow
import numpy as np

from abc import ABC, abstractclassmethod

from metrics import Metric
from data.loader import Dataset
from hyperparameter import Hyperparameter, Categorial


class Searcher(ABC):
    def __init__(self,
                 framework_name: str, *, max_iter: int, is_deterministic: bool):

        self.framework_name = framework_name
        self.max_iter = max_iter
        self.is_deterministic = is_deterministic
    
    @abstractclassmethod
    def find_best_value(self):
        pass

    @abstractclassmethod
    def get_searcher_params(self) -> dict:
        pass

    @abstractclassmethod
    def framework_version(self) -> str:
        pass

    def __str__(self):
        params = self.get_searcher_params()
        params['max_iter'] = self.max_iter
        params['is_deterministic'] = self.is_deterministic
        arguments = ', '.join(f'{key}={value}' for key, value in params.items())
        return f'{self.framework_name}({arguments})'

    def tune(self, estimator,
                   hyperparams: dict[str, Hyperparameter],
                   dataset: Dataset,
                   metric: Metric,
                   experiment_name: str = None):

        self.estimator, self.hyperparams, self.dataset, self.metric = estimator, hyperparams, dataset, metric
        self.is_regression = self.dataset.type == 'regression'
        self.current_step = 0
        self.experiment_name = experiment_name

        return self.find_best_value()
    
    def log_hyperparameters_point(self, arguments, step):
        for name, value in arguments.items():
            x = self.hyperparams[name]
            if isinstance(x, Categorial):
                value = x.values.index(value)
            mlflow.log_metric(f'{self.experiment_name}/{name}', value, step=step)

    def calculate_metric(self, arguments):
        model = self.estimator(**arguments)
        value = self.metric(model, self.dataset)
        self.current_step += 1
        if mlflow.active_run() is not None:
            self.log_hyperparameters_point(arguments, step=self.current_step)
            mlflow.log_metric(f'{self.experiment_name}/metric', np.abs(value), step=self.current_step)
        return value
