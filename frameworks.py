import optuna
import numpy.typing as npt

from abc import ABC, abstractclassmethod
from dataclasses import dataclass, astuple

from metrics import Metric
from data.loader import Dataset

@dataclass
class Hyperparameter:
    name: str
    group: type
    min_value: float
    max_value: float


class Searcher(ABC):
    def __init__(self, max_iter: int):
        self.max_iter = max_iter

    @abstractclassmethod
    def tune(self,
             estimator, 
             hyperparams: list[Hyperparameter],
             features: npt.NDArray, 
             targets, 
             metric: Metric):
        pass


class OptunaSearcher(Searcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'Optuna'

    def tune(self,
             estimator, 
             hyperparams: list[Hyperparameter],
             features: npt.NDArray, 
             targets, 
             metric: Metric):

        self.estimator, self.hyperparams, self.features, self.targets, self.metric = \
            estimator, hyperparams, features, targets, metric
        optuna.logging.disable_default_handler()
        study = optuna.create_study(direction='maximize')
        study.optimize(self.__objective, n_trials=self.max_iter)
        return study.best_value

    def __objective(self, trial: optuna.Trial):
        arguments = {}
        for params in self.hyperparams:
            name, group, min_value, max_value = astuple(params)
            if group is float:
                arguments[name] = trial.suggest_float(name, min_value, max_value)
            elif group is int:
                arguments[name] = trial.suggest_int(name, min_value, max_value)
        model = self.estimator(**arguments)
        return self.metric(model, self.features, self.targets)

    

class HyperoptSearcher(Searcher):
    pass


class iOptSearcher(Searcher):
    pass
