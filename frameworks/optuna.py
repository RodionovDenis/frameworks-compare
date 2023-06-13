import optuna

from dataclasses import astuple
from hyperparameter import Hyperparameter, Numerical, Categorial

from .interface import Searcher


class OptunaSearcher(Searcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'Optuna'

    def find_best_value(self):
        optuna.logging.disable_default_handler()
        study = optuna.create_study(direction='minimize' if self.is_regression else 'maximize')
        study.optimize(self.objective, n_trials=self.max_iter)
        return study.best_value

    def objective(self, trial: optuna.Trial):
        arguments = {name: self.get_value(name, param, trial) for name, param in self.hyperparams.items()}
        return self.calculate_metric_with_log(arguments)
    
    @staticmethod
    def get_value(name: str, param: Hyperparameter, trial: optuna.Trial):
        functions = {
            'int': trial.suggest_int,
            'float': trial.suggest_float
        }
        if isinstance(param, Numerical):
            type, min_v, max_v, log = astuple(param)
            return functions[type](name, min_v, max_v, log=log)
        elif isinstance(param, Categorial):
            return trial.suggest_categorical(name, param.values)
