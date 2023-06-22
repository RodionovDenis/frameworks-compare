import optuna

from dataclasses import astuple
from hyperparameter import Hyperparameter, Numerical, Categorial

from .interface import Searcher


ALGORITHMS = {
    'random': optuna.samplers.RandomSampler,
    'tpe': optuna.samplers.TPESampler,
    'cmaes': optuna.samplers.CmaEsSampler,
    'nsgaii': optuna.samplers.NSGAIISampler,
}


class OptunaSearcher(Searcher):
    def __init__(self, max_iter, *, algorithm: str = 'tpe', is_deterministic=False):
        super().__init__(framework_name='Optuna',
                         max_iter=max_iter,
                         is_deterministic=is_deterministic)

        self.algorithm, self.func_algorithm = algorithm, ALGORITHMS[algorithm]
        self.algorithm_kwargs = {}
        if self.algorithm == 'cmaes':
            self.algorithm_kwargs['warn_independent_sampling'] = False

    def find_best_value(self):
        optuna.logging.disable_default_handler()
        study = optuna.create_study(direction='minimize' if self.is_regression else 'maximize',
                                    sampler=self.func_algorithm(**self.algorithm_kwargs))
        study.optimize(self.objective, n_trials=self.max_iter)
        return study.best_value
    
    def get_searcher_params(self):
        return {'algorithm': self.algorithm}
    
    def framework_version(self):
        return optuna.__version__

    def objective(self, trial: optuna.Trial):
        arguments = {name: self.get_value(name, param, trial) for name, param in self.hyperparams.items()}
        return self.calculate_metric(arguments)
    
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
