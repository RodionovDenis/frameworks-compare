import hyperopt
import numpy as np

from functools import partial
from dataclasses import astuple
from hyperparameter import Numerical, Categorial

from .interface import Searcher


class HyperoptSearcher(Searcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'Hyperopt'
    
    def find_best_value(self):

        arguments = self.__get_space()
        trial = hyperopt.Trials()

        hyperopt.fmin(self.__objective, arguments, hyperopt.tpe.suggest, max_evals=self.max_iter,
                      trials=trial, verbose=False)

        return np.abs(trial.best_trial['result']['loss'])

    def __objective(self, arguments):
        self.__float_to_int(arguments)
        value = self.calculate_metric_with_log(arguments)
        return value if self.is_regression else -value
    
    def __get_space(self):
        arguments = {}
        functions = {
            'float': [hyperopt.hp.uniform, 
                    hyperopt.hp.loguniform],
            'int': [partial(hyperopt.hp.quniform, q=1),
                  partial(hyperopt.hp.qloguniform, q=1)]
        }
        for name, param in self.hyperparams.items():
            if isinstance(param, Numerical):
                type, min_v, max_v, log = astuple(param)
                arguments[name] = functions[type][log](name, np.log(min_v) if log else min_v,
                                                             np.log(max_v) if log else max_v)
            elif isinstance(param, Categorial):
                arguments[name] = hyperopt.hp.choice(name, param.values)
        return arguments
    
    def __float_to_int(self, arguments):
        for name, value in arguments.items():
            x = self.hyperparams[name]
            if isinstance(x, Numerical) and x.type == 'int':
                arguments[name] = int(value + 0.5)
