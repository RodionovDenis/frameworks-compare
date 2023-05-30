import optuna
import hyperopt
import numpy as np

from iOpt.problem import Problem
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters

from abc import ABC, abstractclassmethod
from dataclasses import dataclass, astuple, asdict
from functools import partial

from metrics import Metric
from data.loader import Dataset


class Type:
    @dataclass
    class Number:
        type: type
        min_value: float
        max_value: float
        log: bool
    
    @staticmethod
    def int(min_value, max_value, log=False):
        return Type.Number(int, min_value, max_value, log)
    
    @staticmethod
    def float(min_value, max_value, log=False):
        return Type.Number(float, min_value, max_value, log)


@dataclass
class Hyperparameter:
    name: str
    group: Type.Number


def dict_factory(data):
    result = {}
    for name, value in data:
        if isinstance(value, dict):
            result.update(value)
        elif isinstance(value, type):
            result[name] = value.__name__
        else:
            result[name] = value
    return result


class Searcher(ABC):
    def __init__(self, max_iter: int):
        self.max_iter = max_iter

    @abstractclassmethod
    def tune(self,
             estimator,
             hyperparams: list[Hyperparameter],
             dataset: Dataset,
             metric: Metric):
        pass


class OptunaSearcher(Searcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'Optuna'

    def tune(self,
             estimator, 
             hyperparams: list[Hyperparameter],
             dataset: Dataset,
             metric: Metric):

        self.estimator, self.hyperparams, self.dataset, self.metric = \
            estimator, hyperparams, dataset, metric
        optuna.logging.disable_default_handler()
        study = optuna.create_study(direction='maximize')
        study.optimize(self.__objective, n_trials=self.max_iter)
        return study.best_value

    def __objective(self, trial: optuna.Trial):
        arguments = self.__get_point(trial)
        model = self.estimator(**arguments)
        return self.metric(model, self.dataset)
    
    def __get_point(self, trial: optuna.Trial):
        arguments = {}
        functions = {
            int: trial.suggest_int,
            float: trial.suggest_float
        }
        for params in self.hyperparams:
            name, (type, min_value, max_value, log) = astuple(params)
            value = functions[type](name, min_value, max_value, log=log)
            arguments[name] = value
        return arguments


class HyperoptSearcher(Searcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'Hyperopt'
    
    def tune(self,
             estimator,
             hyperparams: list[Hyperparameter],
             dataset: Dataset,
             metric: Metric):
        arguments = self.__get_space(hyperparams)
        trial = hyperopt.Trials()
        self.estimator, self.dataset, self.metric = \
            estimator, dataset, metric
        hyperopt.fmin(self.__objective, arguments, hyperopt.tpe.suggest, max_evals=self.max_iter,
                      trials=trial, verbose=False)
        return -trial.best_trial['result']['loss']

    def __objective(self, arguments):
        arguments = self.__float_to_int(arguments)
        model = self.estimator(**arguments)
        return -self.metric(model, self.dataset)
    
    def __get_space(self, hyperparams):
        arguments, self.groups = {}, {}
        functions = {
            float: [hyperopt.hp.uniform, 
                    hyperopt.hp.loguniform],
            int: [partial(hyperopt.hp.quniform, q=1),
                  partial(hyperopt.hp.qloguniform, q=1)]
        }
        for params in hyperparams:
            name, (type, min_value, max_value, log) = astuple(params)
            self.groups[name] = type
            arguments[name] = functions[type][log](name,
                                                   np.log(min_value) if log else min_value,
                                                   np.log(max_value) if log else max_value)
            
        return arguments
            
    
    def __float_to_int(self, arguments):
        return {name: int(value) if (self.groups[name] is int) else value for name, value in arguments.items()}


class iOptSearcher(Searcher):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'iOpt'
    
    class __Estimator(Problem):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.estimator, hyperparams, self.dataset, self.metric = args

            self.numberOfFloatVariables = len(hyperparams)
            self.dimension = len(hyperparams)
            self.numberOfObjectives = 1
            
            self.variable_type = []
            self.is_log_variable = []
            for param in hyperparams:
                name, (type, min_value, max_value, log) = astuple(param)

                self.variable_type.append(type)
                self.floatVariableNames.append(name)
                self.lowerBoundOfFloatVariables.append(np.log(min_value) if log else min_value)
                self.upperBoundOfFloatVariables.append(np.log(max_value) if log else max_value)
                self.is_log_variable.append(log)
                

        def Calculate(self, point, functionValue):
            arguments = self.__get_argument_dict(point)
            model = self.estimator(**arguments)
            functionValue.value = -self.metric(model, self.dataset)
            return functionValue

        def __get_argument_dict(self, point):
            arguments = {}
            for name, type, value, log in zip(self.floatVariableNames, self.variable_type, point.floatVariables,
                                              self.is_log_variable):
                value = np.exp(value) if log else value
                value = int(value) if type is int else value
                arguments[name] = value
            return arguments

    def tune(self,
             estimator,
             hyperparams: list[Hyperparameter],
             dataset: Dataset,
             metric: Metric):
        
        problem = self.__Estimator(estimator, hyperparams, dataset, metric)
        framework_params = SolverParameters(itersLimit=self.max_iter)
        solver = Solver(problem, parameters=framework_params)
        solver_info = solver.Solve()
        return -solver_info.bestTrials[0].functionValues[-1].value
    

def get_frameworks(*args, max_iter):
    frameworks = []
    for framework in args:
        frameworks.append(framework(max_iter))
    return frameworks
