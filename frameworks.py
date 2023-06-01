import optuna
import hyperopt
import numpy as np

from iOpt.problem import Problem
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters

from abc import ABC, abstractclassmethod
from dataclasses import dataclass, astuple
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
    
    @dataclass
    class Categorial:
        values: list[str]

    @staticmethod
    def int(min_value, max_value, log=False):
        return Type.Number(int, min_value, max_value, log)

    @staticmethod
    def float(min_value, max_value, log=False):
        return Type.Number(float, min_value, max_value, log)

    @staticmethod
    def choice(*args: str):
        return Type.Categorial([x for x in args])


@dataclass
class Hyperparameter:
    name: str
    group: Type.Number | Type.Categorial


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
        arguments = {param.name: self.get_value(param, trial) for param in self.hyperparams}
        model = self.estimator(**arguments)
        return self.metric(model, self.dataset)
    
    @staticmethod
    def get_value(param: Hyperparameter, trial: optuna.Trial):
        name = param.name
        functions = {
            int: trial.suggest_int,
            float: trial.suggest_float
        }
        if isinstance(param.group, Type.Number):
            type, min_v, max_v, log = astuple(param.group)
            return functions[type](name, min_v, max_v, log=log)
        elif isinstance(param.group, Type.Categorial):
            return trial.suggest_categorical(name, param.group.values)


class HyperoptSearcher(Searcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'Hyperopt'
    
    def tune(self,
             estimator,
             hyperparams: list[Hyperparameter],
             dataset: Dataset,
             metric: Metric):

        self.estimator, self.dataset, self.metric = \
            estimator, dataset, metric

        arguments = self.__get_space(hyperparams)
        trial = hyperopt.Trials()

        hyperopt.fmin(self.__objective, arguments, hyperopt.tpe.suggest, max_evals=self.max_iter,
                      trials=trial, verbose=False)
        return -trial.best_trial['result']['loss']

    def __objective(self, arguments):
        model = self.estimator(**arguments)
        return -self.metric(model, self.dataset)
    
    def __get_space(self, hyperparams: list[Hyperparameter]):
        arguments = {}
        functions = {
            float: [hyperopt.hp.uniform, 
                    hyperopt.hp.loguniform],
            int: [partial(hyperopt.hp.quniform, q=1),
                  partial(hyperopt.hp.qloguniform, q=1)]
        }
        for param in hyperparams:
            name = param.name
            if isinstance(param.group, Type.Number):
                type, min_v, max_v, log = astuple(param.group)
                value = functions[type][log](name, np.log(min_v) if log else min_v,
                                                             np.log(max_v) if log else max_v)
                arguments[name] = value if type is float else int(value)
            elif isinstance(param.group, Type.Categorial):
                arguments[name] = hyperopt.hp.choice(name, param.group.values)

        return arguments


class iOptSearcher(Searcher):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'iOpt'
    
    class __Estimator(Problem):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.estimator, float_hyperparams, discrete_hyperparams, self.dataset, self.metric = args

            self.numberOfFloatVariables = len(float_hyperparams)
            self.numberOfDiscreteVariables = len(discrete_hyperparams)
            self.dimension = len(float_hyperparams) + len(discrete_hyperparams)
            self.numberOfObjectives = 1
            
            self.float_variables_types, self.is_log_float = [], []
            for param in float_hyperparams:
                self.floatVariableNames.append(param.name)
                type, min_v, max_v, log = astuple(param.group)
                self.float_variables_types.append(type)
                self.lowerBoundOfFloatVariables.append(np.log(min_v) if log else min_v)
                self.upperBoundOfFloatVariables.append(np.log(max_v) if log else max_v)
                self.is_log_float.append(log)
                
            for param in discrete_hyperparams:
                self.discreteVariableNames.append(param.name)
                if isinstance(param.group, Type.Number):
                    type, min_v, max_v, log = astuple(param.group)
                    assert type is int, 'Type must be int'
                    assert not log, 'Log must be off'
                    self.discreteVariableValues.append([str(x) for x in range(min_v, max_v + 1)])
                elif isinstance(param.group, Type.Categorial):
                    self.discreteVariableValues.append(param.group.values)      

        def Calculate(self, point, functionValue):
            arguments = self.__get_argument_dict(point)
            model = self.estimator(**arguments)
            functionValue.value = -self.metric(model, self.dataset)
            return functionValue

        def __get_argument_dict(self, point):
            arguments = {}
            for name, type, value, log in zip(self.floatVariableNames, self.float_variables_types, point.floatVariables,
                                              self.is_log_float):
                value = np.exp(value) if log else value
                value = int(value) if type is int else value
                arguments[name] = value
            for name, value in zip(self.discreteVariableNames, point.discreteVariables):
                arguments[name] = int(value) if value.isnumeric() else value

            return arguments

    def tune(self,
             estimator,
             hyperparams: list[Hyperparameter],
             dataset: Dataset,
             metric: Metric):
        
        floats, discretes = self.split_hyperparams(hyperparams)
        problem = self.__Estimator(estimator, floats, discretes, dataset, metric)
        framework_params = SolverParameters(itersLimit=self.max_iter)
        solver = Solver(problem, parameters=framework_params)
        solver_info = solver.Solve()
        return -solver_info.bestTrials[0].functionValues[-1].value
    
    @staticmethod
    def split_hyperparams(hyperparams: list[Hyperparameter]):
        floats, discretes = [], []
        for x in hyperparams:
            if iOptSearcher.is_discrete_hyperparam(x):
                discretes.append(x)
            else:
                floats.append(x)
        return floats, discretes
            
    @staticmethod
    def is_discrete_hyperparam(x: Hyperparameter):
        if isinstance(x.group, Type.Categorial):
            return True
        type, min_v, max_v, log = astuple(x.group)
        return (type is int) and (not log) and (max_v - min_v + 1 <= 100)
    

def get_frameworks(*args, max_iter):
    frameworks = []
    for framework in args:
        frameworks.append(framework(max_iter))
    return frameworks
