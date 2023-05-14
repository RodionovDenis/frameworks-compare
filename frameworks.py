from iOpt.trial import FunctionValue, Point
import numpy.typing as npt
import optuna
import hyperopt

from iOpt.problem import Problem
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters

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
        arguments = {}
        for params in self.hyperparams:
            name, group, min_value, max_value = astuple(params)
            if group is float:
                arguments[name] = trial.suggest_float(name, min_value, max_value)
            elif group is int:
                arguments[name] = trial.suggest_int(name, min_value, max_value)
        model = self.estimator(**arguments)
        return self.metric(model, self.dataset)


class HyperoptSearcher(Searcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'Hyperopt'
    
    def tune(self,
             estimator,
             hyperparams: list[Hyperparameter],
             dataset: Dataset,
             metric: Metric):

        arguments, self.groups = {}, {}
        for params in hyperparams:
            name, group, min_value, max_value = astuple(params)
            self.groups[name] = group
            if group is float:
                arguments[name] = hyperopt.hp.uniform(name, min_value, max_value)
            elif group is int:
                arguments[name] = hyperopt.hp.quniform(name, min_value, max_value, q=1)
        
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
            for param in hyperparams:
                name, group, min_value, max_value = astuple(param)
                self.variable_type.append(group)
                self.floatVariableNames.append(name)
                self.lowerBoundOfFloatVariables.append(min_value)
                self.upperBoundOfFloatVariables.append(max_value)

        def Calculate(self, point, functionValue):
            arguments = self.__get_argument_dict(point)
            model = self.estimator(**arguments)
            functionValue.value = -self.metric(model, self.dataset)
            return functionValue

        def __get_argument_dict(self, point):
            arguments = {}
            for name, group, value in zip(self.floatVariableNames, self.variable_type, point.floatVariables):
                arguments[name] = int(value) if group is int else value
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
