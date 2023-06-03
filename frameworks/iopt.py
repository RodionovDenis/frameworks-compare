import numpy as np

from iOpt.problem import Problem
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters

from dataclasses import astuple
from hyperparameter import Hyperparameter, Numerical, Categorial

from .interface import Searcher


class Estimator(Problem):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.estimator, float_hyperparams, discrete_hyperparams, self.dataset, self.metric = args
        self.numberOfFloatVariables = len(float_hyperparams)
        self.numberOfDiscreteVariables = len(discrete_hyperparams)
        self.dimension = len(float_hyperparams) + len(discrete_hyperparams)
        self.numberOfObjectives = 1

        self.float_variables_types, self.is_log_float = [], []
        for name, param in float_hyperparams.items():
            self.floatVariableNames.append(name)
            type, min_v, max_v, log = astuple(param)
            self.float_variables_types.append(type)
            self.lowerBoundOfFloatVariables.append(np.log(min_v) if log else min_v)
            self.upperBoundOfFloatVariables.append(np.log(max_v) if log else max_v)
            self.is_log_float.append(log)
            
        for name, param in discrete_hyperparams.items():
            self.discreteVariableNames.append(name)
            if isinstance(param, Numerical):
                type, min_v, max_v, log = astuple(param)
                assert type == 'int', 'Type must be int'
                assert not log, 'Log must be off'
                self.discreteVariableValues.append([str(x) for x in range(min_v, max_v + 1)])
            elif isinstance(param, Categorial):
                self.discreteVariableValues.append(param.values)

    def Calculate(self, point, functionValue):
            arguments = self.__get_argument_dict(point)
            functionValue.value = -self.metric(arguments)
            return functionValue

    def __get_argument_dict(self, point):
            arguments = {}
            for name, type, value, log in zip(self.floatVariableNames, self.float_variables_types,
                                              point.floatVariables,
                                              self.is_log_float):
                value = np.exp(value) if log else value
                value = int(value) if type == 'int' else value
                arguments[name] = value
            if point.discreteVariables is not None:
                for name, value in zip(self.discreteVariableNames, point.discreteVariables):
                    arguments[name] = int(value) if value.isnumeric() else value
            return arguments


class iOptSearcher(Searcher):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'iOpt'

    def find_best_value(self):

        floats, discretes = self.split_hyperparams()
        problem = Estimator(self.estimator, floats, discretes, self.dataset, self.calculate_metric_with_log)
        framework_params = SolverParameters(itersLimit=self.max_iter)
        solver = Solver(problem, parameters=framework_params)
        solver_info = solver.Solve()
        return -solver_info.bestTrials[0].functionValues[-1].value
    
    def split_hyperparams(self):
        floats, discretes = {}, {}
        for name, x in self.hyperparams.items():
            if self.is_discrete_hyperparam(x):
                discretes[name] = x
            else:
                floats[name] = x
        return floats, discretes
            
    @staticmethod
    def is_discrete_hyperparam(x: Hyperparameter):
        if isinstance(x, Categorial):
            return True
        type, min_v, max_v, log = astuple(x)
        return (type == 'int') and (not log) and (max_v - min_v + 1 <= 100)