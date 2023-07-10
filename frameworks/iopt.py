import numpy as np
import iOpt

from iOpt.problem import Problem
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters

from dataclasses import astuple
from hyperparameter import Hyperparameter, Numerical, Categorial
from utils import get_commit_hash

from .interface import Searcher


class Estimator(Problem):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.estimator, float_hyperparams, discrete_hyperparams, self.dataset, self.metric = args
        self.is_regression = kwargs['is_regression']

        self.number_of_float_variables = len(float_hyperparams)
        self.number_of_discrete_variables = len(discrete_hyperparams)
        self.dimension = len(float_hyperparams) + len(discrete_hyperparams)
        self.number_of_objectives = 1

        self.float_variables_types, self.is_log_float = [], []
        for name, param in float_hyperparams.items():
            self.float_variable_names.append(name)
            type, min_v, max_v, log = astuple(param)
            self.float_variables_types.append(type)
            self.lower_bound_of_float_variables.append(np.log(min_v) if log else min_v)
            self.upper_bound_of_float_variables.append(np.log(max_v) if log else max_v)
            self.is_log_float.append(log)
            
        for name, param in discrete_hyperparams.items():
            self.discrete_variable_names.append(name)
            if isinstance(param, Numerical):
                type, min_v, max_v, log = astuple(param)
                assert type == 'int', 'Type must be int'
                assert not log, 'Log must be off'
                self.discrete_variable_values.append([str(x) for x in range(min_v, max_v + 1)])
            elif isinstance(param, Categorial):
                self.discrete_variable_values.append(param.values)

    def calculate(self, point, function_value):
            arguments = self.__get_argument_dict(point)
            value = self.metric(arguments)
            function_value.value = value if self.is_regression else -value
            return function_value

    def __get_argument_dict(self, point):
            arguments = {}
            for name, type, value, log in zip(self.float_variable_names, self.float_variables_types,
                                              point.float_variables,
                                              self.is_log_float):
                value = np.exp(value) if log else value
                value = int(value + 0.5) if type == 'int' else value
                arguments[name] = value
            if point.discrete_variables is not None:
                for name, value in zip(self.discrete_variable_names, point.discrete_variables):
                    arguments[name] = int(value) if value.isnumeric() else value
            return arguments


class iOptSearcher(Searcher):
    def __init__(self, max_iter, *, is_deterministic=True, **kwargs):
        super().__init__(framework_name='iOpt',
                         max_iter=max_iter,
                         is_deterministic=is_deterministic)

        self.kwargs = kwargs

    def find_best_value(self):

        floats, discretes = self.split_hyperparams()
        problem = Estimator(self.estimator, floats, discretes, self.dataset, self.calculate_metric,
                            is_regression=self.dataset.type == 'regression')
        framework_params = SolverParameters(iters_limit=self.max_iter, **self.kwargs)
        solver = Solver(problem, parameters=framework_params)
        solver_info = solver.solve()
        return np.abs(solver_info.best_trials[0].function_values[-1].value)
    
    def get_searcher_params(self):
        return self.kwargs.copy()
    
    def framework_version(self):
        return get_commit_hash(iOpt.__path__[0])
    
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
        return (type == 'int') and (not log) and (max_v - min_v + 1 <= 5)
