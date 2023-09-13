from .interface import Searcher, Point

import sklearn
from hyperparameter import Hyperparameter
from data import Dataset
from metrics import Metric
from time import time


class Default(Searcher):
    def __init__(self, max_iter) -> None:
        super().__init__(framework_name='Default',
                         max_iter=max_iter,
                         is_deterministic=True)
        
    def framework_version(self):
        return f'{sklearn.__version__} (sklearn-version)'
    
    def _get_searcher_params(self):
        return {}
    
    def _get_points(self) -> list[Point]:

        model = self.estimator()
        value = self.metric(model, self.dataset)
        return [Point(time(), value, {}) for _ in range(self.max_iter)]
