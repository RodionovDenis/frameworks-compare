from abc import ABC, abstractclassmethod
from dataclasses import dataclass

from .metrics import Metric
from data.loader import Dataset

@dataclass
class Hyperparameter:
    name: str
    group: type
    min_value: float
    max_value: float


class Searcher(ABC):
    def __init__(self, estimator, hyperparams: list[Hyperparameter], max_iter: int):
        self.estimator = estimator
        self.hyperparams = hyperparams
        self.max_iter = max_iter

    @abstractclassmethod
    def tune(inputs, targets, metric: Metric):
        pass

    @abstractclassmethod
    def __str__():
        pass


class OptunaSearcher(Searcher):
    pass


class HyperoptSearcher(Searcher):
    pass


class iOptSearcher(Searcher):
    pass


def run_searchers(searchers: list[Searcher], datasets: list[Dataset]):
    pass




