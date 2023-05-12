from abc import ABC, abstractclassmethod
from sklearn.model_selection import cross_val_score


class Metric(ABC):
    @abstractclassmethod
    def __call__(self, estimator, inputs, targets):
        pass

    @abstractclassmethod
    def __str__(self):
        pass


class CrossValidation(Metric):
    def __init__(self, scoring):
        self.scoring = scoring
    
    def __call__(self, *args):
        estimator, inputs, targets = args
        return cross_val_score(estimator, inputs, targets, scoring=self.scoring)
    
    def get_score(self, *args):
        estimator, inputs, targets = args
        return self.scoring(targets, estimator.predict(inputs))
