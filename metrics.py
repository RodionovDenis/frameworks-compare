from abc import ABC, abstractclassmethod
from sklearn.model_selection import cross_val_score
from functools import partial


class Metric(ABC):
    @abstractclassmethod
    def __call__(self, estimator, inputs, targets, progress_bar=None):
        pass


class CrossValidation(Metric):
    def __init__(self, scoring, **kwargs):
        self.scoring = partial(scoring, **kwargs)
        self.name = f'Cross Validation, scoring {scoring.__name__}'
    
    def __call__(self, *args, progress_bar=None):
        estimator, inputs, targets = args
        value = cross_val_score(estimator, inputs, targets, scoring=self.get_score).mean()
        if progress_bar is not None:
            progress_bar.update(1)
        return value
    
    def get_score(self, *args):
        estimator, inputs, targets = args
        return self.scoring(targets, estimator.predict(inputs))
