from abc import ABC, abstractclassmethod
from typing import Literal

from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from functools import partial
from data.loader import Dataset


class Metric(ABC):
    def __call__(self, estimator, dataset: Dataset):

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', estimator)
        ])

        return cross_val_score(pipeline, dataset.features, dataset.targets,
                               scoring=self.get_score).mean()

    @abstractclassmethod
    def get_score(self, model, x, y):
        pass


class Accuracy(Metric):
    def __init__(self):
        self.name = 'accuracy'

    def get_score(self, model, x, y):
        return accuracy_score(y, model.predict(x))


class F1(Metric):
    def __init__(self, average: Literal['binary', 'macro', 'micro']):
        self.average = average
        self.name = f'f1-score ({self.average})'
    
    def get_score(self, model, x, y):
        return f1_score(y, model.predict(x), average=self.average)


class RocAuc(Metric):
    def __init__(self, average: Literal['binary', 'macro', 'micro']):
        self.average = average
        self.name = f'roc-auc ({self.average})'
    
    def get_score(self, model, x, y):
        if self.average == 'binary':
            return roc_auc_score(y, model.predict_proba(x)[:, 1])
        else:
            return roc_auc_score(y, model.predict_proba(x), average=self.average, multi_class='ovr')


class MSE(Metric):
    def __init__(self):
        self.name = 'mse'
        
    def get_score(self, model, x, y):
        return mean_squared_error(y, model.predict(x))
