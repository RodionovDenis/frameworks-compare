from data import CNAE9, Digits
from frameworks import  Default, OptunaSearcher, HyperoptSearcher, iOptSearcher
from metrics import Accuracy
from experiment import Experiment
from hyperparameter import Numerical

from xgboost import XGBClassifier
from functools import partial


if __name__ == '__main__':
    
    hyperparams = {
        'n_estimators': Numerical('int', 10, 200),
        'max_depth': Numerical('int', 5, 20),
        'min_child_weight': Numerical('int', 1, 10),
        'gamma': Numerical('float', 0.01, 0.6),
        'subsample': Numerical('float', 0.05, 0.95),
        'colsample_bytree': Numerical('float', 0.05, 0.95),
        'learning_rate': Numerical('float', 0.001, 0.1, is_log_scale=True)
    }

    max_iter = 2

    seachers = [
        Default(max_iter),
        OptunaSearcher(max_iter),
        HyperoptSearcher(max_iter),
        iOptSearcher(max_iter)
    ]

    datasets = [CNAE9]

    classifier = partial(XGBClassifier, n_jobs=1)

    experiment = Experiment(classifier, hyperparams, seachers, datasets,
                            Accuracy(preprocessing='standard'))

    result = experiment.run(n_jobs=6, non_deterministic_trials=2, mlflow_uri='http://127.0.0.1:5000')
