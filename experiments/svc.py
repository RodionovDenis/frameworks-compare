from data import CNAE9, Digits
from frameworks import  Default, OptunaSearcher, HyperoptSearcher, iOptSearcher
from metrics import Accuracy
from experiment import Experiment
from hyperparameter import Numerical, Categorial

from sklearn.svm import SVC


if __name__ == '__main__':

    hyperparams = {
        'gamma': Numerical('float', 1e-4, 1e-1, is_log_scale=True),
        'C': Numerical('int', 1, 1e6, is_log_scale=True),
        'kernel': Categorial('linear', 'poly', 'rbf', 'sigmoid')
    }

    max_iter = 5

    seachers = [
        Default(max_iter),
        OptunaSearcher(max_iter),
        HyperoptSearcher(max_iter),
        iOptSearcher(max_iter)
    ]

    datasets = [CNAE9, Digits]

    experiment = Experiment(SVC, hyperparams, seachers, datasets,
                            Accuracy(preprocessing='standard'))

    result = experiment.run(n_jobs=6, non_deterministic_trials=2)
