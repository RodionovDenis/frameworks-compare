from data import CNAE9, Digits
from frameworks import  Default, OptunaSearcher, HyperoptSearcher, iOptSearcher
from metrics import Accuracy
from experiment import Experiment
from hyperparameter import Numerical, Categorial

from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    
    hyperparams = {
        'n_estimators': Numerical('int', 2, 200),
        'criterion': Categorial('gini', 'entropy', 'log_loss'),
        'max_depth': Numerical('int', 2, 15),
        'min_samples_split': Numerical('int', 2, 10),
        'min_samples_leaf': Numerical('int', 2, 10),
    }

    max_iter = 5

    seachers = [
        Default(max_iter),
        OptunaSearcher(max_iter),
        HyperoptSearcher(max_iter),
        iOptSearcher(max_iter)
    ]

    datasets = [CNAE9, Digits]

    experiment = Experiment(RandomForestClassifier, hyperparams, seachers, datasets,
                            Accuracy(preprocessing='standard'))

    result = experiment.run(n_jobs=6, non_deterministic_trials=2)
