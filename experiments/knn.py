from data import BreastCancer
from frameworks import  Default, OptunaSearcher, HyperoptSearcher, iOptSearcher
from metrics import F1
from experiment import Experiment
from hyperparameter import Numerical, Categorial

from sklearn.neighbors import KNeighborsClassifier


if __name__ == '__main__':

    hyperparams = {
        'n_neighbors': Numerical('int', 3, 20),
        'weights': Categorial('uniform', 'distance'),
        'algorithm': Categorial('auto', 'ball_tree', 'kd_tree', 'brute'),
        'p': Numerical('int', 1, 10)
    }

    max_iter = 300

    seachers = [
        Default(max_iter),
        OptunaSearcher(max_iter),
        HyperoptSearcher(max_iter),
        iOptSearcher(max_iter)
    ]

    datasets = [BreastCancer]

    experiment = Experiment(KNeighborsClassifier, 
                            hyperparams, 
                            seachers, 
                            datasets,
                            F1(preprocessing=None, average='binary'))

    result = experiment.run(n_jobs=12, non_deterministic_trials=5)

