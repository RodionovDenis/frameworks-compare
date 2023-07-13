from data import CNAE9, Digits, BreastCancer
from frameworks import  Default, OptunaSearcher, HyperoptSearcher, iOptSearcher
from metrics import Accuracy, F1
from experiment import Experiment
from hyperparameter import Numerical, Categorial

from sklearn.neural_network import MLPClassifier


if __name__ == '__main__':

    hyperparams = {
        'hidden_layer_sizes': Numerical('int', 2, 30),
        'activation': Categorial('identity', 'logistic', 'tanh', 'relu'),
        'solver': Categorial('lbfgs', 'sgd', 'adam'),
        'max_iter': Numerical('int', 100, 700),
    }

    max_iter = 300

    seachers = [
        Default(max_iter),
        OptunaSearcher(max_iter, is_deterministic=False),
        HyperoptSearcher(max_iter, is_deterministic=False),
        iOptSearcher(max_iter)
    ]

    datasets = [CNAE9]

    experiment = Experiment(MLPClassifier, hyperparams, seachers, datasets,
                            Accuracy(preprocessing='standard'))

    result = experiment.run(n_jobs=4, non_deterministic_trials=5, mlflow_uri='http://94.228.124.235:5000/')
