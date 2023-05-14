from data.loader import CNAE9, BreastCancer
from frameworks import Type, Hyperparameter, OptunaSearcher, HyperoptSearcher, iOptSearcher
from metrics import CrossValidation
from experiment import Experiment

from sklearn.metrics import f1_score
from sklearn.svm import SVC


if __name__ == '__main__':
    hyperparams = [
        Hyperparameter('gamma', Type.float, 1e-5, 1e-2),
        Hyperparameter('C', Type.int, 1, 200)
    ]

    seachers = [HyperoptSearcher, OptunaSearcher, iOptSearcher]
    parsers = [CNAE9, BreastCancer]

    experiment = Experiment(SVC, hyperparams, seachers, parsers,
                            CrossValidation(f1_score, average='weighted'))
    
    experiment.set_framework_arguments(max_iter=5)

    experiment.run(default_arguments=True, show_result=True)
