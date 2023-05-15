import data.loader as loader 

from frameworks import Type, Hyperparameter, OptunaSearcher, HyperoptSearcher, iOptSearcher
from metrics import CrossValidation
from experiment import Experiment

from sklearn.metrics import f1_score
from sklearn.svm import SVC


if __name__ == '__main__':
    
    hyperparams = [
        Hyperparameter('gamma', Type.float(1e-3, 1e-1, log=True)),
        Hyperparameter('C', Type.int(1, 1e6, log=True))
    ]

    seachers = [OptunaSearcher, HyperoptSearcher, iOptSearcher]
    parsers = [loader.BreastCancer, loader.Digits, loader.CNAE9, loader.StatlogSegmentation]

    experiment = Experiment(SVC, hyperparams, seachers, parsers,
                            CrossValidation(f1_score, average='weighted'))
    
    experiment.set_framework_arguments(max_iter=10)

    experiment.run(default_arguments=True, show_result=True, n_jobs=16, path_to_folder='result/svc')
