import data.loader as loader 

from frameworks import Type, Hyperparameter, OptunaSearcher, HyperoptSearcher, iOptSearcher
from metrics import CrossValidation
from experiment import Experiment

from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    
    hyperparams = [
        Hyperparameter('n_estimators', Type.int(10, 200)),
        Hyperparameter('max_depth', Type.int(5, 20)),
        Hyperparameter('min_samples_split', Type.int(2, 10))
    ]

    seachers = [OptunaSearcher, HyperoptSearcher, iOptSearcher]
    parsers = [loader.BreastCancer, loader.Digits, loader.CNAE9, loader.StatlogSegmentation]

    experiment = Experiment(RandomForestClassifier, hyperparams, seachers, parsers,
                            CrossValidation(f1_score, average='weighted'))
    
    experiment.set_framework_arguments(max_iter=100)

    experiment.run(default_arguments=True, show_result=True, n_jobs=16, path_to_folder='result/randomforest')
