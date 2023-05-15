import data.loader as loader 

from frameworks import Type, Hyperparameter, OptunaSearcher, HyperoptSearcher, iOptSearcher
from metrics import CrossValidation
from experiment import Experiment

from sklearn.metrics import f1_score
from xgboost import XGBClassifier


if __name__ == '__main__':
    
    hyperparams = [
        Hyperparameter('n_estimators', Type.int(10, 200)),
        Hyperparameter('max_depth', Type.int(5, 20)),
        Hyperparameter('min_child_weight', Type.int(1, 10)),
        Hyperparameter('gamma', Type.float(0.01, 0.6)),
        Hyperparameter('subsample', Type.float(0.05, 0.95)),
        Hyperparameter('colsample_bytree', Type.float(0.05, 0.95)),
        Hyperparameter('learning_rate', Type.float(0.001, 0.1)),
    ]

    seachers = [OptunaSearcher, HyperoptSearcher, iOptSearcher]
    parsers = [loader.BreastCancer, loader.Digits, loader.CNAE9, loader.StatlogSegmentation]

    experiment = Experiment(XGBClassifier, hyperparams, seachers, parsers,
                            CrossValidation(f1_score, average='weighted'))
    
    experiment.set_framework_arguments(max_iter=100)

    experiment.run(default_arguments=True, show_result=True, n_jobs=1, path_to_folder='result/xgboost')
