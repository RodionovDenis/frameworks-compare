import data.loader as loader 

from frameworks import Type, Hyperparameter, OptunaSearcher, HyperoptSearcher, iOptSearcher
from metrics import CrossValidation
from experiment import Experiment

from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from functools import partial


if __name__ == '__main__':
    
    hyperparams = [
        Hyperparameter('n_estimators', Type.int(10, 200)),
        Hyperparameter('max_depth', Type.int(5, 15)),
        Hyperparameter('l2_leaf_reg', Type.int(1, 10)),
        Hyperparameter('learning_rate', Type.float(0.001, 0.1, log=True)),
    ]

    seachers = [OptunaSearcher, HyperoptSearcher, iOptSearcher]

    parsers = [loader.BreastCancer, loader.Digits, loader.CNAE9, loader.StatlogSegmentation, loader.Adult,
               loader.BankMarketing, loader.DryBean, loader.MagicGammaTelescope, loader.Mushroom]
    
    catboost_classifier = partial(CatBoostClassifier, logging_level='Silent', allow_writing_files=False)

    experiment = Experiment(catboost_classifier, hyperparams, seachers, parsers,
                            CrossValidation(f1_score, average='weighted'))

    experiment.run(100,
                   default_arguments=True, show_result=True)
