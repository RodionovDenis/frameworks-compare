import data.loader as loader

from data.loader import get_datasets
from frameworks import  OptunaSearcher, HyperoptSearcher, iOptSearcher, get_frameworks
from metrics import CrossValidation
from experiment import Experiment

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from hyperparameter import Numerical, Categorial


if __name__ == '__main__':
    
    hyperparams = {
        'n_estimators': Numerical('int', 2, 200),
        'criterion': Categorial('gini', 'entropy', 'log_loss'),
        'max_depth': Numerical('int', 2, 15),
        'min_samples_split': Numerical('int', 2, 10),
        'min_samples_leaf': Numerical('int', 2, 10),
    }

    seachers = get_frameworks(OptunaSearcher, HyperoptSearcher, iOptSearcher, max_iter=200)

    parsers = get_datasets(loader.CNAE9)

    experiment = Experiment(RandomForestClassifier, hyperparams, seachers, parsers,
                            CrossValidation(accuracy_score))

    experiment.run(n_jobs=4, mlflow_uri='http://192.168.220.17:8891')
