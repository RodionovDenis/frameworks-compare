import data.loader as loader

from data.loader import get_datasets
from frameworks import OptunaSearcher, HyperoptSearcher, iOptSearcher, get_frameworks
from metrics import CrossValidation
from experiment import Experiment
from hyperparameter import Numerical, Categorial

from sklearn.metrics import f1_score
from sklearn.svm import SVC


if __name__ == '__main__':

    hyperparams = {
        'gamma': Numerical('float', 1e-4, 1e-1, is_log_scale=True),
        'C': Numerical('int', 1, 1e6, is_log_scale=True),
        'kernel': Categorial('linear', 'poly', 'rbf', 'sigmoid')
    }

    frameworks = get_frameworks(OptunaSearcher, HyperoptSearcher, iOptSearcher,
                                max_iter=5)
    
    datasets = get_datasets(loader.BreastCancer)

    experiment = Experiment(SVC, hyperparams, frameworks, datasets,
                            CrossValidation(f1_score, average='binary'))

    experiment.run(n_jobs=2, mlflow_uri='http://127.0.0.1:5000')
