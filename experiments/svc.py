import data.loader as loader

from data.loader import get_datasets
from frameworks import Type, Hyperparameter, OptunaSearcher, HyperoptSearcher, iOptSearcher, get_frameworks
from metrics import CrossValidation
from experiment import Experiment

from sklearn.metrics import f1_score
from sklearn.svm import SVC


if __name__ == '__main__':

    hyperparams = [
        Hyperparameter('gamma', Type.float(1e-4, 1e-1, log=True)),
        Hyperparameter('C', Type.float(1e-1, 1e6, log=True))
    ]

    frameworks = get_frameworks(OptunaSearcher, HyperoptSearcher, iOptSearcher,
                                max_iter=200)
    
    datasets = get_datasets(loader.Adult)

    experiment = Experiment(SVC, hyperparams, frameworks, datasets,
                            CrossValidation(f1_score, average='binary'))

    experiment.run(default=True, show_result=True, n_jobs=4, path_to_folder='datasets_results/svc/adult-binary-f1')
