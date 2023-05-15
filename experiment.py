import pandas as pd
import json

from frameworks import Searcher, Hyperparameter, dict_factory
from data.loader import Parser, get_datasets
from metrics import Metric
from multiprocessing import Pool

from pathlib import Path
from tqdm import tqdm
from itertools import product
from dataclasses import asdict

class Experiment:
    def __init__(self, estimator,
                       hyperparams: list[Hyperparameter],
                       searchers: list[Searcher],
                       parsers: list[Parser],
                       metric: Metric):

        self.estimator = estimator
        self.searcher_classes = searchers
        self.datasets = get_datasets(*parsers)
        self.hyperparams = hyperparams
        self.metric = metric
        self.default_arguments = 'Default Arguments'

    
    def set_framework_arguments(self, **kwargs):
        assert 'max_iter' in kwargs, 'max_iter parameter must have'
        self.max_iter = kwargs['max_iter']
        self.seacher_instances = [searcher(**kwargs) for searcher in self.searcher_classes]

    def run(self, default_arguments=False, show_result=False, path_to_folder=None, n_jobs=1):

        assert 'seacher_instances' in self.__dict__, \
            'First you need to set the parameters using the set_framework_arguments method'

        columns = [self.default_arguments] * default_arguments + [x.name for x in self.seacher_instances]
        frame = pd.DataFrame(index=[x.name for x in self.datasets], columns=columns)

        len_searchers, len_dataset = len(columns), len(self.datasets)

        products = list(product(range(len_searchers), range(len_dataset)))

        with Pool(n_jobs) as pool:
            results = list(tqdm(pool.imap(self.objective, products), total=len(products)))
            for (i, j), value in zip(products, results):
                column = self.seacher_instances[i].name if i < len(self.seacher_instances) else self.default_arguments
                row = self.datasets[j].name
                frame[column][row] = value

        if show_result:
            print(frame)
        
        if isinstance(path_to_folder, str):
            self.__save(frame, path_to_folder)

        return frame
    
    def objective(self, args):
        dataset = self.datasets[args[1]]
        if args[0] == len(self.searcher_classes):
            return self.__default_arguments(dataset)
        searcher = self.seacher_instances[args[0]]
        return searcher.tune(self.estimator, self.hyperparams, dataset, self.metric)
    
    def __default_arguments(self, dataset):
        return self.metric(self.estimator(), dataset)

    def __save(self, frame, path_to_folder: str):
        metainfo = {
            'estimator': self.estimator.__name__,
            'max_iter': self.max_iter,
            'hyperparameters': [asdict(x, dict_factory=dict_factory) for x in  self.hyperparams],
            'metric': self.metric.name
        }
        path = Path(path_to_folder)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / 'metainfo.json', 'w') as f:
            f.write(json.dumps(metainfo, indent=4))
        frame.to_csv(path / 'scores.csv')
