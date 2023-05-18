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
                       metric: Metric,
                       *,
                       path_to_folder=None,
                       n_jobs=1,
                       ):

        self.estimator = estimator
        self.searchers = searchers
        self.datasets = get_datasets(*parsers)
        self.hyperparams = hyperparams
        self.metric = metric

        self.path_to_folder = path_to_folder
        self.n_jobs = n_jobs
        
        self.default_arguments = 'Default Arguments'

    def run(self, max_iter, *, default_arguments=False, show_result=False):
        
        self.searcher_instances = self.__initialized_searchers(max_iter=max_iter)

        columns = [self.default_arguments] * default_arguments + [x.name for x in self.searcher_instances]
        indexes = [x.name for x in self.datasets]
        frame = pd.DataFrame(index=indexes, columns=columns)
        
        len_searchers, len_dataset = len(columns), len(self.datasets)
        products = list(product(range(len_searchers), range(len_dataset)))

        with Pool(self.n_jobs) as pool:
            results = list(tqdm(pool.imap(self.objective, products), total=len(products)))
            for (i, j), value in zip(products, results):
                if i < len(self.searcher_instances):
                    column = self.searcher_instances[i].name 
                else:
                    column = self.default_arguments

                row = self.datasets[j].name
                frame[column][row] = value

        if show_result:
            print(frame)
        
        if isinstance(self.path_to_folder, str):
            self.__save(frame, self.path_to_folder)

        return frame
    
    def __initialized_searchers(self, *args, **kwargs):
        return [searcher(*args, **kwargs) for searcher in self.searchers]
    
    def objective(self, args):
        idx_searcher, idx_dataset = args
        dataset = self.datasets[idx_dataset]

        if idx_searcher == len(self.searchers):
            return self.metric(self.estimator(), dataset)

        searcher = self.searcher_instances[idx_searcher]
        return searcher.tune(self.estimator, self.hyperparams, dataset, self.metric)

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
