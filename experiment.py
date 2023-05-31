import pandas as pd
import json

from frameworks import Searcher, Hyperparameter
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
                       frameworks: list[Searcher],
                       datasets: list[Parser],
                       metric: Metric):

        self.estimator = estimator
        self.frameworks = {x.name: x for x in frameworks}
        self.datasets = {x.name: x for x in datasets}
        self.hyperparams = hyperparams
        self.metric = metric
        
        self.default = 'Default'

    def run(self, path_to_folder=None, default=False, show_result=False, n_jobs=1):

        columns = [self.default] * default + list(self.frameworks)
        indexes = list(self.datasets)
        frame = pd.DataFrame(index=indexes, columns=columns)

        products = list(product(columns, indexes))

        with Pool(n_jobs) as pool:
            result = tqdm(pool.imap_unordered(self.objective, products), total=len(products))
            for (column, row), value in result:
                frame[column][row] = value

        if show_result:
            print(frame)
        
        if isinstance(path_to_folder, str):
            self.__save(frame, path_to_folder)

        return frame
    
    def objective(self, args):
        framework, dataset = self.frameworks.get(args[0]), self.datasets.get(args[1])
        if args[0] == self.default:
            value = self.metric(self.estimator(), dataset)
        else:
            value = framework.tune(self.estimator, self.hyperparams, dataset, self.metric)
        return args, value

    def __save(self, frame, path_to_folder: str):
        metainfo = {
            'estimator': self.estimator.__name__,
            'max_iter': next(iter(self.frameworks.values())).max_iter,
            'hyperparameters': [asdict(x, dict_factory=dict_factory) for x in  self.hyperparams],
            'metric': self.metric.name
        }
        path = Path(path_to_folder)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / 'metainfo.json', 'w') as f:
            f.write(json.dumps(metainfo, indent=4))
        frame.to_csv(path / 'scores.csv')

def dict_factory(data):
    result = {}
    for name, value in data:
        if isinstance(value, dict):
            result.update(value)
        elif isinstance(value, type):
            result[name] = value.__name__
        else:
            result[name] = value
    return result
