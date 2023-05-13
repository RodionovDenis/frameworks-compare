import pandas as pd

from frameworks import Searcher, Hyperparameter
from data.loader import Parser, get_datasets
from metrics import Metric
from tqdm import tqdm
from itertools import product
from functools import partial

class Experiment:
    def __init__(self, estimator, 
                       hyperparams: list[Hyperparameter],
                       searchers: list[Searcher], 
                       parsers: list[Parser],
                       metric: Metric):
        self.estimator = estimator
        self.searchers = searchers
        self.parsers = parsers
        self.hyperparams = hyperparams
        self.metric = metric

    def run(self, *args, **kwargs):
        searchers = [searcher(*args, **kwargs) for searcher in self.searchers]
        datasets = get_datasets(*self.parsers)
        frame = pd.DataFrame(index=[x.name for x in datasets], 
                             columns=[x.name for x in searchers])
        with tqdm(total=len(searchers) * len(datasets) * kwargs['max_iter']) as progress_bar:
            metric = partial(self.metric, progress_bar=progress_bar)
            for searcher, dataset in product(searchers, datasets):
                arguments = self.estimator, self.hyperparams, dataset.features, dataset.targets, metric
                frame[searcher.name][dataset.name] = searcher.tune(*arguments)
        return frame
