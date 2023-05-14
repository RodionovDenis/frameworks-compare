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
        self.searcher_classes = searchers
        self.datasets = get_datasets(*parsers)
        self.hyperparams = hyperparams
        self.metric = metric
        self.default_arguments = 'Default Arguments'

    
    def set_framework_arguments(self, **kwargs):
        assert 'max_iter' in kwargs, 'max_iter parameter must have'
        self.seacher_instances = [searcher(**kwargs) for searcher in self.searcher_classes]
        self.max_iter = kwargs['max_iter']

    def run(self, default_arguments=False):

        assert 'seacher_instances' in self.__dict__, \
            'First you need to set the parameters using the set_framework_arguments method'

        columns = [self.default_arguments] * default_arguments + [x.name for x in self.seacher_instances]
        frame = pd.DataFrame(index=[x.name for x in self.datasets], columns=columns)

        total = len(self.seacher_instances) * len(self.datasets) * self.max_iter \
            + default_arguments * len(self.datasets)

        with tqdm(total=total) as progress_bar:

            metric = partial(self.metric, progress_bar=progress_bar)

            for searcher, dataset in product(self.seacher_instances, self.datasets):
                arguments = self.estimator, self.hyperparams, dataset, metric
                frame[searcher.name][dataset.name] = searcher.tune(*arguments)
            
            if default_arguments:
                self.__calculate_default_arguments(frame, metric)

        return frame
    
    def __calculate_default_arguments(self, frame, metric: Metric):
        for dataset in self.datasets:
            frame[self.default_arguments][dataset.name] = metric(self.estimator(), dataset)
