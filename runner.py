from experiment import Experiment
from argparser import parse_arguments
from frameworks import Default, OptunaSearcher, HyperoptSearcher, iOptSearcher


if __name__ == '__main__':
    arguments = parse_arguments()
    
    seachers = [
        Default(arguments.max_iter),
        OptunaSearcher(arguments.max_iter, algorithm='random'),
        OptunaSearcher(arguments.max_iter, algorithm='tpe'),
        OptunaSearcher(arguments.max_iter, algorithm='cmaes'),
        OptunaSearcher(arguments.max_iter, algorithm='nsgaii'),
        HyperoptSearcher(arguments.max_iter),
        iOptSearcher(arguments.max_iter)
    ]

    experiment = Experiment(arguments.estimator,
                            arguments.hyperparams,
                            seachers,
                            arguments.dataset)
    
    result = experiment.run()
