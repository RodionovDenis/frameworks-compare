from experiment import Experiment
from argparser import parse_arguments
from frameworks import OptunaSearcher, HyperoptSearcher, iOptSearcher, Default


if __name__ == '__main__':
    arguments = parse_arguments()
    
    seachers = [
        Default(arguments.max_iter),
        OptunaSearcher(arguments.max_iter, algorithm='random'),
        OptunaSearcher(arguments.max_iter, algorithm='tpe'),
        OptunaSearcher(arguments.max_iter, algorithm='cmaes'),
        OptunaSearcher(arguments.max_iter, algorithm='nsgaii'),
        HyperoptSearcher(arguments.max_iter),
        iOptSearcher(arguments.max_iter, r=3, refine_solution=True,
                     proportion_of_global_iterations=0.75)
    ]

    experiment = Experiment(arguments.estimator,
                            arguments.hyperparams,
                            seachers,
                            arguments.dataset)
    
    experiment.run(non_deterministic_trials=10, n_jobs=60)
