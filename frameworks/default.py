from .interface import Searcher

import mlflow


class Default(Searcher):
    def __init__(self, *args) -> None:
        super().__init__(*args, name='Default', is_deterministic=True)
    
    def find_best_value(self):
        model = self.estimator()
        value = self.metric(model, self.dataset)
        if mlflow.active_run() is not None:
            self.log_values(value)
        return value
    
    def searcher_params(self):
        pass
    
    def log_values(self, value):
        for _ in range(self.max_iter):
            self.current_step += 1
            mlflow.log_metric(self.log_name, value, step=self.current_step)
