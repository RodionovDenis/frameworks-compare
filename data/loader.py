from abc import ABC, abstractclassmethod


class Dataset(ABC):
    @abstractclassmethod
    def load_dataset():
        pass


def get_datasets(datasets: list[Dataset]):
    pass
