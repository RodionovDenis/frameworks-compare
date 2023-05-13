import requests
import zipfile
import os
import numpy as np
import numpy.typing as npt

from urllib.parse import urlencode
from abc import ABC, abstractclassmethod
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class Dataset:
    name: str
    features: npt.NDArray
    targets: npt.NDArray


class Parser(ABC):
    def __init__(self, path):
        self.path = Path(__file__).parent / 'datasets' / path
    
    @abstractclassmethod
    def load_dataset() -> Dataset:
        pass


class UCIParser(Parser):
    def __init__(self, path, *, spliter, skips='?'):
        super().__init__(f'uci-datasets/{path}')
        self.spliter = spliter
        self.skips = skips

    def load_uci(self):
        with open(self.path) as f:
            data = []
            while (row := f.readline()).strip():
                input_ = [x.strip() for x in row.split(self.spliter)]
                if self.skips in input_:
                    continue
                data.append(input_)
        return data
    
    def preprocess_features(self, features, skip=None):
        number_features = len(features[0])
        return np.array(
            [self.preprocess_feature([sample[i] for sample in features])
             for i in range(number_features) if (skip is None) or (i != skip)]
        ).T
    
    def preprocess_feature(self, feature):
        try:
            return [float(value) for value in feature]
        except ValueError:
            return LabelEncoder().fit_transform(feature).tolist()

    def preprocess_target(self, targets):
        return LabelEncoder().fit_transform(targets)
    
    @staticmethod
    def separate_target(data, index_target):
        features, targets = [], []
        if index_target < 0:
            index_target = len(data[0]) + index_target
        for sample in data:
            features.append([x for i, x in enumerate(sample) if i != index_target])
            targets.append(sample[index_target])
        return features, targets
    
    def parse_text_file(self, index_target, feature_skip=None):
        data = self.load_uci()
        features, targets = self.separate_target(data, index_target)
        return self.preprocess_features(features, skip=feature_skip), \
               self.preprocess_target(targets)
    

class Adult(UCIParser):
    def __init__(self):
        super().__init__('adult/adult.data', spliter=',')
    
    def load_dataset(self) -> Dataset:
        features, targets = self.parse_text_file(-1)
        return Dataset('Adult', features, targets)


class CNAE9(UCIParser):
    def __init__(self):
        super().__init__('cnae-9/CNAE-9.data', spliter=',')
    
    def load_dataset(self) -> Dataset:
        features, targets = self.parse_text_file(0)
        return Dataset('CNAE-9', features, targets)
    

class BreastCancer(UCIParser):
    def __init__(self):
        super().__init__('breast cancer/breast-cancer-wisconsin.data', spliter=',')
    
    def load_dataset(self) -> Dataset:
        features, targets = self.parse_text_file(-1, feature_skip=0)
        return Dataset('Breast Cancer', features, targets)


def get_datasets(*args) -> list[Dataset]:
    result = []
    for dataset in args:
        instance = dataset()
        result.append(instance.load_dataset())
    return result


if __name__ == '__main__':
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = 'https://disk.yandex.ru/d/1RHWkpKcmMWWcg'

    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']

    download_response = requests.get(download_url)
    with open('datasets.zip', 'wb') as f:
        f.write(download_response.content)

    with zipfile.ZipFile('datasets.zip') as f:
        f.extractall()

    os.remove('datasets.zip')
