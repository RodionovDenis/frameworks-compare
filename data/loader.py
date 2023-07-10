import requests
import zipfile
import os
import numpy as np
import numpy.typing as npt

from urllib.parse import urlencode
from typing import Literal
from abc import ABC, abstractclassmethod
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer, load_digits
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Dataset:
    name: str
    features: npt.NDArray
    targets: npt.NDArray
    type: Literal['classification', 'regression'] = 'classification'


class Parser(ABC):
    def __init__(self, path):
        self.path = Path(__file__).parent / 'datasets' / path
    
    @abstractclassmethod
    def load_dataset() -> Dataset:
        pass


class UCIParser(Parser):
    def __init__(self, path, *, spliter, skips='?', regression=False):
        super().__init__(f'uci-datasets/{path}')
        self.spliter = spliter
        self.skips = skips
        self.regression = regression

    def load_uci(self, sample_skip=None):
        with open(self.path) as f:
            data = []
            sample = -1
            while (row := f.readline()).strip():
                sample += 1
                if (sample_skip is not None) and (sample == sample_skip):
                    continue
                input_ = [x.strip() for x in row.split(self.spliter)]
                if self.skips in input_:
                    continue
                data.append(input_)
        return data
    
    def preprocess_features(self, features, skip=None):
        number_features = len(features[0])
        return np.array(
            [self.preprocess_feature([sample[i] for sample in features])
             for i in range(number_features) if (skip is None) or self.feature_skip_condition(i, skip)]
        ).T
    
    @staticmethod
    def feature_skip_condition(i, skip):
        return (isinstance(skip, int) and i != skip) or \
               (isinstance(skip, list) and i not in skip)
    
    def preprocess_feature(self, feature):
        try:
            return [float(value) for value in feature]
        except ValueError:
            return LabelEncoder().fit_transform(feature).tolist()

    def preprocess_target(self, targets):
        if self.regression:
            return [float(value) for value in targets]
        return LabelEncoder().fit_transform(targets)
    
    @staticmethod
    def separate_target(data, index_target: int | list[int]):
        if isinstance(index_target, int):
            index_target = [index_target]
        for i, x in enumerate(index_target):
            index_target[i] = len(data[0]) + x if x < 0 else x
        features, targets = [], []
        for sample in data:
            features.append([x for i, x in enumerate(sample) if i not in index_target])
            target = [sample[i] for i in index_target]
            if len(target) == 1:
                targets.append(target[0])
            else:
                targets.append(target.index('1'))
        return features, targets
    
    def parse_text_file(self, index_target, sample_skip=None, feature_skip=None):
        data = self.load_uci(sample_skip)
        features, targets = self.separate_target(data, index_target)
        return self.preprocess_features(features, skip=feature_skip), \
               self.preprocess_target(targets)
    

class BreastCancer:
    def load_dataset(self) -> Dataset:
        x, y = load_breast_cancer(return_X_y=True)
        return Dataset('Breast Cancer', x, y ^ 1)
    

class Digits:
    def load_dataset(self) -> Dataset:
        x, y = load_digits(return_X_y=True)
        return Dataset('Digits', x, y)


class Adult(UCIParser):
    def __init__(self):
        super().__init__('adult/adult.data', spliter=',')
    
    def load_dataset(self) -> Dataset:
        features, targets = self.parse_text_file(index_target=-1)
        return Dataset('Adult', features, targets)


class BankMarketing(UCIParser):
    def __init__(self):
        super().__init__('bank-marketing/bank.data', spliter=';')
    
    def load_dataset(self) -> Dataset:
        features, targets = self.parse_text_file(index_target=-1, sample_skip=0)
        return Dataset('Bank-marketing', features, targets)


class CNAE9(UCIParser):
    def __init__(self):
        super().__init__('cnae-9/CNAE-9.data', spliter=',')
    
    def load_dataset(self) -> Dataset:
        features, targets = self.parse_text_file(index_target=0)
        return Dataset('CNAE-9', features, targets)


class StatlogSegmentation(UCIParser):
    def __init__(self):
        super().__init__('statlog-segmentation/segment.dat', spliter=None)

    def load_dataset(self) -> Dataset:
        features, targets = self.parse_text_file(index_target=-1)
        return Dataset('Statlog Segmentation', features, targets)


class DryBean(UCIParser):
    def __init__(self):
        super().__init__('dry-bean/Dry_Bean_Dataset.csv', spliter=',')

    def load_dataset(self) -> Dataset:
        features, targets = self.parse_text_file(sample_skip=0,  index_target=-1)
        return Dataset('Dry Bean', features, targets)
    

class MagicGammaTelescope(UCIParser):
    def __init__(self):
        super().__init__('magic-gamma-telescope/magic04.data', spliter=',')

    def load_dataset(self) -> Dataset:
        features, targets = self.parse_text_file(index_target=-1)
        return Dataset('Magic Gamma Telescope', features, targets)
    

class Mushroom(UCIParser):
    def __init__(self):
        super().__init__('mushroom/agaricus-lepiota.data', spliter=',')

    def load_dataset(self) -> Dataset:
        features, targets = self.parse_text_file(index_target=0)
        return Dataset('Mushroom', features, targets)
    

class Semeion(UCIParser):
    def __init__(self):
        super().__init__('semeion/semeion.data', spliter=None)
    
    def load_dataset(self) -> Dataset:
        indexes = np.arange(-10, 0).tolist()
        features, targets = self.parse_text_file(index_target=indexes)
        return Dataset('Semeion', features, targets)
    

class WineQuality(UCIParser):
    def __init__(self):
        super().__init__('wine-quality/wine.csv', spliter=',', regression=True)
        
    def load_dataset(self) -> Dataset:
        features, targets = self.parse_text_file(index_target=-1, sample_skip=0, feature_skip=0)
        return Dataset('Wine Quality', features, targets, 'regression')


class Ecoli(UCIParser):
    def __init__(self):
        super().__init__('ecoli/ecoli.data', spliter=None)
        
    def load_dataset(self) -> Dataset:
        features, targets = self.parse_text_file(index_target=-1, feature_skip=0)
        indexes = np.isin(targets, [2, 3, 6], invert=True)
        return Dataset('Ecoli', features[indexes], 
                                LabelEncoder().fit_transform(targets[indexes]))


class CreditApproval(UCIParser):
    def __init__(self):
        super().__init__('credit-approval/crx.data', spliter=',')
        
    def load_dataset(self) -> Dataset:
        features, targets = self.parse_text_file(index_target=-1)
        return Dataset('Credit Approval', features, targets)


class Balance(UCIParser):
    def __init__(self):
        super().__init__('balance/balance-scale.data', spliter=',')
        
    def load_dataset(self) -> Dataset:
        features, targets = self.parse_text_file(index_target=0)
        return Dataset('Balance', features, targets)


class Parkinsons(UCIParser):
    def __init__(self):
        super().__init__('parkinsons/parkinsons.data', spliter=',')
    
    def load_dataset(self) -> Dataset:
        features, targets = self.parse_text_file(index_target=-7, feature_skip=0, sample_skip=0)
        return Dataset('Parkinsons', features, targets)
    

class Zoo(UCIParser):
    def __init__(self):
        super().__init__('zoo/zoo.data', spliter=',')
    
    def load_dataset(self) -> Dataset:
        features, targets = self.parse_text_file(index_target=-1, feature_skip=0)
        return Dataset('Zoo', features, targets)


class CylinderBands(UCIParser):
    def __init__(self):
        super().__init__('cylinder-bands/bands.data', spliter=',')
    
    def load_dataset(self) -> Dataset:
        features, targets = self.parse_text_file(index_target=-1, feature_skip=[0, 1])
        return Dataset('Cylinder Bands', features, targets)


class ConnectionBenchVowel(UCIParser):
    def __init__(self):
        super().__init__('connection-bench-vowel/vowel.data', spliter=',')
        
    def load_dataset(self) -> Dataset:
        features, targets = self.parse_text_file(index_target=-1, feature_skip=0)
        return Dataset('Connection Bench Vowel', features, targets)


def get_datasets(*args) -> list[Dataset]:
    result = []
    for dataset in args:
        instance = dataset()
        result.append(instance.load_dataset())
    return result


if __name__ == '__main__':
    
    path = Path(__file__).parent
    
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = 'https://disk.yandex.ru/d/1RHWkpKcmMWWcg'

    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']

    download_response = requests.get(download_url)
    with open(path / 'datasets.zip', 'wb') as f:
        f.write(download_response.content)

    with zipfile.ZipFile(path / 'datasets.zip') as f:
        f.extractall(path=path)

    os.remove(path / 'datasets.zip')
