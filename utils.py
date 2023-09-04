import subprocess
import pandas as pd
from matplotlib.pyplot import savefig


def get_commit_hash(path_repository: str = '.'):
    try:
        result = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                         cwd=path_repository)
        return result.decode('ascii').strip()
    except subprocess.CalledProcessError:
        pass


def save_result(filename: str, dataframe: pd.DataFrame):
    axes = dataframe.plot.bar(figsize=(10, 10))
    axes.legend(loc='lower center')
    savefig(filename)
    
