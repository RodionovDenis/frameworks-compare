import subprocess

def get_commit_hash(path_repository: str = '.'):
    try:
        result = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                         cwd=path_repository)
        return result.decode('ascii').strip()
    except subprocess.CalledProcessError:
        pass
