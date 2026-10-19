import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def provenance(config, root=None):
    root = Path(root or Path(__file__).resolve().parents[2])
    files = list((root/'utils').rglob('*.py')) + list((root/'configuration').glob('*.py'))
    files += [root/name for name in ['requirements.txt','run_experiments.py','main.py'] if (root/name).exists()]
    hashes = {str(path.relative_to(root)).replace('\\','/'):file_hash(path) for path in sorted(files)}
    def git(*args):
        try:
            return subprocess.check_output(['git','-C',str(root),*args], stderr=subprocess.DEVNULL,
                                           text=True, timeout=10).strip()
        except (OSError,subprocess.SubprocessError):
            return None
    packages = {}
    for name in ['numpy','pandas','scikit-learn','scipy','shap','lime','xgboost','numba']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'unavailable'
    data = Path(config['data_path']).resolve()
    payload = {'protocol':'controlled-xai-v4','code_sha256':hashes,
               'data_path':str(data),'data_sha256':file_hash(data),'config':config,
               'packages':packages,'python':platform.python_version(),
               'git_commit':git('rev-parse','HEAD'),'git_branch':git('branch','--show-current'),
               'git_status':git('status','--short'),'platform':platform.platform()}
    fingerprint = {'code':hashes,'data':payload['data_sha256'],'config':config,'packages':packages,
                   'python':payload['python'],'platform':payload['platform']}
    payload['fingerprint'] = hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
    return payload
