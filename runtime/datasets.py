"""Bind external frozen RL inputs. No RL training examples are bundled."""
from pathlib import Path
from metatool.release import ROOT,read,sha

def bind_datasets(train_data,validation_data):
    paths={'train.parquet':Path(train_data).resolve(),'validation.parquet':Path(validation_data).resolve()}
    if paths['train.parquet']==paths['validation.parquet']:raise ValueError('Training and validation inputs must be separate files')
    expected=read(ROOT/'configs/dataset_profile.json');result={}
    for name,path in paths.items():
        if not path.is_file():raise FileNotFoundError(f'External RL data is missing: {path}')
        actual=sha(path)
        if actual!=expected[name]['sha256']:
            raise ValueError(f'{name} differs from the frozen RL dataset; API-Bank evaluation JSON must not be substituted')
        result[name]=dict(path=str(path),sha256=actual,rows=expected[name]['rows'])
    return result
