"""Create an isolated RL work directory from an explicit local model."""
import shutil
from pathlib import Path
from metatool.release import ROOT as PACKAGE,sha,read,atomic,verify_release,disjoint_work,model_identity
from runtime.model import bind_input

def install(work,model,train_data,validation_data):
    verify_release();work=disjoint_work(work);model=Path(model).resolve()
    if work==model or work in model.parents or model in work.parents:
        raise ValueError('Model and work must be disjoint')
    if work.exists():raise FileExistsError('Use a new work directory; existing runs are never overwritten')
    from runtime.datasets import bind_datasets
    inputs=bind_datasets(train_data,validation_data)
    initial=bind_input(model)
    work.mkdir(parents=True,exist_ok=False);(work/'src').mkdir()
    for src,dst in [(PACKAGE/'vendor/verl/verl',work/'src/verl'),(PACKAGE/'metatool',work/'src/metatool')]:
        shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (work/'data').mkdir()
    for name,info in inputs.items():(work/'data'/name).symlink_to(info['path'])
    (work/'model').symlink_to(model,target_is_directory=True)
    (work/'assets').mkdir();shutil.copy2(PACKAGE/'configs/route.json',work/'assets/branch_protocol.json')
    record=dict(initial_model=initial,dataset_inputs=inputs,
        source={str(p.relative_to(work/'src')):sha(p) for p in sorted((work/'src').rglob('*')) if p.is_file()},
        dataset_sha256={n:sha(work/'data'/n) for n in ['train.parquet','validation.parquet']},
        protocol_sha256=sha(work/'assets/branch_protocol.json'))
    atomic(work/'INSTALL.json',record);return record
