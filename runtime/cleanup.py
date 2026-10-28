"""Retire only this RL run's checkpoints after verified model export."""
import shutil
from pathlib import Path
from metatool.release import atomic,read,sha,model_identity

def exported(work):
    run=Path(work)/'runs/metatool_seed42'
    return (run/'MERGE_COMPLETE.json').is_file() and (run/'TRAINING_SUMMARY.json').is_file()

def cleanup(work):
    work=Path(work).resolve();run=work/'runs/metatool_seed42'
    if not exported(work) or read(run/'TRAINING_SUMMARY.json').get('complete') is not True:
        raise ValueError('Complete training and export are required for checkpoint retirement')
    if (run/'merged_hf').is_symlink() or model_identity(run/'merged_hf')!=read(run/'MERGE_COMPLETE.json'):
        raise ValueError('Final exported model differs; checkpoints preserved')
    owner=read(run/'EXPERIMENT.json')['run_uid']
    targets=[run/'checkpoints/global_step_50',run/'checkpoints/global_step_100']
    for target in targets:
        ancestors=[target,*[a for a in target.parents if a!=work and work in a.parents]]
        if any(a.is_symlink() for a in ancestors) or (target.exists() and any(p.is_symlink() for p in target.rglob('*'))):
            raise ValueError('Symlink checkpoint retirement refused')
    record=dict(run_uid=owner,final_export_sha256=sha(run/'MERGE_COMPLETE.json'),
                retired=[str(t.relative_to(work)) for t in targets],exports_preserved=True,input_model_preserved=True)
    atomic(run/'CHECKPOINT_RETIREMENT_INTENT.json',record)
    for target in targets:
        if target.exists():shutil.rmtree(target)
    atomic(run/'CHECKPOINT_RETIREMENT_COMPLETE.json',record)
    return record
