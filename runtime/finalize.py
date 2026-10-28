"""Recover post-training stages only. This program has NO training entry point."""
from __future__ import annotations
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import sys
from runtime.install import PACKAGE,sha
from runtime.experiment import environment,run_command
from runtime.preflight import verify_install
from metatool.release import model_identity


def finalize(work,run):
    verify_install(work)
    if run.resolve().parent != (work/'runs').resolve():raise ValueError('Wrong work/run binding')
    experiment=json.loads((run/'EXPERIMENT.json').read_text())
    for name,h in experiment['package_files'].items():
        if not (PACKAGE/name).is_file() or sha(PACKAGE/name)!=h:raise ValueError(f'Package changed after run registration: {name}')
    env=environment(work);env['METATOOL_RUN']=str(run)
    status=json.loads((run/'STATUS.json').read_text());stage='training_audit'
    def save():
        (run/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    def command(name,args,command_env=env,cwd=None):
        nonlocal stage
        stage=name;status.update(status='running',stage=stage);save()
        index=1
        while (run/f'{name}_{index:02d}.log').exists():index+=1
        run_command(args,run/f'{name}_{index:02d}.log',command_env,cwd)
    try:
        sys.path.insert(0,str(work/'src'))
        from metatool.rl.full_recovery import latest, read, atomic_json,retire_50_after_final
        committed=latest(run,allow_final_model_only=True)
        if committed['step']!=100:raise ValueError('No completed step100; use explicit resume for training')
        if not (run/'checkpoint_manifest.json').exists():
            atomic_json(run/'checkpoint_manifest.json',dict(best_step=100,path=committed['path'],
                selection={'step':100,'scores':None,'selection_rule':'fixed_final_step'},
                full_training_state=committed['full_training_state'],exact_training_resume=False,
                recovered_from_committed_full_marker=True))
        event=run/'checkpoint_events/step_100.json';event_doc=read(event)
        if event_doc['state']!='COMMITTED':
            atomic_json(run/'checkpoint_events/step_100_pre_reconcile.json',event_doc)
            atomic_json(event,{**event_doc,'state':'COMMITTED','reconciled_from_full_marker':True})
        retire_50_after_final(run)
        command('training_audit' ,[sys.executable,PACKAGE/'scripts/runtime_entry.py','summarize','--run',run,'--tokenizer',work/'model'])
        sys.path.insert(0,str(work/'src'))
        from metatool.rl.full_recovery import ensure_merge_space
        from metatool.rl.checkpoint_safety import estimate_model_bytes
        merged=run/'merged_hf';marker=run/'MERGE_COMPLETE.json' 
        if marker.exists():
            if json.loads(marker.read_text())!=model_identity(merged):raise ValueError('Merged model identity changed')
        else:
            if merged.exists() and any(merged.iterdir()):
                raise ValueError('Partial merge exists: preserve and inspect it before retrying merger; NEVER retrain')
            ensure_merge_space(run,estimate_model_bytes(work/'model'))
            command('merge',[sys.executable,'-m','verl.model_merger','merge','--backend','fsdp',
                '--local_dir',run/'checkpoints/global_step_100/actor','--target_dir',merged,
                '--use_cpu_initialization'],cwd=work/'src')
            marker.write_text(json.dumps(model_identity(merged),indent=2)+'\n')
        status.update(status='completed',stage='merged',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
        print(json.dumps({'completed_training_and_merge':True,'merged_model':str(merged),'evaluation':'python run.py evaluate'}))
    except BaseException as exc:
        status.update(status='failed',stage=stage,error=repr(exc));save();raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    with (a.run/'.finalize.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finalize(a.work.resolve(),a.run.resolve())
