"""One fixed 100-step training run with explicit, audited full-state recovery."""
from __future__ import annotations
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from runtime.install import PACKAGE,sha

def environment(work):
    env=os.environ.copy()
    env.pop('RAY_ADDRESS',None)  # every recovery attempt gets fresh Ray workers
    env.update(VLLM_USE_V1='0',PYTHONHASHSEED='42',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HYDRA_FULL_ERROR='1',  # Preserve the successful metatool recovery's full traceback setting.
        CUDA_LAUNCH_BLOCKING='0',TORCH_SHOW_CPP_STACKTRACES='1',PYTHONFAULTHANDLER='1',
        RAY_DEDUP_LOGS='0',METATOOL_CUDA_SYNC_BOUNDARIES='1',
        EXPERIMENT_NAME='qwen3_8B',WITHLENGTH='0',REFINEDREWARD='0',COARSEREWARD='0',STRICTMATCH='0',
        CORRECTMAX1='0',MAX1STEP30MAX3='0',SCHEDULEREWARD='1',SCHEDULELENGTH='0',
        USE_THINKING_SEMANTIC='0',INTERMEDIATEREWARD='0',METATOOL_QUIET_REWARD='1',
        TMPDIR=str(Path('/tmp')/('metatool_'+sha(work/'INSTALL.json')[:8])),
        RAY_TMPDIR=str(Path('/tmp')/('mt_ray_'+sha(work/'INSTALL.json')[:8])),
        PYTHONPATH=str(work/'src')+os.pathsep+str(PACKAGE),
        METATOOL_WRAP_DEADLINE_TS=os.environ.get('METATOOL_WRAP_DEADLINE_TS','inf'))
    for key,leaf in [('PIP_CACHE_DIR','pip'),('TORCH_HOME','torch'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),
                     ('VLLM_CACHE_ROOT','vllm'),('XDG_CACHE_HOME','xdg'),('HF_HOME','hf')]:
        env[key]=str(work/'cache'/leaf)
    for key in ['TMPDIR','RAY_TMPDIR','PIP_CACHE_DIR','TORCH_HOME','TRITON_CACHE_DIR',
                'VLLM_CACHE_ROOT','XDG_CACHE_HOME','HF_HOME','TORCHINDUCTOR_CACHE_DIR']:
        Path(env[key]).mkdir(parents=True,exist_ok=True)
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    return env

def run_command(command,log,env,cwd=None):
    print(json.dumps({'stage_log':str(log),'command':list(map(str,command))}),flush=True)
    with log.open('x',encoding='utf-8') as stream:
        done=subprocess.run(list(map(str,command)),env=env,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT)
    if done.returncode: raise RuntimeError(f'Command failed ({done.returncode}); preserve and inspect {log}')

def main():
    import uuid
    from runtime.preflight import verify_install
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--run-id',default='metatool_seed42')
    p.add_argument('--training-budget-authorized',action='store_true')
    p.add_argument('--resume',action='store_true',help='Explicitly continue latest committed checkpoint; never auto-retry CUDA failure')
    p.add_argument('--retry-preflight',action='store_true')
    a=p.parse_args();work=a.work.resolve()
    if not a.training_budget_authorized:raise ValueError('Pass --training-budget-authorized for the already-approved single run')
    if a.resume and a.retry_preflight:raise ValueError('Choose resume or retry-preflight')
    if '/' in a.run_id or a.run_id in {'.','..'} or not a.run_id.strip() or any(c.isspace() for c in str(work)):
        raise ValueError('Invalid work/run path')
    install_record=verify_install(work)
    sys.path.insert(0,str(work/'src'))
    from metatool.rl import full_recovery as recovery
    run=work/'runs'/a.run_id
    if not a.resume and not a.retry_preflight:run.mkdir(parents=True,exist_ok=False)
    lock=(run/'.launcher.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=environment(work);env.update(METATOOL_RUN=str(run),METATOOL_FULL_RECOVERY='1',METATOOL_RUNTIME_MODE='training')
    env.pop('METATOOL_RESUME_CHECKPOINT',None)
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    stage='preflight'
    if a.resume or a.retry_preflight:
        spec=recovery.read(run/'EXPERIMENT.json')
        if spec['recovery_schema']!=recovery.SCHEMA:raise ValueError('Old runs cannot be relabelled as full recovery')
        for name,h in spec['package_files'].items():
            if sha(PACKAGE/name)!=h:raise ValueError(f'Package changed since run registration: {name}')
        if a.retry_preflight:
            prior=recovery.read(run/'STATUS.json')
            if prior['stage']!='preflight' or (run/'training_updates').exists():raise ValueError('Retry-preflight is only for a failed CPU/setup gate')
    else:
        spec=json.loads((PACKAGE/'configs/rl.json').read_text())
        spec.update(run_uid=uuid.uuid4().hex,recovery_schema=recovery.SCHEMA,start_utc=start,
                    command=['bash',str(PACKAGE/'scripts/train_rl.sh'),str(work),str(run)],
                    package_files=json.loads((PACKAGE/'MANIFEST.json').read_text()))
        spec['initial_model']=install_record['initial_model']
        spec['dataset_inputs']=install_record['dataset_inputs']
        recovery.atomic_json(run/'EXPERIMENT.json',spec)
    attempts=run/'attempts';attempts.mkdir(exist_ok=True)
    index=0
    while (attempts/f'attempt_{index:03d}').exists():index+=1
    attempt=attempts/f'attempt_{index:03d}';attempt.mkdir()
    env.update(METATOOL_ATTEMPT_DIR=str(attempt),METATOOL_COMPILER_ERROR_DIR=str(attempt/'compiler_errors'),
               METATOOL_CUDA_TRACE_DIR=str(attempt/'cuda_runtime'))
    status=dict(status='running',stage=stage,run=str(run),attempt=str(attempt),started_utc=start)
    def save_status():recovery.atomic_json(run/'STATUS.json',status)
    save_status()
    (attempt/'launcher_pid.txt').write_text(str(os.getpid())+'\n')
    try:
        checkpoint=recovery.latest(run,deep=True,allow_final_model_only=True) if a.resume else None
        if checkpoint and checkpoint['step']==100:
            # A completed endpoint needs no actor initialization or optimizer replay.
            stage='finalize';status['stage']=stage;save_status()
            run_command([sys.executable,PACKAGE/'scripts/runtime_entry.py','finalize','--work',work,'--run',run],attempt/'finalize.log',env)
            return
        # Prevent overlap with surviving old workers; never kill unrelated GPU jobs.
        running=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                               text=True,capture_output=True,check=True).stdout.strip()
        if running:raise RuntimeError(f'GPU compute processes still present ({running}); inspect ownership before starting')
        if checkpoint:
            if not checkpoint['full_training_state']:raise ValueError('No complete optimizer state')
            recovery.archive_suffix(run,checkpoint,attempt)
            env['METATOOL_RESUME_CHECKPOINT']=checkpoint['path']
        command=[sys.executable,PACKAGE/'scripts/runtime_entry.py','preflight','--work',work]
        if checkpoint:command+=['--run',run]
        run_command(command,attempt/'preflight.log',env)
        for name in ['INSTALL.json','PREFLIGHT.json','DISK_BUDGET.json']:
            shutil.copyfile(work/name,attempt/name)
            if not (run/name).exists():shutil.copyfile(work/name,run/name)
        recovery.atomic_json(attempt/'ATTEMPT.json',dict(started_utc=start,run_uid=spec['run_uid'],
            resume_checkpoint=checkpoint['path'] if checkpoint else None,
            completed_step_at_start=checkpoint['step'] if checkpoint else 0,
            scheduler_total_steps=100,automatic_gpu_retries=0))
        stage='training';status['stage']=stage;save_status()
        run_command(spec['command'],attempt/'train.log',env)
        endpoint=recovery.latest(run,allow_final_model_only=True)
        if endpoint['step']!=100:raise RuntimeError('Training ended before step100; preserve checkpoint and resume explicitly')
        stage='finalize';status['stage']=stage;save_status()
        run_command([sys.executable,PACKAGE/'scripts/runtime_entry.py','finalize','--work',work,'--run',run],attempt/'finalize.log',env)
        print(json.dumps({'status':'completed','run':str(run),'report':str(run/'TRAINING_SUMMARY.json')}))
    except BaseException as exc:
        status.update(status='failed',stage=stage,error=repr(exc),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save_status();raise

if __name__=='__main__':main()
