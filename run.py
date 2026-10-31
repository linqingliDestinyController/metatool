#!/usr/bin/env python3
"""metatool Stage 2: direct RL training from a local model, export and evaluation."""
from pathlib import Path
import argparse,os,subprocess,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/verl'));sys.path.insert(0,str(ROOT))

def child(module,args,env=None):
    e=os.environ.copy();e['PYTHONPATH']=os.pathsep.join([str(ROOT),str(ROOT/'vendor/verl'),e.get('PYTHONPATH','')])
    e.update(PYTHONHASHSEED='42',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',HYDRA_FULL_ERROR='1',TORCH_SHOW_CPP_STACKTRACES='1',
        PYTHONFAULTHANDLER='1',PYTHONDONTWRITEBYTECODE='1',RAY_DEDUP_LOGS='0')
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:e[name]='1'
    if env:e.update(env)
    subprocess.run([sys.executable,'-m',module,*map(str,args)],env=e,cwd=ROOT,check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('train',aliases=['rl']);q.add_argument('--model',type=Path,help='Required for a new run; complete local HF model')
    q.add_argument('--work',type=Path,required=True);q.add_argument('--gpus',default='0,1,2,3,4,5')
    q.add_argument('--train-data',type=Path,help='Explicit frozen RL train.parquet path (bundled data/rl or external); required for a new run')
    q.add_argument('--validation-data',type=Path,help='Explicit frozen RL validation.parquet path (bundled data/rl or external); required for a new run')
    modes=q.add_mutually_exclusive_group();modes.add_argument('--resume',action='store_true');modes.add_argument('--retry-preflight',action='store_true')
    q=s.add_parser('finalize');q.add_argument('--work',type=Path,required=True)
    q=s.add_parser('score');q.add_argument('--predictions',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q=s.add_parser('evaluate');q.add_argument('--model',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--gpu',default='0')
    a=p.parse_args()
    if a.command=='score':
        child('metatool.evaluation.score',['--predictions',a.predictions.resolve(),'--output',a.output.resolve()],{'CUDA_VISIBLE_DEVICES':''});return
    if a.command=='evaluate':
        child('metatool.evaluation.generate',['--model',a.model.resolve(),'--output',a.output.resolve()],{'CUDA_VISIBLE_DEVICES':a.gpu,'VLLM_USE_V1':'1'});return
    from metatool.release import disjoint_work,verify_release
    from runtime.cleanup import exported,cleanup
    work=disjoint_work(a.work);verify_release()
    if a.command in {'train','rl'}:
        from runtime.install import install
        from runtime.preflight import verify_install
        devices=[x.strip() for x in a.gpus.split(',')]
        if len(devices)!=6 or len(set(devices))!=6 or not all(devices):
            p.error('This unchanged RL profile requires six distinct visible GPU identifiers')
        if not work.exists():
            if a.resume or a.retry_preflight:p.error('Cannot resume or retry a missing work directory')
            if a.model is None:p.error('--model is required for a new run')
            if a.train_data is None or a.validation_data is None:
                p.error('--train-data and --validation-data are required; use explicit paths to bundled data/rl parquet or external files matching configs/dataset_profile.json; API-Bank is evaluation-only')
            install(work,a.model.resolve(),a.train_data.resolve(),a.validation_data.resolve())
        record=verify_install(work)
        if a.model is not None and str(a.model.resolve())!=record['initial_model']['path']:
            raise ValueError('This work directory is bound to a different input model; use a new work directory')
        for name,arg in [('train.parquet',a.train_data),('validation.parquet',a.validation_data)]:
            if arg is not None and str(arg.resolve())!=record['dataset_inputs'][name]['path']:
                raise ValueError('Existing work directory is bound to different RL data')
        if exported(work):
            cleanup(work);print('Completed export verified; no training started.');return
        args=['--work',work,'--training-budget-authorized']
        if a.resume:args+=['--resume']
        if a.retry_preflight:args+=['--retry-preflight']
        child('runtime.experiment',args,{'CUDA_VISIBLE_DEVICES':','.join(devices)})
        cleanup(work)
    elif a.command=='finalize':
        if not exported(work):
            child('runtime.finalize',['--work',work,'--run',work/'runs/metatool_seed42'])
        cleanup(work)
if __name__=='__main__':main()
