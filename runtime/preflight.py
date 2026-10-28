"""Necessary local integrity, tokenizer, environment and capacity checks only."""
import argparse,importlib.metadata,json,os,shutil
from pathlib import Path
from runtime.install import PACKAGE,sha,read,model_identity
from metatool.release import verify_release

def verify_install(work):
    verify_release();work=Path(work);record=read(work/'INSTALL.json')
    for name,h in record['source'].items():
        if sha(work/'src'/name)!=h:raise ValueError(f'Installed source differs: {name}')
    expected=read(PACKAGE/'configs/dataset_profile.json')
    if record['dataset_sha256']!={n:v['sha256'] for n,v in expected.items()}:
        raise ValueError('Registered datasets differ from the frozen RL contract')
    for name,h in record['dataset_sha256'].items():
        if str((work/'data'/name).resolve())!=record['dataset_inputs'][name]['path']:
            raise ValueError('External dataset path changed')
        if sha(work/'data'/name)!=h:raise ValueError(f'Data differs: {name}')
    if sha(work/'assets/branch_protocol.json')!=record['protocol_sha256']:raise ValueError('Route protocol differs')
    return record

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--run',type=Path)
    a=p.parse_args();work=a.work.resolve();record=verify_install(work)
    expected={'torch':'2.6.0+cu124','transformers':'4.51.3','vllm':'0.8.5.post1','ray':'2.47.1'}
    versions={k:importlib.metadata.version(k) for k in expected}
    if versions!=expected:raise ValueError(f'Frozen runtime versions differ: {versions}')
    if os.environ.get('VLLM_USE_V1')!='0':raise ValueError('Training requires vLLM engine 0')
    import torch
    from transformers import AutoTokenizer,AutoConfig
    from metatool.rl.branch_credit import load_protocol
    from metatool.rl.checkpoint_safety import estimate_model_bytes
    from metatool.rl.full_recovery import disk_plan,latest
    from runtime.model import verify_input
    model=(work/'model').resolve()
    identity_check=verify_input(model,record['initial_model'])
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    cfg=AutoConfig.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    if (len(tok),cfg.vocab_size,cfg.num_hidden_layers,cfg.model_type)!=(151669,151936,36,'qwen3'):
        raise ValueError('This frozen profile requires its Qwen3-8B tokenizer/model')
    protocol=load_protocol(work/'assets/branch_protocol.json')
    for text,ids in [('<think>\n',[151667,198]),('</think>',[151668])]:
        if tok.encode(text,add_special_tokens=False)!=ids:raise ValueError('Native decision tokens differ')
    if set(range(*protocol['continue_token_range']))&set(tok.all_special_ids):raise ValueError('Lexical support differs')
    if torch.cuda.device_count()!=6:raise ValueError('Expose six GPUs for this frozen RL profile')
    gpu=[dict(name=torch.cuda.get_device_name(i),bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(6)]
    if any(g['bytes']<75*1024**3 for g in gpu):raise ValueError('The unchanged batch requires 80 GB class GPUs')
    if a.run:
        cp=latest(a.run)
        if cp['step']!=50 or not cp['full_training_state']:raise ValueError('Resume requires complete step50 state')
    storage=disk_plan(shutil.disk_usage(work).free,estimate_model_bytes(model))
    (work/'DISK_BUDGET.json').write_text(json.dumps(storage,indent=2)+'\n')
    if not storage['final_write_fits' if a.run else 'startup_fits']:raise OSError('Insufficient actual free disk; see DISK_BUDGET.json')
    report=dict(versions=versions,gpu=gpu,storage=storage,source_install_sha256=sha(work/'INSTALL.json'),
                tokenizer_length=len(tok),input_identity_check=identity_check,model_loaded=False,new_training_updates=0)
    (work/'PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
