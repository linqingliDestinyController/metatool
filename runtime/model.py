"""Bind a user-supplied local HF model without loading model weights."""
from pathlib import Path
from metatool.release import ROOT,read,sha,model_identity

def validate_layout(model):
    model=Path(model).resolve()
    if not model.is_dir() or not (model/'config.json').is_file():
        raise ValueError('--model must be a complete local Hugging Face model directory')
    if (model/'adapter_config.json').exists():
        raise ValueError('Supply a merged full model, not an adapter directory')
    cfg=read(model/'config.json');profile=read(ROOT/'configs/model_profile.json')
    for name,value in profile.items():
        if name in {'tokenizer_length','minimum_context','tokenizer_sha256'}:continue
        if cfg.get(name)!=value:raise ValueError(f'Model profile mismatch at {name}: {cfg.get(name)!r} != {value!r}')
    if cfg.get('max_position_embeddings',0)<profile['minimum_context']:
        raise ValueError('Model context is shorter than the frozen training profile')
    for name,expected in profile['tokenizer_sha256'].items():
        if not (model/name).is_file() or sha(model/name)!=expected:
            raise ValueError(f'Tokenizer/chat template differs: {name}; provide the matching original tokenizer; expected hashes are in configs/model_profile.json')
    if (model/'chat_template.jinja').exists():
        raise ValueError('External chat_template.jinja would override the frozen template; use the bundled tokenizer contract')
    shards={p.name for p in model.glob('*.safetensors')}
    if not shards:raise ValueError('No local safetensors weights; model weights are not included in this code package')
    index=model/'model.safetensors.index.json'
    if index.exists():
        mapping=read(index).get('weight_map',{})
        if not mapping or set(mapping.values())!=shards or any(Path(n).name!=n for n in mapping.values()):
            raise ValueError('Safetensors index is incomplete or references unexpected shard files')
    elif shards!={'model.safetensors'}:
        raise ValueError('Multiple shards require a complete model.safetensors.index.json')
    return cfg

def file_stats(model):
    model=Path(model)
    files=[*model.glob('*.safetensors'),*model.glob('*.json'),*model.glob('*.txt'),*model.glob('*.jinja')]
    result={}
    for path in sorted(set(files)):
        stat=path.stat()
        result[path.name]=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns,inode=stat.st_ino)
    return result

def bind_input(model):
    model=Path(model).resolve();validate_layout(model)
    before=file_stats(model);identity=model_identity(model);after=file_stats(model)
    if before!=after:raise ValueError('Input model changed while hashing')
    return dict(path=str(model),identity=identity,file_stats=after,role='rl_initialization')

def verify_input(model,binding):
    model=Path(model).resolve()
    if str(model)!=binding['path']:raise ValueError('Input model path changed after registration')
    validate_layout(model);before=file_stats(model)
    if before==binding['file_stats']:
        return 'bound_sha256_with_unchanged_size_mtime_ctime_inode'
    identity=model_identity(model)
    if before!=file_stats(model) or identity!=binding['identity']:
        raise ValueError('Input model content changed after registration')
    return 'full_sha256_rechecked'
