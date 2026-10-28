"""Generate one frozen evaluation suite; commit actual raw outputs before parsing."""
import argparse,fcntl,hashlib,importlib.metadata,json,os
from pathlib import Path
from metatool.release import ROOT,read,atomic,digest,sha,model_identity
from metatool.io import rows
from metatool.evaluation.api_score import load_cases

def generate(model,output):
    suite='api'
    if os.environ.get('VLLM_USE_V1')!='1':raise ValueError('Evaluation requires VLLM_USE_V1=1')
    expected={'torch':'2.6.0+cu124','transformers':'4.51.3','vllm':'0.8.5.post1'}
    if {k:importlib.metadata.version(k) for k in expected}!=expected:raise ValueError('Evaluation runtime differs from the frozen profile')
    from transformers import AutoTokenizer
    from metatool.rl.exact_token_layout import response_byte_layout
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    if len(tok)!=151669:raise ValueError('Frozen Qwen3 tokenizer required')
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=read(ROOT/'configs/api_eval.json')
    spec=dict(suite=suite,profile=cfg,model=model_identity(model));fingerprint=digest(spec)
    atomic(output/'generation_spec.json',dict(fingerprint=fingerprint,**spec))
    if (output/'complete.json').exists():
        done=read(output/'complete.json')
        if done['fingerprint']!=fingerprint or sha(output/'predictions.jsonl')!=done['sha256']:raise ValueError('Completed output differs')
        print('Completed output verified; no model loaded.');return
    requests=[]
    cases=load_cases(ROOT/'data/api_bank')
    for key,c in cases.items():
        msgs=[{'role':'system','content':c['system']},{'role':'user','content':c['user']}]
        text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        ids=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True)
        requests.append(dict(id=key,prompt_token_ids=ids,prompt_sha256=hashlib.sha256(text.encode()).hexdigest()))
    if digest([{'prompt_token_ids':r['prompt_token_ids']} for r in requests])!=cfg['prompts_sha256']:
        raise ValueError('API prompt token sequences changed')
    chunk=64;settings=cfg['llm_settings'];base_params={k:v for k,v in cfg['sampling'].items() if k!='effective_temperature'}
    base_params['allowed_token_ids']=list(range(len(tok)))
    llm=None;result=[]
    for start in range(0,len(requests),chunk):
        selected=requests[start:start+chunk];path=output/f'chunk_{start//chunk:03d}.raw.json'
        if path.exists():raw=read(path)
        else:
            from vllm import LLM,SamplingParams
            if llm is None:llm=LLM(model=str(model),**settings)
            params=SamplingParams(**base_params)
            generated=llm.generate([{'prompt_token_ids':r['prompt_token_ids']} for r in selected],sampling_params=params,use_tqdm=True)
            records=[dict(prompt_token_ids=list(g.prompt_token_ids),outputs=[dict(text=o.text,token_ids=list(o.token_ids),finish_reason=o.finish_reason,stop_reason=o.stop_reason) for o in g.outputs]) for g in generated]
            raw=dict(fingerprint=fingerprint,start=start,requests_sha256=digest(selected),records=records)
            raw['payload_sha256']=digest(raw);atomic(path,raw)
        body={k:v for k,v in raw.items() if k!='payload_sha256'}
        if digest(body)!=raw['payload_sha256'] or raw['fingerprint']!=fingerprint or raw['start']!=start or raw['requests_sha256']!=digest(selected):raise ValueError('Raw chunk binding changed')
        if len(raw['records'])!=len(selected):raise ValueError('Missing output; raw evidence retained')
        for req,record in zip(selected,raw['records']):
            if record['prompt_token_ids']!=req['prompt_token_ids'] or len(record['outputs'])!=1:raise ValueError('Reordered prompt or multiple outputs')
            o=record['outputs'][0];ids=o['token_ids']
            if not ids or any(type(i) is not int or not 0<=i<len(tok) for i in ids):raise ValueError('Invalid actual generated IDs')
            decoded,offsets=response_byte_layout(ids,tok)
            row={k:v for k,v in req.items() if k!='prompt_token_ids'}
            row.update(response_token_ids=ids,raw_decoded=decoded,response_byte_offsets=offsets,finish_reason=o['finish_reason'],stop_reason=o['stop_reason'])
            row.update(output=o['text'],output_tokens=len(ids),truncated=o['finish_reason']=='length')
            result.append(row)
    dest=output/'predictions.jsonl';text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result)
    if dest.exists() and dest.read_text()!=text:raise ValueError('Committed aggregate differs')
    if not dest.exists():
        tmp=dest.with_suffix('.pending');tmp.write_text(text);os.replace(tmp,dest)
    atomic(output/'complete.json',dict(fingerprint=fingerprint,n=len(result),sha256=sha(dest)))
    from metatool.evaluation.api_score import score,load_predictions
    metrics,items=score(cases,load_predictions(dest))
    atomic(output/'scores.json',dict(metrics=metrics,items=items));print(json.dumps(metrics,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();generate(a.model.resolve(),a.output.resolve())
if __name__=='__main__':main()
