"""Sample STOP/CONTINUE from identical actual IDs; auxiliary-only outputs."""
from __future__ import annotations
import os
import random
from pathlib import Path
from metatool.rl.branch_credit import (digest,load_protocol,seed_for,write_json_exclusive,
                                         intermediate_prefix,MODES)
from metatool.rl.cuda_runtime_guard import checked_generate, emit, success_only_rng

class ContinueOnly:
    """V0 one-token conditional distribution; includes whitespace lexical IDs."""
    def __init__(self,lo=0,hi=151643): self.lo,self.hi=lo,hi
    def __call__(self,past_tokens,logits):
        import torch
        if past_tokens or not 0<=self.lo<self.hi<=logits.shape[-1]:
            raise ValueError("Use only at one intermediate decision token")
        if not bool(torch.isfinite(logits[self.lo:self.hi]).all()):
            raise ValueError("Nonfinite continuation logits")
        result=torch.full_like(logits,float('-inf'))
        result[self.lo:self.hi]=logits[self.lo:self.hi]
        return result

def probe_requests(prompt_ids,prefixes,modes,replicates,active,step,protocol):
    if len({len(v) for v in (prompt_ids,prefixes,modes,replicates,active)})!=1:
        raise ValueError("Probe metadata dimensions differ")
    jobs=[]
    for base,prefix,mode,rep,on in zip(prompt_ids,prefixes,modes,replicates,active):
        base=list(map(int,base));prefix=list(map(int,prefix));rep=int(rep);on=bool(on)
        if not base or len(base)>3072 or any(not 0<=i<151669 for i in base):
            raise ValueError("Invalid original prompt IDs")
        if mode not in MODES or rep not in (0,1):raise ValueError("Invalid branch")
        if on and intermediate_prefix(prefix,protocol)[0]!=prefix:
            raise ValueError("Active comparison must use a real 66-token intermediate state")
        if not on and prefix!=protocol['open_ids']:raise ValueError("Invalid inactive collective slot")
        state=base+prefix
        jobs.append(dict(base_ids=base,prefix_ids=prefix,prompt_sha256=digest(base),
            state_sha256=digest(state),mode=mode,replicate=rep,active=on,
            first_seed=seed_for(step,state,mode,rep,'first'),
            continuation_seed=seed_for(step,state,mode,rep,'tail')))
    return jobs

def generate_probes(rollout,prompts):
    import numpy as np
    import torch
    from tensordict import TensorDict
    from vllm import SamplingParams
    from verl import DataProto
    from metatool.rl.exact_token_layout import response_byte_layout
    if os.environ.get('VLLM_USE_V1')!='0' or rollout.config.tensor_model_parallel_size!=1 or rollout.lora_kwargs:
        raise ValueError("Keep the audited dense V0 synchronous TP=1 engine")
    protocol=load_protocol(prompts.meta_info['branch_protocol_path'])
    if rollout._metatool_tokenizer_len!=151669:raise ValueError("Tokenizer identity changed")
    non=prompts.non_tensor_batch;step=int(prompts.meta_info['global_steps'])
    jobs=probe_requests(non['raw_prompt_ids'],non['branch_prefix_ids'],non['branch_mode'],
        non['branch_replicate'],non['branch_active'],step,protocol)
    keys=list(non['branch_key']);rank=torch.distributed.get_rank()
    if len(keys)!=len(jobs) or len(keys)!=len(set(keys)):raise ValueError("Bad distributed probe keys")
    raw_path=Path(os.environ['METATOOL_RUN'])/'branch_probes'/f'step_{step:03d}_rank_{rank}.json'
    if raw_path.exists():raise FileExistsError("Never regenerate committed probes")
    first_raw=[];tail_raw=[];first={};tails={};settings=protocol['sampling']
    def params(seed,budget,restricted=False):
        return SamplingParams(n=1,temperature=1.,top_p=1.,top_k=-1,max_tokens=budget,
            seed=seed,stop_token_ids=settings['stop_token_ids'],detokenize=False,logprobs=0,
            logits_processors=[ContinueOnly(*protocol['continue_token_range'])] if restricted else None)
    def preserve(outputs,indices):
        return [dict(key=keys[indices[j]] if j<len(indices) else None,prompt_ids=list(o.prompt_token_ids) if o.prompt_token_ids is not None else None,
            samples=[dict(token_ids=list(s.token_ids),finish_reason=s.finish_reason,stop_reason=s.stop_reason)
                     for s in o.outputs]) for j,o in enumerate(outputs)]
    py_rng,np_rng=random.getstate(),np.random.get_state()
    try:
        with success_only_rng(torch, [torch.cuda.current_device()], "branch_rng_context"):
            indices=[i for i,j in enumerate(jobs) if j['active'] and j['mode']=='continue']
            if indices:
                inputs=[dict(prompt_token_ids=jobs[i]['base_ids']+jobs[i]['prefix_ids']) for i in indices]
                outs=checked_generate(rollout.inference_engine,device=torch.cuda,
                    vocab_size=rollout._metatool_tokenizer_len,prompts=inputs,
                    sampling_params=[params(jobs[i]['first_seed'],1,True) for i in indices],use_tqdm=False)
                first_raw=preserve(outs,indices)
                if len(outs)!=len(indices):raise RuntimeError("Missing first-stage sample")
                for i,req,o in zip(indices,inputs,outs):
                    if list(o.prompt_token_ids)!=req['prompt_token_ids'] or len(o.outputs)!=1:
                        raise RuntimeError("First-stage binding mismatch")
                    ids=list(o.outputs[0].token_ids)
                    if len(ids)!=1 or not 0<=ids[0]<151643:raise RuntimeError("Continuation mask failed")
                    first[i]=ids
            indices=[i for i,j in enumerate(jobs) if j['active']]
            if indices:
                inputs=[];samplers=[]
                for i in indices:
                    j=jobs[i];decision=[protocol['close_id']] if j['mode']=='stop' else first[i]
                    inputs.append(dict(prompt_token_ids=j['base_ids']+j['prefix_ids']+decision))
                    samplers.append(params(j['continuation_seed'],2048-len(j['prefix_ids'])-1))
                outs=checked_generate(rollout.inference_engine,device=torch.cuda,
                    vocab_size=rollout._metatool_tokenizer_len,prompts=inputs,sampling_params=samplers,use_tqdm=False)
                tail_raw=preserve(outs,indices)
                if len(outs)!=len(indices):raise RuntimeError("Missing tail sample")
                for i,req,o in zip(indices,inputs,outs):
                    if list(o.prompt_token_ids)!=req['prompt_token_ids'] or len(o.outputs)!=1:
                        raise RuntimeError("Tail request binding mismatch")
                    tails[i]=o.outputs[0]
        records=[]
        for i,j in enumerate(jobs):
            shared=dict(**j,key=keys[i],used_in_ppo=False)
            if not j['active']:
                records.append(dict(**shared,response_token_ids=[],output=None,assistant_tokens=0,
                    incremental_tokens=0,sampled_tokens=0,first_sampled_ids=[],continuation_ids=[],
                    finish_reason='inactive',stop_reason=None,truncated=False,byte_alignment_verified=False))
                continue
            decision=[protocol['close_id']] if j['mode']=='stop' else first[i]
            sample=tails[i];tail=list(sample.token_ids);ids=j['prefix_ids']+decision+tail
            if sample.finish_reason not in ('stop','length') or not tail or len(ids)>2048:
                raise RuntimeError("Incomplete/over-budget branch; preserve evidence")
            if any(type(t) is not int or not 0<=t<151669 for t in ids):
                raise RuntimeError("Invalid generated ID: never delete or re-tokenize it")
            text,_=response_byte_layout(ids,rollout.tokenizer)
            records.append(dict(**shared,response_token_ids=ids,output=text,assistant_tokens=len(ids),
                incremental_tokens=1+len(tail),sampled_tokens=len(tail)+(j['mode']=='continue'),
                first_sampled_ids=first.get(i,[]),forced_decision_ids=decision if j['mode']=='stop' else [],
                continuation_ids=tail,continuation_prompt_sha256=digest(j['base_ids']+j['prefix_ids']+decision),
                finish_reason=sample.finish_reason,stop_reason=sample.stop_reason,
                truncated=sample.finish_reason=='length' and ids[-1] not in settings['stop_token_ids'],
                byte_alignment_verified=True))
        if any(j['active'] for j in jobs) and rollout._metatool_sync_model._metatool_mask_calls==0:
            raise RuntimeError("Valid-ID model mask not called")
        write_json_exclusive(raw_path,dict(step=step,rank=rank,first_returned=first_raw,
            continuation_returned=tail_raw,records=records,actual_outputs=sum(j['active'] for j in jobs),
            collective_slots=len(jobs),sampled_tokens_are_auxiliary_only=True))
    except BaseException as exc:
        try:
            write_json_exclusive(raw_path.with_suffix('.failure.json'),dict(step=step,rank=rank,error=repr(exc),
                jobs=jobs,first_returned=first_raw,continuation_returned=tail_raw))
        except BaseException as log_exc:
            emit('diagnostic_write_error',primary_error=repr(exc),secondary_error=repr(log_exc))
        raise
    finally:
        random.setstate(py_rng);np.random.set_state(np_rng)
    padded=torch.full((len(records),2048),rollout.pad_token_id,dtype=torch.long,
                      device=prompts.batch['input_ids'].device)
    for i,r in enumerate(records):
        if r['active']:
            padded[i,:r['assistant_tokens']]=torch.tensor(r['response_token_ids'],device=padded.device)
    lengths=torch.tensor([r['assistant_tokens'] for r in records],dtype=torch.long,device=padded.device)
    def objects(values):
        a=np.empty(len(values),dtype=object);a[:]=values;return a
    return DataProto(batch=TensorDict(dict(responses=padded,branch_response_lengths=lengths),
        batch_size=[len(records)]),non_tensor_batch=dict(branch_trace=objects(records),branch_key=objects(keys)))
