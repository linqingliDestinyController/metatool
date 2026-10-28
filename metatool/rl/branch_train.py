"""metatool controller: bind main sampled prefixes -> paired returns -> actor slots."""
from __future__ import annotations
import os
from pathlib import Path
from collections import Counter
from metatool.rl.branch_credit import (ROUTE_FIELDS,MODES,action_of,branch_target,digest,
    load_protocol,route_slots,select_states,write_json_exclusive,prefix_complete_utf8)

def validate_training_config(trainer):
    c=trainer.config;a=c.actor_rollout_ref.actor
    expected=[(c.data.train_batch_size,96),(c.actor_rollout_ref.rollout.n,4),
        (c.trainer.n_gpus_per_node,6),(c.trainer.nnodes,1),(a.ppo_mini_batch_size,48),
        (a.ppo_epochs,1),(a.ulysses_sequence_parallel_size,1),
        (c.actor_rollout_ref.rollout.tensor_model_parallel_size,1),
        (c.actor_rollout_ref.rollout.temperature,1.)]
    if any(x!=y for x,y in expected) or trainer.async_rollout_mode or a.shuffle or a.use_fused_kernels:
        raise ValueError("Preserve the actual six-rank synchronous training layout")
    if c.algorithm.get('meta_control_strength',0)!=0 or not c.algorithm.get('absolute_action_penalty_enabled',False):
        raise ValueError("Keep centered offset zero and absolute action penalty enabled")
    if float(a.branch_route_coef)!=.01:raise ValueError("Registered route coefficient changed")

def score_probe_records(records,selected,protocol):
    from metatool.rl.validation import score_one
    keys={f'{j}:{m}:{r}' for j in range(12) for m in MODES for r in (0,1)}
    keyed={r['key']:r for r in records}
    if len(records)!=48 or set(keyed)!=keys or len(selected)!=12:
        raise RuntimeError("Require all 48 dispatch rows, including explicit inactive rows")
    targets=[]
    for j,s in enumerate(selected):
        outcomes={m:[] for m in MODES};costs={m:[] for m in MODES};outputs=[]
        for mode in MODES:
            for rep in (0,1):
                r=keyed[f'{j}:{mode}:{rep}']
                if (r['mode'],r['replicate'],r['active'])!=(mode,rep,s['active']):
                    raise RuntimeError("Wrong probe condition")
                if r['prompt_sha256']!=digest(s['prompt_ids']) or r['state_sha256']!=digest(s['prompt_ids']+s['prefix_ids']):
                    raise RuntimeError("Probe is not from the same question AND prefix")
                if r['base_ids']!=s['prompt_ids'] or r['prefix_ids']!=s['prefix_ids'] or r['used_in_ppo'] is not False:
                    raise RuntimeError("Broken actual-ID provenance")
                if not s['active']:
                    if r['response_token_ids'] or r['output'] is not None or r['assistant_tokens'] or r['incremental_tokens']:
                        raise RuntimeError("Inactive slot is not a model prediction")
                    outputs.append(r);continue
                if s['expected_action']!='call' or action_of(s['ground_truth'])!='call' or not r['byte_alignment_verified']:
                    raise RuntimeError("Only exact-Call feedback is eligible")
                decision=[protocol['close_id']] if mode=='stop' else r['first_sampled_ids']
                if len(decision)!=1 or (mode=='continue' and not 0<=decision[0]<151643):
                    raise RuntimeError("Incorrect branch intervention")
                if r['response_token_ids']!=s['prefix_ids']+decision+r['continuation_ids']:
                    raise RuntimeError("Returned IDs differ from raw prefix+decision+tail")
                if r['assistant_tokens']!=len(r['response_token_ids']) or r['incremental_tokens']!=1+len(r['continuation_ids']):
                    raise RuntimeError("Cost must exclude the common prefix")
                if r['sampled_tokens']!=len(r['continuation_ids'])+(mode=='continue'):
                    raise RuntimeError("Forced STOP token is not newly sampled")
                if r['forced_decision_ids']!=([protocol['close_id']] if mode=='stop' else []):
                    raise RuntimeError("Forced intervention record mismatch")
                fmt,exact,decision_ok=score_one(r['output'],s['ground_truth'],r['truncated'])
                if exact is None:raise RuntimeError("Respond proxy must never enter this objective")
                success=int(exact)
                outcomes[mode].append(success);costs[mode].append(r['incremental_tokens'])
                outputs.append({**r,'proxy_success':success,'proxy_format_valid':fmt})
        result=branch_target(outcomes['stop'],outcomes['continue'],stop_tokens=costs['stop'],
            continue_tokens=costs['continue'],active=s['active'],protocol=protocol)
        targets.append({**s,**result,'stop_successes':outcomes['stop'],
            'continue_successes':outcomes['continue'],'outputs':outputs})
    return targets

def run_training_probes(trainer,gen_batch,unrepeated_batch,main_output):
    import numpy as np
    from verl import DataProto
    validate_training_config(trainer)
    if len(unrepeated_batch)!=96 or len(gen_batch)!=384 or len(main_output)!=384:
        raise ValueError("Need the current batch before balancing/repeating labels")
    step=int(trainer.global_steps);p=load_protocol(trainer.config.actor_rollout_ref.actor.branch_protocol_path)
    prompts=[]
    for i in range(96):
        row=i*4;ids=gen_batch.batch['input_ids'][row];mask=gen_batch.batch['attention_mask'][row].bool()
        prompts.append(ids[mask].detach().cpu().tolist())
    # main_response_lengths comes directly from vLLM's returned token_ids BEFORE padding.
    # An EOS-derived attention mask is NOT a substitute for this provenance.
    traces=main_output.non_tensor_batch['metatool_main_trace']
    actual=[]
    for row,t in enumerate(traces):
        if t['prompt_ids']!=prompts[row//4] or t['prompt_sha256']!=digest(prompts[row//4]):
            raise RuntimeError("Main response/prompt binding mismatch")
        ids=t['response_token_ids']
        if main_output.batch['responses'][row,:len(ids)].detach().cpu().tolist()!=ids:
            raise RuntimeError("Main sampled IDs differ from PPO tensor")
        actual.append(ids)
    from metatool.rl.rollout_log import save_main_rollouts
    save_main_rollouts(step, traces, unrepeated_batch, trainer.tokenizer)
    gt=[v['ground_truth'] for v in unrepeated_batch.non_tensor_batch['reward_model']]
    selected,selection=select_states(prompts,gt,actual,step,p,
        prefix_byte_check=lambda ids:prefix_complete_utf8(ids,trainer.tokenizer))
    uids=unrepeated_batch.non_tensor_batch['uid'].tolist()
    for s in selected:
        s['source_uid']=uids[s['source_batch_index']] if s['active'] else None
        if s['active']:s['source_main_trace']=dict(traces[s['source_row']])
    # Retain the full original response only for selected states; other eligibility
    # decisions are completely replayable from the first 66 IDs in selection.
    write_json_exclusive(Path(os.environ['METATOOL_RUN'])/'branch_probes'/f'step_{step:03d}_selection.json',
        dict(step=step,selection=selection,selected=selected,main_rows=384,main_prompts=96))
    rows=[];bases=[];prefixes=[];modes=[];reps=[];active=[];keys=[]
    for j,s in enumerate(selected):
        for mode in MODES:
            for rep in (0,1):
                rows.append(s['source_row'] if s['active'] else 0);bases.append(s['prompt_ids'])
                prefixes.append(s['prefix_ids']);modes.append(mode);reps.append(rep);active.append(s['active']);keys.append(f'{j}:{mode}:{rep}')
    def objects(values):
        a=np.empty(len(values),dtype=object);a[:]=values;return a
    probe=DataProto.from_dict(tensors={k:gen_batch.batch[k][rows].clone() for k in ('input_ids','attention_mask','position_ids')},
        non_tensors={k:objects(v) for k,v in dict(raw_prompt_ids=bases,branch_prefix_ids=prefixes,
            branch_mode=modes,branch_replicate=reps,branch_active=active,branch_key=keys).items()},
        meta_info=dict(branch_probe=True,global_steps=step,branch_protocol_path=trainer.config.actor_rollout_ref.actor.branch_protocol_path))
    output=trainer.actor_rollout_wg.generate_sequences(probe);records=output.non_tensor_batch['branch_trace'].tolist()
    for i,r in enumerate(records):
        n=int(output.batch['branch_response_lengths'][i].item())
        if output.batch['responses'][i,:n].detach().cpu().tolist()!=r['response_token_ids']:
            raise RuntimeError("Aux tensor/raw ID binding mismatch")
    targets=score_probe_records(records,selected,p);counts=Counter(t['winner'] for t in targets)
    write_json_exclusive(Path(os.environ['METATOOL_RUN'])/'branch_probes'/f'step_{step:03d}_targets.json',
        dict(step=step,objective_version=42,protocol=p,
            scope='runtime_smoke_only' if trainer.config.trainer.get('cuda_smoke_rounds',0) else 'training_only',natural_ppo_rows=384,
            auxiliary_dispatch_rows=48,actual_auxiliary_outputs=sum(active),targets=targets,
            selected_active=selection['selected_active'],eligible_prompts=selection['eligible_prompts'],
            feedback_limit='Exact Call only; no automatic Respond semantics; inactive rows are NOT outputs'))
    # The main raw trace is not fed through PPO or propagated into unrelated workers.
    main_output.non_tensor_batch.pop('metatool_main_trace')
    metrics={'branch/probe_outputs':float(sum(active)),'branch/dispatch_rows':48.,
        'branch/active_states':float(selection['selected_active']),'branch/eligible_prompts':float(selection['eligible_prompts']),
        'branch/stop_winners':float(counts['stop']),'branch/continue_winners':float(counts['continue']),
        'branch/ties':float(counts['tie']),'branch/inactive_slots':float(counts['inactive']),
        'branch/cost_informative':float(sum(t['target_reason']=='both_correct_cost' for t in targets)),
        'branch/probe_assistant_tokens':float(sum(r['assistant_tokens'] for r in records)),
        'branch/probe_incremental_tokens':float(sum(r['incremental_tokens'] for r in records)),
        'branch/probe_sampled_tokens':float(sum(r['sampled_tokens'] for r in records)),
        'branch/probe_truncated':float(sum(r['truncated'] for r in records)),
        'branch/mean_margin':sum(t['margin'] for t in targets)/12}
    return targets,metrics

def attach_route_slots(batch,targets,tokenizer,protocol_path):
    import torch
    p=load_protocol(protocol_path)
    if len(batch)!=384 or len(targets)!=12:raise ValueError("Fixed collective slot layout changed")
    states=[t['prompt_ids']+t['prefix_ids'] for t in targets]
    width=max(map(len,states));device=batch.batch['responses'].device
    ids=torch.full((384,width),tokenizer.pad_token_id,dtype=torch.long,device=device)
    mask=torch.zeros_like(ids);slots=torch.zeros(384,dtype=torch.bool,device=device)
    labels=torch.zeros(384,dtype=torch.long,device=device);margins=torch.zeros(384,device=device)
    weights=torch.zeros(384,device=device);active=torch.zeros(384,dtype=torch.bool,device=device)
    for slot,state,t in zip(route_slots(),states,targets):
        if digest(state)!=t['state_sha256'] or t['importance_weight']!=1.:raise ValueError("State/estimand changed")
        ids[slot,-len(state):]=torch.tensor(state,device=device);mask[slot,-len(state):]=1
        slots[slot]=True;labels[slot]=t['target'];margins[slot]=t['margin'];weights[slot]=1.;active[slot]=t['active']
    steps={t['training_step'] for t in targets}
    if len(steps)!=1 or not 1<=next(iter(steps))<=100:raise ValueError("Mixed-step targets")
    batch.meta_info['branch_step']=next(iter(steps))
    for key,tensor in zip(ROUTE_FIELDS,(slots,ids,mask,labels,margins,weights,active)):batch.batch[key]=tensor
    batch.meta_info['global_token_num']=list(batch.meta_info['global_token_num'])+list(map(len,states))
    return {'branch/aux_state_tokens':float(sum(map(len,states))),'branch/aux_slots':12.}
