"""Audit actual metatool dispatch, generated samples, rewards, credit and six ranks."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path

def read(p):return json.loads(p.read_text())

def audit(run,tokenizer_path):
    from transformers import AutoTokenizer
    from metatool.rl.branch_credit import digest,intermediate_prefix,prefix_complete_utf8
    from metatool.rl.branch_train import score_probe_records
    from metatool.rl.branch_metrics import reduce_rank_rows,FIELDS
    from metatool.rl.exact_token_layout import response_byte_layout
    tok=AutoTokenizer.from_pretrained(str(tokenizer_path),local_files_only=True,trust_remote_code=False)
    from metatool.rl.full_recovery import completed_metrics,latest
    metrics,completion_evidence=completed_metrics(run)
    endpoint=latest(run,allow_final_model_only=True)
    if endpoint["step"]!=100:raise ValueError("Final model checkpoint was never committed")
    if [r['step'] for r in metrics]!=list(range(1,101)):raise ValueError('Require exactly continuous steps 1..100')
    if list((run/'branch_probes').glob('*.failure.json')):raise ValueError('Probe failure evidence exists')
    checkpoint=read(run/'checkpoint_manifest.json')
    if checkpoint['best_step']!=100 or checkpoint['selection']['selection_rule']!='fixed_final_step':
        raise ValueError('Require fixed step100 endpoint with committed full-state provenance')
    counts=Counter({k:0 for k in ('active_states','informative_states','actual_auxiliary_outputs',
        'assistant_tokens_including_reused_prefix','incremental_assistant_tokens','new_sampled_tokens')})
    winners=Counter();reasons=Counter();active_prob=0.;eligible=0
    for m in metrics:
        step=m['step'];folder=run/'branch_probes'
        for k,v in m.items():
            if isinstance(v,(int,float)) and not math.isfinite(v):raise ValueError(f'Nonfinite metric {step}/{k}')
        if m['branch_actor/local_optimizer_steps']!=2 or m['branch/aux_slots']!=12 or m['branch/dispatch_rows']!=48:
            raise ValueError('Fixed optimizer/dispatch layout changed')
        if m['metacontrol/strength']!=0:raise ValueError('Unexpected offset enabled')
        doc=read(folder/f'step_{step:03d}_targets.json');sel=read(folder/f'step_{step:03d}_selection.json')
        ts=doc['targets'];p=doc['protocol'];s=sel['selected']
        if doc['objective_version']!=42 or len(ts)!=12 or len(s)!=12:raise ValueError('Wrong objective or state count')
        if sel['main_rows']!=384 or sel['main_prompts']!=96:raise ValueError('Main batch changed')
        candidate_rows=sel['selection']['candidates'];candidate_eligible=[]
        if [c['source_batch_index'] for c in candidate_rows]!=list(range(96)):
            raise ValueError('Require all 96 original prompt candidates')
        for c in candidate_rows:
            base=c['prompt_ids']
            if c['prompt_sha256']!=digest(base) or c['ordering_key']!=digest([42,step,base]):
                raise ValueError('Prompt ordering hash changed')
            order=sorted(range(4),key=lambda r:digest([42,step,base,'replicate',r]))
            expected_order=order if c['expected_action']=='call' else []
            if [r['replicate'] for r in c['replicates']]!=expected_order:
                raise ValueError('Replicate ordering or Respond exclusion changed')
            good=[]
            for rep in c['replicates']:
                prefix,reason=intermediate_prefix(rep['inspected_ids'],p)
                if prefix is not None and not prefix_complete_utf8(prefix,tok):reason='incomplete_utf8_at_fixed_checkpoint'
                if reason!=rep['reason']:raise ValueError('Prefix eligibility replay differs')
                if reason=='eligible':good.append(rep['replicate'])
            if good:candidate_eligible.append((c['ordering_key'],c['source_batch_index'],good[0]))
        expected_pairs=[(i,rep) for _,i,rep in sorted(candidate_eligible)[:12]]
        if [(t['source_batch_index'],t['source_replicate']) for t in s if t['active']]!=expected_pairs:
            raise ValueError('Outcome-independent selection differs from saved prefix evidence')
        if len(candidate_eligible)!=doc['eligible_prompts']:raise ValueError('Eligible denominator mismatch')
        raw=[]
        for rank in range(6):
            r=read(folder/f'step_{step:03d}_rank_{rank}.json')
            if (r['step'],r['rank'],len(r['records']))!=(step,rank,8):raise ValueError('Bad distributed dispatch')
            raw.extend(r['records'])
            keyed={v['key']:v for v in r['records']}
            first={v['key']:v for v in r['first_returned']};tails={v['key']:v for v in r['continuation_returned']}
            if len(first)!=len(r['first_returned']) or len(tails)!=len(r['continuation_returned']):
                raise ValueError('Duplicate raw requests')
            if set(tails)!={k for k,v in keyed.items() if v['active']} or set(first)!={k for k,v in keyed.items() if v['active'] and v['mode']=='continue'}:
                raise ValueError('Returned sampler requests differ from active dispatch')
            for key,v in keyed.items():
                if not v['active']:continue
                tail=tails[key]
                decision=[p['close_id']] if v['mode']=='stop' else v['first_sampled_ids']
                if tail['prompt_ids']!=v['base_ids']+v['prefix_ids']+decision or len(tail['samples'])!=1:
                    raise ValueError('Raw continuation prompt mismatch')
                if tail['samples'][0]['token_ids']!=v['continuation_ids']:raise ValueError('Raw tail mismatch')
                sample=tail['samples'][0]
                if sample['finish_reason']!=v['finish_reason'] or sample['stop_reason']!=v['stop_reason']:
                    raise ValueError('Raw termination metadata differs')
                truncated=sample['finish_reason']=='length' and v['response_token_ids'][-1] not in p['sampling']['stop_token_ids']
                if truncated!=v['truncated']:
                    raise ValueError('Raw truncation flag differs')
                if v['mode']=='continue':
                    if first[key]['prompt_ids']!=v['base_ids']+v['prefix_ids'] or first[key]['samples'][0]['token_ids']!=decision:
                        raise ValueError('Raw decision sample mismatch')
                decoded,_=response_byte_layout(v['response_token_ids'],tok)
                if decoded!=v['output']:raise ValueError('Saved output differs from actual returned IDs')
        recomputed=score_probe_records(raw,s,p)
        if recomputed!=ts:raise ValueError('Exact-Call rewards/targets differ on replay')
        actor=read(run/'branch_actor_metrics'/f'step_{step:03d}.json')
        if actor['fields']!=list(FIELDS) or actor['world_size']!=6:raise ValueError('Missing six-rank evidence')
        agg=reduce_rank_rows(actor['rank_rows'])
        for k,v in agg.items():
            if not math.isclose(v,m[k],rel_tol=1e-6,abs_tol=1e-8) or v!=actor['aggregated'][k]:
                raise ValueError('Six-rank metric replay differs')
        for j,t in enumerate(ts):
            row=dict(zip(FIELDS,actor['rank_rows'][j%6][j//6]))
            for k,v in dict(active=float(t['active']),margin=t['margin'],target_continue=t['target'],
                            importance_weight=1.,effective_weight=t['margin']).items():
                if not math.isclose(row[k],v,rel_tol=1e-6,abs_tol=1e-8):raise ValueError('Feedback did not reach its actor slot')
            if t['active']:
                trace=t['source_main_trace'];ids=trace['response_token_ids']
                if ids!=t['source_response_ids'] or intermediate_prefix(ids,p)[0]!=t['prefix_ids'] or not prefix_complete_utf8(t['prefix_ids'],tok):
                    raise ValueError('Prefix is not from the saved actual main rollout')
                if trace['prompt_ids']!=t['prompt_ids'] or digest(t['prompt_ids']+t['prefix_ids'])!=t['state_sha256']:
                    raise ValueError('Main prompt/state binding mismatch')
                active_prob+=row['p_continue'];counts['active_states']+=1
            winners[t['winner']]+=1;reasons[t['target_reason']]+=1
            counts['informative_states']+=t['margin']>0
            for v in t['outputs']:
                counts['actual_auxiliary_outputs']+=v['active']
                counts['assistant_tokens_including_reused_prefix']+=v['assistant_tokens']
                counts['incremental_assistant_tokens']+=v['incremental_tokens']
                counts['new_sampled_tokens']+=v['sampled_tokens']
        n=sum(t['active'] for t in ts)
        if n!=doc['selected_active'] or 4*n!=doc['actual_auxiliary_outputs'] or m['branch/probe_outputs']!=4*n:
            raise ValueError('Inactive rows counted as generated outputs')
        eligible+=doc['eligible_prompts']
    lineage=[read(p) for p in sorted((run/'attempts').glob('*/RESUME_LINEAGE.json'))]
    extra=sum(r['extra_completed_optimizer_updates_per_rank'] for r in lineage)
    result=dict(complete=True,steps=100,optimizer_steps_per_rank=200,main_training_outputs=38400,
        optimizer_step_scope='retained checkpoint lineage only',
        extra_completed_optimizer_updates_per_rank=extra,
        incomplete_rank_updates_may_add_unmeasured_cost=bool(lineage),
        recovery_lineage=lineage,bitwise_gpu_equivalence_verified=False,
        completion_evidence=completion_evidence,
        collective_state_slots=1200,collective_dispatch_rows=4800,eligible_prompts_across_steps=eligible,
        **counts,winners=dict(winners),target_reasons=dict(reasons),
        mean_p_continue_active=active_prob/counts['active_states'] if counts['active_states'] else None,
        method_received_comparison_feedback=counts['informative_states']>0,
        limits=['Actual auxiliary output count is <=4800; inactive dispatch rows are NOT outputs.',
                'p_continue is a conditional lexical-token mass at an intermediate state.',
                'This is one development candidate, not isolated component causality or generalization.'])
    (run/'TRAINING_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True)
    a=p.parse_args();print(json.dumps(audit(a.run,a.tokenizer),indent=2))
