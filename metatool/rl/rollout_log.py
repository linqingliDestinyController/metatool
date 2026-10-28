"""Persist every actual main response without changing sampling or training tensors."""
import json, os
from pathlib import Path
from metatool.rl.branch_credit import write_json_exclusive

def save_main_rollouts(step, traces, batch, tokenizer):
    rows=[]
    for i, trace in enumerate(traces):
        record=dict(trace)
        record.update(row=i, replicate=i%4, source_batch_index=i//4,
                      uid=str(batch.non_tensor_batch['uid'][i//4]),
                      ground_truth=batch.non_tensor_batch['reward_model'][i//4]['ground_truth'])
        record['raw_decoded']=tokenizer.decode(record['response_token_ids'],
            skip_special_tokens=False,clean_up_tokenization_spaces=False)
        rows.append(record)
    if len(rows)!=384: raise ValueError('Incomplete main rollout collection')
    write_json_exclusive(Path(os.environ['METATOOL_RUN'])/'main_rollouts'/f'step_{step:03d}.json',
                         dict(step=step,records=rows,actual_outputs=len(rows)))
