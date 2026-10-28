"""Explicit DP aggregation; no changes to gradient/optimizer collectives."""
from __future__ import annotations
import math
import os
from pathlib import Path
from metatool.rl.branch_credit import write_json_exclusive

FIELDS = ('active', 'loss', 'p_continue', 'margin', 'importance_weight', 'effective_weight',
          'target_continue', 'informative', 'aux_backward', 'coefficient')

def local_rows(metrics):
    if metrics['branch_actor/local_optimizer_steps'] != [2]:
        raise ValueError('Require two optimizer steps per rank, NOT twelve global optimizer steps')
    values = [metrics['branch_actor/' + k] for k in FIELDS]
    if any(len(v) != 2 for v in values):
        raise ValueError('Require two auxiliary states per rank')
    rows = [[float(values[j][i]) for j in range(len(FIELDS))] for i in range(2)]
    if not all(math.isfinite(v) for row in rows for v in row):
        raise ValueError('Nonfinite actor metric')
    return rows

def reduce_rank_rows(rank_rows):
    """Pure reconstruction of means over all ranks and BOTH sequential updates."""
    if not rank_rows or any(len(rows) != 2 or any(len(row) != len(FIELDS) for row in rows) for rows in rank_rows):
        raise ValueError('Incomplete auxiliary metrics')
    if not all(math.isfinite(v) for rows in rank_rows for row in rows for v in row):
        raise ValueError('Nonfinite rank metrics')
    n = 2 * len(rank_rows)
    result = {'branch_actor/' + k: sum(row[j] for rows in rank_rows for row in rows)/n
              for j, k in enumerate(FIELDS)}
    result.update({'branch_actor/metric_world_size': len(rank_rows),
                   'branch_actor/global_aux_states': n,
                   'branch_actor/optimizer_steps_per_rank': 2,
                   'branch_actor/local_optimizer_steps': 2})
    # First DataProto metadata survives concatenation: preserve rank0 explicitly.
    result.update({'branch_actor_rank0/' + k: sum(row[j] for row in rank_rows[0])/2
                   for j, k in enumerate(FIELDS)})
    return result

def gather_rank_rows(rows, device):
    """Real tensor collective; also exercised by the two-rank CPU/Gloo test."""
    import torch
    import torch.distributed as dist
    tensor = torch.tensor(rows, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return [value.cpu().tolist() for value in gathered]

def aggregate_actor_metrics(metrics, *, step, device, expected_world_size=6):
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_world_size() != expected_world_size or expected_world_size != 6:
        raise ValueError('metatool production logging requires exactly six DP ranks, TP=SP=1')
    if not 1 <= step <= 100:
        raise ValueError('Missing/invalid branch_step')
    rank_rows = gather_rank_rows(local_rows(metrics), device)
    aggregated = reduce_rank_rows(rank_rows)
    if dist.get_rank() == 0:
        write_json_exclusive(Path(os.environ['METATOOL_RUN'])/'branch_actor_metrics'/f'step_{step:03d}.json',
            {'step': step, 'world_size': 6, 'fields': list(FIELDS), 'rank_rows': rank_rows,
             'aggregated': aggregated,
             'scope': '12 auxiliary states: 6 DP ranks x 2 sequential optimizer minibatches; other actor metrics retain inherited scope'})
    result = {k:v for k,v in metrics.items() if not k.startswith('branch_actor/')}
    result.update({k:[v] for k,v in aggregated.items()})
    return result
