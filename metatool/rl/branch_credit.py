"""metatool: outcome-first comparisons at ONE actual intermediate Think state.

Pure helpers. Auxiliary returns label STOP/CONTINUE; generated branch tokens
are not added to the original PPO trajectories. Missing states are explicit
zero-gradient collective slots, never invented model outputs.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

PROBE_PROMPTS = 12
PER_MODE = 2
NATURAL_PROMPTS = 96
NATURAL_DRAWS = 4
DP = 6
LOCAL_MINIBATCH = 32
MODES = ("stop", "continue")
ROUTE_FIELDS = ("branch_slot", "branch_input_ids", "branch_attention_mask",
                "branch_target", "branch_margin", "branch_importance_weight", "branch_active")

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def seed_for(step, state_ids, mode, replicate, stage):
    return int(digest([42, int(step), list(state_ids), mode, int(replicate), stage])[:8], 16) % (2**31-1)

def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def action_of(gt):
    call = "<tool_call>" in gt and "</tool_call>" in gt
    response = "<response>" in gt and "</response>" in gt
    if call == response:
        raise ValueError("Ambiguous original action label; do not relabel")
    return "call" if call else "respond"

def validate_protocol(p):
    expected = dict(schema_version=42, objective="metatool_stop_continue",
        model_tokenizer_length=151669, open_ids=[151667,198], close_id=151668,
        content_tokens_before_decision=64, continue_token_range=[0,151643],
        probe_budget=2048, per_mode=2, max_states=12, route_coefficient=.01,
        cost_tie_margin=.10, minimum_cost_gap=32,
        sampling=dict(temperature=1.,top_p=1.,top_k=-1,stop_token_ids=[151643,151645]))
    if any(p.get(k) != v for k,v in expected.items()):
        raise ValueError("The registered metatool protocol changed")
    return p

def load_protocol(path):
    return validate_protocol(json.loads(Path(path).read_text()))

def continue_ids(protocol):
    return range(*protocol["continue_token_range"])

def intermediate_prefix(ids, protocol):
    """Inspect ONLY the first 66 real response IDs, never the eventual outcome.

    Exact native open + 64 lexical tokens. No response re-tokenization and no
    moving a cut according to the future close token, success, or final length.
    """
    cut = len(protocol["open_ids"]) + protocol["content_tokens_before_decision"]
    if len(ids) < cut:
        return None, "shorter_than_checkpoint"
    prefix = list(ids[:cut])
    if prefix[:2] != protocol["open_ids"]:
        return None, "non_native_open"
    lo, hi = protocol["continue_token_range"]
    if any(type(i) is not int or not lo <= i < hi for i in prefix[2:]):
        return None, "closed_or_special_before_checkpoint"
    return prefix, "eligible"

def prefix_complete_utf8(prefix, tokenizer):
    """Reject a fixed cut inside a byte-BPE character; never move/re-tokenize it."""
    from metatool.rl.exact_token_layout import _BYTE_DECODER
    tokens=tokenizer.convert_ids_to_tokens(prefix[2:])
    raw=b''.join(bytes(_BYTE_DECODER[c] for c in token) for token in tokens)
    try:
        raw.decode('utf-8',errors='strict')
    except UnicodeDecodeError:
        return False
    return True

def select_states(prompts, ground_truths, responses, step, protocol, prefix_byte_check=None):
    if (len(prompts),len(ground_truths),len(responses)) != (96,96,384):
        raise ValueError("Need 96 original questions and their four actual responses")
    candidates = []; audit = []
    for i, (base, gt) in enumerate(zip(prompts, ground_truths)):
        base = list(map(int, base)); action = action_of(gt)
        item = dict(source_batch_index=i, prompt_ids=base, prompt_sha256=digest(base), expected_action=action,
                    ordering_key=digest([42,step,base]), replicates=[])
        chosen = None
        order = sorted(range(4), key=lambda r:digest([42,step,base,"replicate",r]))
        if action == "call":
            for rep in order:
                ids = responses[i*4+rep]
                prefix, reason = intermediate_prefix(ids, protocol)
                if prefix is not None and prefix_byte_check is not None and not prefix_byte_check(prefix):
                    prefix,reason=None,"incomplete_utf8_at_fixed_checkpoint"
                item["replicates"].append(dict(replicate=rep, inspected_ids=list(ids[:66]), reason=reason))
                if chosen is None and prefix is not None:
                    chosen = dict(active=True, source_batch_index=i, source_replicate=rep,
                        source_row=i*4+rep, prompt_ids=base, prompt_sha256=digest(base),
                        ground_truth=gt, expected_action="call", prefix_ids=prefix,
                        state_sha256=digest(base+prefix), source_response_ids=list(ids))
        if chosen is not None:
            candidates.append((item["ordering_key"], chosen))
        audit.append(item)
    candidates.sort(key=lambda pair:(pair[0],pair[1]["source_batch_index"]))
    selected = [item for _,item in candidates[:12]]
    # Real model context but no sampled comparison and exactly zero loss.
    # This is needed so every FSDP rank enters both forwards/backwards.
    while len(selected) < 12:
        selected.append(dict(active=False, source_batch_index=None, source_replicate=None,
            source_row=None, prompt_ids=list(map(int,prompts[0])),
            prompt_sha256=digest(list(map(int,prompts[0]))), ground_truth=None,
            expected_action=None, prefix_ids=list(protocol["open_ids"]),
            state_sha256=digest(list(map(int,prompts[0]))+protocol["open_ids"]),
            source_response_ids=None))
    for j,item in enumerate(selected):
        item.update(slot_index=j, training_step=int(step), importance_weight=1.0)
    return selected, dict(candidates=audit, eligible_prompts=len(candidates),
                         selected_active=sum(t["active"] for t in selected))

def branch_target(stop_successes, continue_successes, *, stop_tokens, continue_tokens,
                  active=True, protocol):
    if not active:
        if any((stop_successes,continue_successes,stop_tokens,continue_tokens)):
            raise ValueError("Inactive slots must not contain generated outcomes")
        return dict(target=0, margin=0., winner="inactive", target_reason="inactive",
                    q_stop=None,q_continue=None,success_delta=0.,cost_gap=None)
    for values in (stop_successes,continue_successes):
        if len(values)!=2 or any(x not in (0,1,False,True) for x in values):
            raise ValueError("Require two exact-Call outcomes per branch")
    for values in (stop_tokens,continue_tokens):
        if len(values)!=2 or any(type(x) is not int or not 1<=x<=2048 for x in values):
            raise ValueError("Require real incremental assistant token counts")
    qs=(1+sum(stop_successes))/4; qc=(1+sum(continue_successes))/4
    delta=qc-qs; gap=(sum(continue_tokens)-sum(stop_tokens))/2
    target=int(delta>0); margin=abs(delta); reason="success" if delta else "tie"
    if not delta and sum(stop_successes)==sum(continue_successes)==2 and abs(gap)>=protocol["minimum_cost_gap"]:
        target=int(gap<0); margin=protocol["cost_tie_margin"]; reason="both_correct_cost"
    return dict(target=target, margin=margin,
        winner=MODES[target] if margin else "tie", target_reason=reason,
        q_stop=qs,q_continue=qc,success_delta=delta,cost_gap=gap)

def route_slots(total_rows=384, dp=6, local_minibatch=32):
    if (total_rows,dp,local_minibatch)!=(384,6,32):
        raise ValueError("Frozen six-rank layout changed")
    return [rank*64+mini*32 for mini in range(2) for rank in range(6)]

def write_json_exclusive(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
