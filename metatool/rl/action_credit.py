"""Action-credit correction and trajectory decision monitoring for metatool."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Decisions:
    deliberation: str | None
    action: str | None
    deliberation_char: int | None
    action_char: int | None

    @property
    def mode(self):
        if self.deliberation == "think":
            return "think"
        if self.deliberation == "fast" and self.action == "call":
            return "tool"
        if self.deliberation == "fast" and self.action == "respond":
            return "direct"
        return "invalid"


def decisions(text: str) -> Decisions:
    """Read existing structure; do not add a literal <direct> or control token.

    The first non-whitespace content token versus </think> is a proxy for the
    compute decision. The first distinguishing tag-name token after </think>
    is a proxy for call/respond. The common opening <think> is NOT a gate.
    An unclosed thought is still 'think', never cheap/empty reasoning.
    """
    opening = re.match(r"\s*<think>", text)
    if not opening:
        return Decisions(None, None, None, None)
    end = text.find("</think>", opening.end())
    stop = end if end >= 0 else len(text)
    body = re.search(r"\S", text[opening.end():stop])
    if body:
        mode, char = "think", opening.end() + body.start()
    elif end >= 0:
        mode, char = "fast", end
    else:
        return Decisions(None, None, None, None)
    terminal = re.match(r"\s*<(tool_call|response)>", text[end + len("</think>"):]) if end >= 0 else None
    if terminal:
        action = {"tool_call": "call", "response": "respond"}[terminal.group(1)]
        action_char = end + len("</think>") + terminal.start(1)
    else:
        action, action_char = None, None
    return Decisions(mode, action, char, action_char)


ABSOLUTE_ACTION_PENALTY = 0.05


def structural_action(text, response_ids, tokenizer):
    """Recognize one closed call or response branch without consulting ground truth."""
    payload = text
    if response_ids[-1] == tokenizer.eos_token_id:
        suffix = tokenizer.decode([response_ids[-1]], skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
        if not suffix or not payload.endswith(suffix):
            raise RuntimeError("Generated special-token suffix disagrees with sampled IDs")
        payload = payload[:-len(suffix)]

    call = "<tool_call>" in payload or "</tool_call>" in payload
    respond = "<response>" in payload or "</response>" in payload
    if call and respond:
        return None, None, "mixed"
    tags = ("<think>", "</think>", "<tool_call>", "</tool_call>",
            "<response>", "</response>")
    counts = [payload.count(tag) for tag in tags]
    if counts[0] != 1 or counts[1] != 1 or (counts[2], counts[3], counts[4], counts[5]) not in (
        (1, 1, 0, 0), (0, 0, 1, 1)
    ):
        reason = "unclosed" if any(a > b for a, b in ((counts[0], counts[1]),
                     (counts[2], counts[3]), (counts[4], counts[5]))) else "undecidable"
        return None, None, reason
    if re.fullmatch(r"\s*<think>.*?</think>\s*<(tool_call|response)>.*?</\1>\s*",
                    payload, flags=re.S) is None:
        return None, None, "undecidable"
    choice = decisions(text)
    expected = "call" if counts[2] else "respond"
    if choice.action != expected or choice.action_char is None:
        raise RuntimeError("Structural action and sampled action token disagree")
    return choice.action, choice.action_char, "valid"


def expected_action(ground_truth):
    """Use the existing training answer to identify call versus direct response."""
    call = "<tool_call>" in ground_truth and "</tool_call>" in ground_truth
    respond = "<response>" in ground_truth and "</response>" in ground_truth
    if call == respond:
        return None
    return "call" if call else "respond"


def posterior_deltas(uids, labels, successes):
    """Return group-centered Beta(1+s, 1+n-s) posterior means in [-1, 1].

    Compare only different observed choices for the SAME prompt UID. Groups
    with one choice, no success or no failure produce no extra signal. The
    four on-policy samples are reused; there is no forced alternative rollout.
    Estimates are intentionally local to this batch, avoiding cross-question
    difficulty confounding and stale global success tables.
    """
    if not (len(uids) == len(labels) == len(successes)):
        raise ValueError("Mismatched feedback lengths")
    if any(s not in (0, 1, False, True) for s in successes):
        raise ValueError("Feedback must be binary completion/decision success")
    groups = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[uid].append(i)
    result = [0.0] * len(uids)
    mixed = informative = 0
    for indices in groups.values():
        known = [i for i in indices if labels[i] is not None]
        counts = Counter(labels[i] for i in known)
        if len(counts) < 2:
            continue
        mixed += 1
        if not 0 < sum(successes[i] for i in known) < len(known):
            continue
        hits = Counter()
        for i in known:
            hits[labels[i]] += int(successes[i])
        means = {m: (1.0 + hits[m]) / (2.0 + n) for m, n in counts.items()}
        center = sum(counts[m] * means[m] for m in counts) / len(known)
        for i in known:
            result[i] = means[labels[i]] - center
        informative += int(any(abs(result[i]) > 1e-12 for i in known))
    return result, {"groups": len(groups), "mixed_groups": mixed, "informative_groups": informative}


def bounded_advantage(advantage, delta, strength):
    """Scalar reference for the tensor adapter; preserve sign and zero exactly."""
    if not all(math.isfinite(x) for x in (advantage, delta, strength)):
        raise ValueError("Non-finite credit input")
    if not 0 <= strength <= 0.2 or not -1 <= delta <= 1:
        raise ValueError("Credit exceeds the supported conservative bound")
    return advantage + strength * abs(advantage) * delta


def apply_meta_control(data, tokenizer, strength=0.0, absolute_action_penalty_enabled=False):
    """Called AFTER original std-normalized GRPO and BEFORE actor.update_actor.

    Apply the fixed negative correction at one actual wrong-action token.
    Group statistics are logged; their centered offset has zero strength.
    Rewards, returns and token weights remain intact.
    """
    strength = float(strength)
    if not math.isfinite(strength) or strength != 0:
        raise ValueError("metatool uses zero group-centered offset")
    if not isinstance(absolute_action_penalty_enabled, bool):
        raise ValueError("absolute_action_penalty_enabled must be boolean")
    if strength == 0 and not absolute_action_penalty_enabled:
        return {}

    import torch
    from metatool.rl.validation import score_one
    from metatool.rl.exact_token_layout import response_byte_layout, span_to_token_span

    batch = data.batch
    mask = batch["response_mask"]
    ids = batch["responses"]
    old_adv = batch["advantages"]
    weights = batch["token_weight"]
    if not (mask.shape == ids.shape == old_adv.shape == weights.shape):
        raise ValueError("Response/advantage/weight shape mismatch")
    uids = data.non_tensor_batch["uid"].tolist()
    if any(n != 4 for n in Counter(uids).values()):
        raise ValueError("This matched experiment requires rollout.n=4 per UID")

    choices, success, positions, need_tool = [], [], [], []
    absolute_counts = Counter()
    absolute_positions = []
    for i in range(ids.shape[0]):
        n = int(mask[i].sum().item())
        if n <= 0 or not bool(mask[i, :n].bool().all()) or bool(mask[i, n:].bool().any()):
            raise ValueError("Expected nonempty contiguous generated response mask")
        row = ids[i, :n].detach().cpu().tolist()
        text, offsets = response_byte_layout(row, tokenizer)
        choice = decisions(text)
        gt = data.non_tensor_batch["reward_model"][i]["ground_truth"]
        # Feedback reuses the frozen training validation definition. This
        # is a training-only choice monitor, NOT an API-Bank test scorer.
        truncated = n == ids.shape[1] and row[-1] != tokenizer.eos_token_id
        _, tool_ok, decision_ok = score_one(text, gt, truncated)
        success.append(int(tool_ok if tool_ok is not None else decision_ok))
        need_tool.append("<tool_call>" in gt)
        choices.append(choice)
        spans = []
        for char in (choice.deliberation_char, choice.action_char):
            if char is None:
                spans.append(None)
            else:
                # One actual sampled token, not a retokenized string or a
                # length-amplified block. Overlapping gates are averaged below.
                lo, _ = span_to_token_span(text, char, char + 1, offsets)
                spans.append(lo if weights[i, lo].item() > 0 else None)
        positions.append(spans)
        if absolute_action_penalty_enabled:
            action, action_char, status = structural_action(text, row, tokenizer)
            absolute_counts[status] += 1
            if status == "valid":
                target = expected_action(gt)
                if target is None:
                    absolute_counts["ambiguous_ground_truth"] += 1
                elif action == target:
                    absolute_counts["correct"] += 1
                else:
                    lo, _ = span_to_token_span(text, action_char, action_char + 1, offsets)
                    if lo >= n or weights[i, lo].item() <= 0:
                        raise RuntimeError("Wrong action token is not a weighted generated position")
                    absolute_positions.append((i, lo))
                    absolute_counts["wrong"] += 1

    compute_delta, compute_stats = posterior_deltas(uids, [c.deliberation for c in choices], success)
    if absolute_action_penalty_enabled:
        # The inactive group-centered call/respond correction is replaced, not stacked.
        action_delta = [0.0] * len(choices)
        action_stats = {"groups": len(set(uids)), "mixed_groups": 0, "informative_groups": 0}
    else:
        action_delta, action_stats = posterior_deltas(uids, [c.action for c in choices], success)
    row_indices, token_indices, deltas = [], [], []
    for i, spans in enumerate(positions):
        by_token = defaultdict(list)
        for token, delta in zip(spans, (compute_delta[i], action_delta[i])):
            if token is not None:
                by_token[token].append(delta)
        for token, values in by_token.items():
            delta = sum(values) / len(values)
            if abs(delta) > 1e-12:
                row_indices.append(i)
                token_indices.append(token)
                deltas.append(delta)

    changed = 0
    max_relative_change = 0.0
    zero_grpo_penalized = sum(old_adv[i, token].item() == 0 for i, token in absolute_positions)
    if absolute_action_penalty_enabled and absolute_positions:
        # Clone even when advantages and returns share storage after GRPO.
        updated = batch["advantages"].clone()
        ri = torch.tensor([p[0] for p in absolute_positions], device=updated.device, dtype=torch.long)
        ti = torch.tensor([p[1] for p in absolute_positions], device=updated.device, dtype=torch.long)
        updated[ri, ti] = updated[ri, ti] - ABSOLUTE_ACTION_PENALTY
        if not bool(torch.isfinite(updated[ri, ti]).all()):
            raise RuntimeError("Non-finite absolute action advantage")
        batch["advantages"] = updated
        changed = int((updated != old_adv).sum().item())

    size = len(choices)
    modes = Counter(c.mode for c in choices)
    logs = {"metacontrol/strength": strength, "metacontrol/changed_tokens": changed,
            "metacontrol/max_relative_change": max_relative_change,
            "metacontrol/proxy_success": sum(success) / size}
    for stage, stats in (("compute", compute_stats), ("action", action_stats)):
        logs.update({f"metacontrol/{stage}_{k}": v for k, v in stats.items()})
    logs.update({f"metacontrol/mode_{m}": modes[m] / size for m in ("direct", "tool", "think", "invalid")})
    for needed, label in ((True, "need_tool"), (False, "no_tool")):
        selected = [i for i, value in enumerate(need_tool) if value == needed]
        if selected:
            logs[f"metacontrol/{label}_n"] = len(selected)
            logs[f"metacontrol/{label}_success"] = sum(success[i] for i in selected) / len(selected)
    if absolute_action_penalty_enabled:
        logs["metacontrol/absolute_action_penalty"] = ABSOLUTE_ACTION_PENALTY
        logs["metacontrol/absolute_action_changed_tokens"] = len(absolute_positions)
        logs["metacontrol/absolute_action_zero_grpo_penalized"] = zero_grpo_penalized
        for label in ("valid", "correct", "wrong", "mixed", "unclosed",
                      "undecidable", "ambiguous_ground_truth"):
            logs[f"metacontrol/absolute_action_{label}"] = absolute_counts[label]
    return logs
