"""metatool intermediate stop/continue gradient, accumulated with PPO before the SAME optimizer step."""
from __future__ import annotations

def route_loss_from_logits(logits, think_ids, close_id, target, margin, coefficient, importance_weight=None):
    import torch
    # Native close token versus lexical continuation mass at the SAME intermediate state.
    # This is a conditional decision probability, not a calibrated probability of useful reasoning.
    z = logits.float()
    think_ids = torch.as_tensor(think_ids, device=z.device, dtype=torch.long)
    log_t = torch.logsumexp(z.index_select(-1, think_ids), dim=-1)
    log_f = z[..., close_id]
    normalizer = torch.logaddexp(log_f, log_t)
    log_p = torch.stack((log_f-normalizer, log_t-normalizer), dim=-1)
    y = target.detach().long().reshape(-1)
    w = margin.detach().float().reshape(-1)
    iw = (torch.ones_like(w) if importance_weight is None else
          importance_weight.detach().float().reshape(-1))
    if len(iw) != len(w) or not bool((torch.isfinite(iw) & (iw > 0) & (iw <= 12)).all()):
        raise ValueError("Invalid detached action importance weight")
    if not 0 <= coefficient <= .1:
        raise ValueError("Invalid route coefficient")
    if z.ndim != 2 or z.shape[0] != len(y) or len(y) != len(w):
        raise ValueError("Route batch shape mismatch")
    if not bool(((y == 0) | (y == 1)).all()) or not bool(((w >= 0) & (w <= .5)).all()):
        raise ValueError("Invalid detached route feedback")
    loss = -(log_p.gather(-1, y[:, None]).squeeze(-1) * w * iw).mean() * coefficient
    if not bool(torch.isfinite(loss)): raise RuntimeError("Non-finite route loss")
    return loss, log_p

def backward_route(actor, mini_batch):
    import torch
    from metatool.rl.branch_credit import load_protocol, continue_ids, intermediate_prefix
    if actor.ulysses_sequence_parallel_size != 1 or actor.use_fused_kernels:
        raise ValueError("metatool state forward requires SP=1 and unfused logits")
    if bool(getattr(getattr(actor.actor_module, "cpu_offload", None), "offload_params", False)):
        raise ValueError("FSDP CPUOffload cannot accumulate this auxiliary gradient")
    selected = mini_batch.batch["branch_slot"].bool()
    if len(selected) != 32 or int(selected.sum().item()) != 1:
        raise ValueError("Exactly ONE route slot per rank/local optimizer minibatch is required")
    if not hasattr(actor, "_branch_protocol"):
        actor._branch_protocol = load_protocol(actor.config.branch_protocol_path)
    protocol = actor._branch_protocol
    ids = mini_batch.batch["branch_input_ids"][selected]
    mask = mini_batch.batch["branch_attention_mask"][selected].bool()
    ids = ids[mask].reshape(1, -1)
    active = bool(mini_batch.batch["branch_active"][selected][0].item())
    if active:
        prefix = ids[0, -66:].tolist()
        if ids.shape[1] <= 66 or intermediate_prefix(prefix, protocol)[0] != prefix:
            raise ValueError("Actor did not receive the actual intermediate state")
    elif ids[0, -2:].tolist() != protocol["open_ids"]:
        raise ValueError("Invalid zero-gradient collective placeholder")
    target = mini_batch.batch["branch_target"][selected]
    margin = mini_batch.batch["branch_margin"][selected]
    importance = mini_batch.batch["branch_importance_weight"][selected]
    if not active and float(margin[0]) != 0:
        raise ValueError("Inactive collective slot has nonzero credit")
    if float(importance[0]) != 1.0:
        raise ValueError("metatool samples Call-prefix states, not Call/Respond strata")
    position_ids = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    # One unpadded sequence on every rank, including zero-margin ties. Thus every
    # rank enters the same FSDP forward/backward collectives. No submodule bypass.
    with torch.autocast(device_type=actor.device_name, dtype=torch.bfloat16):
        out = actor.actor_module(input_ids=ids, attention_mask=None,
                                 position_ids=position_ids, use_cache=False)
        loss, log_p = route_loss_from_logits(out.logits[:, -1, :],
            list(continue_ids(protocol)), protocol["close_id"], target, margin,
            float(actor.config.branch_route_coef), importance)
    loss.backward()  # No zero_grad, clipping, optimizer.step, or /32 here.
    return {"branch_actor/active": float(active), "branch_actor/loss": float(loss.detach()),
            "branch_actor/p_continue": float(log_p[0, 1].detach().exp()),
            "branch_actor/margin": float(margin[0]),
            "branch_actor/importance_weight": float(importance[0]),
            "branch_actor/effective_weight": float(importance[0] * margin[0]),
            "branch_actor/target_continue": float(target[0]),
            "branch_actor/informative": float(margin[0] > 0),
            "branch_actor/aux_backward": 1.0,
            "branch_actor/coefficient": float(actor.config.branch_route_coef)}
