"""Lazy Torch state helpers shared by real workers/driver and native CPU tests."""
from __future__ import annotations
import copy
import os
from pathlib import Path
import random

from metatool.rl.full_recovery import atomic_json, save_tensor_file


def cpu_rng_state():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def restore_cpu_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def optimizer_summary(optimizer, scheduler, step):
    values = [int(v["step"].item()) if hasattr(v["step"], "item") else int(v["step"])
              for v in optimizer.state.values() if "step" in v]
    if not values or min(values) != 2 * step or max(values) != 2 * step:
        raise ValueError(f"Adam state update count differs from 2*completed_step={2*step}")
    return {"completed_step": step, "optimizer_state_entries": len(values),
            "optimizer_step_min": min(values), "optimizer_step_max": max(values),
            "scheduler_last_epoch": scheduler.state_dict()["last_epoch"],
            "learning_rates": list(scheduler.get_last_lr())}


def counter_state(counter):
    if counter is None:
        return None
    state = dict(vars(counter))
    if any(not isinstance(v, int) for v in state.values()):
        raise ValueError("Unexpected vLLM counter state; inspect installed 0.8.5.post1 API")
    return state


def set_counter(counter, state):
    if state is None:
        if counter is not None:
            raise ValueError("vLLM counter schema changed")
        return
    if counter is None or set(vars(counter)) != set(state):
        raise ValueError("vLLM counter schema changed")
    vars(counter).update(state)


def capture_worker(worker, step):
    from metatool.rl.cuda_runtime_guard import ensure_healthy
    ensure_healthy()
    if getattr(worker, "_recovery_completed_step", 0) != step:
        raise ValueError("Worker did not complete this outer update")
    engine = worker.rollout.inference_engine
    if engine.llm_engine.has_unfinished_requests():
        raise ValueError("Only quiescent rollout engines can be checkpointed")
    manager = worker.rollout_sharding_manager
    summary = optimizer_summary(worker.actor_optimizer, worker.actor_lr_scheduler, step)
    return {"completed_step": step, "summary": summary,
            "rng": worker.checkpoint_manager.get_rng_state(),
            "gen_random_states": manager.gen_random_states.clone().cpu(),
            "torch_random_states": manager.torch_random_states.clone().cpu(),
            "request_counter": counter_state(getattr(engine, "request_counter", None)),
            "sequence_counter": counter_state(getattr(engine.llm_engine, "seq_counter", None))}


def save_worker(worker, directory, step):
    import torch.distributed as dist
    directory = Path(directory)
    state = capture_worker(worker, step)
    rank, world = dist.get_rank(), dist.get_world_size()
    save_tensor_file(state, directory / f"runtime_world_size_{world}_rank_{rank}.pt")
    atomic_json(directory / f"runtime_summary_rank_{rank}.json", state["summary"])
    return state["summary"]


def restore_worker(worker, directory):
    import torch
    import torch.distributed as dist
    from metatool.rl.cuda_runtime_guard import ensure_healthy
    ensure_healthy()
    rank, world = dist.get_rank(), dist.get_world_size()
    path = Path(directory) / f"runtime_world_size_{world}_rank_{rank}.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    actual = optimizer_summary(worker.actor_optimizer, worker.actor_lr_scheduler, state["completed_step"])
    if actual != state["summary"]:
        raise ValueError("Restored optimizer/scheduler differs from saved runtime summary")
    worker._recovery_completed_step = state["completed_step"]
    manager = worker.rollout_sharding_manager
    manager.gen_random_states = state["gen_random_states"].clone()
    manager.torch_random_states = state["torch_random_states"].clone()
    engine = worker.rollout.inference_engine
    if engine.llm_engine.has_unfinished_requests():
        raise ValueError("Cannot resume into an active rollout engine")
    set_counter(getattr(engine, "request_counter", None), state["request_counter"])
    set_counter(getattr(engine.llm_engine, "seq_counter", None), state["sequence_counter"])
    worker.checkpoint_manager.load_rng_state(state["rng"])
    atomic_json(Path(os.environ["METATOOL_ATTEMPT_DIR"]) / f"RESTORED_RANK_{rank}.json", actual)
    return actual


VALIDATION_FIELDS = ("validation_initial", "validation_best", "validation_bad_consecutive",
                   "validation_last_val_step", "validation_stop")


def driver_state(trainer, phase):
    return {"step": trainer.global_steps, "epoch": trainer._recovery_epoch,
            "total_training_steps": trainer.total_training_steps,
            "phase": phase, "train_dataloader": copy.deepcopy(trainer.train_dataloader.state_dict()),
            "val_dataloader": copy.deepcopy(trainer.val_dataloader.state_dict()),
            "rng": cpu_rng_state(), "max_steps_duration": getattr(trainer, "max_steps_duration", 0),
            "validation_state": {k: getattr(trainer, k) for k in VALIDATION_FIELDS}}


def resume_iterator(loader, restore_after_creation=False):
    """Restored mid-epoch iterator creation must not consume extra driver RNG."""
    saved = cpu_rng_state() if restore_after_creation else None
    iterator = iter(loader)
    if saved is not None:
        restore_cpu_rng(saved)
    return iterator
