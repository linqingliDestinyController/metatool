"""Full recovery hooks for the frozen metatool driver, without changing its losses."""
from __future__ import annotations
import os
from pathlib import Path

from metatool.rl import full_recovery as disk
from metatool.rl.recovery_state import driver_state, restore_cpu_rng, resume_iterator


def enabled(trainer):
    return os.environ.get("METATOOL_FULL_RECOVERY") == "1" and trainer.config.actor_rollout_ref.actor.get("branch_route_enabled", False)


def initialize(trainer):
    # The candidate is dense, synchronous TP=SP=1 GRPO, no changing curriculum.
    # Reject unsupported dynamic state rather than silently dropping it.
    from verl.experimental.dataset.sampler import AbstractCurriculumSampler
    if trainer.use_critic or trainer.use_reference_policy or trainer.use_rm or trainer.async_rollout_mode:
        raise ValueError("Full recovery supports only this frozen actor-only synchronous candidate")
    if isinstance(trainer.train_dataloader.sampler, AbstractCurriculumSampler) or hasattr(trainer.train_dataset, "on_batch_end"):
        raise ValueError("Dynamic dataset/curriculum state is outside this checkpoint contract")
    if trainer.total_training_steps != 100:
        raise ValueError("Scheduler horizon must stay 100 across all attempts")
    trainer._recovery_epoch = 0
    trainer._recovery_resume_iterator = False
    trainer._recovery_pending_validation = False
    trainer._recovery_loaded = False


def save(trainer):
    from metatool.rl.checkpoint_safety import estimate_model_bytes
    if not hasattr(trainer, "_snapshot_model_bytes"):
        trainer._snapshot_model_bytes = estimate_model_bytes(trainer.config.actor_rollout_ref.model.path)
    result = disk.save_checkpoint(os.environ["METATOOL_RUN"], trainer.global_steps,
        trainer.resource_pool_manager.get_n_gpus(), trainer._snapshot_model_bytes,
        writer=lambda actor: trainer.actor_rollout_wg.save_checkpoint(
            str(actor), None, trainer.global_steps, max_ckpt_to_keep=None),
        driver_writer=lambda path: disk.save_tensor_file(driver_state(trainer, "after_update_before_validation"), path))
    print(f"[METATOOL_FULL_CHECKPOINT] step={trainer.global_steps} path={result['path']}", flush=True)


def boundary(trainer):
    disk.promote_boundary(os.environ["METATOOL_RUN"], trainer.global_steps,
        worker_writer=lambda path: trainer.actor_rollout_wg.save_recovery_runtime(str(path), trainer.global_steps),
        driver_writer=lambda path: disk.save_tensor_file(driver_state(trainer, "between_steps"), path))
    if trainer.global_steps == 50:
        from metatool.rl.checkpoint_safety import estimate_model_bytes
        size = estimate_model_bytes(trainer.config.actor_rollout_ref.model.path)
        disk.require_final_write_space(os.environ["METATOOL_RUN"], size)


def load(trainer):
    import torch
    run = Path(os.environ["METATOOL_RUN"])
    record = disk.latest(run)
    explicit = Path(os.environ["METATOOL_RESUME_CHECKPOINT"]).resolve()
    if explicit != Path(record["path"]) or record["step"] >= 100:
        raise ValueError("Resume only the latest committed step <100; step100 goes to finalize.py")
    path = explicit / "boundary_ready" if record["boundary"] else explicit
    state = torch.load(path / "driver.pt", map_location="cpu", weights_only=False)
    if (state["step"], state["total_training_steps"], state["phase"]) != (record["step"], 100, record["phase"]):
        raise ValueError("Driver/checkpoint phase differs")
    runtime = path if record["boundary"] else explicit / "actor"
    trainer.actor_rollout_wg.load_checkpoint(str(explicit / "actor"), del_local_after_load=False,
                                           recovery_runtime_path=str(runtime))
    trainer.global_steps = state["step"]
    trainer._recovery_epoch = state["epoch"]
    trainer.train_dataloader.load_state_dict(state["train_dataloader"])
    trainer.val_dataloader.load_state_dict(state["val_dataloader"])
    for key, value in state["validation_state"].items():
        setattr(trainer, key, value)
    trainer.max_steps_duration = state["max_steps_duration"]
    restore_cpu_rng(state["rng"])
    trainer._recovery_loaded = True
    trainer._recovery_resume_iterator = True
    trainer._recovery_pending_validation = record["phase"] != "between_steps"
    disk.atomic_json(Path(os.environ["METATOOL_ATTEMPT_DIR"]) / "RESTORED_DRIVER.json", {
        "completed_step": state["step"], "next_optimizer_step": state["step"] + 1,
        "epoch": state["epoch"], "phase": state["phase"],
        "full_training_state_restored": True, "bitwise_gpu_equivalence_verified": False})
    return state["step"]


def finish_pending(trainer):
    if not trainer._recovery_pending_validation:
        return
    # Model update already committed. Complete only the missing scheduled
    # validation; never repeat update_actor for this completed checkpoint step.
    step = trainer.global_steps
    if step % int(trainer.config.trainer.test_freq) == 0:
        metrics = trainer._validate()
        trainer._validation_after_validation(metrics)
    boundary(trainer)
    trainer._recovery_pending_validation = False


def iterator(trainer):
    restore = trainer._recovery_resume_iterator
    trainer._recovery_resume_iterator = False
    return resume_iterator(trainer.train_dataloader, restore)
