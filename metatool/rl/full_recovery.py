"""Step50 complete recovery, step100 BF16 endpoint, then retire step50.

No torch import here. Big tensor files are hashed while torch.save writes them;
on resume their hashes are checked once before a GPU training process is started.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid

SCHEMA = "metatool-recover50-final100-2"
SAVE_STEPS = (50, 100)
GIB = 1024 ** 3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=float)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    fsync_dir(path.parent)


def read(path):
    return json.loads(Path(path).read_text())


def binding(run):
    run = Path(run).resolve()
    doc = read(run / "EXPERIMENT.json")
    if doc.get("recovery_schema") != SCHEMA or not doc.get("run_uid"):
        raise ValueError("Only this full-recovery candidate can resume; unbound model-only runs cannot")
    return {"run": str(run), "run_uid": doc["run_uid"],
            "experiment_sha256": sha(run / "EXPERIMENT.json")}


def disk_plan(free_bytes, model_bytes):
    # FP32 model + two FP32 Adam moments, 10% serialization allowance + 512 MiB.
    full = math.ceil(3 * model_bytes * 1.10) + 512 * 1024 ** 2
    final = math.ceil(model_bytes * .5 * 1.10) + 256 * 1024 ** 2
    reserve = 10 * GIB
    return dict(free_bytes=int(free_bytes), existing_files_already_deducted=True,
                model_fp32_bytes=model_bytes, full_checkpoint_upper_bytes=full,
                final_bf16_upper_bytes=final, merge_upper_bytes=final, reserve_bytes=reserve,
                startup_required_bytes=full + reserve,
                startup_fits=free_bytes >= full + reserve,
                final_write_required_bytes=final + reserve,
                final_write_fits=free_bytes >= final + reserve,
                conservative_new_run_peak_bytes=full + final + reserve,
                conservative_entire_run_fits=free_bytes >= full + final + reserve,
                phase50_size_recheck_required=True,
                delete_previous_before_commit=False)


def require_final_write_space(run, model_bytes):
    plan = disk_plan(shutil.disk_usage(run).free, model_bytes)
    atomic_json(Path(run) / "FINAL_WRITE_SPACE.json", plan)
    if not plan["final_write_fits"]:
        raise OSError("Not enough actual FREE space for final model plus reserve; step50 is preserved")
    return plan


def final_model_state(state_dict):
    """Match the frozen merger's BF16 output, without changing live FP32 state."""
    return {name: value.bfloat16() if value.is_floating_point() else value
            for name, value in state_dict.items()}


def required_files(step, world_size=6):
    names = {"actor/huggingface/config.json"}
    for rank in range(world_size):
        names.add(f"actor/model_world_size_{world_size}_rank_{rank}.pt")
        if step == 50:
            names.update(f"actor/{kind}_world_size_{world_size}_rank_{rank}.pt"
                         for kind in ("optim", "extra_state", "runtime"))
            names.add(f"actor/runtime_summary_rank_{rank}.json")
        else:
            names.add(f"actor/final_summary_rank_{rank}.json")
    if step == 50:
        names.add("driver.pt")
    return names


class HashWriter:
    """Streaming torch.save ZIP sink. Does not hold tensor payloads in RAM."""
    def __init__(self, stream):
        self.stream, self.digest, self.count = stream, hashlib.sha256(), 0

    def write(self, data):
        written = self.stream.write(data)
        if written != len(data):
            raise OSError("Short checkpoint write")
        self.digest.update(data)
        self.count += written
        return written

    def flush(self):
        self.stream.flush()

    def tell(self):
        return self.count


def save_tensor_file(value, path, torch_module=None):
    if torch_module is None:
        import torch as torch_module
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        sink = HashWriter(stream)
        torch_module.save(value, sink)
        sink.flush()
        os.fsync(stream.fileno())
    item = {"bytes": sink.count, "sha256": sink.digest.hexdigest()}
    atomic_json(str(path) + ".integrity.json", item)
    return item


def inventory(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Checkpoint must not contain symlinks")
        if not path.is_file() or path.name.endswith(".integrity.json"):
            continue
        if path.suffix == ".pt":
            info = read(str(path) + ".integrity.json")
            if info["bytes"] != path.stat().st_size or info["bytes"] <= 0:
                raise ValueError("Incomplete tensor file")
        else:
            info = {"bytes": path.stat().st_size, "sha256": sha(path)}
        result[str(path.relative_to(root))] = info
    return result


def check_files(root, files, deep=False):
    for name, info in files.items():
        path = Path(root) / name
        if Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink():
            raise ValueError("Unsafe checkpoint path")
        if not path.is_file() or path.stat().st_size != info["bytes"]:
            raise ValueError(f"Missing/truncated checkpoint file: {path}")
        # Small state/config always verified; big tensors streamed once on resume.
        if (deep or path.suffix != ".pt" or info["bytes"] < 16 << 20) and sha(path) != info["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {path}")


def validate(path, run, deep=False, allow_final_model_only=False):
    path, run = Path(path).resolve(), Path(run).resolve()
    if path.parent != run / "checkpoints" or not re.fullmatch(r"global_step_\d+", path.name):
        raise ValueError("Checkpoint belongs to another run")
    doc = read(path / "FULL_CHECKPOINT.json")
    if doc["schema"] != SCHEMA or doc["binding"] != binding(run):
        raise ValueError("Checkpoint source/run binding differs")
    if path.name != f"global_step_{doc['step']}" or doc["world_size"] != 6:
        raise ValueError("Checkpoint step or six-rank layout differs")
    if doc["step"] not in SAVE_STEPS or doc["full_training_state"] != (doc["step"] == 50):
        raise ValueError("Only step50 is a complete training recovery checkpoint")
    if not required_files(doc["step"]).issubset(doc["files"]):
        raise ValueError("Checkpoint manifest is missing required state")
    if not doc["full_training_state"] and not allow_final_model_only:
        raise ValueError("Step100 is a final model, not resumable training state")
    check_files(path, doc["files"], deep)
    boundary = read(path / "BOUNDARY.json") if (path / "BOUNDARY.json").exists() else None
    if boundary:
        if boundary["step"] != doc["step"] or boundary["binding"] != doc["binding"]:
            raise ValueError("Boundary binding differs")
        check_files(path, boundary["files"], deep)
    return {**doc, "path": str(path), "boundary": boundary,
            "phase": "between_steps" if boundary else "after_update_before_validation"}


def latest(run, deep=False, allow_final_model_only=False):
    run = Path(run).resolve()
    found = []
    for path in (run / "checkpoints").glob("global_step_*"):
        if path.is_symlink():
            raise ValueError("Symlink checkpoint rejected")
        if (path / "FULL_CHECKPOINT.json").is_file():
            found.append(path)
    if not found:
        raise ValueError("No committed full checkpoint. Never warm-start or use an incomplete save")
    # If100 is committed, finalization need not reread the older ~100 GB state.
    selected = max(found, key=lambda p: int(p.name.removeprefix("global_step_")))
    return validate(selected, run, deep, allow_final_model_only)


def owned_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def save_checkpoint(run, step, world_size, model_bytes, writer, driver_writer):
    run = Path(run).resolve()
    if world_size != 6 or step not in SAVE_STEPS:
        raise ValueError("Frozen six-rank save schedule changed")
    bind = binding(run)
    root = run / "checkpoints"
    root.mkdir(exist_ok=True)
    if root.is_symlink():
        raise ValueError("Checkpoint root must be owned regular directory")
    final, staging = root / f"global_step_{step}", root / f".writing_step_{step}"
    if final.exists() or staging.exists():
        raise FileExistsError("Never overwrite a committed or incomplete checkpoint")
    previous = validate(root / "global_step_50", run) if step == 100 else None
    plan = disk_plan(shutil.disk_usage(root).free, model_bytes)
    enough = plan["startup_fits"] if step == 50 else plan["final_write_fits"]
    if not enough:
        raise OSError("Insufficient FREE space; do not delete the recovery checkpoint to make room")
    transaction = {"binding": bind, "step": step, "previous_step": 50 if previous else None,
                   "mode": "stage_verify_commit_then_retire50",
                   "old_checkpoint_survives_write_failure": bool(previous),
                   "disk": plan, "state": "WRITING"}
    event = run / "checkpoint_events" / f"step_{step:03d}.json"
    atomic_json(event, transaction)
    staging.mkdir()
    atomic_json(staging / "OWNER.json", bind)
    writer(staging / "actor")
    if step == 50:
        driver_writer(staging / "driver.pt")
    files = inventory(staging)
    required = required_files(step, world_size)
    if not required.issubset(files):
        raise ValueError(f"Incomplete checkpoint: {sorted(required - set(files))}")
    if step == 100 and any(re.match(r"actor/(optim|extra_state|runtime)_", name) or name == "driver.pt" for name in files):
        raise ValueError("Final step100 must save model-only state")
    for rank in range(world_size):
        summary = read(staging / "actor" / f"{'runtime' if step == 50 else 'final'}_summary_rank_{rank}.json")
        if (summary["completed_step"], summary["optimizer_step_min"], summary["optimizer_step_max"]) != (step, 2 * step, 2 * step):
            raise ValueError("Actual Adam updates do not match the completed training step")
    # Read back the smaller final model ONCE before committing/deleting step50.
    # Step50 hashes are streamed during save and deeply checked on explicit resume.
    if step == 100:
        check_files(staging, files, deep=True)
    doc = dict(schema=SCHEMA, binding=bind, step=step, world_size=world_size,
               total_training_steps=100, full_training_state=(step == 50), files=files,
               exact_training_resume=(step == 50), final_bf16=(step == 100),
               final_payload_verified_before_commit=(step == 100),
               bitwise_gpu_equivalence_verified=False,
               contents=(["model_fp32", "optimizer", "scheduler", "worker_rng", "rollout_rng",
                         "driver_rng", "train_and_validation_dataloader", "epoch", "completed_step"]
                         if step == 50 else ["model_bf16", "HF config/tokenizer", "rank_update_counts"]),
               selection_rule="fixed_final_step" if step == 100 else "recovery_only")
    atomic_json(staging / "FULL_CHECKPOINT.json", doc)
    fsync_dir(staging)
    staging.rename(final)
    fsync_dir(root)
    result = validate(final, run, allow_final_model_only=True)
    atomic_json(run / "checkpoint_metadata" / f"step_{step}" / "FULL_CHECKPOINT.json", doc)
    atomic_json(run / "recovery_manifest.json", {k: v for k, v in result.items() if k != "files"})
    if step == 100:
        atomic_json(run / "checkpoint_manifest.json", dict(best_step=100, path=str(final),
                    selection={"step": 100, "scores": None, "selection_rule": "fixed_final_step"},
                    full_training_state=False, exact_training_resume=False,
                    bitwise_gpu_equivalence_verified=False))
    transaction.update(state="COMMITTED", bytes=owned_size(final))
    atomic_json(event, transaction)
    if step == 100:
        retire_50_after_final(run)
    return result


def retire_50_after_final(run):
    """Idempotent retirement; a interrupted deletion cannot block the final model."""
    run = Path(run).resolve()
    final = validate(run / "checkpoints/global_step_100", run, allow_final_model_only=True)
    if not final.get("final_payload_verified_before_commit"):
        raise ValueError("Final payload must be verified before deleting step50")
    root = run / "checkpoints"
    old, retiring = root / "global_step_50", root / ".retiring_step_50"
    event = run / "STEP50_RETIREMENT.json"
    marker_hash = sha(Path(final["path"]) / "FULL_CHECKPOINT.json")
    if old.is_symlink() or retiring.is_symlink():
        raise ValueError("Symlink checkpoint retirement rejected")
    if old.exists():
        prior = validate(old, run)
        if retiring.exists():
            raise ValueError("Ambiguous prior retirement; preserve both paths")
        atomic_json(run / "checkpoint_metadata/step_50/FULL_CHECKPOINT.json", read(old / "FULL_CHECKPOINT.json"))
        if (old / "BOUNDARY.json").exists():
            atomic_json(run / "checkpoint_metadata/step_50/BOUNDARY.json", read(old / "BOUNDARY.json"))
        atomic_json(event, {"binding": final["binding"], "state": "FINAL_VERIFIED",
                    "final_marker_sha256": marker_hash, "retired_step": prior["step"],
                    "bytes_before_delete": owned_size(old)})
        old.rename(retiring)
        fsync_dir(root)
    if retiring.exists():
        proof = read(event)
        if proof["binding"] != final["binding"] or proof["final_marker_sha256"] != marker_hash:
            raise ValueError("Retirement proof differs from the final checkpoint")
        if any(p.is_symlink() for p in retiring.rglob("*")):
            raise ValueError("Symlink inside retirement directory rejected")
        shutil.rmtree(retiring)
        fsync_dir(root)
    if event.exists():
        atomic_json(event, {**read(event), "state": "RETIRED", "remaining_checkpoint": final["path"]})


def promote_boundary(run, step, worker_writer, driver_writer):
    """Small state commit after validation/step bookkeeping; no weight rewrite."""
    run = Path(run).resolve()
    if step != 50:
        return  # The final model will never resume training.
    path = run / "checkpoints" / f"global_step_{step}"
    doc = validate(path, run)
    if doc["boundary"]:
        raise ValueError("Boundary already committed")
    temp = path / (".boundary_" + uuid.uuid4().hex)
    temp.mkdir()
    worker_writer(temp)
    driver_writer(temp / "driver.pt")
    files = inventory(temp)
    required = {"driver.pt"} | {f"runtime_world_size_6_rank_{r}.pt" for r in range(6)}
    if not required.issubset(files):
        raise ValueError("Incomplete post-validation boundary state")
    final = path / "boundary_ready"
    if final.exists():
        raise ValueError("Uncommitted boundary directory exists; preserve it")
    temp.rename(final)
    fsync_dir(path)
    atomic_json(path / "BOUNDARY.json", dict(step=step, binding=doc["binding"],
                files={"boundary_ready/" + n: v for n, v in files.items()}))


def completed_metrics(run, total_steps=100):
    run = Path(run)
    lines = [line for line in (run / "train_metrics.jsonl").read_text().splitlines() if line.strip()] if (run / "train_metrics.jsonl").exists() else []
    rows, torn_tail = [], False
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except ValueError:
            if i != len(lines) - 1:
                raise ValueError("Malformed metrics row before the tail; inspect original evidence")
            torn_tail = True  # only the tail can be completed via a real ledger below
    keys = [int(r["step"]) for r in rows]
    if keys != sorted(set(keys)) or any(not 1 <= k <= total_steps for k in keys):
        raise ValueError("Duplicate/out-of-order canonical training rows")
    by_step = {r["step"]: r for r in rows}
    fallback = []
    for step in range(1, total_steps + 1):
        ledger = read(run / "training_updates" / f"step_{step:03d}.json")
        if ledger["step"] != step or ledger["metrics"].get("branch_actor/local_optimizer_steps") != 2:
            raise ValueError("Missing or invalid completed-update evidence")
        if step not in by_step:
            # A resumed checkpoint can have a completed optimizer update but no
            # timing/validation row. Preserve that distinction, never invent it.
            if step not in SAVE_STEPS:
                raise ValueError("Missing metrics away from a committed recovery point")
            event = read(run / "checkpoint_events" / f"step_{step:03d}.json")
            if event["state"] != "COMMITTED":
                raise ValueError("Uncommitted update cannot fill a missing metrics row")
            by_step[step] = {**ledger["metrics"], "step": step}
            fallback.append(step)
    return [by_step[s] for s in range(1, total_steps + 1)], {
        "missing_rows_read_from_real_update_ledgers": fallback,
        "torn_final_metrics_line_preserved": torn_tail,
        "train_metrics_modified_by_audit": False, "timing_or_validation_values_invented": False}


def archive_suffix(run, checkpoint, attempt):
    """Keep original evidence; canonical view contains only the retained lineage."""
    run, attempt = Path(run).resolve(), Path(attempt).resolve()
    if attempt.parent != run / "attempts":
        raise ValueError("Attempt archive must belong to this run")
    archive = attempt / "rolled_back_evidence"
    archive.mkdir(exist_ok=False)
    step, phase = checkpoint["step"], checkpoint["phase"]
    moved = []
    # A crash may follow directory rename but precede the tiny COMMITTED journal.
    # The full marker and verified tensor hashes, not the stale journal, decide.
    event = run / "checkpoint_events" / f"step_{step:03d}.json"
    old_event = read(event)
    if old_event["state"] != "COMMITTED":
        atomic_json(archive / "checkpoint_event_before_reconcile.json", old_event)
        atomic_json(event, {**old_event, "state": "COMMITTED", "reconciled_from_full_marker": True})
    cp = Path(checkpoint["path"])
    if phase != "between_steps":
        for path in [cp / "boundary_ready", *cp.glob(".boundary_*")]:
            if path.exists():
                if path.is_symlink():
                    raise ValueError("Symlink boundary rejected")
                dest = archive / "uncommitted_boundaries" / path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, dest)
    # A prior stage-then-rotate save could fail while its older full checkpoint
    # remains valid. Keep metadata before reclaiming only that unusable staging
    # directory; otherwise its bytes or filename would block a later save.
    discarded_staging = []
    for path in (run / "checkpoints").glob(".writing_step_*"):
        if path.name != ".writing_step_100" or path.is_symlink() or read(path / "OWNER.json") != checkpoint["binding"]:
            raise ValueError("Unexpected staging owner/commit; inspect without deleting")
        if any(p.is_symlink() for p in path.rglob("*")):
            raise ValueError("Symlink in staging rejected")
        files = {str(p.relative_to(path)): p.stat().st_size for p in path.rglob("*") if p.is_file()}
        for name in files:
            if name.endswith(".pt") and not re.fullmatch(r"(actor/(model|optim|extra_state|runtime)_world_size_6_rank_[0-5]|driver)\.pt", name):
                raise ValueError("Unexpected tensor in staging; preserve and inspect")
        dest = archive / "incomplete_checkpoint_metadata" / path.name
        shutil.copytree(path, dest, ignore=shutil.ignore_patterns("*.pt"))
        atomic_json(dest / "DISCARDED_INCOMPLETE_FILES.json", {"files_and_bytes": files,
            "committed_checkpoint": False, "reason": "explicit resume uses the verified complete step50 checkpoint"})
        shutil.rmtree(path)
        discarded_staging.append(str(path))
    for folder in ("main_rollouts", "branch_probes", "branch_actor_metrics", "training_updates", "validation", "checkpoint_events"):
        for f in sorted((run / folder).glob("step_*")):
            match = re.match(r"step_(\d+)", f.name)
            remove = match and int(match[1]) > step
            remove = remove or (folder == "validation" and phase != "between_steps" and match and int(match[1]) == step)
            if remove and f.is_file() and not f.is_symlink():
                dest = archive / folder / f.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(f, dest)
                moved.append(str(f.relative_to(run)))
    for name in ("train_metrics.jsonl", "validation_metrics.jsonl"):
        path = run / name
        if not path.exists():
            continue
        original = path.read_text()
        rows = original.splitlines(keepends=True)
        keep = []
        for line in rows:
            try:
                n = json.loads(line)["step"]
            except (ValueError, KeyError):
                break  # preserve torn tail in archive, never score it
            if n < step or (n == step and phase == "between_steps"):
                keep.append(line)
        os.replace(path, archive / name)
        with path.open("x") as stream:
            stream.write("".join(keep))
            stream.flush(); os.fsync(stream.fileno())
    dropped_updates = []
    for name in moved:
        match = re.fullmatch(r"training_updates/step_(\d+)\.json", name)
        if match:
            ledger = read(archive / name)
            if ledger.get("phase") == "actor_update_returned_before_validation" and ledger.get("metrics", {}).get("branch_actor/local_optimizer_steps") == 2:
                dropped_updates.append(int(match[1]))
    dropped_updates = sorted(set(dropped_updates))
    result = dict(resume_from_step=step, resume_phase=phase, next_optimizer_step=step + 1,
                  completed_updates_discarded=dropped_updates,
                  extra_completed_optimizer_updates_per_rank=2 * len(dropped_updates),
                  incomplete_rank_updates_may_add_unmeasured_cost=True,
                  discarded_incomplete_owned_staging=discarded_staging,
                  moved_files=moved, original_evidence_archive=str(archive))
    atomic_json(attempt / "RESUME_LINEAGE.json", result)
    return result


def ensure_merge_space(run, model_bytes):
    """Only finalization: retire step50 after verified step100, then check space."""
    run = Path(run).resolve()
    if not read(run / "TRAINING_SUMMARY.json")["complete"]:
        raise ValueError("Training audit must pass before final merge")
    final = latest(run, allow_final_model_only=True)
    if final["step"] != 100:
        raise ValueError("Final model not committed")
    retire_50_after_final(run)
    plan = disk_plan(shutil.disk_usage(run).free, model_bytes)
    if plan["free_bytes"] < plan["merge_upper_bytes"] + plan["reserve_bytes"]:
        raise OSError("Not enough free space for final BF16 HF merge; preserve step100")
    atomic_json(run / "FINALIZATION_STORAGE.json", {"step50_retired": True, "step100_model_preserved": True, "disk": plan})
