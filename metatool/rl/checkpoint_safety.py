"""CPU-only shared checkpoint utilities. Recovery policy lives in full_recovery.py."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import struct


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, default=float)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def estimate_model_bytes(model_path):
    """Read safetensors headers only; conservatively allow FP32 saved weights."""
    count = 0
    seen = set()
    files = sorted(Path(model_path).glob("*.safetensors"))
    if not files:
        raise ValueError("No local safetensors headers for disk-budget estimation")
    for path in files:
        with path.open("rb") as stream:
            head = stream.read(8)
            if len(head) != 8:
                raise ValueError(f"Truncated safetensors header: {path}")
            size = struct.unpack("<Q", head)[0]
            if not 0 < size <= 64 * 1024 ** 2:
                raise ValueError(f"Invalid safetensors header length: {path}")
            entries = json.loads(stream.read(size))
        for name, entry in entries.items():
            if name == "__metadata__":
                continue
            if name in seen:
                raise ValueError(f"Duplicate tensor name: {name}")
            seen.add(name)
            count += math.prod(entry["shape"])
    if count <= 0:
        raise ValueError("Empty model metadata")
    return count * 4


def record_completed_update(run, step, epoch, metrics):
    """Durable evidence before validation; not a fabricated train_metrics row."""
    path = Path(run) / "training_updates" / f"step_{step:03d}.json"
    if path.exists():
        raise FileExistsError("This optimizer step already has a completion record")
    atomic_json(path, {"step": step, "epoch": epoch, "phase": "actor_update_returned_before_validation",
                       "metrics": metrics, "record_type": "update_ledger_not_checkpoint"})
