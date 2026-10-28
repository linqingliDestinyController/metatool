"""CPU-only diagnostics around the real CUDA rollout; never retry a failed engine.

No torch import here. A caught failure must not trigger tensor formatting, RNG
restoration, memory profiling, empty_cache(), or a distributed collective.
"""
from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import GeneratorType
from contextlib import contextmanager

_context = contextvars.ContextVar("metatool_cuda_trace", default=None)
_failed_process = False


def emit(event, **fields):
    """Best effort CPU evidence. A full disk must not mask the primary error."""
    row = {"time_ns": time.time_ns(), "pid": os.getpid(), "event": event,
           **(_context.get() or {}), **fields}
    try:
        line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
        root = os.environ.get("METATOOL_CUDA_TRACE_DIR")
        if root:
            folder = Path(root)
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / f"pid_{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
        if event.endswith("error"):
            sys.stderr.write("[METATOOL_CUDA_PRIMARY] " + line)
            sys.stderr.flush()
    except BaseException:
        # Diagnostics are secondary; even stderr may be unavailable on failure.
        pass


def record_failure(stage, exc):
    global _failed_process
    _failed_process = True
    emit("runtime_error", stage=stage, exception_type=type(exc).__name__,
         exception=str(exc), traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
         cleanup_skipped=True, retry_allowed=False)


def ensure_healthy():
    if _failed_process:
        raise RuntimeError("This rollout process has failed. Start a fresh process; no in-process CUDA retry.")


def cuda_boundary(device, stage):
    """Serialize outstanding work before storage lifetime changes, if enabled."""
    ensure_healthy()
    emit("boundary_begin", stage=stage)
    try:
        if os.environ.get("METATOOL_CUDA_SYNC_BOUNDARIES", "1") == "1":
            device.synchronize()
    except BaseException as exc:
        record_failure(stage, exc)
        raise
    emit("boundary_ok", stage=stage)


@contextmanager
def success_only_context(manager, stage):
    """Restore GPU state on success only; all failures abort this worker."""
    if isinstance(getattr(manager, "gen", None), GeneratorType):
        raise TypeError("Unsafe generator context: use success_only_rng instead")
    manager.__enter__()
    try:
        yield manager
    except BaseException as exc:
        record_failure(stage, exc)
        raise
    else:
        manager.__exit__(None, None, None)


@contextmanager
def success_only_rng(torch_module, devices, stage="branch_rng_context"):
    """Snapshot RNG explicitly; never create a fork_rng generator finalizer.

    On success restore exactly the CPU and listed CUDA RNG states. Any failure
    aborts this worker, so its saved RNG state is discarded without CUDA calls.
    """
    ensure_healthy()
    cpu_state = torch_module.get_rng_state()
    device_states = [(device, torch_module.cuda.get_rng_state(device)) for device in devices]
    try:
        yield
    except BaseException as exc:
        record_failure(stage, exc)
        raise
    else:
        torch_module.set_rng_state(cpu_state)
        for device, state in device_states:
            torch_module.cuda.set_rng_state(state, device)


@contextmanager
def sampling_override(params, updates):
    previous = {}
    try:
        for key, value in updates.items():
            if hasattr(params, key):
                previous[key] = getattr(params, key)
                setattr(params, key, value)
        yield
    finally:
        # These are ordinary Python attributes, so safe even after a CUDA error.
        for key, value in previous.items():
            setattr(params, key, value)


def trace_rollout(func):
    @functools.wraps(func)
    def wrapper(self, prompts, *args, **kwargs):
        ensure_healthy()
        meta = prompts.meta_info
        token = _context.set({"step": meta.get("global_steps"),
                              "branch_probe": bool(meta.get("branch_probe", False)),
                              "validate": bool(meta.get("validate", False)),
                              "rank": getattr(self, "rank", os.environ.get("RANK")),
                              "call_id": f"{os.getpid()}-{time.time_ns()}"})
        emit("rollout_begin", cuda_launch_blocking=os.environ.get("CUDA_LAUNCH_BLOCKING"),
             sync_boundaries=os.environ.get("METATOOL_CUDA_SYNC_BOUNDARIES", "1"))
        try:
            result = func(self, prompts, *args, **kwargs)
            emit("rollout_ok")
            return result
        except BaseException as exc:
            record_failure("worker.generate_sequences", exc)
            raise
        finally:
            _context.reset(token)
    return wrapper


def checked_generate(engine, *, prompts, sampling_params, device, vocab_size, **kwargs):
    """Validate actual prompt IDs, then call the unchanged vLLM sampler once."""
    ensure_healthy()
    # vLLM inputs here are CPU lists. Never convert GPU tensors in error handlers.
    ids = [[int(t) for t in row["prompt_token_ids"]] for row in prompts]
    for row in ids:
        if not row or any(t < 0 or t >= vocab_size for t in row):
            raise ValueError("Invalid actual prompt token ID before vLLM dispatch")
    fingerprint = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    emit("sample_begin", prompt_sha256=fingerprint, prompt_lengths=[len(row) for row in ids],
         batch_size=len(ids), prefix_caching=bool(engine.llm_engine.cache_config.enable_prefix_caching))
    cuda_boundary(device, "before_vllm_generate")
    try:
        result = engine.generate(prompts=prompts, sampling_params=sampling_params, **kwargs)
        cuda_boundary(device, "after_vllm_generate")
    except BaseException as exc:
        record_failure("vllm.generate", exc)
        raise
    emit("sample_ok", returned_requests=len(result))
    return result


def reset_prefix_cache_checked(engine):
    """Check the result instead of silently ignoring a refused cache reset."""
    ensure_healthy()
    if engine.llm_engine.has_unfinished_requests():
        raise RuntimeError("Weight synchronization attempted with unfinished vLLM requests")
    if engine.llm_engine.cache_config.enable_prefix_caching:
        if engine.reset_prefix_cache() is not True:
            raise RuntimeError("vLLM prefix-cache reset refused after weight synchronization")
    emit("weights_cache_ready", prefix_caching=bool(engine.llm_engine.cache_config.enable_prefix_caching))
