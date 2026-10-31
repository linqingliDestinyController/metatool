# Metatool 
This is the demo RL training code for the paper:"MetaTool : Thinking Just Enough, Calling Tools When Needed" . 

# Model Structure
![main](main.png)

## Data

| File | Examples | Usage |
|---|---:|---|
| `data/rl/train.parquet` | 3,907 | Training |
| `data/rl/validation.parquet` | 80 | Validation |
| `data/api_bank/level-1-api_processed.json` | 399 | API-Bank Level 1 |
| `data/api_bank/level-2-api_processed.json` | 67 | API-Bank Level 2 |
| `data/api_bank/level-3-api_processed.json` | 131 | API-Bank Level 3 |

Training and validation paths can point to the bundled parquet files or external copies matching `configs/dataset_profile.json`. API-Bank data is used for evaluation.

## Run

Run from the project directory:

```bash
python run.py train \
  --model /absolute/path/to/Qwen3-8B \
  --train-data "$PWD/data/rl/train.parquet" \
  --validation-data "$PWD/data/rl/validation.parquet" \
  --work /absolute/path/to/metatool_rl_work \
  --gpus 0,1,2,3,4,5
```

Provide a complete local Hugging Face model with safetensors weights and its tokenizer. Model metadata and tokenizer files must match `configs/model_profile.json`, including the tokenizer and chat-template hashes. The model directory must contain full weights rather than an adapter or unmerged FSDP checkpoint.

`WORK` must be a new directory outside the project tree and separate from the input model. Model and dataset paths are registered when the run is created. The six-GPU configuration requires at least 75 GiB of memory per GPU.

## Checkpointing and Recovery

Each completed training step saves 384 main rollouts with generated token IDs, prompt IDs, decoded text, and labels. Auxiliary branch outputs, state selections, and supervision targets are also saved. Inactive placeholder slots do not count as generated outputs.

At step 50, a full checkpoint saves the model, optimizer, scheduler, RNG states, driver and dataloader states, and supported rollout states. Recovery becomes available after this checkpoint has been successfully committed.

At step 100, the final model checkpoint is saved, verified, and committed before the step-50 checkpoint is deleted. This deletion does not wait for `merged_hf` export. The step-100 checkpoint contains model state. After the final model has been exported and verified, the remaining training checkpoints are removed; the input model, final model, rollout evidence, and logs are retained.

```bash
# Resume from the committed full step-50 checkpoint.
python run.py train --work /absolute/path/to/metatool_rl_work \
  --gpus 0,1,2,3,4,5 --resume

# Retry preflight after a failure before any optimizer updates.
python run.py train --work /absolute/path/to/metatool_rl_work \
  --gpus 0,1,2,3,4,5 --retry-preflight

# Complete export or cleanup after training.
python run.py finalize --work /absolute/path/to/metatool_rl_work
```

Recovery uses the registered model and dataset paths. There is no recovery checkpoint before the complete step-50 checkpoint is committed. The launcher does not automatically retry a failed training run. Manual resume may repeat updates made after step 50 and before the failure.

| Output | Path relative to `WORK` |
|---|---|
| Final model | `runs/metatool_seed42/merged_hf/` |
| Main rollouts on the retained trajectory | `runs/metatool_seed42/main_rollouts/` |
| Auxiliary outputs and targets | `runs/metatool_seed42/branch_probes/` |
| Rollback evidence | `runs/metatool_seed42/attempts/attempt_*/rolled_back_evidence/` |
| Launch and error logs | `runs/metatool_seed42/attempts/` |
| Training audit logs | `runs/metatool_seed42/training_audit_*.log` |
| Model export logs | `runs/metatool_seed42/merge_*.log` |
| Training counts | `runs/metatool_seed42/TRAINING_SUMMARY.json` |
| Registered model and dataset inputs | `INSTALL.json` |

Complete rollout records include the rollback evidence directories. `TRAINING_SUMMARY.json` is produced after the training audit succeeds. Early input-binding errors may appear in the terminal before attempt logs are created.

Generated rollouts are stored under `WORK`, not in `data/`. Disk checks use the actual remaining free space. The step-50 checkpoint is retained until the step-100 checkpoint has been committed and verified.

## API-Bank Evaluation

```bash
FINAL=/absolute/path/to/metatool_rl_work/runs/metatool_seed42/merged_hf
python run.py evaluate --model "$FINAL" \
  --output /absolute/path/to/metatool_rl_work/eval_api --gpu 0
```

Evaluation uses 597 examples: 399 at Level 1, 67 at Level 2, and 131 at Level 3. The parser, soft-match rules, and sampling settings are defined by the evaluation implementation. Raw generation chunks are saved before parsing and scoring. Generation runs in a separate vLLM process.

Score existing predictions on CPU:

```bash
CUDA_VISIBLE_DEVICES="" python run.py score \
  --predictions /absolute/path/to/predictions.jsonl \
  --output /absolute/path/to/api_scores.json
```

## Environment

The reference environment is specified in `configs/environment.json`:

| Dependency | Version |
|---|---|
| Python | 3.10.21 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 4.51.3 |
| vLLM | 0.8.5.post1 |
| Ray | 2.47.1 |
| CUDA | 12.4 |

Reuse a compatible environment and install dependencies from `requirements.txt`. FlashAttention must match the installed PyTorch and CUDA versions.

## Project Structure

- `run.py`: training, recovery, export, evaluation, and scoring entry point.
- `metatool/rl/`: token credit assignment, action correction, Stop/Continue supervision, rollout logging, and recovery.
- `metatool/evaluation/`: API-Bank inference and scoring.
- `runtime/`: input registration, training launch, model export, and cleanup.
- `vendor/verl/`: distributed training engine and compatibility layers.
- `configs/`: model, tokenizer, dataset, environment, and training configuration.
- `data/rl/`: training and validation parquet files.
- `data/api_bank/`: API-Bank evaluation inputs.

The `metatool` Python module and `metatool_seed42` run directory names are used by the implementation. The distributed controller in `vendor/verl/verl/single_controller/` uses Ray to coordinate multiple GPU workers. Third-party compatibility components retain their original licenses.

## CPU Checks

```bash
CUDA_VISIBLE_DEVICES="" python tests_cpu.py
```

The default checks cover small-tensor PPO and routing gradients and CPU RNG. Input-registration and launch-dispatch checks additionally accept `--model-metadata`, `--train-data`, and `--validation-data`, using miniature weight containers and mocked GPU subprocesses.

Regenerate the release manifest after changing project files:

```bash
CUDA_VISIBLE_DEVICES="" python scripts/make_manifest.py
```

## License

See `LICENSE`, `NOTICE`, and `licenses/`.
