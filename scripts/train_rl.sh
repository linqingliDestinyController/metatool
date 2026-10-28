#!/usr/bin/env bash
set -euo pipefail
# Internal entry point. Use python run.py rl to prepare and launch.
[[ $# -eq 2 ]] || { echo "usage: train.sh WORK RUN" >&2; exit 2; }
BASE="$1"
RUN="$2"
RUN_ID="$(basename -- "$RUN")"
SRC="$BASE/src"
[[ -f "$BASE/PREFLIGHT.json" && -f "$RUN/EXPERIMENT.json" ]] || exit 2
[[ "${VLLM_USE_V1:-}" == 0 ]] || { echo 'Training requires VLLM_USE_V1=0' >&2; exit 2; }
export METATOOL_RUN="$RUN"
export PYTHONPATH="$SRC"
export METATOOL_CUDA_SYNC_BOUNDARIES=1
export METATOOL_CUDA_TRACE_DIR="$RUN/cuda_runtime"
# Normal training is asynchronous within each CUDA boundary. Keep the reviewed
# storage-lifetime synchronizations, without global per-kernel debug blocking.
export CUDA_LAUNCH_BLOCKING=0
export METATOOL_FULL_RECOVERY=1
export METATOOL_CUDA_TRACE_DIR="$METATOOL_ATTEMPT_DIR/cuda_runtime"
RESUME_MODE=disable
if [[ -n "${METATOOL_RESUME_CHECKPOINT:-}" ]]; then RESUME_MODE=auto; fi
cd "$SRC"
exec python -u -m verl.trainer.main_metatool \
  data.train_files="$BASE/data/train.parquet" \
  data.val_files="$BASE/data/validation.parquet" \
  data.train_batch_size=96 data.val_batch_size=80 \
  data.max_prompt_length=3072 data.max_response_length=2048 \
  data.dataloader_num_workers=2 data.validation_shuffle=False '+data.seed=42' \
  actor_rollout_ref.model.path="$BASE/model" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=48 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.shuffle=False \
  actor_rollout_ref.actor.use_fused_kernels=False \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  +actor_rollout_ref.actor.branch_route_enabled=True \
  +actor_rollout_ref.actor.branch_route_coef=0.01 \
  "+actor_rollout_ref.actor.branch_protocol_path=$BASE/assets/branch_protocol.json" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6144 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean-weighted \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]' \
  'actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]' \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.enforce_eager=True \
  '+actor_rollout_ref.rollout.enable_prefix_caching=False' \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.max_model_len=6144 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.01 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  '+actor_rollout_ref.rollout.seed=42' \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=True \
  +algorithm.meta_control_strength=0.0 \
  algorithm.absolute_action_penalty_enabled=true \
  algorithm.use_kl_in_reward=False \
  trainer.logger=console trainer.n_gpus_per_node=6 trainer.nnodes=1 \
  trainer.critic_warmup=0 trainer.save_freq=50 trainer.test_freq=50 \
  trainer.val_before_train=True trainer.resume_mode="$RESUME_MODE" \
  trainer.project_name=metatool \
  trainer.experiment_name="$RUN_ID" \
  trainer.default_local_dir="$RUN/checkpoints" \
  trainer.total_epochs=4 trainer.total_training_steps=100 \
  hydra.run.dir="$METATOOL_ATTEMPT_DIR/hydra"
