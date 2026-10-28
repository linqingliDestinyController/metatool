# metatool — Stage 2

独立在线 RL 代码包，直接从 `--model` 指定的本地完整 HF 模型开始。当前沿用 **Qwen3-8B、6 GPU、100 步**训练配置。

包内 `data/` 只包含 API-Bank 三个评测输入文件：

| 文件 | 样本数 |
|---|---:|
| `data/api_bank/level-1-api_processed.json` | 399 |
| `data/api_bank/level-2-api_processed.json` | 67 |
| `data/api_bank/level-3-api_processed.json` | 131 |

包内不含偏好训练实现、偏好正负样本、Natural/dev80 数据、RL 训练 parquet、tokenizer 文件或历史模型预测。这些数据没有换一个目录继续打包。

## 直接运行

模型权重与原有 RL 训练/验证 parquet 由外部路径提供。API-Bank 是评测输入，不用于训练。

```bash
python run.py train \
  --model /absolute/path/to/Qwen3-8B \
  --train-data /absolute/path/to/train.parquet \
  --validation-data /absolute/path/to/validation.parquet \
  --work /absolute/path/to/metatool_rl_work \
  --gpus 0,1,2,3,4,5
```

首次运行必须指定这三项输入。模型须为完整且兼容的 HF 目录，不能是 adapter 或 FSDP 分片。`WORK` 使用代码包外的新目录，并与输入模型分离。训练输入与 `configs/dataset_profile.json` 的原有 RL 数据哈希绑定；不能用 API-Bank 的 JSON 替代 parquet。

启动器只做必要的输入、tokenizer、核心环境、GPU 可用性和磁盘检查，然后训练。不执行前置微调、额外 GPU smoke 或实验矩阵。模型首次注册保存完整 SHA256，文件大小、mtime、ctime 和 inode 未变化时复用内容绑定，发生变化则重新核对内容。

输入模型可以是原始兼容模型，也可以是已有的兼容完整模型，不要求其他阶段的目录或导出记录。**从原始基座直接进行 RL 是新的仅 RL 实验，其结果需要重测。** 本包未提供该实验的成绩。

## 保存与恢复

每步保存全部 384 条主 rollout 的实际 IDs、prompt IDs、解码文本和标签。辅助分支输出、选择和目标继续保存；无效占位不计作真实生成。

第 50 步保存完整模型、优化器、scheduler、RNG、driver/dataloader 和支持的 rollout 状态。第 100 步提交并校验后才删除第 50 步。最终模型导出并校验后清除本运行的训练 checkpoint，保留输入模型、最终模型、全部 rollout 和日志。

```bash
# 修复故障后，从完整50步状态继续。
python run.py train --work /absolute/path/to/metatool_rl_work \
  --gpus 0,1,2,3,4,5 --resume

# 仅启动检查失败且没有优化更新时使用。
python run.py train --work /absolute/path/to/metatool_rl_work \
  --gpus 0,1,2,3,4,5 --retry-preflight

# 训练已完成，仅恢复导出或清理，不训练。
python run.py finalize --work /absolute/path/to/metatool_rl_work
```

恢复会使用已登记的模型和数据路径，不能换输入。第 50 步前发生故障时没有相应恢复点；不会自动重复已消耗的预算。完整分布式 GPU 恢复的逐位一致性未在此次交付中重新测试。

| 输出 | 相对 WORK 的路径 |
|---|---|
| 最终模型 | `runs/metatool_seed42/merged_hf/` |
| 全部主 rollout | `runs/metatool_seed42/main_rollouts/` |
| 辅助输出及目标 | `runs/metatool_seed42/branch_probes/` |
| 每次启动与错误日志 | `runs/metatool_seed42/attempts/` |
| 训练计数 | `runs/metatool_seed42/TRAINING_SUMMARY.json` |
| 模型和外部数据绑定 | `INSTALL.json` |

运行生成的 rollout 在 WORK 下保存，不会添加到发布包的 `data/`。磁盘检查使用扣除已有文件后的实际可用空间；不会提前删除第 50 步来腾空间。

## API-Bank 评测

```bash
FINAL=/absolute/path/to/metatool_rl_work/runs/metatool_seed42/merged_hf
python run.py evaluate --model "$FINAL" \
  --output /absolute/path/to/metatool_rl_work/eval_api --gpu 0
```

保持固定的 597 个样本、原解析与 soft-match 规则、采样参数。先保存原始生成 chunk，再解析评分。评测使用独立 vLLM 进程。

已有预测可用 CPU 评分：

```bash
CUDA_VISIBLE_DEVICES="" python run.py score \
  --predictions /absolute/path/to/predictions.jsonl \
  --output /absolute/path/to/api_scores.json
```

## 环境

优先复用满足核心版本的环境：Python 3.10.21、Torch 2.6.0+cu124、Transformers 4.51.3、vLLM 0.8.5.post1、Ray 2.47.1。版本见 `configs/environment.json`。

新环境安装示例：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install flash-attn --no-build-isolation
```

FlashAttention 须与 PyTorch/CUDA 匹配。原归档未提供完整依赖及编译器锁文件，此次没有新建 GPU 环境验证安装流程。模型结构、原生 token IDs、tokenizer/chat template、batch 和分布式布局与当前配置绑定；其元数据约束见 `configs/model_profile.json`，实际 tokenizer 文件由输入模型提供。

## 目录与 CPU 验证

- `metatool/rl/`：token 信用、动作修正、停止/继续监督、rollout 记录与恢复。
- `vendor/verl/`：当前训练使用的分布式引擎与适配。
- `runtime/`：外部输入绑定、启动、模型导出和清理。
- `metatool/evaluation/`：API-Bank 推理和评分。
- `data/api_bank/`：唯一随包提供的数据。

`vendor/verl/verl/third_party/` 是依赖兼容层，不是额外训练方法或数据。`vllm/` 由当前生成及权重同步路径使用；`torch/` 是旧版 PyTorch 的 FSDP2 状态加载兼容实现，当前 FSDP1 配置不启用；`sglang/` 服务于备用推理后端，当前 vLLM 配置不启用。保留这些通用兼容分支及其版权信息，不把它们作为 MetaTool 的方法贡献。未使用的数学题和搜索工具示例已移除。

`vendor/verl/verl/single_controller/` 是当前训练直接依赖的多卡任务调度层：通过 Ray 创建 GPU worker、分配资源、分发 batch 和远程方法调用、收集各 rank 的结果。这里的 single 指统一控制进程；实际计算由多个 GPU worker 完成。MetaTool 的决策和损失实现位于 `metatool/rl/`。

```bash
CUDA_VISIBLE_DEVICES="" python tests_cpu.py
```

默认只检查小张量 PPO/路线梯度及 CPU RNG。需要检查输入绑定和启动调度时，额外提供 `--model-metadata`、`--train-data` 和 `--validation-data`；检查只使用模型目录中的元数据，采用微型权重容器和模拟 GPU 子进程，不加载语言模型权重。

本次没有真实 GPU 训练或生成新预测。检查结果见 `checks/`。执行提示在 `docs/CODEX_PROMPT_ZH.txt`；许可见 `LICENSE` 和 `NOTICE`。
