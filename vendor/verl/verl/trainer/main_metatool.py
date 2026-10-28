
import os
import socket
import torch
import hydra
import ray
from omegaconf import OmegaConf
from verl import DataProto
from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_type
from verl.workers.reward_manager.abstract import AbstractRewardManager


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    run_ppo(config)


# Define function to run PPO-like training process
def run_ppo(config) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
               for distributed PPO training, including Ray initialization settings,
               model paths, and training hyperparameters.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with local cluster configuration
        # Set environment variables in runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM log level, and allow runtime LoRA updates
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from config
        runtime_env = get_ppo_ray_runtime_env()
        task_keys = (
            "METATOOL_RUN", "METATOOL_WRAP_DEADLINE_TS", "PYTHONPATH", "PYTHONHASHSEED",
            "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "EXPERIMENT_NAME", "METATOOL_QUIET_REWARD",
            "WITHLENGTH", "REFINEDREWARD", "COARSEREWARD", "STRICTMATCH",
            "CORRECTMAX1", "MAX1STEP30MAX3", "SCHEDULEREWARD", "SCHEDULELENGTH",
            "USE_THINKING_SEMANTIC", "INTERMEDIATEREWARD", "TMPDIR", "PIP_CACHE_DIR",
            "TORCH_HOME", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "RAY_TMPDIR",
            "XDG_CACHE_HOME", "HF_HOME", "VLLM_USE_V1",
            "CUDA_LAUNCH_BLOCKING", "TORCH_SHOW_CPP_STACKTRACES", "HYDRA_FULL_ERROR",
            "METATOOL_CUDA_SYNC_BOUNDARIES", "METATOOL_CUDA_TRACE_DIR", "METATOOL_COMPILER_ERROR_DIR",
            "TORCHINDUCTOR_CACHE_DIR", "PYTHONFAULTHANDLER", "RAY_DEDUP_LOGS",
            "METATOOL_FULL_RECOVERY", "METATOOL_ATTEMPT_DIR", "METATOOL_RESUME_CHECKPOINT",
        )
        runtime_env["env_vars"].update({key: os.environ[key] for key in task_keys if key in os.environ})
        ray.init(runtime_env=runtime_env, num_cpus=config.ray_init.num_cpus,
                 _temp_dir=os.environ.get("RAY_TMPDIR", "/tmp/metatool_ray"))

    # Create remote instance of TaskRunner class and
    # remotely execute the `run` method of TaskRunner instance and wait for completion
    if (
        is_cuda_available
        and config.trainer.get("profile_steps") is not None
        and len(config.trainer.get("profile_steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available on CUDA platform. Please run 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] Get timeline trace file path from config, defaults to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # Please ensure main_task is not scheduled on head node
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to support distributed execution across multiple nodes and GPUs.
    """

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        worker processes, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters
                   required to set up and run the PPO training process.
        """
        # Print initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        import faulthandler
        import signal
        faulthandler.register(signal.SIGUSR2, all_threads=True)
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        import random
        import numpy as np
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download checkpoints from HDFS to local machine.
        # `use_shm` determines whether to use shared memory, which may lead to faster model loading if enabled
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # For multimodal LLM, may be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Define worker process class based on actor policy.
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                # import warnings
                # warnings.warn(f"Legacy worker impl is going to be deprecated, will be removed in the future. \
                #   Please set trainer.use_legacy_worker_impl = false to switch to the new worker implementation.")
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker process classes.
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Define resource pool specifications.
        # Map roles to resource pools.
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # We should adopt multi-source reward functions here:
        # - For rule-based rm, we directly call reward scores
        # - For model-based rm, we call the model
        # - For code-related prompts, if there are test cases, we send to sandbox
        # Finally, we combine all rewards together
        # Reward type depends on the data labels
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Add reference policy worker process if using KL loss or KL reward.
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Load reward manager for training and validation.
        reward_fn = AbstractRewardManager(
             tokenizer, num_examine=0
        )
        val_reward_fn = AbstractRewardManager(
             tokenizer, num_examine=1
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize PPO trainer.
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # Instantiate trainer to get global step count
        import verl.trainer.main_metatool
        verl.trainer.main_metatool._global_trainer = trainer
        # Initialize trainer workers.
        trainer.init_workers()
        # Start training process.
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True):
    """Create dataset.

    Arguments:
        data_paths: List of data file paths.
        data_config: Data configuration.
        tokenizer (Tokenizer): Tokenizer.
        processor (Processor): Processor.

    Returns:
        dataset (Dataset): Dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Check if custom dataset class is specified in data config
    # and if path to custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Validate that custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"Custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    else:
        # If no custom class specified, use default RLHFDataset class
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate dataset using determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create sampler for dataset.

    Arguments:
        data_config: Data configuration.
        dataset (Dataset): Dataset.

    Returns:
        sampler (Sampler): Sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_type(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum learning, num_workers must be 0 to prevent data caching. "
            "If the data loader caches data before batches complete, "
            "the curriculum sampler will have no chance to reorder it."
        )

    # Use sampler to facilitate checkpoint recovery.
    # If random shuffle is enabled in data config, create random sampler.
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If random shuffle is disabled, use sequential sampler to traverse dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
