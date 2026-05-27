import contextlib
import csv
import inspect
import io
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from shutil import rmtree
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union, cast
from scipy.stats import spearmanr
from datetime import datetime

import datasets
import deepspeed
import deepspeed.utils.zero_to_fp32
import diffusers
import imageio.v2 as imageio
import imageio.v3 as iio
import numpy as np
import torch
import transformers
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from diffusers.training_utils import EMAModel

# import mlflow
# import humanize
from einops import rearrange, repeat
from PIL import Image
from rich.logging import RichHandler
from stable_diffusion_application.kaimm import Stability
from stable_diffusion_application.kaimm.accelerate_dataset import skip_first_batches
from stable_diffusion_application.kaimm.accelerate_utils import load_deepspeed_state, load_rng_state, prepare, save_deepspeed_state, set_seed
from stable_diffusion_application.kaimm.logging import get_logger
from stable_diffusion_application.kaimm.state import PartialState
from tqdm.auto import tqdm

from assets.models_eval import *

from ..configs.experiment_config import ExperimentConfig
from ..data import DataConfig
from ..pipelines.base_pipeline import BasePipelineConfig, PipelineMixin
from ..utils import *
from .optimizers import *
from .schedulers import SchedulerConfig
from .stability import StabilityConfig

torch._dynamo.config.cache_size_limit = 128


@dataclass
class TrainerConfig(ExperimentConfig):
    """Configuration for training regimen"""

    _target: Type = field(default_factory=lambda: Trainer)
    """target class to instantiate"""

    seed: Optional[int] = 0
    """Random seed."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps."""
    use_ema: bool = False
    """Whether to use EMA."""
    ema_decay: float = 0.9993
    """EMA decay."""
    ema_start_step: int = 0
    """EMA start step."""
    data_per_ema: int = 800_000
    """data per EMA decay."""
    save_ema: bool = True
    """Whether to save the whole ema model."""

    max_steps: Optional[int] = -1
    """Maximal steps to train."""
    num_epochs: int = 1
    """Number of epochs to train."""
    data_per_save: int = 500_000
    """Number of data between saves."""
    data_per_val: int = 500_000
    """Number of data between validation."""
    step_per_save: Optional[int] = None
    """Step per save. If not None, it will overwrite data_per_save."""
    step_per_val: Optional[int] = None
    """Step per validation. If not None, it will overwrite data_per_val."""
    step_per_valmetrics: Optional[int] = None
    """Step per validation for metrics. """
    step_per_valloss: Optional[int] = None
    """Step per valloss for metrics. """
    step_per_ema: Optional[int] = None
    """Step per EMA. If not None, it will overwrite data_per_ema."""
    step_per_measure_time: Optional[int] = 1
    """Step per meature time."""
    measure_time: bool = False
    """Whether to measure time"""
    verbose: bool = False
    step_per_tb: Optional[int] = 1

    train_data: DataConfig = DataConfig()
    """train video data loader configuration"""
    aux_data: DataConfig = DataConfig()
    """auxiliary data loader configuration"""
    joint_train_prob: float = 0.0
    """probability of using aux_data in joint training"""

    trainable_modules: Optional[List[Type]] = field(default_factory=lambda: [])
    """List of modules to train. Each element can be a string or a dictionary. \
    If it is a string, it is the name of the module. If it is a dictionary, \
    it must have a key "module" which is the name of the module. \
    e.g. [{"unet": ["motion_modules"]}] will only train the motion_modules in unet."""
    nontrainable_modules: Optional[List[Type]] = field(default_factory=lambda: [])
    """List of modules to not train. Each element can be a string or a dictionary. """

    optimizer: AdamWOptimizerConfig = AdamWOptimizerConfig()
    """Optimizer configuration"""
    scheduler: SchedulerConfig = SchedulerConfig()
    """Scheduler configuration"""

    pipeline: BasePipelineConfig = BasePipelineConfig()
    """Configuration for pipeline initialization."""

    val_at_begin: bool = True
    """Whether to validate at the training begins."""
    valmetrics_at_begin: bool = False
    """Whether to validate for metrics at the training begins."""
    valloss_at_begin: bool = False
    """Whether to val loss at the training begins."""
    num_valloss_timesteps: int = 20
    """Number of timesteps for Val loss."""
    val_data: DataConfig = DataConfig(num_samples=1)
    """Val data loader configuration"""
    valmetrics_data: DataConfig = DataConfig(num_samples=1)
    """Val data for metrics loader configuration"""
    valloss_data: DataConfig = DataConfig(num_samples=1)
    """Val data for metrics loader configuration"""
    val_types: List[str] = field(default_factory=lambda: ["gt", "t2v", "vae"])
    """Types of validation samples to generate."""

    resume: bool = False
    """Whether to resume training."""
    resume_ckpt_path: Optional[str] = None
    """Checkpoint path to resume training."""
    resume_no_skip: bool = False
    """Whether to not skip dataloader when resume training."""
    add_skipped_step: Optional[int] = 0
    """Additional skipped dataloader steps when resume training."""

    stability: StabilityConfig = StabilityConfig()
    """Configuration for stability."""
    deepspeed: str = "src/engine/zero_stage2_config.json"
    """Configuration for deepspeed initialization."""
    training_precision: Literal["fp16", "fp32", "bf16"] = "fp16"
    """Training precision."""
    delete_deepspeed_weights: bool = True
    skip_train_data: Optional[int] = None
    skip_aux_data: Optional[int] = None
    tb_add_image: bool = False
    tb_add_video: bool = False

    num_frames: Optional[int] = None
    """global num_frames"""
    height: Optional[int] = None
    """global height"""
    width: Optional[int] = None
    """global width"""

    def __post_init__(self):
        if self.num_frames is not None:
            self.train_data.num_frames = self.num_frames
            self.val_data.num_frames = self.num_frames
            if "num_frames" in self.pipeline.call:
                self.pipeline.call["num_frames"] = self.num_frames
            log_to_rank0(f"Replacing num_frames in train_data, val_data, and call with {self.num_frames}")
        if self.height is not None:
            self.train_data.height = self.height
            self.aux_data.height = self.height
            self.val_data.height = self.height
            if "height" in self.pipeline.call:
                self.pipeline.call["height"] = self.height
            log_to_rank0(f"Replacing height in train_data, val_data, and call with {self.height}")
        if self.width is not None:
            self.train_data.width = self.width
            self.aux_data.width = self.width
            self.val_data.width = self.width
            if "width" in self.pipeline.call:
                self.pipeline.call["width"] = self.width
            log_to_rank0(f"Replacing width in train_data, val_data, and call with {self.width}")


@contextlib.contextmanager
def ema_context(ema_model, zero_optimizer):
    # 在进入上下文时执行的操作：存储EMA状态
    ema_model.store(zero_optimizer.single_partition_of_fp32_groups)
    ema_model.copy_to(zero_optimizer.single_partition_of_fp32_groups)
    zero_optimizer.step_no_grad()
    yield  # 允许代码块运行
    # 在退出上下文时执行的操作：恢复EMA状态
    ema_model.restore(zero_optimizer.single_partition_of_fp32_groups)
    zero_optimizer.step_no_grad()


class Manager(torch.nn.Module):
    def __init__(self, pipeline):
        super().__init__()
        module_names, _ = pipeline._get_signature_keys(pipeline)
        module_names.add('transformer_ref')
        module_names = sorted(module_names)  # To ensure every rank gets the same module_names
        for module_name in module_names:
            module = getattr(pipeline, module_name, None)
            if module is not None:
                if isinstance(module, torch.nn.Module) or isinstance(module, torch.nn.Parameter):
                    setattr(self, module_name, module)

        def _remove_frozen_module():
            for module_name in module_names:
                module = getattr(pipeline, module_name, None)
                if module and isinstance(module, torch.nn.Module):
                    rg = next(module.parameters()).requires_grad
                    if not rg:
                        delattr(self, module_name)

        self.remove_frozen_module = _remove_frozen_module

        def _recover_frozen_module():
            for module_name in module_names:
                module = getattr(pipeline, module_name, None)
                if module and isinstance(module, torch.nn.Module):
                    setattr(self, module_name, module)

        self.recover_frozen_module = _recover_frozen_module

    def save_ckpts(self, ds_state_dict, path):
        if hasattr(self, "unet"):
            self.unet.save_ckpt(ds_state_dict, path)
        if hasattr(self, "visual_tokenizer"):
            save_model_from_ds("visual_tokenizer", ds_state_dict, path)
            if hasattr(self, "discriminator"):
                save_model_from_ds("discriminator", ds_state_dict, path)
        if hasattr(self, "transformer"):
            save_model_from_ds("transformer", ds_state_dict, path)


class Trainer:
    def __init__(self, config: TrainerConfig):
        # make 3dvae faster when using AMD
        if is_amd():
            wrap_conv3d()
        self.config = config
        config.set_timestamp()
        self.exp_root = config.get_base_dir()
        os.makedirs(self.exp_root, exist_ok=True)

        # init deepspeed, state, logger
        self.init_state_and_logger()
        # init pipeline, manager, dataloader, optimizer, lr_scheduler, trainning info, stability
        self.init_train()
        self.pipeline.train_status = TrainingStatus.INIT

        (
            self.manager,
            self.optimizer,
            self.lr_scheduler,
            self.train_dataloader,
            self.val_dataloader,
            self.valmetrics_dataloader,
            self.valloss_dataloader,
            self.aux_dataloader,
        ) = prepare(
            self.manager,
            self.optimizer,
            self.lr_scheduler,
            self.train_dataloader,
            self.val_dataloader,
            self.valmetrics_dataloader,
            self.valloss_dataloader,
            self.aux_dataloader,
        )
        self.manager, self.optimizer, _, _ = deepspeed.initialize(
            model=self.manager, optimizer=self.optimizer, config=self.deepspeed_config, dist_init_required=True
        )

        # for ema
        if self.config.use_ema:
            self.ema_model = EMAModel(self.optimizer.single_partition_of_fp32_groups, decay=self.config.ema_decay, update_after_step=self.config.ema_start_step)
            self.ema_model.to(self.state.device, torch.float32)

        self.logger_tb = self.manager.monitor.tb_monitor
        if self.state.is_main_process:
            config.print_to_terminal()
            config.save_config()

        self.timer = Timer()

    def init_state_and_logger(self):
        with io.open(self.config.deepspeed, "r", encoding="utf-8") as f:
            deepspeed_config = json.load(f)
        now = datetime.now()
        port = (now.hour * 100 + now.minute) % 10000
        deepspeed.init_distributed(distributed_port=10000+port,)
        state = PartialState()

        # for logger
        self.logger = get_logger(__name__, log_level="INFO")
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[RichHandler(console=CONSOLE)],
        )

        if self.config.optimizer.max_grad_norm is not None:
            deepspeed_config["gradient_clipping"] = self.config.optimizer.max_grad_norm
        deepspeed_config["train_micro_batch_size_per_gpu"] = self.config.train_data.batch_size

        # TODO: turn to mlflow
        if deepspeed_config["tensorboard"]["enabled"]:
            deepspeed_config["tensorboard"]["output_path"] = str(self.exp_root)

        deepspeed_config["fp16"]["enabled"] = True if self.config.training_precision == "fp16" else False
        deepspeed_config["bf16"]["enabled"] = True if self.config.training_precision == "bf16" else False
        if deepspeed_config["fp16"]["enabled"]:
            state.mixed_precision = "fp16"
            self.dtype = torch.float16
        elif deepspeed_config["bf16"]["enabled"]:
            state.mixed_precision = "bf16"
            self.dtype = torch.bfloat16
        else:
            state.mixed_precision = "fp32"
            self.dtype = torch.float32

        deepspeed_config["gradient_accumulation_steps"] = self.config.gradient_accumulation_steps
        state.gradient_accumulation_steps = deepspeed_config["gradient_accumulation_steps"]
        deepspeed_config["train_batch_size"] = (
            self.config.train_data.batch_size * torch.distributed.get_world_size() * deepspeed_config["gradient_accumulation_steps"]
        )
        self.deepspeed_config = deepspeed_config
        self.logger.info(state)

        if state.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        self.state = state
        if self.config.seed is None:
            self.config.seed = int(time.time())
        set_seed(self.config.seed, device_specific=True)

    def init_train(self):
        # for pipeline and manager
        self.pipeline = self.config.pipeline.from_pretrained()
        self.manager = Manager(self.pipeline)
        self.pipeline.to(self.state.device)

        # for dataloader
        self.train_data = self.config.train_data.setup()
        self.train_dataloader = self.train_data.dataloader
        self.val_data = self.config.val_data.setup()
        self.val_dataloader = self.val_data.dataloader

        if self.config.step_per_valloss is not None or self.config.valloss_at_begin:
            self.valloss_data = self.config.valloss_data.setup()
            self.valloss_dataloader = self.valloss_data.dataloader
        else:
            self.valloss_data = None
            self.valloss_dataloader = None

        if self.config.step_per_valmetrics is not None or self.config.valmetrics_at_begin:
            self.valmetrics_data = self.config.valmetrics_data.setup()
            self.valmetrics_dataloader = self.valmetrics_data.dataloader
        else:
            self.valmetrics_data = None
            self.valmetrics_dataloader = None

        if self.config.joint_train_prob > 0:
            self.aux_data = self.config.aux_data.setup()
            self.aux_dataloader = self.aux_data.dataloader
        else:
            self.aux_data = None
            self.aux_dataloader = None

        # for lr
        if self.config.optimizer.scale_lr:
            self.config.optimizer.lr *= self.state.gradient_accumulation_steps * self.config.train_data.batch_size * self.state.num_processes

        # for requires_grad
        self.manager.requires_grad_(False)
        assert len(self.config.trainable_modules) > 0, "trainable_modules must be set"
        self.set_modules_requires_grad(self.config.trainable_modules)
        if self.config.nontrainable_modules is not None:
            self.set_modules_requires_grad(self.config.nontrainable_modules, requires_grad=False)

        # for optimizer
        # different lr for base and reward head
        # base_parameters = [param for name, param in self.manager.named_parameters() if 'reward' not in name and param.requires_grad]
        # reward_parameters = [param for name, param in self.manager.named_parameters() if 'reward' in name and param.requires_grad]
        # num_optimized_params = sum([param.numel() for param in base_parameters + reward_parameters])
        # log_to_rank0("Total of optimized parameters:", num_optimized_params)

        # base_lr = self.config.optimizer.lr / 10
        # reward_lr = self.config.optimizer.lr

        # self.optimizer = self.config.optimizer.setup([
        #     {'params': base_parameters, 'lr': base_lr},
        #     {'params': reward_parameters, 'lr': reward_lr}
        # ])
        parameters_to_optmize = [param for param in self.manager.parameters() if param.requires_grad]
        num_optimized_params = sum([param.numel() for param in parameters_to_optmize])
        log_to_rank0("Total of optimized parameters:", num_optimized_params)
        self.optimizer = self.config.optimizer.setup(parameters_to_optmize)

        # for trainning info
        self.global_batch_size = self.state.num_processes * self.state.gradient_accumulation_steps * self.config.train_data.batch_size
        self.num_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.state.gradient_accumulation_steps / self.state.num_processes)
        self.config.max_steps = max(self.config.num_epochs * self.num_steps_per_epoch, self.config.max_steps)
        self.config.num_epochs = max(math.ceil(self.config.max_steps / self.num_steps_per_epoch), self.config.num_epochs)
        self.validation_rank_num = min(math.ceil(self.config.val_data.num_samples / self.config.val_data.batch_size), torch.distributed.get_world_size())
        self.validation_process_group = torch.distributed.new_group(ranks=[i for i in range(self.validation_rank_num)])
        if self.config.step_per_valmetrics is not None:
            self.validation_metrics_rank_num = min(
                math.ceil(len(self.valmetrics_dataloader) / self.config.valmetrics_data.batch_size), torch.distributed.get_world_size()
            )
            self.validation_metrics_process_group = torch.distributed.new_group(ranks=[i for i in range(self.validation_metrics_rank_num)])
        if self.config.step_per_valloss is not None:
            self.validation_loss_rank_num = min(
                math.ceil(len(self.valloss_dataloader) / self.config.valloss_data.batch_size), torch.distributed.get_world_size()
            )
            self.validation_loss_process_group = torch.distributed.new_group(ranks=[i for i in range(self.validation_loss_rank_num)])

        # for lr_cheduler
        self.lr_scheduler = self.config.scheduler.setup(
            self.optimizer, gradient_accumulation_steps=self.state.gradient_accumulation_steps, num_training_steps=self.config.max_steps
        )

        # for stability
        self.stability_server = Stability(self.config.stability)
        self.stability_server.consecutive_anomalies = 0

    def set_modules_requires_grad(self, module_info, requires_grad=True, target=None):
        if target is None:
            target = self.manager
        for module in module_info:
            if isinstance(module, str):
                self.logger.info(f"[init train] set {module} requires grad")
                model = getattr(target, module)
                model.requires_grad_(requires_grad)
            else:
                module, suffix = list(module.keys())[0], list(module.values())[0]
                self.logger.info(f"[init train] set {module}: {suffix} requires grad")
                model = getattr(target, module)
                for name, module in model.named_modules():
                    if name.endswith(tuple(suffix)):
                        for params in module.parameters():
                            params.requires_grad_(requires_grad)

    def save_ckpt(self, global_step):
        num_trained_data = global_step * self.global_batch_size
        save_root = os.path.join(self.exp_root, "checkpoints", f"checkpoint-{num_trained_data}")
        ds_save_path = os.path.join(save_root, "deepspeed")

        if self.config.use_ema:
            ema_path = os.path.join(save_root, "ema")
            os.makedirs(ema_path, exist_ok=True)
            torch.save(self.ema_model.state_dict(), os.path.join(ema_path, f"ema_r{self.state.process_index:05d}.pt"))

            if self.config.save_ema:
                with ema_context(self.ema_model, self.optimizer):
                    # self.manager.save_ckpts(self.manager.state_dict(), ema_path)  # TODO: fix multi card ema
                    # TODO: save ema ckpt
                    if hasattr(self.manager, "transformer"):
                        if self.deepspeed_config["zero_optimization"]["stage"] == 3:
                            params = [param for param in self.manager.transformer.parameters()]
                            params[0].all_gather(params)
                        if self.state.is_main_process:
                            torch.save(self.manager.transformer.state_dict(), os.path.join(ema_path, "ema.ckpt"))
                        if self.deepspeed_config["zero_optimization"]["stage"] == 3:
                            params[0].partition(params)

        self.manager.remove_frozen_module()
        save_deepspeed_state(self.manager, self.lr_scheduler, ds_save_path, self.state.process_index)
        self.manager.recover_frozen_module()
        self.logger.info(f"Saved state to {ds_save_path}")
        torch.distributed.barrier()
        if self.config.delete_deepspeed_weights and self.state.is_main_process:
            self.logger.info("start save torch model weights.")
            ds_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(ds_save_path)
            self.manager.save_ckpts(ds_dict, save_root)
            # get deepspeed model ckpt name
            latest_path_tag_or_ref = os.path.join(ds_save_path, "latest")
            if os.path.isfile(latest_path_tag_or_ref):
                with open(latest_path_tag_or_ref, "r") as fd:
                    tag = fd.read().strip()
                    # get deepspeed model ckpt path
                    latest_path = os.path.join(ds_save_path, tag)
            # get list of deepspeed model ckpt path
            model_ckpt_files = deepspeed.utils.zero_to_fp32.get_model_state_files(latest_path)
            for file_path in model_ckpt_files:
                o_d = torch.load(file_path, map_location="cpu")
                # remove deepspeed model weight from ckpt only
                o_d["module"] = {}
                # remove deepspeed model freeze weights if possible
                if "frozen_param_fragments" in o_d:
                    del o_d["frozen_param_fragments"]
                torch.save(o_d, file_path)
            self.logger.info("finish save torch model weights.")

    def load_ckpt(self, ckpt_path=None, no_skip=False, add_skipped_step=0):
        if ckpt_path is None:
            dirs = os.listdir(os.path.join(self.exp_root, "checkpoints"))
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            ckpt_path = os.path.join(self.exp_root, "checkpoints", dirs[-1])

        if not os.path.exists(ckpt_path):
            self.logger.error(f"Checkpoint '{ckpt_path}' does not exist.")
            raise ValueError()

        self.logger.info(f"Resuming from checkpoint {ckpt_path}")
        self.manager.remove_frozen_module()
        load_deepspeed_state(self.manager, self.lr_scheduler, os.path.join(ckpt_path, "deepspeed"), self.state.process_index)
        self.manager.recover_frozen_module()
        if self.config.use_ema:
            self.ema_model.load_state_dict(
                torch.load(os.path.join(ckpt_path, "ema", f"ema_r{self.state.process_index:05d}.pt"), map_location=lambda storage, loc: storage)
            )
            self.ema_model.to(self.state.device)

        num_trained_data = int(ckpt_path.split("-")[-1])
        global_step = num_trained_data // (self.state.gradient_accumulation_steps * self.config.train_data.batch_size * self.state.num_processes)
        first_epoch = global_step // self.num_steps_per_epoch

        if no_skip:
            skipped_dataloader = self.train_dataloader
        else:
            global_accumulation_step = global_step * self.state.gradient_accumulation_steps
            skipped_step = global_accumulation_step % (self.num_steps_per_epoch * self.state.gradient_accumulation_steps)
            skipped_dataloader = skip_first_batches(self.train_dataloader, num_batches=skipped_step + add_skipped_step)

        # reload random states
        load_rng_state(os.path.join(ckpt_path, "deepspeed"), self.state.process_index)

        return first_epoch, global_step, skipped_dataloader

    def get_batch(self, train_dataloader, aux_dataloader, global_step):
        dataloader_iter = iter(train_dataloader)
        while True:
            if self.config.joint_train_prob > 0:
                is_joint_training = torch.rand(1).to(self.state.device) < self.config.joint_train_prob
                # 将is_joint_training广播至多机多卡
                if torch.distributed.get_world_size() > 1:
                    torch.distributed.broadcast(is_joint_training, src=0)
            else:
                is_joint_training = False
            with measure_time(
                "Get Batch", self.config.measure_time and global_step % self.config.step_per_measure_time == 0, timer=self.timer, verbose=self.config.verbose
            ):
                if is_joint_training:
                    # 触发混合训练
                    # is_update_step = False
                    is_update_step = True
                    try:
                        batch = next(aux_dataloader_iter)
                    except Exception as e:
                        aux_dataloader_iter = iter(aux_dataloader)
                        batch = next(aux_dataloader_iter)
                else:
                    is_update_step = True
                    try:
                        batch = next(dataloader_iter)

                        if self.train_data.config.random_num_frames:
                            max_latent_num_frames = (self.train_data.num_frames - 1) // self.train_data.temporal_unit_size + 1
                            latent_num_frames = torch.randint(1, max_latent_num_frames + 1, ()).to(self.state.device)
                            if torch.distributed.get_world_size() > 1:
                                torch.distributed.broadcast(latent_num_frames, src=0)
                            sample_num_frames = (latent_num_frames - 1) * self.train_data.temporal_unit_size + 1

                            if "vae_latents" in batch:
                                latent_params = batch["vae_latents"]
                                if not isinstance(latent_params, list):
                                    latent_params = [latent_params]
                                tmp_list = []
                                for latent_param in latent_params:
                                    tmp_list.append(latent_param[:, :latent_num_frames])
                                batch["vae_latents"] = tmp_list
                            elif "data" in batch:
                                samples = batch["data"]
                                if not isinstance(samples, list):
                                    samples = [samples]
                                tmp_list = []
                                for sample in samples:
                                    tmp_list.append(sample[:, :sample_num_frames])
                                batch["data"] = tmp_list
                            batch["num_frames"] = sample_num_frames.item()

                    except StopIteration:
                        break  # 视频数据遍历完毕,结束当前epoch
            yield is_update_step, batch

    def train(self):
        first_epoch, global_step = 0, 0

        if self.config.resume:
            first_epoch, global_step, skipped_dataloader = self.load_ckpt(
                self.config.resume_ckpt_path, self.config.resume_no_skip, self.config.add_skipped_step
            )

        num_trained_data = global_step * self.global_batch_size
        milestone_val = num_trained_data + self.config.data_per_val
        milestone_save = num_trained_data + self.config.data_per_save
        if self.config.use_ema:
            milestone_ema = num_trained_data + self.config.data_per_ema

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(0, self.config.max_steps), disable=not self.state.is_main_process)
        progress_bar.set_description(self.config.experiment_name)

        if self.config.val_at_begin:
            self.validate(global_step)
            self.validate_image(global_step)
            if self.config.use_ema:
                with ema_context(self.ema_model, self.optimizer):
                    self.validate(global_step, "ema")
                    self.validate_image(global_step, "ema")

        if self.config.valloss_at_begin:
            self.validate_loss(global_step)
            if self.config.use_ema:
                with ema_context(self.ema_model, self.optimizer):
                    self.validate_loss(global_step, "ema")

        if self.config.valmetrics_at_begin:
            self.validate_metrics(global_step)
            if self.config.use_ema:
                with ema_context(self.ema_model, self.optimizer):
                    self.validate_metrics(global_step, "ema")

        for epoch in range(first_epoch, self.config.num_epochs):
            train_dataloader = self.train_dataloader
            aux_dataloader = self.aux_dataloader
            if self.config.resume and epoch == first_epoch:
                # using skipped_dataloader in the first epoch to skip batchs already used
                train_dataloader = skipped_dataloader
                progress_bar.update(global_step)

            if self.config.skip_train_data is not None and epoch == first_epoch:
                log_to_rank0(f"skip train data: {self.config.skip_train_data}")
                total_batches = len(self.train_data.dataset) // self.train_data.config.batch_size // torch.distributed.get_world_size()
                skip_batches = self.config.skip_train_data // self.train_data.config.batch_size // torch.distributed.get_world_size()
                train_dataloader = skip_first_batches(self.train_dataloader, num_batches=skip_batches % total_batches)

            if self.aux_data is not None and self.config.skip_aux_data is not None and epoch == first_epoch:
                log_to_rank0(f"skip aux data: {self.config.skip_aux_data}")
                total_batches = len(self.aux_data.dataset) // self.aux_data.config.batch_size // torch.distributed.get_world_size()
                skip_batches = self.config.skip_aux_data // self.aux_data.config.batch_size // torch.distributed.get_world_size()
                aux_dataloader = skip_first_batches(self.aux_dataloader, num_batches=skip_batches % total_batches)

            for is_update_step, batch in self.get_batch(train_dataloader, aux_dataloader, global_step):
                if global_step >= self.config.max_steps:
                    break
                
                # For VAE debugging purpose only.
                # with torch.no_grad():
                #     for param in self.manager.parameters():
                #         param.data = param.data.clone().contiguous()

                self.manager.train()
                self.pipeline.train_status = TrainingStatus.TRAINING

                # forward
                with measure_time(
                    "Forward", self.config.measure_time and global_step % self.config.step_per_measure_time == 0, timer=self.timer, verbose=self.config.verbose
                ):
                    batch = self._cast_inputs(batch)
                    loss_dict = self.pipeline.forward(batch)
                    total_loss = loss_dict["total_loss"]

                # backpropagate
                with measure_time(
                    "Backward", self.config.measure_time and global_step % self.config.step_per_measure_time == 0, timer=self.timer, verbose=self.config.verbose
                ): 
                    # torch.cuda.empty_cache()
                    self.manager.backward(total_loss)
                    # torch.cuda.empty_cache()
                if self.stability_server.stability_protection:
                    world_loss = total_loss.detach().clone()
                    torch.distributed.all_reduce(world_loss)
                    world_loss = world_loss / torch.distributed.get_world_size()

                    if "world_loss" in set(inspect.signature(self.stability_server.track_loss_anomaly).parameters.keys()):
                        should_continue = self.stability_server.track_loss_anomaly(total_loss.detach().item(), world_loss.detach().item())
                    else:
                        should_continue = self.stability_server.track_loss_anomaly(total_loss.detach().item())
                    self.stability_server.handle_loss_anomalies()
                else:
                    should_continue = True

                with measure_time(
                    "Optimizer",
                    self.config.measure_time and global_step % self.config.step_per_measure_time == 0,
                    timer=self.timer,
                    verbose=self.config.verbose,
                ):
                    if should_continue:
                        self.manager.step()
                    else:
                        self.manager.optimizer.zero_grad(set_to_none=True)
                        if self.manager.optimizer.cpu_offload:
                            self.manager.optimizer.reset_cpu_buffers()
                        else:
                            self.manager.optimizer.averaged_gradients = {}

                    self.lr_scheduler.step()

                if self.manager.is_gradient_accumulation_boundary():
                    if not should_continue:
                        log_to_rank0(f"Loss anomaly detected! skip step {global_step}")
                        continue

                    logs = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "global_bs": self.global_batch_size,
                        "steps_per_save": (
                            self.config.data_per_save // self.global_batch_size if self.config.step_per_save is None else self.config.step_per_save
                        ),
                        "steps_per_val": self.config.data_per_val // self.global_batch_size if self.config.step_per_val is None else self.config.step_per_val,
                        "steps_per_valmetrics": self.config.step_per_valmetrics,
                        "steps_per_valloss": self.config.step_per_valloss,
                        "loss": total_loss.item(),
                        "lr": self.lr_scheduler.get_last_lr()[0],
                        "grad_norm": float(self.manager.optimizer._global_grad_norm),
                        "memory alloca:": torch.cuda.memory_allocated(),
                        "memory max": torch.cuda.max_memory_allocated(),
                        "aux data": not is_update_step,
                    }
                    if is_update_step:
                        global_step += 1
                        num_trained_data += self.global_batch_size
                        logs.update({"global_step": global_step})
                        progress_bar.set_postfix(**logs, refresh=False)
                        progress_bar.update()

                        if self.config.step_per_tb is not None and global_step % self.config.step_per_tb == 0:
                            for k, v in loss_dict.items():
                                avg_v = np.mean(list(map(lambda x: x.cpu(), self.gather_tensors(v.detach()))))
                                # std_v = np.std(list(map(lambda x: x.cpu(), self.gather_tensors(v.detach()))))
                                if self.state.is_main_process:
                                    self.logger_tb.summary_writer.add_scalar(f"Loss/{k}", avg_v, global_step)
                                    # self.logger_tb.summary_writer.add_scalar(f"Loss/{f'{k}_std'}", std_v, global_step)
                                del avg_v
                                # del std_v

                            if self.state.is_main_process:
                                self.logger_tb.summary_writer.add_scalar(f"Loss/grad_norm", self.manager.optimizer._global_grad_norm, global_step)

                        if self.config.measure_time and global_step % self.config.step_per_measure_time == 0 and self.state.is_main_process:
                            for k, v in self.timer.timers.items():
                                if k not in ["Save", "Validation", "Metrics", "ValLoss"]:
                                    self.logger_tb.summary_writer.add_scalar(f"Time/{k}", v[-1], global_step)

                        if self.config.use_ema:
                            if (self.config.step_per_ema is not None and global_step % self.config.step_per_ema == 0) or (
                                self.config.step_per_ema is None and num_trained_data >= milestone_ema
                            ):
                                self.ema_model.step(self.optimizer.single_partition_of_fp32_groups)
                                if self.config.step_per_ema is None and num_trained_data >= milestone_ema:
                                    milestone_ema += self.config.data_per_ema

                        if (self.config.step_per_save is not None and global_step % self.config.step_per_save == 0) or (
                            self.config.step_per_save is None and num_trained_data >= milestone_save
                        ):
                            with measure_time("Save", self.config.measure_time, timer=self.timer, verbose=self.config.verbose):
                                try:
                                    self.save_ckpt(global_step)
                                except Exception as e:
                                    print("Save exception:", e)
                            if self.config.measure_time and self.state.is_main_process:
                                self.logger_tb.summary_writer.add_scalar(f"Time/Save", self.timer.timers["Save"][-1], global_step)
                            if self.config.step_per_save is None and num_trained_data >= milestone_save:
                                milestone_save += self.config.data_per_save

                        if (self.config.step_per_val is not None and global_step % self.config.step_per_val == 0) or (
                            self.config.step_per_val is None and num_trained_data >= milestone_val
                        ):
                            with measure_time("Validation", self.config.measure_time, timer=self.timer, verbose=self.config.verbose):
                                self.validate(global_step)
                                self.validate_image(global_step)
                                if self.config.use_ema:
                                    with ema_context(self.ema_model, self.optimizer):
                                        self.validate(global_step, "ema")
                                        self.validate_image(global_step, "ema")
                            if self.config.measure_time and self.state.is_main_process:
                                self.logger_tb.summary_writer.add_scalar(f"Time/Validation", self.timer.timers["Validation"][-1], global_step)
                            if self.config.step_per_val is None and num_trained_data >= milestone_val:
                                milestone_val += self.config.data_per_val

                        if self.config.step_per_valloss is not None and global_step % self.config.step_per_valloss == 0:
                            with measure_time("ValLoss", self.config.measure_time, timer=self.timer, verbose=self.config.verbose):
                                self.validate_loss(global_step)
                                if self.config.use_ema:
                                    with ema_context(self.ema_model, self.optimizer):
                                        self.validate_loss(global_step, name="ema")
                            if self.config.measure_time and self.state.is_main_process:
                                self.logger_tb.summary_writer.add_scalar(f"Time/ValLoss", self.timer.timers["ValLoss"][-1], global_step)

                        if self.config.step_per_valmetrics is not None and global_step % self.config.step_per_valmetrics == 0:
                            with measure_time("Metrics", self.config.measure_time, timer=self.timer, verbose=self.config.verbose):
                                self.validate_metrics(global_step)
                                if self.config.use_ema:
                                    with ema_context(self.ema_model, self.optimizer):
                                        self.validate_metrics(global_step, "ema")
                            if self.config.measure_time and self.state.is_main_process:
                                self.logger_tb.summary_writer.add_scalar(f"Time/Metrics", self.timer.timers["Metrics"][-1], global_step)

                    else:
                        progress_bar.set_postfix(**logs, refresh=False)
                        progress_bar.update()
        
        with measure_time("Save", self.config.measure_time, timer=self.timer, verbose=self.config.verbose):
            try:
                self.save_ckpt(global_step)
            except Exception as e:
                print("Save exception:", e)
        if self.config.step_per_valloss is not None:
            with measure_time("ValLoss", self.config.measure_time, timer=self.timer, verbose=self.config.verbose):
                self.validate_loss(global_step)
                if self.config.use_ema:
                    with ema_context(self.ema_model, self.optimizer):
                        self.validate_loss(global_step, name="ema")
            if self.config.measure_time and self.state.is_main_process:
                self.logger_tb.summary_writer.add_scalar(f"Time/ValLoss", self.timer.timers["ValLoss"][-1], global_step)

    def save_video(self, videos, col_name, global_step, type_name="validation"):
        video = np.transpose(videos, (0, 2, 3, 1))
        filename = f"{self.config.experiment_name}_{type_name}_{col_name}_step{global_step}.mp4"
        filepath = os.path.join(self.exp_root, type_name)
        os.makedirs(filepath, exist_ok=True)
        imageio.mimsave(os.path.join(filepath, filename), video, fps=self.config.train_data.sample_fps)

        ## Optional: convert video to gif, requires ffmpeg
        # video2gif(os.path.join(filepath, filename), fps=self.config.train_data.sample_fps)
        # os.remove(os.path.join(filepath, filename))

    @torch.no_grad()
    def validate_image(self, global_step, type_name="validation"):
        _num_frames = self.pipeline.pipeline_config.call["num_frames"]
        _fps = self.pipeline.pipeline_config.call["fps"]
        self.pipeline.pipeline_config.call["num_frames"] = 1
        self.pipeline.pipeline_config.call["fps"] = 0
        self.validate(global_step, type_name)
        self.pipeline.pipeline_config.call["num_frames"] = _num_frames
        self.pipeline.pipeline_config.call["fps"] = _fps

    @torch.no_grad()
    def validate(self, global_step, type_name="validation"):
        torch.distributed.barrier()
        log_to_rank0(f"Validating at step {global_step}...")

        self.pipeline.train_status = TrainingStatus.VALIDATING
        if self.pipeline.pipeline_config.distributed_clip:
            recover_model(self.pipeline.clip_g_model)
            recover_model(self.pipeline.clip_l_model)

        if torch.distributed.get_rank() < self.validation_rank_num:
            self.manager.eval()
            videos = []
            for batch in tqdm(self.val_dataloader):
                batch = self._cast_inputs(batch)
                video = []
                for op in self.config.val_types:
                    video.append(getattr(self.pipeline, f"val_{op}")(batch))

                try:
                    video = torch.cat(video, 3).mul_(0.5).add_(0.5).clamp_(0, 1)
                except:
                    video = pack_data([v[None] for v in video])[0]
                    video = rearrange(video, "num bf c h w -> bf c h (num w)")
                    video = video.mul_(0.5).add_(0.5).clamp_(0, 1)
                video = rearrange(video, "(b f) c h w -> b f c h w", f=self.pipeline.pipeline_config.call["num_frames"])
                videos.append(video)
            videos = torch.vstack(videos)

            output_tensors = self.gather_tensors(videos, self.validation_process_group)
            videos = torch.vstack(output_tensors)
            for num_cols in range(int(math.sqrt(videos.shape[0])), 0, -1):
                if videos.shape[0] % num_cols == 0:
                    videos = rearrange(videos, "(b r) f c h w -> f c (b h) (r w)", r=num_cols)
                    break

            # numpy无法将bf16转为uint8
            videos = videos.cpu().mul(255).to(torch.uint8).numpy()
            col_name = "_".join(self.config.val_types)
            # torch.cuda.empty_cache()

            if self.state.is_main_process:
                if videos.shape[0] == 1:
                    img_pil = Image.fromarray(np.transpose(videos[0], (1, 2, 0)))
                    filename = f"{self.config.experiment_name}_{type_name}_{col_name}_step{global_step}.png"
                    filepath = os.path.join(self.exp_root, type_name)
                    os.makedirs(filepath, exist_ok=True)
                    img_pil.save(os.path.join(filepath, filename))
                    if self.config.tb_add_image:
                        self.logger_tb.summary_writer.add_image(f"{type_name}/{col_name}_image", videos[0], global_step)
                else:
                    self.save_video(videos, col_name, global_step, type_name)
                    if self.config.tb_add_video:
                        self.logger_tb.summary_writer.add_video(f"{type_name}/{col_name}", videos[None, ...], global_step, self.config.train_data.sample_fps)

        if self.pipeline.pipeline_config.distributed_clip:
            partition_model(self.pipeline.clip_g_model)
            partition_model(self.pipeline.clip_l_model)

        torch.cuda.synchronize()
        torch.distributed.barrier()

    @torch.no_grad()
    def validate_loss(self, global_step, name="model"):
        if hasattr(self.pipeline, "get_val_loss"):
            torch.distributed.barrier()
            log_to_rank0(f"Validating loss at step {global_step}...")
            if torch.distributed.get_rank() < self.validation_loss_rank_num:
                self.manager.eval()
                val_loss = {}
                for i, batch in enumerate(tqdm(self.valloss_dataloader)):
                    if i == 0:
                        val_loss.update(getattr(self.pipeline, "get_val_loss")(batch, self.config.num_valloss_timesteps).items())
                    else:
                        for k, v in getattr(self.pipeline, "get_val_loss")(batch, self.config.num_valloss_timesteps).items():
                            val_loss[k] += v
                for k, v in val_loss.items():
                    val_loss[k] = self.gather_tensors(torch.cat(v, 0), self.validation_loss_process_group)
                    val_loss[k] = torch.cat(val_loss[k], dim=0).mean(0)
                if self.state.is_main_process:
                    for k, v in val_loss.items():
                        self.logger_tb.summary_writer.add_scalar(f"Validation/{name}/{k}", v.item(), global_step)
            torch.cuda.synchronize()
            torch.distributed.barrier()
        else:
            log_to_rank0(f"Pipeline {self} does NOT support valloss.")
    
    # compute spearman ratio
    # @torch.no_grad()
    # def validate_loss(self, global_step, name="model"):
    #     def compute_spearman(pred, label):
    #         # Convert torch tensors to numpy arrays
    #         pred_np = pred.cpu().numpy()
    #         label_np = label.cpu().numpy()

    #         # Calculate Spearman correlation
    #         spearman_corr, _ = spearmanr(label_np, pred_np)

    #         return spearman_corr

    #     if hasattr(self.pipeline, "get_spearman"):
    #         torch.distributed.barrier()
    #         log_to_rank0(f"spearman at step {global_step}...")
    #         if torch.distributed.get_rank() < self.validation_loss_rank_num:
    #             self.manager.eval()
    #             spearman = {}
    #             for i, batch in enumerate(tqdm(self.valloss_dataloader)):
    #                 if i == 0:
    #                     spearman.update(getattr(self.pipeline, "get_spearman")(batch).items())
    #                 else:
    #                     for k, v in getattr(self.pipeline, "get_spearman")(batch).items():
    #                         spearman[k] += v
    #             for k, v in spearman.items():
    #                 spearman[k] = self.gather_tensors(torch.cat(v, 0), self.validation_loss_process_group)
    #                 spearman[k] = torch.cat(spearman[k], dim=0)
    #             if self.state.is_main_process:
    #                 for k, v in spearman.items():
    #                     if 'predicted' in k:
    #                         test_step = k.split('-')[-1]
    #                         spearman[k] = compute_spearman(spearman[k], spearman[f"label"])
    #                         key_name = f"spearman-{test_step}"
    #                     else:
    #                         spearman[k] = spearman[k].float().mean()
    #                         key_name = k
    #                     self.logger_tb.summary_writer.add_scalar(f"Validation/{name}/{key_name}", spearman[k].item(), global_step)
    #         torch.cuda.synchronize()
    #         torch.distributed.barrier()
    #     else:
    #         log_to_rank0(f"Pipeline {self} does NOT support spearman.")

    @torch.no_grad()
    def validate_metrics(self, global_step, name="model"):
        torch.distributed.barrier()
        # torch.cuda.empty_cache()

        model_iq = IQAIGCModelV2().cuda()
        model_vq = VQAIGCModelV2().cuda()
        model_umt = UMTScoreModel().cuda()
        model_motionsmooth = MotionSmoothModel().cuda()
        model_iq.eval()
        model_vq.eval()
        model_umt.eval()
        model_motionsmooth.eval()
        log_to_rank0(f"Validating metrics at step {global_step}...")

        if torch.distributed.get_rank() < self.validation_metrics_rank_num:
            self.manager.eval()
            vq_scores_vq, vq_scores_dq, vq_scores_a, vq_scores_overall = [], [], [], []
            iq_scores_q, iq_scores_a = [], []
            umt_scores, motionsmooth_scores = [], []
            for _, batch in enumerate(tqdm(self.valmetrics_dataloader)):
                batch = self._cast_inputs(batch)
                prompts = batch["prompts"]
                video = getattr(self.pipeline, "val_t2v")(batch)
                video = rearrange(video, "(b f) c h w -> b f h w c", f=self.config.valmetrics_data.num_frames)
                video = video.mul_(0.5).add_(0.5).clamp_(0, 1)
                video = video.cpu().mul(255).to(torch.uint8).numpy()
                iq_score_a, iq_score_q = model_iq(video, prompts)
                iq_scores_q.append(iq_score_q)
                iq_scores_a.append(iq_score_a)
                vq_score_vq, vq_score_dq, vq_score_a, vq_score_overall = model_vq(video, prompts)
                vq_scores_vq.append(vq_score_vq)
                vq_scores_dq.append(vq_score_dq)
                vq_scores_a.append(vq_score_a)
                vq_scores_overall.append(vq_score_overall)
                umt_score = model_umt(video, prompts)
                umt_scores.append(umt_score)
                motionsmooth_score = model_motionsmooth(video)
                motionsmooth_scores.append(motionsmooth_score)

            vq_scores_vq, vq_scores_dq, vq_scores_a, vq_scores_overall, iq_scores_q, iq_scores_a, umt_scores, motionsmooth_scores = (
                torch.cat(vq_scores_vq, 0),
                torch.cat(vq_scores_dq, 0),
                torch.cat(vq_scores_a, 0),
                torch.cat(vq_scores_overall, 0),
                torch.cat(iq_scores_q, 0),
                torch.cat(iq_scores_a, 0),
                torch.cat(umt_scores, 0),
                torch.cat(motionsmooth_scores, 0),
            )
            vq_scores_vq_all = self.gather_tensors(vq_scores_vq, self.validation_metrics_process_group)
            vq_scores_dq_all = self.gather_tensors(vq_scores_dq, self.validation_metrics_process_group)
            vq_scores_a_all = self.gather_tensors(vq_scores_a, self.validation_metrics_process_group)
            vq_scores_overall_all = self.gather_tensors(vq_scores_overall, self.validation_metrics_process_group)
            iq_scores_q_all = self.gather_tensors(iq_scores_q, self.validation_metrics_process_group)
            iq_scores_a_all = self.gather_tensors(iq_scores_a, self.validation_metrics_process_group)
            umt_scores_all = self.gather_tensors(umt_scores, self.validation_metrics_process_group)
            motionsmooth_scores_all = self.gather_tensors(motionsmooth_scores, self.validation_metrics_process_group)

            vq_score_vq_mean = torch.cat(vq_scores_vq_all, dim=0).mean(0)
            vq_score_dq_mean = torch.cat(vq_scores_dq_all, dim=0).mean(0)
            vq_score_a_mean = torch.cat(vq_scores_a_all, dim=0).mean(0)
            vq_score_overall_mean = torch.cat(vq_scores_overall_all, dim=0).mean(0)
            iq_score_q_mean = torch.cat(iq_scores_q_all, dim=0).mean(0)
            iq_score_a_mean = torch.cat(iq_scores_a_all, dim=0).mean(0)
            umt_score_mean = torch.cat(umt_scores_all, dim=0).mean(0)
            motionsmooth_score_mean = torch.cat(motionsmooth_scores_all, dim=0).mean(0)
            fitted_overall_score, fitted_vq_score, fitted_t2valign_score, fitted_dq_score = get_overall_score_v2(
                iq_score_a=iq_score_a_mean,
                iq_score_q=iq_score_q_mean,
                vq_score_vq=vq_score_vq_mean,
                vq_score_dq=vq_score_dq_mean,
                vq_score_a=vq_score_a_mean,
                vq_score_overall=vq_score_overall_mean,
                umt_score=umt_score_mean,
                motionsmooth_score=motionsmooth_score_mean,
            )

            if self.state.is_main_process:
                self.logger_tb.summary_writer.add_scalar(f"Metrics/fitted_overall_score/{name}", fitted_overall_score.item(), global_step)
                self.logger_tb.summary_writer.add_scalar(f"Metrics/fitted_vq_score/{name}", fitted_vq_score.item(), global_step)
                self.logger_tb.summary_writer.add_scalar(f"Metrics/fitted_dq_score/{name}", fitted_dq_score.item(), global_step)
                self.logger_tb.summary_writer.add_scalar(f"Metrics/fitted_t2valign_score/{name}", fitted_t2valign_score.item(), global_step)

        del model_iq, model_vq, model_umt, model_motionsmooth
        # torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.distributed.barrier()

    @torch.no_grad()
    def test(self, mode="video"):
        self.manager.eval()
        output_root = os.path.join(self.exp_root, "video")
        os.makedirs(output_root, exist_ok=True)

        cnt = 0
        for i, batch in enumerate(tqdm(self.val_dataloader)):
            batch = self._cast_inputs(batch)
            pipeline_cls = self.pipeline if hasattr(self.pipeline, "prepare_call_kwargs") else PipelineMixin
            call_kwargs = pipeline_cls.prepare_call_kwargs(pipeline=self.pipeline, batch=batch)

            # 执行pipeline前向
            call_kwargs["output_type"] = "np"
            if isinstance(self.pipeline, StableVideoDiffusionPipeline):
                res = self.pipeline(**call_kwargs).frames
            else:
                res = self.pipeline(**call_kwargs).images
            if isinstance(res, list):
                res = np.concatenate(res, 0)

            if "control_image" in call_kwargs:
                for control in call_kwargs["control_image"]:
                    control = rearrange(control.cpu().numpy(), "b c h w -> b h w c")
                    res = np.concatenate([res, control], 2)

            if mode == "video":
                res = rearrange(res, "(b f) h w c -> b f h w c", f=self.config.val_data.num_frames)
            res = (res * 255).astype(np.uint8)

            # 结果落盘
            result_csv = []
            for j in range(len(res)):
                video = res[j]
                data_path = (os.path.splitext(os.path.basename(batch["data_paths"][j]))[0] + ".") if "data_paths" in batch else ""
                prompt = batch["prompts"][j] if "prompts" in batch else ""
                name_endfix = get_test_video_basename(self.config.pipeline, call_kwargs, self.config.val_data, prompt)
                if mode == "video":
                    if "index" in batch:
                        video_name = batch["index"][j] + ".mp4"
                    else:
                        video_name = f"R{torch.distributed.get_rank()}L{cnt}_{data_path}{name_endfix}.mp4"
                    output_path = os.path.join(output_root, video_name)
                    iio.imwrite(output_path, video, fps=self.config.val_data.sample_fps)
                elif mode == "image":
                    # image_name = f"R{torch.distributed.get_rank()}L{cnt}_{data_path}{name_endfix}.jpg"
                    words = "".join(char for char in prompt.strip() if char.isalnum() or char.isspace()).split()
                    linked_words = "_".join(words[:10])
                    image_name = linked_words + ".jpg"
                    output_path = os.path.join(output_root, image_name)
                    img_pil = Image.fromarray(video)
                    img_pil.save(output_path)
                result_csv.append([prompt, data_path, output_path])
                # video2gif(output_path, fps=self.config.val_data.sample_fps)
                cnt += 1

            all_csv = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(all_csv, result_csv)

            if self.state.is_main_process:
                all_results_csv = [item for sublist in all_csv for item in sublist]
                csv_path = os.path.join(self.exp_root, "results.csv")
                with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["prompts", "input_data_path", "videos"])
                    writer.writerows(all_results_csv)
                    log_to_rank0(f"Result csv written to {csv_path}.")
                if os.path.exists(os.path.join(self.exp_root, "tensorboard")):
                    rmtree(os.path.join(self.exp_root, "tensorboard"))

        # torch.cuda.empty_cache()

    def _cast_inputs(self, inputs):
        if isinstance(inputs, (list, tuple)):
            new_inputs = []
            for v in inputs:
                new_inputs.append(self._cast_inputs(v))
            return inputs.__class__(new_inputs)
        elif isinstance(inputs, dict):
            new_inputs = {}
            for k, v in inputs.items():
                new_inputs[k] = self._cast_inputs(v)
            return new_inputs
        elif isinstance(inputs, torch.Tensor):
            if torch.is_floating_point(inputs):
                inputs = inputs.to(dtype=self.dtype)
            return inputs.to(device=self.state.device)
        else:
            return inputs

    def gather_tensors(self, tensors, group_size=None):
        all_tensors = [torch.zeros_like(tensors) for _ in range(torch.distributed.get_world_size(group=group_size))]
        torch.distributed.all_gather(all_tensors, tensors, group=group_size)
        return all_tensors
