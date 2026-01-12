import os
import sys
import time
import hydra
import torch
import wandb
import logging
import warnings
import threading
import itertools
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from einops import rearrange
from accelerate import Accelerator
from torchvision import utils
import torch.distributed as dist
from pathlib import Path
from collections import OrderedDict
from hydra.types import RunMode
from hydra.core.hydra_config import HydraConfig
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from dino_wm.metrics.image_metrics import eval_images
from dino_wm.utils import slice_trajdict_with_t, cfg_to_dict, seed, sample_tensors
import dino_wm.models
import dino_wm.datasets
sys.modules['models'] = dino_wm.models
sys.modules['datasets'] = dino_wm.datasets

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

class DINOWM:
    def __init__(self, cfg):
        self.cfg = cfg

        if HydraConfig.get().mode == RunMode.MULTIRUN:
            log.info(" Multirun setup begin...")
            log.info(f"SLURM_JOB_NODELIST={os.environ['SLURM_JOB_NODELIST']}")
            log.info(f"DEBUGVAR={os.environ['DEBUGVAR']}")
            # ==== init ddp process group ====
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            try:
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=timedelta(minutes=5),  # Set a 5-minute timeout
                )
                log.info("Multirun setup completed.")
            except Exception as e:
                log.error(f"DDP setup failed: {e}")
                raise
            torch.distributed.barrier()
            # # ==== /init ddp process group ====

        self.accelerator = Accelerator(log_with="wandb")
        self.device = self.accelerator.device
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.num_reconstruct_samples = self.cfg.training.num_reconstruct_samples
        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0

        assert cfg.training.batch_size % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Batch_size: {cfg.training.batch_size} num_processes: {self.accelerator.num_processes}."
        )

        OmegaConf.set_struct(cfg, False)
        cfg.effective_batch_size = cfg.training.batch_size
        cfg.gpu_batch_size = cfg.training.batch_size // self.accelerator.num_processes
        OmegaConf.set_struct(cfg, True)

        self.accelerator.wait_for_everyone()

        seed(cfg.training.seed)

        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.decoder = None
        self.train_encoder = self.cfg.model.train_encoder
        self.train_predictor = self.cfg.model.train_predictor
        self.train_decoder = self.cfg.model.train_decoder
        log.info(f"Train encoder, predictor, decoder:\
            {self.cfg.model.train_encoder}\
            {self.cfg.model.train_predictor}\
            {self.cfg.model.train_decoder}")

        self._keys_to_save = [
            "epoch",
        ]
        self._keys_to_save += (
            ["encoder"] if self.train_encoder else []
        )
        self._keys_to_save += (
            ["predictor"]
            if self.train_predictor and self.cfg.has_predictor
            else []
        )
        self._keys_to_save += (
            ["decoder"] if self.train_decoder else []
        )
        self._keys_to_save += ["action_encoder", "proprio_encoder"]

        self.init_models()

    def save_ckpt(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            ckpt = {}
            for k in self._keys_to_save:
                if not isinstance(self.__dict__[k], int):
                    self.__dict__[k].eval()
                if self.__dict__[k] is None:
                    continue
                if hasattr(self.__dict__[k], "module"):
                    ckpt[k] = self.accelerator.unwrap_model(self.__dict__[k])
                if isinstance(self.__dict__[k], torch.optim.Optimizer):
                    ckpt[k] = self.__dict__[k].state_dict()
                else:
                    ckpt[k] = self.__dict__[k]
            Path("checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, "checkpoints/model_latest.pth")
            # torch.save(ckpt, f"checkpoints/model_{self.epoch}.pth")
            log.info("Saved model to {}".format(os.getcwd()))
            ckpt_path = os.path.join(os.getcwd(), f"checkpoints/model_latest.pth")
            for k in self._keys_to_save:
                if not isinstance(self.__dict__[k], int):
                    self.__dict__[k].train()  # set back to train mode
        else:
            ckpt_path = None
        model_name = self.cfg["saved_folder"].split("outputs/")[-1]
        model_epoch = self.epoch
        return ckpt_path, model_name, model_epoch

    def load_ckpt(self, filename="model_latest.pth"):
        ckpt = torch.load(filename)
        for k, v in ckpt.items():
            self.__dict__[k] = v
        not_in_ckpt = set(self._keys_to_save) - set(ckpt.keys())
        if len(not_in_ckpt):
            log.warning("Keys not found in ckpt: %s", not_in_ckpt)

    def init_models(self):
        model_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest.pth"
        if model_ckpt.exists():
            self.load_ckpt(model_ckpt)
            log.info(f"Resuming from epoch {self.epoch}: {model_ckpt}")

        # initialize encoder
        if self.encoder is None:
            self.encoder = hydra.utils.instantiate(
                self.cfg.encoder,
            )
        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if "proprio_encoder" in self.cfg:
            if self.proprio_encoder is None:
                self.proprio_encoder = hydra.utils.instantiate(
                    self.cfg.proprio_encoder,
                    in_chans=self.cfg.proprio_dim,
                    emb_dim=self.cfg.proprio_emb_dim,
                )
            proprio_emb_dim = self.proprio_encoder.emb_dim
            print(f"Proprio encoder type: {type(self.proprio_encoder)}")
            self.proprio_encoder = self.accelerator.prepare(self.proprio_encoder)
        else:
            proprio_emb_dim = 0
            self.proprio_encoder = None

        if self.action_encoder is None:
            self.action_encoder = hydra.utils.instantiate(
                self.cfg.action_encoder,
                in_chans=self.cfg.action_dim,
                emb_dim=self.cfg.action_emb_dim,
            )
        action_emb_dim = self.action_encoder.emb_dim
        print(f"Action encoder type: {type(self.action_encoder)}")

        self.action_encoder = self.accelerator.prepare(self.action_encoder)

        # initialize predictor
        if self.encoder.latent_ndim == 1:  # if feature is 1D
            num_patches = 1
        else:
            decoder_scale = 16  # from vqvae
            num_side_patches = self.cfg.img_size // decoder_scale
            num_patches = num_side_patches**2

        if self.cfg.concat_dim == 0:
            num_patches += 2

        if self.cfg.has_predictor:
            if self.predictor is None:
                pred_dim = self.encoder.emb_dim
                pred_dim += action_emb_dim * self.cfg.num_action_repeat * self.cfg.concat_dim
                if self.proprio_encoder is not None:
                    pred_dim += proprio_emb_dim * self.cfg.num_proprio_repeat * self.cfg.concat_dim
                self.predictor = hydra.utils.instantiate(
                    self.cfg.predictor,
                    num_patches=num_patches,
                    num_frames=self.cfg.num_hist,
                    dim=pred_dim,
                )
            if not self.train_predictor:
                for param in self.predictor.parameters():
                    param.requires_grad = False

        # initialize decoder
        if self.cfg.has_decoder:
            if self.decoder is None:
                if self.cfg.env.decoder_path is not None:
                    decoder_path = os.path.join(
                        self.base_path, self.cfg.env.decoder_path
                    )
                    ckpt = torch.load(decoder_path)
                    if isinstance(ckpt, dict):
                        self.decoder = ckpt["decoder"]
                    else:
                        self.decoder = torch.load(decoder_path)
                    log.info(f"Loaded decoder from {decoder_path}")
                else:
                    self.decoder = hydra.utils.instantiate(
                        self.cfg.decoder,
                        emb_dim=self.encoder.emb_dim,  # 384
                    )
            if not self.train_decoder:
                for param in self.decoder.parameters():
                    param.requires_grad = False
        self.encoder, self.predictor, self.decoder = self.accelerator.prepare(
            self.encoder, self.predictor, self.decoder
        )
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=self.cfg.concat_dim,
            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
        )

    def openloop_rollout(
        self, dset, num_rollout=10, rand_start_end=False, min_horizon=2, mode="train"
    ):
        np.random.seed(self.cfg.training.seed)
        min_horizon = min_horizon + self.cfg.num_hist
        plotting_dir = f"rollout_plots/e{self.epoch}_rollout"
        if self.accelerator.is_main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        logs = {}

        # rollout with both num_hist and 1 frame as context
        num_past = [(self.cfg.num_hist, ""), (1, "_1framestart")]

        # sample traj
        for idx in range(num_rollout):
            valid_traj = False
            while not valid_traj:
                traj_idx = np.random.randint(0, len(dset))
                datapack = dset[traj_idx]
                if len(datapack) == 3:
                    obs, act, _ = datapack
                else:
                    obs, act, _, _ = datapack
                act = act.to(self.device)
                is_stacked_already = (self.action_encoder.in_chans == act.shape[-1])  # HACK: data might be stacked already
                # if rand_start_end:
                #     if obs["visual"].shape[0] > min_horizon + 1:
                #         start = np.random.randint(
                #             0,
                #             obs["visual"].shape[0] - min_horizon - 1,
                #         )
                #     else:
                #         start = 0
                #     max_horizon = (obs["visual"].shape[0] - start - 1)
                #     if max_horizon > min_horizon:
                #         valid_traj = True
                #         horizon = np.random.randint(min_horizon, max_horizon + 1)
                # else:
                valid_traj = True
                start = 0
                horizon = (obs["visual"].shape[0] - 1)
                if not is_stacked_already:
                    horizon = horizon // self.cfg.frameskip

            if is_stacked_already:
                for k in obs.keys():
                    obs[k] = obs[k][start : start + horizon + 1]
                act = act[start : start + horizon]
            else:
                for k in obs.keys():
                    obs[k] = obs[k][start : start + horizon * self.cfg.frameskip + 1 : self.cfg.frameskip]
                act = act[start : start + horizon * self.cfg.frameskip]
                act = rearrange(act, "(h f) d -> h (f d)", f=self.cfg.frameskip)

            obs_g = {}
            for k in obs.keys():
                obs_g[k] = obs[k][-1].unsqueeze(0).unsqueeze(0).to(self.device)
            z_g = self.model.encode_obs(obs_g)
            actions = act.unsqueeze(0)

            for past in num_past:
                n_past, postfix = past

                obs_0 = {}
                for k in obs.keys():
                    obs_0[k] = (
                        obs[k][:n_past].unsqueeze(0).to(self.device)
                    )  # unsqueeze for batch, (b, t, c, h, w)

                z_obses, z = self.model.rollout(obs_0, actions)
                z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
                div_loss = self.err_eval_single(z_obs_last, z_g)

                for k in div_loss.keys():
                    log_key = f"z_{k}_err_rollout{postfix}"
                    if log_key in logs:
                        logs[f"z_{k}_err_rollout{postfix}"].append(
                            div_loss[k]
                        )
                    else:
                        logs[f"z_{k}_err_rollout{postfix}"] = [
                            div_loss[k]
                        ]

                if self.cfg.has_decoder:
                    visuals = self.model.decode_obs(z_obses)[0]["visual"]
                    imgs = torch.cat([obs["visual"], visuals[0].cpu()], dim=0)
                    self.plot_imgs(
                        imgs,
                        obs["visual"].shape[0],
                        f"{plotting_dir}/e{self.epoch}_{mode}_{idx}{postfix}.png",
                    )
                    pred_img_np = visuals[0].detach().cpu().numpy() # (t, 3, h, w)
                    gt_img_np = obs["visual"].detach().cpu().numpy()
                    concat_img = np.concatenate((pred_img_np, gt_img_np), axis=-1) # (t, 3, h, w * 3)
                    concat_img = (concat_img + 1.0) / 2.0
                    concat_img = np.clip(concat_img, 0.0, 1.0)
                    concat_img = (concat_img * 255.0).astype(np.uint8)
                    self.wandb_run.log(
                        {f"{mode}_vis/rollout_{idx}": wandb.Video(concat_img, caption=f"rollout_{idx}")}
                    )
        logs = {
            key: sum(values) / len(values) for key, values in logs.items() if values
        }
        return logs

