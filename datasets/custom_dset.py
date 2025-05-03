import concurrent.futures
import glob
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Any
import copy 

import cv2
import h5py
import numpy as np
import torch
import zarr
import zarr.storage
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.kinematics_helper import KinHelper
import torch
import decord
import numpy as np
from einops import rearrange

from ext_utils.imagecodecs_numcodecs import Jpeg2k, register_codecs
from ext_utils.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
    get_twenty_times_normalizer_from_stat,
)
from ext_utils.pytorch_util import dict_apply
from ext_utils.replay_buffer import ReplayBuffer
from ext_utils.sampler import SequenceSampler, get_val_mask
from .traj_dset import TrajDataset, get_train_val_sliced, TrajSlicerDataset

decord.bridge.set_bridge("torch")
register_codecs()

def normalizer_from_stat(stat: Dict[str, np.ndarray]) -> LinearNormalizer:
    max_abs = np.maximum(stat["max"].max(), np.abs(stat["min"]).max())
    scale = np.full_like(stat["max"], fill_value=1 / max_abs)
    offset = np.zeros_like(stat["max"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )

class CustomDataset(TrajDataset):
    def __init__(
        self,
        dataset_dir: str,
        horizon: int,
        val_horizon: int = 0,
        skip_frame: int = 1,
        pad_before: int = 1,
        pad_after: int = 7,
        seed: int = 42,
        val_ratio: float = 0.1,
        skip_idx: int = 20,
        use_cache: bool = True,
        delta_action: bool = False,
        action_mode: str = "bimanual_push",
        shape_meta: Optional[Dict[str, Any]] = None,
    ):
        # assign config
        seq_horizon = (horizon + 1) * skip_frame
        pad_before = pad_before
        pad_after = pad_after
        use_cache = use_cache
        seed = seed
        val_ratio = val_ratio
        self.val_horizon = (val_horizon + 1) * skip_frame
        self.skip_idx = skip_idx
        self.action_mode = action_mode

        replay_buffer = None
        with h5py.File(f"{dataset_dir}/episode_0.hdf5") as file:
            self.robot_bases = file["robot_bases"][0].copy()
        if use_cache:
            cache_info_str = ""
            obs_shape_meta = shape_meta["obs"]
            for _, attr in obs_shape_meta.items():
                type = attr.get("type", "low_dim")
            cache_zarr_path = os.path.join(
                dataset_dir, f"cache{cache_info_str}.zarr.zip"
            )
            print("Acquiring lock on cache.")
        print("Loading cached ReplayBuffer from Disk.")
        with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
            replay_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store, store=zarr.MemoryStore()
            )
        print("Loaded!")
        self.replay_buffer = replay_buffer

        rgb_keys = list()
        depth_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "depth":
                depth_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=seq_horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.lowdim_keys = lowdim_keys
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.downsample_horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.dataset_dir = dataset_dir
        self.skip_frame = skip_frame
        self.delta_action = delta_action
        self.action_dim = replay_buffer["action"].shape[-1] * skip_frame
        if self.delta_action:
            self.kin_helper = KinHelper("trossen_vx300s")

    def get_seq_length(self, idx):
        epi_idx = np.searchsorted(self.sampler.episode_ends, idx)
        episode_start = self.sampler.episode_ends[idx - 1] if idx > 0 else 0
        episode_end = self.sampler.episode_ends[idx]
        episode_len = episode_end - episode_start
        return episode_len
    
    def get_normalizer(self, mode: str = "none", **kwargs: dict) -> LinearNormalizer:
        """Return a normalizer for the dataset."""
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer["action"])
        if self.delta_action:
            this_normalizer = get_twenty_times_normalizer_from_stat(stat)
        else:
            this_normalizer = normalizer_from_stat(stat)
        normalizer["action"] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith("pos"):
                # this_normalizer = get_range_normalizer_from_stat(stat)
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("qpos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("vel"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        for key in self.depth_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer
    def get_validation_dataset(self) -> "CustomDataset":
        """Return a validation dataset."""
        val_set = copy.copy(self)
        val_set.is_val = True
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.val_horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
            skip_idx=self.skip_idx,
        )
        val_set.train_mask = self.val_mask
        return val_set
    
    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        obs_dict = dict()
        skip_start = np.random.randint(0, self.skip_frame) + self.skip_frame
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(sample[key], -1, 1).astype(np.float32) / 255.0
            obs_dict[key] = obs_dict[key][skip_start :: self.skip_frame]
            # T,C,H,W
            del sample[key]
        for key in self.depth_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint16 image to float32
            obs_dict[key] = np.moveaxis(sample[key], -1, 1).astype(np.float32) / 1000.0
            obs_dict[key] = obs_dict[key][skip_start :: self.skip_frame]
            # T,C,H,W
            del sample[key]
        for key in self.lowdim_keys:
            obs_dict[key] = sample[key].astype(np.float32)
            obs_dict[key] = obs_dict[key][skip_start :: self.skip_frame]
            del sample[key]

        actions = sample["action"].astype(np.float32)
        action_dim = actions.shape[-1]
        downsample_horizon = actions.shape[0] // self.skip_frame - 1
        action_len = downsample_horizon * self.skip_frame
        action_start = skip_start - self.skip_frame
        actions = actions[action_start : action_start + action_len]
        actions = actions.reshape(downsample_horizon, self.skip_frame, action_dim)
        if self.delta_action:
            joint_pos = obs_dict["joint_pos"].astype(np.float32)
            robot_bases = self.robot_bases
            # compute ee pos in robot_bases[0]
            num_robot = joint_pos.shape[1] // 7
            world_t_ee_pose = np.zeros((joint_pos.shape[0], num_robot, 4, 4))
            for i in range(num_robot):
                for t in range(joint_pos.shape[0]):
                    joint_fk = np.concatenate(
                        [
                            joint_pos[t, i * 7 : (i + 1) * 7],
                            joint_pos[t, i * 7 + 6 : (i + 1) * 7],
                        ]
                    )
                    ee_pose = self.kin_helper.compute_fk_from_link_idx(
                        joint_fk, [self.kin_helper.sapien_eef_idx]
                    )[0]
                    world_t_ee_pose[t, i] = robot_bases[i] @ ee_pose
            if self.action_mode == "bimanual_push":
                d_actions = np.zeros_like(actions)
                d_actions[..., :2] = actions[..., :2] - world_t_ee_pose[:, 0:1, :2, 3]
                d_actions[..., 2:] = actions[..., 2:] - world_t_ee_pose[:, 1:2, :2, 3]
                actions = d_actions
            elif self.action_mode == "single_ee":
                d_actions = np.zeros_like(actions)
                d_actions[..., :3] = actions[..., :3] - world_t_ee_pose[:, 1:2, :3, 3]
                d_actions[..., 3:4] = (
                    actions[..., 3:4] - joint_pos[:, 13:14][:, None]
                ) / 100
            else:
                raise NotImplementedError
        actions = actions.reshape(downsample_horizon, self.skip_frame * action_dim)
        data = {
            "visual": torch.from_numpy(list(obs_dict.values())[0]),
        }
        return data, torch.from_numpy(actions), {}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return data


def load_custom_slice_train_val(
    dataset_dir: str,
    pad_before: int = 1,
    pad_after: int = 7,
    seed: int = 42,
    val_ratio: float = 0.1,
    skip_idx: int = 20,
    use_cache: bool = True,
    delta_action: bool = False,
    action_mode: str = "bimanual_push",
    shape_meta: Optional[Dict[str, Any]] = None,
    frameskip: int = 1,
    num_pred: int = 0,
    num_hist: int = 0,
):
    dset = CustomDataset(
        dataset_dir=dataset_dir,
        horizon=num_hist + num_pred,
        val_horizon=num_hist + num_pred,
        skip_frame=frameskip,
        pad_before=pad_before,
        pad_after=pad_after,
        seed=seed,
        val_ratio=val_ratio,
        skip_idx=skip_idx,
        use_cache=use_cache,
        delta_action=delta_action,
        action_mode=action_mode,
        shape_meta=shape_meta
    )
    dset_train = dset
    dset_val = dset.get_validation_dataset()

    datasets = {}
    datasets['train'] = dset_train
    datasets['valid'] = dset_val
    traj_dset = {}
    traj_dset['train'] = dset_train
    traj_dset['valid'] = dset_val
    return datasets, traj_dset
