import os
from typing import Dict, Optional, Any
import copy 

import h5py
import numpy as np
import torch
import zarr
from yixuan_utilities.kinematics_helper import KinHelper
import torch
import decord
import numpy as np
from filelock import FileLock

from dino_wm.ext_utils.imagecodecs_numcodecs import register_codecs
from dino_wm.ext_utils.replay_buffer import ReplayBuffer
from dino_wm.ext_utils.sampler import SequenceSampler, get_val_mask
from .traj_dset import TrajDataset

decord.bridge.set_bridge("torch")
register_codecs()

def load_replay_buffer(dataset_dir: str) -> ReplayBuffer:
    replay_buffer = None
    cache_info_str = ""
    cache_zarr_path = os.path.join(dataset_dir, f"cache{cache_info_str}.zarr.zip")
    cache_lock_path = cache_zarr_path + ".lock"
    print("Acquiring lock on cache.")
    with FileLock(cache_lock_path):
        print("Loading cached ReplayBuffer from Disk.")
        with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
            replay_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store, store=zarr.MemoryStore()
            )
        print("Loaded!")
    return replay_buffer

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
        skip_idx: int = 20,
        use_cache: bool = True,
        delta_action: bool = False,
        action_mode: str = "bimanual_push",
        shape_meta: Optional[Dict[str, Any]] = None,
    ):
        # assign config
        seq_horizon = (horizon + 1) * skip_frame
        self.val_horizon = (val_horizon + 1) * skip_frame
        self.skip_idx = skip_idx
        self.action_mode = action_mode

        train_dir = os.path.join(dataset_dir, "train")
        replay_buffer = load_replay_buffer(train_dir, use_cache, shape_meta)
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

        train_mask = np.ones((self.replay_buffer.n_episodes,), dtype=bool)
        all_keys = list(self.replay_buffer.keys())

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            keys=all_keys,
            skip_frame=skip_frame,
            keys_to_keep_intermediate=["action"],
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.lowdim_keys = lowdim_keys
        self.train_mask = train_mask
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.dataset_dir = dataset_dir
        self.skip_frame = skip_frame
        self.action_dim = replay_buffer["action"].shape[-1] * skip_frame
        self.use_cache = use_cache

    def get_seq_length(self, idx):
        episode_start = self.sampler.episode_ends[idx - 1] if idx > 0 else 0
        episode_end = self.sampler.episode_ends[idx]
        episode_len = episode_end - episode_start
        return episode_len

    def get_validation_dataset(self) -> "CustomDataset":
        """Return a validation dataset."""
        val_set = copy.copy(self)
        val_set.is_val = True
        val_dir = os.path.join(self.dataset_dir, "val")
        shape_meta = self.shape_meta
        use_cache = self.use_cache
        val_set.replay_buffer = load_replay_buffer(val_dir, use_cache, shape_meta)
        val_mask = np.ones((val_set.replay_buffer.n_episodes,), dtype=bool)
        val_set.sampler = SequenceSampler(
            replay_buffer=val_set.replay_buffer,
            sequence_length=self.val_horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_mask,
            skip_idx=self.skip_idx,
            skip_frame=self.skip_frame,
            keys_to_keep_intermediate=["action"],
        )
        val_set.train_mask = val_mask
        return val_set

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(sample[key], -1, 1).astype(np.float32) / 255.0
            obs_dict[key] = (obs_dict[key] - 0.5) * 2.0
            # T,C,H,W
            del sample[key]
        for key in self.depth_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint16 image to float32
            obs_dict[key] = np.moveaxis(sample[key], -1, 1).astype(np.float32) / 1000.0
            # T,C,H,W
            del sample[key]
        for key in self.lowdim_keys:
            obs_dict[key] = sample[key].astype(np.float32)
            # obs_dict[key] = obs_dict[key][skip_start :: self.skip_frame]
            del sample[key]

        actions = sample["action"].astype(np.float32)
        # action_dim = actions.shape[-1]
        # downsample_horizon = actions.shape[0] // self.skip_frame - 1
        # action_len = downsample_horizon * self.skip_frame
        # action_start = skip_start - self.skip_frame
        # actions = actions[action_start : action_start + action_len]
        # actions = actions.reshape(downsample_horizon, self.skip_frame, action_dim)
        # actions = actions.reshape(downsample_horizon, self.skip_frame * action_dim)
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
    val_horizon: int = 0,
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
    
    traj_dset = CustomDataset(
        dataset_dir=dataset_dir,
        horizon=val_horizon,
        val_horizon=val_horizon,
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
    traj_dset_train = traj_dset
    traj_dset_val = traj_dset.get_validation_dataset()
    traj_dset = {}
    
    traj_dset['train'] = traj_dset_train
    traj_dset['valid'] = traj_dset_val
    return datasets, traj_dset
