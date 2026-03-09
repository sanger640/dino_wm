import json
import torch
import numpy as np
from pathlib import Path
import cv2
from einops import rearrange
from typing import Callable, Optional, List
import os
import lmdb
import time

# FIX 1: Prevent OpenCV from spawning its own threads within PyTorch workers
cv2.setNumThreads(0)

from .traj_dset import TrajDataset, get_train_val_sliced

class LazyVideo:
    def __init__(self, lmdb_env, flat_keys_dict, start_idx, num_frames, cam_prefixes, transform=None):
        self.lmdb_env = lmdb_env
        self.flat_keys_dict = flat_keys_dict
        self.start_idx = start_idx
        self.num_frames = num_frames
        self.cam_prefixes = cam_prefixes
        self.transform = transform
        self.shape = (num_frames, len(cam_prefixes), 3, 224, 224)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, item):
        if isinstance(item, slice):
            indices = range(*item.indices(self.num_frames))
        elif isinstance(item, int):
            indices = [item]
        else:
            raise TypeError("LazyVideo indices must be integers or slices")

        loaded_imgs = []
        # txn is process-safe because of the PID check in the main dataset class
        with self.lmdb_env.begin(write=False) as txn:
            for action_idx in indices:
                safe_idx = min(action_idx, self.num_frames - 1)
                global_idx = self.start_idx + safe_idx
                t_imgs = []
                
                for prefix in self.cam_prefixes:
                    key = self.flat_keys_dict[prefix][global_idx]
                    img_bytes = txn.get(key)
                    
                    if img_bytes is not None:
                        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        t_imgs.append(img)
                    else:
                        t_imgs.append(np.zeros((224, 224, 3), dtype=np.uint8))
                
                loaded_imgs.append(np.stack(t_imgs))

        images = np.stack(loaded_imgs)
        images = torch.from_numpy(images).float() / 255.0
        images = rearrange(images, "t v h w c -> t v c h w")

        if self.transform:
            t, v, c, h, w = images.shape
            images = rearrange(images, "t v c h w -> (t v) c h w")
            images = self.transform(images)
            images = rearrange(images, "(t v) c h w -> t v c h w", t=t, v=v)

        return images[0] if isinstance(item, int) else images

class JengaDataset(TrajDataset):
    def __init__(self, data_path, n_rollout=None, transform=None, normalize_action=True, cam_prefixes=None):
        self.data_path = Path(data_path)
        self.lmdb_path = self.data_path / "jenga_images.lmdb"
        self.cache_path = self.data_path / "prebaked_keys_flat.pth"
        self.transform = transform
        self.normalize_action = normalize_action
        self.cam_prefixes = cam_prefixes or ["cam1", "cam2"]
        
        # FIX 2: Initialize LMDB placeholders as None
        self.lmdb_env = None 
        self._lmdb_pid = -1
        
        self.rank = int(os.environ.get("RANK", 0))

        if self.cache_path.exists():
            print(f"[Rank {self.rank}] 🚀 Loading giant flat cache...")
            cache = torch.load(self.cache_path)
            self.flat_keys = cache['flat_keys']
            self.episodes_meta = cache['episodes_meta']
            self.action_mean, self.action_std = cache['action_mean'], cache['action_std']
            self.proprio_mean, self.proprio_std = cache['proprio_mean'], cache['proprio_std']
        else:
            # (Standard Rank 0 baking logic remains the same...)
            if self.rank == 0: self._bake_flat_array(n_rollout)
            else:
                while not self.cache_path.exists(): time.sleep(2)
                cache = torch.load(self.cache_path)
                self.flat_keys, self.episodes_meta = cache['flat_keys'], cache['episodes_meta']
                self.action_mean, self.action_std = cache['action_mean'], cache['action_std']
                self.proprio_mean, self.proprio_std = cache['proprio_mean'], cache['proprio_std']

        self.action_dim = self.proprio_dim = self.state_dim = 4

    def _init_lmdb(self):
        """
        FIX 3: PID-Aware initialization. This prevents Segmentation Faults 
        by ensuring workers don't share memory pointers with the main process.
        """
        current_pid = os.getpid()
        if self._lmdb_pid != current_pid:
            # We have been forked! Close old handles and reset.
            self.lmdb_env = None
            self._lmdb_pid = current_pid

        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(
                str(self.lmdb_path), 
                readonly=True, 
                lock=False, 
                readahead=False, 
                meminit=False
            )

    def get_seq_length(self, idx): return self.episodes_meta[idx]["length"]
    def __len__(self): return len(self.episodes_meta)

    def __getitem__(self, idx):
        self._init_lmdb()
        meta = self.episodes_meta[idx]
        act = (meta["act_vec"] - self.action_mean) / self.action_std
        proprio = (meta["proc_vec"] - self.proprio_mean) / self.proprio_std
        visual_loader = LazyVideo(self.lmdb_env, self.flat_keys, meta["start_idx"], meta["length"], self.cam_prefixes, self.transform)
        return {"visual": visual_loader, "proprio": proprio}, act, meta["proc_vec"], {}
        
def load_jenga_slice_train_val(
    transform,
    n_rollout=None,
    data_path=None,
    normalize_action=True,
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=1,
    action_freq=10, # Keep in signature so Hydra config doesn't crash
    img_freq=30,    # Keep in signature so Hydra config doesn't crash
    cam_serials=None, 
):
    dset = JengaDataset(
        data_path=data_path,
        n_rollout=n_rollout,
        transform=transform,
        normalize_action=normalize_action,
        cam_prefixes=["cam1", "cam2"]
    )
    
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        num_frames=num_hist + num_pred,
        frameskip=frameskip,
    )

    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": dset_train, "valid": dset_val}
    
    return datasets, traj_dset