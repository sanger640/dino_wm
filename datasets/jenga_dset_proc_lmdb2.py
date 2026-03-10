import torch
import numpy as np
from pathlib import Path
import cv2
from einops import rearrange
import os
import lmdb
import pickle

cv2.setNumThreads(0)
from .traj_dset import TrajDataset, get_train_val_sliced

class LazyVideo:
    """Fetches images directly from the LMDB on the fly."""
    def __init__(self, lmdb_env, episode_keys, cam_prefixes, num_frames, transform=None):
        self.lmdb_env = lmdb_env
        self.episode_keys = episode_keys 
        self.cam_prefixes = cam_prefixes
        self.num_frames = num_frames
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
        with self.lmdb_env.begin(write=False) as txn:
            for idx in indices:
                t_imgs = []
                for prefix in self.cam_prefixes:
                    cam_keys = self.episode_keys.get(prefix, [])
                    
                    if len(cam_keys) > 0:
                        safe_idx = min(idx, len(cam_keys) - 1)
                        key = cam_keys[safe_idx].encode('ascii')
                        img_bytes = txn.get(key)
                        
                        if img_bytes is not None:
                            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            t_imgs.append(img)
                        else:
                            t_imgs.append(np.zeros((224, 224, 3), dtype=np.uint8))
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
        self.lmdb_path = self.data_path / "jenga_unified.lmdb"
        self.transform = transform
        self.normalize_action = normalize_action
        self.cam_prefixes = cam_prefixes or ["cam1", "cam2"]
        
        self.lmdb_env = None 
        self._lmdb_pid = -1

        # --- Instant Initialization via __metadata__ ---
        print(f"⚡ Rank {os.environ.get('RANK', 0)} fetching metadata from LMDB...")
        
        # Temporarily open LMDB just to grab the metadata
        with lmdb.open(str(self.lmdb_path), readonly=True, lock=False) as temp_env:
            with temp_env.begin(write=False) as txn:
                meta_bytes = txn.get(b"__metadata__")
                if meta_bytes is None:
                    raise ValueError("Could not find __metadata__ key in LMDB!")
                self.metadata = pickle.loads(meta_bytes)

        # Extract stats
        self.episodes_dict = self.metadata["episodes"]
        self.episode_names = list(self.episodes_dict.keys())
        
        if n_rollout:
            self.episode_names = self.episode_names[:n_rollout]

        self.seq_lengths = torch.tensor([self.episodes_dict[ep]["seq_len"] for ep in self.episode_names])
        self.action_dim = self.proprio_dim = self.state_dim = 4

        # Load normalization stats
        if self.normalize_action:
            stats = self.metadata["stats"]
            self.action_mean = torch.from_numpy(stats["action_mean"])
            self.action_std = torch.from_numpy(stats["action_std"])
            self.proprio_mean = torch.from_numpy(stats["proprio_mean"])
            self.proprio_std = torch.from_numpy(stats["proprio_std"])
        else:
            self.action_mean = self.proprio_mean = torch.zeros(4)
            self.action_std = self.proprio_std = torch.ones(4)

        print(f"✅ Ready. Loaded {len(self.episode_names)} episodes instantly.")

    def _init_lmdb(self):
        """Ensures PyTorch multiprocessing workers don't share LMDB environments."""
        current_pid = os.getpid()
        if self._lmdb_pid != current_pid:
            self.lmdb_env = None
            self._lmdb_pid = current_pid

        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(
                str(self.lmdb_path), 
                readonly=True, lock=False, readahead=False, meminit=False
            )

    def __getitem__(self, idx):
        self._init_lmdb()
        ep_name = self.episode_names[idx]
        
        # 1. Fetch vectors directly from LMDB
        with self.lmdb_env.begin(write=False) as txn:
            act_bytes = txn.get(f"{ep_name}_actions".encode('ascii'))
            proc_bytes = txn.get(f"{ep_name}_proprio".encode('ascii'))
            
        act_vec = torch.from_numpy(pickle.loads(act_bytes))
        proc_vec = torch.from_numpy(pickle.loads(proc_bytes))
        
        # 2. Normalize
        act = (act_vec - self.action_mean) / self.action_std
        proprio = (proc_vec - self.proprio_mean) / self.proprio_std
        
        # 3. Setup Video Loader
        ep_keys = self.episodes_dict[ep_name]["keys"]
        visual_loader = LazyVideo(
            self.lmdb_env, ep_keys, self.cam_prefixes, len(act_vec), self.transform
        )
        
        return {"visual": visual_loader, "proprio": proprio}, act, proc_vec, {}

    def get_seq_length(self, idx): return self.seq_lengths[idx]
    def __len__(self): return len(self.episode_names)

def load_jenga_slice_train_val(
    transform, n_rollout=None, data_path=None, normalize_action=True,
    split_ratio=0.9, num_hist=0, num_pred=0, frameskip=1, action_freq=10, img_freq=10, cam_serials=None, **kwargs
):
    dset = JengaDataset(data_path, n_rollout, transform, normalize_action)
    
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        num_frames=num_hist + num_pred,
        frameskip=frameskip,
    )
    return {"train": train_slices, "valid": val_slices}, {"train": dset_train, "valid": dset_val}