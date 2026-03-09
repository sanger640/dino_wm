import torch
import numpy as np
from pathlib import Path
import cv2
from einops import rearrange
import os
import lmdb
import json

# Prevent OpenCV from spawning extra threads that choke the CPU
cv2.setNumThreads(0)

from .traj_dset import TrajDataset, get_train_val_sliced

class LazyVideo:
    """
    Ultra-lean reader using sequential timestamp keys.
    Bypasses serialization bottlenecks to maximize H100 throughput.
    """
    def __init__(self, lmdb_env, episode_keys, cam_prefixes, num_frames, transform=None):
        self.lmdb_env = lmdb_env
        self.episode_keys = episode_keys  # New: Pass the actual string keys!
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
                    cam_keys = self.episode_keys[prefix]
                    
                    if len(cam_keys) > 0:
                        # Safety Map: If ratio is 1.1, clamp to the last available image index
                        safe_idx = min(idx, len(cam_keys) - 1)
                        key = cam_keys[safe_idx].encode('ascii')
                        img_bytes = txn.get(key)
                        
                        if img_bytes is not None:
                            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            t_imgs.append(img)
                        else:
                            print("oh boi")
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
        # ... [Keep your existing init setup (paths, episodes_root, etc.)] ...
        self.data_path = Path(data_path)
        self.lmdb_path = self.data_path / "jenga_images.lmdb"
        self.transform = transform
        self.normalize_action = normalize_action
        self.cam_prefixes = cam_prefixes or ["cam1", "cam2"]
        
        self.lmdb_env = None 
        self._lmdb_pid = -1

        episodes_root = self.data_path / "episodes"
        self.episode_dirs = sorted(
            [p for p in episodes_root.iterdir() if p.is_dir()],
            key=lambda x: int(x.name) if x.name.isdigit() else x.name
        )
        if n_rollout:
            self.episode_dirs = self.episode_dirs[:n_rollout]

        self.episodes_data = []
        self.seq_lengths = []
        all_actions = []
        all_proprios = []
        
        print(f"🔍 Rank {os.environ.get('RANK', 0)} parsing JSONs...")

        for ep_dir in self.episode_dirs:
            traj_files = list(ep_dir.glob("trajectory_*.json"))
            if not traj_files: continue
            
            try:
                with open(traj_files[0], 'r') as f:
                    data = json.load(f)
                
                waypoints = data.get('waypoints', [])
                if len(waypoints) < 2: continue

                act_vec = torch.tensor([w['position'] + [float(w['gripper'])] for w in waypoints], dtype=torch.float32)
                proc_vec = torch.tensor([w['proc_pos'] + [float(w['proc_gripper'])] for w in waypoints], dtype=torch.float32)

                self.seq_lengths.append(len(waypoints))
                self.episodes_data.append({
                    "name": ep_dir.name,
                    "act_vec": act_vec,
                    "proc_vec": proc_vec
                })
                all_actions.append(act_vec)
                all_proprios.append(proc_vec)
            except Exception as e:
                continue

        # --- NEW: Lightning-Fast Key Scanner ---
        # This builds a tiny list of correct string keys for each episode in < 1 second.
        print(f"⚡ Rank {os.environ.get('RANK', 0)} linking timestamps to actions...")
        self.episode_keys = {ep["name"]: {cam: [] for cam in self.cam_prefixes} for ep in self.episodes_data}
        
        with lmdb.open(str(self.lmdb_path), readonly=True, lock=False).begin() as txn:
            for key_bytes, _ in txn.cursor():
                key_str = key_bytes.decode('ascii')
                parts = key_str.split('_')
                if len(parts) >= 3:
                    ep_name, cam_name = parts[0], parts[1]
                    if ep_name in self.episode_keys and cam_name in self.cam_prefixes:
                        self.episode_keys[ep_name][cam_name].append(key_str)
                        
        # Sort chronologically so Action 0 gets Timestamp 0
        for ep_name in self.episode_keys:
            for cam in self.cam_prefixes:
                self.episode_keys[ep_name][cam].sort(key=lambda x: int(x.split('_')[-1]))
        # ---------------------------------------

        self.seq_lengths = torch.tensor(self.seq_lengths)
        self.action_dim = self.proprio_dim = self.state_dim = 4

        # Calculate Normalization Statistics
        if self.normalize_action and len(all_actions) > 0:
            act_cat = torch.cat(all_actions, dim=0)
            prop_cat = torch.cat(all_proprios, dim=0)
            self.action_mean = act_cat.mean(0)
            self.action_std = act_cat.std(0) + 1e-6
            self.proprio_mean = prop_cat.mean(0)
            self.proprio_std = prop_cat.std(0) + 1e-6
        else:
            self.action_mean = self.proprio_mean = torch.zeros(4)
            self.action_std = self.proprio_std = torch.ones(4)

    def _init_lmdb(self):
        # ... [Keep exact same _init_lmdb logic] ...
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
        ep = self.episodes_data[idx]
        
        act = (ep["act_vec"] - self.action_mean) / self.action_std
        proprio = (ep["proc_vec"] - self.proprio_mean) / self.proprio_std
        
        # We now pass the specific list of timestamped keys for THIS episode
        visual_loader = LazyVideo(
            self.lmdb_env, self.episode_keys[ep["name"]], self.cam_prefixes, len(ep["act_vec"]), self.transform
        )
        
        return {"visual": visual_loader, "proprio": proprio}, act, ep["proc_vec"], {}

    def get_seq_length(self, idx): return self.seq_lengths[idx]
    def __len__(self): return len(self.episodes_data)


def load_jenga_slice_train_val(
    transform, n_rollout=None, data_path=None, normalize_action=True,
    split_ratio=0.9, num_hist=0, num_pred=0, frameskip=1, action_freq=10, img_freq=10, cam_serials=None, **kwargs
):
    # Pass action_freq/img_freq in signature just to absorb Hydra's config arguments safely
    dset = JengaDataset(data_path, n_rollout, transform, normalize_action)
    
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        num_frames=num_hist + num_pred,
        frameskip=frameskip,
    )
    return {"train": train_slices, "valid": val_slices}, {"train": dset_train, "valid": dset_val}