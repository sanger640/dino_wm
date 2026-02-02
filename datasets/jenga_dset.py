import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from einops import rearrange
from typing import Callable, Optional, List
import os
import time

from .traj_dset import TrajDataset, get_train_val_sliced

class LazyVideo:
    """
    A helper class that acts like a Tensor but loads images from disk only when sliced.
    Handles the mapping from Action Index (10Hz) -> Image Index (30Hz).
    """
    def __init__(self, img_dir, timestamps, cam_prefixes, img_ext, freq_ratio, transform=None):
        self.img_dir = img_dir
        self.timestamps = timestamps
        self.cam_prefixes = cam_prefixes
        self.img_ext = img_ext
        self.freq_ratio = freq_ratio
        self.transform = transform
        
        # We pretend to have the length of the actions, not images
        self.shape = (len(timestamps) // freq_ratio, len(cam_prefixes), 3, 224, 224) 

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        # Handle slicing (e.g., [start:end:step]) provided by TrajSlicerDataset
        # start_time = time.time()
        # print("woohoo")
        if isinstance(item, slice):
            start = item.start if item.start is not None else 0
            stop = item.stop if item.stop is not None else len(self)
            step = item.step if item.step is not None else 1
            indices = range(start, stop, step)
        elif isinstance(item, int):
            indices = [item]
        else:
            raise TypeError("LazyVideo indices must be integers or slices")

        loaded_imgs = []
        for action_idx in indices:
            # MAP 10Hz ACTION INDEX -> 30Hz IMAGE INDEX
            img_idx = action_idx * self.freq_ratio
            
            t_imgs = []
            # Check bounds
            if img_idx < len(self.timestamps):
                ts = self.timestamps[img_idx]
                for prefix in self.cam_prefixes:
                    fname = f"{prefix}_{ts}{self.img_ext}"
                    img_path = self.img_dir / fname
                    try:
                        img = Image.open(img_path).convert('RGB')
                        t_imgs.append(np.array(img))
                    except Exception:
                        # Return black frame on failure
                        print("img exception dude")
                        t_imgs.append(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                t_imgs = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in self.cam_prefixes]
                print("ummm chief?")
            loaded_imgs.append(np.stack(t_imgs))

        # Stack Time: (T, Views, H, W, C)
        images = np.stack(loaded_imgs)
        images = torch.from_numpy(images).float()
        
        # Rearrange to (T, V, C, H, W)
        images = rearrange(images, "t v h w c -> t v c h w") / 255.0

        # Apply Transforms
        if self.transform:
            t, v, c, h, w = images.shape
            images = rearrange(images, "t v c h w -> (t v) c h w")
            images = self.transform(images)
            images = rearrange(images, "(t v) c h w -> t v c h w", t=t, v=v)

        # print("doneoo")
        # print(time.time() - start_time)   
        # If single index was requested, remove the time dimension (to behave like a tensor)
        if isinstance(item, int):
            return images[0]
            
        return images

# --- Updated Dataset Class ---
class JengaDataset(TrajDataset):
    def __init__(
        self,
        data_path: str,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
        img_ext: str = ".png",
        action_freq: int = 10,
        img_freq: int = 30,
        cam_prefixes: List[str] = None, 
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        self.img_ext = img_ext
        self.cam_prefixes = cam_prefixes or ["cam1", "cam2"]
        self.freq_ratio = int(img_freq / action_freq) 

        self.episode_dirs = sorted(
            [p for p in (self.data_path / "episodes").iterdir() if p.is_dir()],
            key=lambda x: int(x.name) if x.name.isdigit() else x.name
        )
        
        if n_rollout:
            self.episode_dirs = self.episode_dirs[:n_rollout]

        self.episodes_data = []
        self.seq_lengths = []
        all_actions = []

        print(f"Loading {len(self.episode_dirs)} episodes...")

        for ep_dir in self.episode_dirs:
            # 1. Load JSON
            json_files = list(ep_dir.glob("trajectory_*.json"))
            if not json_files: continue
            
            with open(json_files[0], 'r') as f:
                traj_data = json.load(f)
            
            waypoints = traj_data['waypoints']
            img_dir = ep_dir / "rgb_frames"
            
            # 2. Scan and Sort Images
            cam1_files = list(img_dir.glob(f"{self.cam_prefixes[0]}_*{self.img_ext}"))
            if not cam1_files: continue

            def get_ts(p):
                try: return int(p.stem.split('_')[-1])
                except ValueError: return 0

            cam1_files.sort(key=get_ts)
            valid_timestamps = [get_ts(f) for f in cam1_files]

            # 3. Validation
            n_images = len(valid_timestamps)
            n_actions = len(waypoints)
            max_valid_actions = n_images // self.freq_ratio
            final_len = min(n_actions, max_valid_actions)
            
            if final_len < 2: continue 

            self.seq_lengths.append(final_len)
            
            # 4. Load Vectors
            pos = np.array([wp['position'] for wp in waypoints], dtype=np.float32)[:final_len]
            # ori = np.array([wp['orientation'] for wp in waypoints], dtype=np.float32)[:final_len]
            grip = np.array([[float(wp['gripper'])] for wp in waypoints], dtype=np.float32)[:final_len]
            
            traj_vec = np.concatenate([pos, grip], axis=-1)
            
            # Store metadata instead of loading images
            self.episodes_data.append({
                "traj": torch.from_numpy(traj_vec),
                "timestamps": valid_timestamps, 
                "img_dir": img_dir
            })
            
            all_actions.append(torch.from_numpy(traj_vec))

        self.seq_lengths = torch.tensor(self.seq_lengths)
        
        # 5. Stats
        if self.normalize_action and len(all_actions) > 0:
            all_actions_cat = torch.cat(all_actions, dim=0)
            self.action_mean = all_actions_cat.mean(dim=0)
            self.action_std = all_actions_cat.std(dim=0) + 1e-6
            self.proprio_mean = self.action_mean
            self.proprio_std = self.action_std
        else:
            self.action_mean = torch.zeros(4)
            self.action_std = torch.ones(4)
            self.proprio_mean = torch.zeros(4)
            self.proprio_std = torch.ones(4)
            
        self.action_dim = 4
        self.proprio_dim = 4
        # IMPORTANT: TrajSlicerDataset checks for this
        self.state_dim = 4 

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def __getitem__(self, idx):
        """
        Returns full episode data, but images are wrapped in LazyVideo.
        TrajSlicerDataset calls this, then slices the result.
        """
        ep_data = self.episodes_data[idx]
        
        # Prepare vectors
        traj_data = ep_data["traj"]
        act = (traj_data - self.action_mean) / self.action_std
        proprio = (traj_data - self.proprio_mean) / self.proprio_std
        state = traj_data 

        # Prepare Lazy Video Wrapper
        # This object is lightweight. The actual loading happens 
        # when TrajSlicerDataset does: obs['visual'][start:end]
        visual_loader = LazyVideo(
            img_dir=ep_data["img_dir"],
            timestamps=ep_data["timestamps"],
            cam_prefixes=self.cam_prefixes,
            img_ext=self.img_ext,
            freq_ratio=self.freq_ratio,
            transform=self.transform
        )

        obs = {"visual": visual_loader, "proprio": proprio}
        
        # Returns standard tuple expected by TrajSlicerDataset
        return obs, act, state, {}

    def __len__(self):
        return len(self.seq_lengths)

def load_jenga_slice_train_val(
    transform,
    n_rollout=None,
    data_path=None,
    normalize_action=True,
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=1,
    action_freq=10,
    img_freq=30,
    cam_serials=None, 
):
    dset = JengaDataset(
        data_path=data_path,
        n_rollout=n_rollout,
        transform=transform,
        normalize_action=normalize_action,
        action_freq=action_freq,
        img_freq=img_freq,
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