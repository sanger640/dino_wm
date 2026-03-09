import torch
import numpy as np
from pathlib import Path
import cv2
import pickle
from einops import rearrange
from typing import Callable, Optional, List
import lmdb

from .traj_dset import TrajDataset, get_train_val_sliced

class LazyVideo:
    """
    Reads images instantly from the LMDB key-value store using cv2 byte decoding.
    """
    def __init__(self, lmdb_env, ep_name, timestamps, cam_prefixes, freq_ratio, transform=None):
        self.lmdb_env = lmdb_env
        self.ep_name = ep_name
        self.timestamps = timestamps
        self.cam_prefixes = cam_prefixes
        self.freq_ratio = freq_ratio
        self.transform = transform
        
        self.shape = (len(timestamps) // freq_ratio, len(cam_prefixes), 3, 224, 224) 

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
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
        
        with self.lmdb_env.begin(write=False) as txn:
            for action_idx in indices:
                img_idx = action_idx * self.freq_ratio
                t_imgs = []
                
                if img_idx < len(self.timestamps):
                    ts = self.timestamps[img_idx]
                    for prefix in self.cam_prefixes:
                        key = f"{self.ep_name}_{prefix}_{ts}".encode('ascii')
                        img_bytes = txn.get(key)
                        
                        if img_bytes is not None:
                            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            t_imgs.append(img)
                        else:
                            print(f"Missing key in LMDB: {key}")
                            t_imgs.append(np.zeros((224, 224, 3), dtype=np.uint8))
                else:
                    t_imgs = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in self.cam_prefixes]
                
                loaded_imgs.append(np.stack(t_imgs))

        images = np.stack(loaded_imgs)
        images = torch.from_numpy(images).float()
        
        images = rearrange(images, "t v h w c -> t v c h w") / 255.0

        if self.transform:
            t, v, c, h, w = images.shape
            images = rearrange(images, "t v c h w -> (t v) c h w")
            images = self.transform(images)
            images = rearrange(images, "(t v) c h w -> t v c h w", t=t, v=v)

        if isinstance(item, int):
            return images[0]
            
        return images


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
        self.lmdb_path = self.data_path / "jenga_images.lmdb"
        self.pkl_path = self.data_path / "jenga_metadata.pkl"
        
        self.transform = transform
        self.normalize_action = normalize_action
        self.cam_prefixes = cam_prefixes or ["cam1", "cam2"]
        self.freq_ratio = int(img_freq / action_freq) 

        # MULTIPROCESSING SAFETY: Do not open LMDB yet
        self.lmdb_env = None 

        if not self.pkl_path.exists():
            raise RuntimeError(f"Metadata pickle not found at {self.pkl_path}. Please run create_metadata_pkl.py first!")

        # --- FAST LOAD: Read all metadata instantly from the pickle ---
        with open(self.pkl_path, 'rb') as f:
            metadata = pickle.load(f)
            
        self.episodes_data = metadata["episodes_data"]
        
        if n_rollout:
            self.episodes_data = self.episodes_data[:n_rollout]
            self.seq_lengths = torch.tensor(metadata["seq_lengths"][:n_rollout])
        else:
            self.seq_lengths = torch.tensor(metadata["seq_lengths"])

        if self.normalize_action:
            self.action_mean = torch.from_numpy(metadata["stats"]["action_mean"])
            self.action_std = torch.from_numpy(metadata["stats"]["action_std"])
            self.proprio_mean = torch.from_numpy(metadata["stats"]["proprio_mean"])
            self.proprio_std = torch.from_numpy(metadata["stats"]["proprio_std"])
        else:
            self.action_mean = torch.zeros(4)
            self.action_std = torch.ones(4)
            self.proprio_mean = torch.zeros(4)
            self.proprio_std = torch.ones(4)
            
        self.action_dim = 4
        self.proprio_dim = 4
        self.state_dim = 4

    def _init_lmdb(self):
        """Lazily initialize the LMDB connection exactly once per DataLoader worker."""
        if self.lmdb_env is None:
            if not self.lmdb_path.exists():
                raise RuntimeError(f"LMDB database not found at {self.lmdb_path}.")
            self.lmdb_env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,     
                readahead=False, 
                meminit=False    
            )

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def __getitem__(self, idx):
        self._init_lmdb()

        ep_data = self.episodes_data[idx]
        
        # Convert numpy arrays stored in the pickle to torch tensors
        act_data = torch.from_numpy(ep_data["act_vec"])
        proc_data = torch.from_numpy(ep_data["proc_vec"])
        
        act = (act_data - self.action_mean) / self.action_std
        proprio = (proc_data - self.proprio_mean) / self.proprio_std
        state = proc_data 

        visual_loader = LazyVideo(
            lmdb_env=self.lmdb_env,
            ep_name=ep_data["ep_name"],
            timestamps=ep_data["timestamps"],
            cam_prefixes=self.cam_prefixes,
            freq_ratio=self.freq_ratio,
            transform=self.transform
        )

        obs = {"visual": visual_loader, "proprio": proprio}
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