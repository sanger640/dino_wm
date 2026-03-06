import os
import lmdb
from pathlib import Path
from tqdm import tqdm
import json

# --- CONFIGURATION ---
DATA_PATH = Path("/home/sanger/panda_express/tasks/jenga_mujoco") # Update this to your Jenga dataset root
LMDB_PATH = DATA_PATH / "jenga_images.lmdb"
CAM_PREFIXES = ["cam1", "cam2"]
IMG_EXT = ".png"

# 100GB Map Size (This is just a virtual memory limit, it won't actually take 100GB on disk)
MAP_SIZE = 1099511627776 

def pack_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset path {DATA_PATH} does not exist.")
        
    episodes_dir = DATA_PATH / "episodes"
    episode_dirs = sorted(
        [p for p in episodes_dir.iterdir() if p.is_dir()],
        key=lambda x: int(x.name) if x.name.isdigit() else x.name
    )

    print(f"Found {len(episode_dirs)} episodes. Creating LMDB at {LMDB_PATH}...")
    
    # Open LMDB Environment
    env = lmdb.open(str(LMDB_PATH), map_size=MAP_SIZE, writemap=True)
    
    total_images_written = 0

    with env.begin(write=True) as txn:
        for ep_dir in tqdm(episode_dirs, desc="Packing Episodes"):
            ep_name = ep_dir.name
            img_dir = ep_dir / "rgb_frames"
            
            # Find all timestamps from cam1 to know what to pack
            cam1_files = list(img_dir.glob(f"{CAM_PREFIXES[0]}_*{IMG_EXT}"))
            
            def get_ts(p):
                try: return int(p.stem.split('_')[-1])
                except ValueError: return 0
                
            timestamps = sorted([get_ts(f) for f in cam1_files])
            
            for ts in timestamps:
                for prefix in CAM_PREFIXES:
                    img_path = img_dir / f"{prefix}_{ts}{IMG_EXT}"
                    
                    if img_path.exists():
                        # Read raw bytes directly (bypassing PIL/OpenCV decoding for now)
                        with open(img_path, 'rb') as f:
                            img_bytes = f.read()
                        
                        # Create a unique key: e.g., "001_cam1_10"
                        key = f"{ep_name}_{prefix}_{ts}".encode('ascii')
                        
                        # Store in database
                        txn.put(key, img_bytes)
                        total_images_written += 1

    env.close()
    print(f"\nSuccessfully packed {total_images_written} images into LMDB!")
    print(f"Database size: {os.path.getsize(LMDB_PATH) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    pack_dataset()