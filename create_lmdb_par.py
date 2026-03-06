import os
import lmdb
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# --- CONFIG ---
DATA_PATH = Path("/home/sanger/panda_express/tasks/jenga_mujoco")
LMDB_PATH = DATA_PATH / "jenga_images.lmdb"
NUM_WORKERS = multiprocessing.cpu_count() # Use all cores
MAP_SIZE = 1024**4  # 1 Terabyte (Virtual limit)

def process_episode(ep_dir):
    """Worker function to read one episode's images into memory."""
    results = []
    ep_name = ep_dir.name
    img_dir = ep_dir / "rgb_frames"
    
    # We only care about cam1/cam2 .png files
    for cam in ["cam1", "cam2"]:
        img_files = list(img_dir.glob(f"{cam}_*.png"))
        for img_path in img_files:
            ts = img_path.stem.split('_')[-1]
            key = f"{ep_name}_{cam}_{ts}".encode('ascii')
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            results.append((key, img_bytes))
    return results

def pack_parallel():
    episodes_dir = DATA_PATH / "episodes"
    episode_dirs = sorted([p for p in episodes_dir.iterdir() if p.is_dir()])
    
    print(f"🚀 Packing {len(episode_dirs)} episodes using {NUM_WORKERS} workers...")
    
    env = lmdb.open(str(LMDB_PATH), map_size=MAP_SIZE, writemap=True)
    
    with env.begin(write=True) as txn:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # We process in chunks to avoid overwhelming RAM
            chunk_size = 100 
            for i in range(0, len(episode_dirs), chunk_size):
                chunk = episode_dirs[i : i + chunk_size]
                futures = [executor.submit(process_episode, d) for d in chunk]
                
                for f in tqdm(futures, desc=f"Writing Chunk {i//chunk_size + 1}"):
                    ep_results = f.result()
                    for key, val in ep_results:
                        txn.put(key, val)
    
    env.close()
    print(f"✅ Success! LMDB size: {os.path.getsize(LMDB_PATH) / 1024**2:.2f} MB")

if __name__ == "__main__":
    pack_parallel()