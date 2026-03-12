import os
import json
import pickle
import lmdb
import numpy as np
import argparse  # Added for CLI support
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Pack Jenga episodes into LMDB")
parser.add_argument("--data_path", type=str, 
                    default="/home/ali313/links/scratch/jenga_mujoco_noise",
                    help="Path to the source dataset directory")
parser.add_argument("--map_size_gb", type=int, default=50, 
                    help="LMDB map size in GB")
args = parser.parse_args()

# --- CONFIG (Dynamically set from CLI) ---
DATA_PATH = Path(args.data_path)
LMDB_PATH = DATA_PATH / "jenga_unified.lmdb"
NUM_WORKERS = min(32, multiprocessing.cpu_count())
# Increased default to 500GB based on your previous error
MAP_SIZE = args.map_size_gb * 1024**3

def process_episode(ep_dir):
    """Reads JSON + Images, returns LMDB key-value pairs and metadata stats."""
    ep_name = ep_dir.name
    
    # 1. Parse JSON
    traj_files = list(ep_dir.glob("trajectory_*.json"))
    if not traj_files: return None
    
    try:
        with open(traj_files[0], 'r') as f:
            data = json.load(f)
            
        waypoints = data.get('waypoints', [])
        if len(waypoints) < 2: return None

        act_vec = np.array([w['position'] + [float(w['gripper'])] for w in waypoints], dtype=np.float32)
        proc_vec = np.array([w['proc_pos'] + [float(w['proc_gripper'])] for w in waypoints], dtype=np.float32)
    except Exception:
        return None

    kv_pairs = []
    
    # 2. Serialize and store vectors
    kv_pairs.append((f"{ep_name}_actions".encode('ascii'), pickle.dumps(act_vec)))
    kv_pairs.append((f"{ep_name}_proprio".encode('ascii'), pickle.dumps(proc_vec)))

    # 3. Read Images and track valid keys
    img_dir = ep_dir / "rgb_frames"
    valid_keys = {"cam1": [], "cam2": []}
    
    for cam in ["cam1", "cam2"]:
        img_files = sorted(list(img_dir.glob(f"{cam}_*.png")), key=lambda x: int(x.stem.split('_')[-1]))
        for img_path in img_files:
            ts = img_path.stem.split('_')[-1]
            key_str = f"{ep_name}_{cam}_{ts}"
            
            with open(img_path, 'rb') as f:
                kv_pairs.append((key_str.encode('ascii'), f.read()))
                
            valid_keys[cam].append(key_str)

    # 4. Return the data to be written, plus the stats for the global metadata
    ep_info = {
        "name": ep_name,
        "seq_len": len(act_vec),
        "keys": valid_keys,
        "actions": act_vec,
        "proprios": proc_vec
    }
    
    return kv_pairs, ep_info

def pack_parallel():
    episodes_dir = DATA_PATH / "episodes"
    episode_dirs = sorted([p for p in episodes_dir.iterdir() if p.is_dir()])
    
    print(f"🚀 Packing {len(episode_dirs)} episodes into Unified LMDB using {NUM_WORKERS} workers...")
    
    env = lmdb.open(str(LMDB_PATH), map_size=MAP_SIZE, writemap=True)
    
    all_episodes_meta = {}
    all_actions = []
    all_proprios = []
    
    with env.begin(write=True) as txn:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            chunk_size = 100 
            for i in range(0, len(episode_dirs), chunk_size):
                chunk = episode_dirs[i : i + chunk_size]
                futures = [executor.submit(process_episode, d) for d in chunk]
                
                for f in tqdm(futures, desc=f"Writing Chunk {i//chunk_size + 1}"):
                    res = f.result()
                    if res is not None:
                        kv_pairs, ep_info = res
                        
                        # Write bytes to LMDB
                        for k, v in kv_pairs:
                            txn.put(k, v)
                            
                        # Accumulate stats for metadata
                        all_episodes_meta[ep_info["name"]] = {
                            "seq_len": ep_info["seq_len"],
                            "keys": ep_info["keys"]
                        }
                        all_actions.append(ep_info["actions"])
                        all_proprios.append(ep_info["proprios"])

    # --- Compute Global Metadata ---
    print("🧠 Computing normalization statistics...")
    act_cat = np.concatenate(all_actions, axis=0)
    prop_cat = np.concatenate(all_proprios, axis=0)
    
    metadata = {
        "episodes": all_episodes_meta,
        "stats": {
            "action_mean": act_cat.mean(axis=0),
            "action_std": act_cat.std(axis=0) + 1e-6,
            "proprio_mean": prop_cat.mean(axis=0),
            "proprio_std": prop_cat.std(axis=0) + 1e-6,
        }
    }

    # Write the master metadata key
    print("📝 Writing global __metadata__ key...")
    with env.begin(write=True) as txn:
        txn.put(b"__metadata__", pickle.dumps(metadata))
        
    env.close()
    print(f"✅ Success! Unified LMDB size: {os.path.getsize(LMDB_PATH) / 1024**3:.2f} GB")

if __name__ == "__main__":
    pack_parallel()