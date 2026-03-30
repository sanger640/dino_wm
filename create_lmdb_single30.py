import os
import json
import pickle
import lmdb
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# --- CONFIG ---
DATA_PATH = Path("/home/sanger/wksp/panda_express/tasks/jenga_mujoco_noise")
LMDB_PATH = DATA_PATH / "jenga_single.lmdb" # Overwriting the same file
NUM_WORKERS = min(32, multiprocessing.cpu_count())
MAP_SIZE = 20*1024**3 # 1 Terabyte virtual limit


def process_episode(ep_dir):
    """Reads JSON + Images, aligns them by timestamp, returns LMDB pairs."""
    ep_name = ep_dir.name
    err = 0
    # 1. Parse JSON for Actions & Timestamps
    traj_files = list(ep_dir.glob("trajectory_*.json"))
    if not traj_files: return None
    
    try:
        with open(traj_files[0], 'r') as f:
            data = json.load(f)
            
        waypoints = data.get('waypoints', [])
        if len(waypoints) < 2: return None

        # Extract Action Data AND Timestamps
        act_times = np.array([w['timestamp'] for w in waypoints])
        act_raw = np.array([w['position'] + [float(w['gripper'])] for w in waypoints], dtype=np.float32)
        proc_raw = np.array([w['proc_pos'] + [float(w['proc_gripper'])] for w in waypoints], dtype=np.float32)
    except Exception:
        return None

    # 2. Parse Images for Timestamps
    img_dir = ep_dir / "rgb_frames"
    cam = "cam2"
    img_files = sorted(list(img_dir.glob(f"{cam}_*.png")), key=lambda x: int(x.stem.split('_')[-1]))
    if not img_files: return None
    
    # Convert image filename milliseconds back to seconds for comparison
    img_times = np.array([int(p.stem.split('_')[-1]) / 1000.0 for p in img_files])

    # 3. ALIGNMENT (Action-Centric Match)
    aligned_act = []
    aligned_proc = []
    valid_keys = {cam: []}
    kv_pairs = []
    
    # For every action waypoint, find the closest image frame in time
    for i, a_time in enumerate(act_times):
        # Calculate time difference to all images
        time_diffs = np.abs(img_times - a_time)
        closest_img_idx = np.argmin(time_diffs)
        
        # Guardrail: If the closest image is more than 0.1s off, skip this step
        if time_diffs[closest_img_idx] > 0.1:
            err+=1
            print(f"err : {err}")
            continue 
            
        img_path = img_files[closest_img_idx]
        ts_str = img_path.stem.split('_')[-1]
        key_str = f"{ep_name}_{cam}_{ts_str}"
        
        # Store the aligned pair
        aligned_act.append(act_raw[i])
        aligned_proc.append(proc_raw[i])
        valid_keys[cam].append(key_str)
        
        # Write the image bytes
        with open(img_path, 'rb') as f:
            kv_pairs.append((key_str.encode('ascii'), f.read()))

    # Reject if alignment gutted the episode
    if len(aligned_act) < 2: return None

    # Convert aligned lists back to numpy arrays
    final_act = np.array(aligned_act, dtype=np.float32)
    final_proc = np.array(aligned_proc, dtype=np.float32)

    # 4. Serialize and store vectors
    kv_pairs.append((f"{ep_name}_actions".encode('ascii'), pickle.dumps(final_act)))
    kv_pairs.append((f"{ep_name}_proprio".encode('ascii'), pickle.dumps(final_proc)))

    # Return clean, 1-to-1 matched metadata
    ep_info = {
        "name": ep_name,
        "seq_len": len(final_act),
        "keys": valid_keys,
        "actions": final_act,
        "proprios": final_proc
    }
    
    return kv_pairs, ep_info

def pack_parallel():
    episodes_dir = DATA_PATH / "episodes"
    episode_dirs = sorted([p for p in episodes_dir.iterdir() if p.is_dir()])
    
    print(f"🚀 Packing {len(episode_dirs)} episodes into Single-Cam LMDB using {NUM_WORKERS} workers...")
    
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
    print(f"✅ Success! Single-Cam LMDB size: {os.path.getsize(LMDB_PATH) / 1024**3:.2f} GB")

if __name__ == "__main__":
    pack_parallel()