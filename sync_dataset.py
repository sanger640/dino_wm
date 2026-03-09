import os
import json
import glob
from pathlib import Path
import numpy as np
from tqdm import tqdm

DATA_DIR = Path("/home/ali313/scratch/jenga_mujoco_noise/episodes")

def sync_episode(ep_dir):
    json_files = list(ep_dir.glob("trajectory_*.json"))
    if not json_files:
        return False
    
    # Load the action trajectory
    traj_path = json_files[0]
    with open(traj_path, 'r') as f:
        traj_data = json.load(f)
    
    # Get all action timestamps
    action_timestamps = np.array([wp['timestamp'] for wp in traj_data['waypoints']])
    
    # Get all image timestamps (from cam1 filenames)
    img_dir = ep_dir / "rgb_frames"
    cam1_files = sorted(list(img_dir.glob("cam1_*.png")))
    
    if not cam1_files:
        return False

    img_timestamps = []
    img_filenames = []
    for f in cam1_files:
        # Extract the integer timestamp from the filename (e.g. cam1_1772267583737.png)
        ts_str = f.stem.split('_')[1]
        img_timestamps.append(int(ts_str) / 1000.0) # convert back to seconds
        img_filenames.append(f.name)
        
    img_timestamps = np.array(img_timestamps)
    
    # The Mapping dictionary
    sync_map = {}
    
    # For every action, find the closest image
    for action_idx, action_ts in enumerate(action_timestamps):
        # Calculate absolute difference between this action and ALL images
        time_diffs = np.abs(img_timestamps - action_ts)
        
        # Find the index of the image with the smallest time difference
        closest_img_idx = np.argmin(time_diffs)
        
        # Check if the closest image is actually close (e.g., within 0.1 seconds)
        # If it's way off, something went horribly wrong in recording
        if time_diffs[closest_img_idx] > 0.15:
             print(f"[Warning] Ep {ep_dir.name} Act {action_idx} has no image within 0.15s!")
             # You could choose to drop the episode here, but let's keep it for now
        
        sync_map[str(action_idx)] = {
            "img_idx": int(closest_img_idx),
            "cam1_file": img_filenames[closest_img_idx],
            "cam2_file": img_filenames[closest_img_idx].replace("cam1", "cam2"),
            "time_error_ms": float(time_diffs[closest_img_idx] * 1000)
        }
        
    # Save the mapping alongside the trajectory
    map_path = ep_dir / "sync_map.json"
    with open(map_path, 'w') as f:
        json.dump(sync_map, f, indent=2)
        
    return True

if __name__ == "__main__":
    episodes = sorted([p for p in DATA_DIR.iterdir() if p.is_dir()])
    print(f"Found {len(episodes)} episodes. Starting synchronization...")
    
    success_count = 0
    for ep in tqdm(episodes):
        if sync_episode(ep):
            success_count += 1
            
    print(f"Finished! Synchronized {success_count}/{len(episodes)} episodes.")