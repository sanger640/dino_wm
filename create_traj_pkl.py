import json
import numpy as np
from pathlib import Path
import pickle
import torch
import argparse

def create_metadata_pickle(
    data_path: str, 
    action_freq: int = 10, 
    img_freq: int = 30, 
    cam_prefixes: list = None,
    img_ext: str = ".png"
):
    if cam_prefixes is None:
        cam_prefixes = ["cam1", "cam2"]
        
    data_path = Path(data_path)
    freq_ratio = int(img_freq / action_freq)
    
    episode_dirs = sorted(
        [p for p in (data_path / "episodes").iterdir() if p.is_dir()],
        key=lambda x: int(x.name) if x.name.isdigit() else x.name
    )
    
    episodes_data = []
    seq_lengths = []
    all_actions = []
    all_proprios = []

    print(f"Scanning {len(episode_dirs)} episodes to build metadata pickle...")

    for ep_dir in episode_dirs:
        json_files = list(ep_dir.glob("trajectory_*.json"))
        if not json_files: 
            continue
        
        with open(json_files[0], 'r') as f:
            traj_data = json.load(f)
        
        waypoints = traj_data['waypoints']
        img_dir = ep_dir / "rgb_frames"
        
        cam1_files = list(img_dir.glob(f"{cam_prefixes[0]}_*{img_ext}"))
        if not cam1_files: 
            continue

        def get_ts(p):
            try: return int(p.stem.split('_')[-1])
            except ValueError: return 0

        cam1_files.sort(key=get_ts)
        valid_timestamps = [get_ts(f) for f in cam1_files]

        n_images = len(valid_timestamps)
        n_actions = len(waypoints)
        max_valid_actions = n_images // freq_ratio
        final_len = min(n_actions, max_valid_actions)
        
        if final_len < 2: 
            continue 

        seq_lengths.append(final_len)
        
        # Keep as numpy arrays in the pickle for maximum compatibility
        cmd_pos = np.array([wp['position'] for wp in waypoints], dtype=np.float32)[:final_len]
        cmd_grip = np.array([[float(wp['gripper'])] for wp in waypoints], dtype=np.float32)[:final_len]
        act_vec = np.concatenate([cmd_pos, cmd_grip], axis=-1)
        
        proc_pos = np.array([wp['proc_pos'] for wp in waypoints], dtype=np.float32)[:final_len]
        proc_grip = np.array([[float(wp['proc_gripper'])] for wp in waypoints], dtype=np.float32)[:final_len]
        proc_vec = np.concatenate([proc_pos, proc_grip], axis=-1)
        
        episodes_data.append({
            "ep_name": ep_dir.name, 
            "act_vec": act_vec,
            "proc_vec": proc_vec,
            "timestamps": valid_timestamps, 
        })
        
        all_actions.append(torch.from_numpy(act_vec))
        all_proprios.append(torch.from_numpy(proc_vec))

    # Calculate global normalization stats
    print("Calculating normalization statistics...")
    if len(all_actions) > 0:
        all_actions_cat = torch.cat(all_actions, dim=0)
        action_mean = all_actions_cat.mean(dim=0).numpy()
        action_std = (all_actions_cat.std(dim=0) + 1e-6).numpy()
        
        all_proprios_cat = torch.cat(all_proprios, dim=0)
        proprio_mean = all_proprios_cat.mean(dim=0).numpy()
        proprio_std = (all_proprios_cat.std(dim=0) + 1e-6).numpy()
    else:
        action_mean, proprio_mean = np.zeros(4), np.zeros(4)
        action_std, proprio_std = np.ones(4), np.ones(4)

    # Package everything into a single dictionary
    metadata = {
        "episodes_data": episodes_data,
        "seq_lengths": seq_lengths,
        "stats": {
            "action_mean": action_mean,
            "action_std": action_std,
            "proprio_mean": proprio_mean,
            "proprio_std": proprio_std
        }
    }

    out_file = data_path / "jenga_metadata.pkl"
    with open(out_file, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"Successfully saved metadata to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to your dataset root")
    args = parser.parse_args()
    create_metadata_pickle(args.data_path)