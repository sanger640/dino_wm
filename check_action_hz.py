import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("/home/ali313/scratch/jenga_mujoco_noise/episodes")

def analyze_action_frequency():
    episodes = sorted([p for p in DATA_DIR.iterdir() if p.is_dir()])
    print(f"Scanning {len(episodes)} episodes for action timestamps...\n")
    
    all_time_diffs = []
    episode_hz_means = []

    for ep_dir in tqdm(episodes, desc="Processing Episodes"):
        json_files = list(ep_dir.glob("trajectory_*.json"))
        if not json_files:
            continue
        
        with open(json_files[0], 'r') as f:
            traj_data = json.load(f)
            
        waypoints = traj_data.get('waypoints', [])
        if len(waypoints) < 2:
            continue
            
        # Extract timestamps
        timestamps = np.array([wp['timestamp'] for wp in waypoints])
        
        # Calculate the time difference between consecutive waypoints
        diffs = np.diff(timestamps)
        
        # Filter out zero diffs just in case of a duplicate log
        diffs = diffs[diffs > 0]
        
        if len(diffs) > 0:
            all_time_diffs.extend(diffs)
            
            # Average Hz for this specific episode
            ep_mean_hz = np.mean(1.0 / diffs)
            episode_hz_means.append(ep_mean_hz)

    # Calculate Global Statistics
    all_time_diffs = np.array(all_time_diffs)
    all_hz = 1.0 / all_time_diffs
    
    print("\n" + "="*40)
    print("🎬 ACTION FREQUENCY REPORT (Hz)")
    print("="*40)
    print(f"Total Waypoint Transitions Analyzed : {len(all_hz):,}")
    print(f"Mean Frequency                      : {np.mean(all_hz):.2f} Hz")
    print(f"Median Frequency                    : {np.median(all_hz):.2f} Hz")
    print(f"Standard Deviation                  : ±{np.std(all_hz):.2f} Hz")
    print("-" * 40)
    print(f"Fastest Spike (Max Hz)              : {np.max(all_hz):.2f} Hz")
    print(f"Slowest Drop (Min Hz)               : {np.min(all_hz):.2f} Hz")
    print("="*40)
    
    # Check for extreme lag spikes
    lag_spikes = np.sum(all_hz < 5.0) # Number of times the loop dropped below 5Hz
    if lag_spikes > 0:
        print(f"⚠️ Warning: Detected {lag_spikes} transitions that dropped below 5 Hz.")
    else:
        print("✅ No severe lag spikes detected.")

if __name__ == "__main__":
    analyze_action_frequency()