"""
Computes per-patch FTLE statistics from known-safe trajectories.
Used to generate patch_stats.npz for the ftle_calibrated robustness mode.

Identifies background/shadow patches (consistently high FTLE even in safe situations)
so they can be normalized out at inference, reducing false positives.

Usage:
    # First start server_ablation.py in any ftle mode:
    #   python server_ablation.py mode=ftle
    #
    # Then run this script:
    python calibrate_patches.py \
        --lmdb /path/to/jenga_single.lmdb \
        --labels /path/to/labels.json \
        --output outputs/patch_stats.npz \
        --n-perturb 50 --noise-std 0.05

The labels.json should map episode names to 0 (safe) or 1 (failure):
    {"episode_001": 0, "episode_002": 1, ...}

The output patch_stats.npz contains:
    p95: (196,) - 95th percentile FTLE per patch across safe trajectories
    mean: (196,) - mean FTLE per patch
    std: (196,) - std FTLE per patch
"""

import argparse
import json
import os
import zmq
import lmdb
import pickle
import numpy as np
import cv2
from pathlib import Path


NUM_HIST = 3
NUM_PRED = 8
DINO_PORT = 5556
NUM_PATCHES = 196


def process_image(img_bytes):
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.transpose(img, (2, 0, 1))


def center_crop_resize(img_hwc, size=224):
    h, w = img_hwc.shape[:2]
    scale = size / min(h, w)
    img_hwc = cv2.resize(img_hwc, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = img_hwc.shape[:2]
    y0 = (sh - size) // 2
    x0 = (sw - size) // 2
    return img_hwc[y0:y0 + size, x0:x0 + size]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lmdb",      required=True)
    p.add_argument("--labels",    required=True, help="JSON file: {episode_name: 0|1}")
    p.add_argument("--output",    default="outputs/patch_stats.npz")
    p.add_argument("--server-ip", default="localhost")
    p.add_argument("--port",      type=int, default=DINO_PORT)
    p.add_argument("--n-perturb", type=int, default=50)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--max-steps-per-ep", type=int, default=None,
                   help="Limit steps per episode for speed (default: all)")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.labels) as f:
        labels = json.load(f)
    safe_episodes = {k for k, v in labels.items() if v == 0}
    print(f"Found {len(safe_episodes)} safe episodes for calibration")

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{args.server_ip}:{args.port}")
    print(f"Connected to server at {args.server_ip}:{args.port}")

    all_patch_ftles = []  # accumulate (196,) vectors

    env = lmdb.open(args.lmdb, readonly=True, lock=False)
    with env.begin() as txn:
        metadata = pickle.loads(txn.get(b"__metadata__"))
        ep_names = list(metadata["episodes"].keys())

        for ep_idx, ep_name in enumerate(ep_names):
            if ep_name not in safe_episodes:
                continue

            ep_meta = metadata["episodes"][ep_name]
            c2_keys = ep_meta["keys"]["cam2"]
            act_all  = pickle.loads(txn.get(f"{ep_name}_actions".encode()))
            prop_all = pickle.loads(txn.get(f"{ep_name}_proprio".encode()))

            actual_len = min(len(c2_keys), len(act_all), len(prop_all))
            max_start = actual_len - (NUM_HIST + NUM_PRED) - 1
            if max_start <= 0:
                continue

            steps = range(0, max_start + 1, NUM_PRED)
            if args.max_steps_per_ep is not None:
                steps = list(steps)[:args.max_steps_per_ep]

            print(f"  [{ep_idx+1}] {ep_name}: {len(list(steps))} windows")

            for start in steps:
                vis_frames, skip = [], False
                for t_off in range(NUM_HIST + NUM_PRED):
                    idx = start + t_off
                    if idx >= len(c2_keys):
                        skip = True
                        break
                    img_bytes = txn.get(c2_keys[idx].encode())
                    if img_bytes is None:
                        skip = True
                        break
                    vis_frames.append(process_image(img_bytes))

                if skip:
                    break

                # Build batch: original + N perturbed
                clean_actions = act_all[start:start + NUM_HIST + NUM_PRED]
                batch_size = args.n_perturb
                batch_actions = np.tile(clean_actions[np.newaxis], (batch_size, 1, 1)).astype(np.float32)
                # Perturb all but the first
                T, D = clean_actions.shape
                noise = np.random.normal(0, args.noise_std, (batch_size - 1, T, 3))
                batch_actions[1:, :, :3] += noise

                vis_hist = np.tile(
                    np.stack(vis_frames[:NUM_HIST])[np.newaxis],
                    (batch_size, 1, 1, 1, 1)
                )
                prop_hist = np.tile(
                    prop_all[start:start + NUM_HIST][np.newaxis],
                    (batch_size, 1, 1)
                )

                socket.send_pyobj({
                    "visual":  vis_hist.astype(np.uint8),
                    "proprio": prop_hist.astype(np.float32),
                    "actions": batch_actions,
                })
                resp = socket.recv_pyobj()

                if "error" in resp:
                    print(f"    Server error: {resp['error']}")
                    continue

                patch_ftles = resp.get("worst_patch_ftles")
                if patch_ftles is not None and patch_ftles.shape == (NUM_PATCHES,):
                    all_patch_ftles.append(patch_ftles)

    env.close()

    if len(all_patch_ftles) == 0:
        print("ERROR: No patch FTLE data collected. Make sure labels.json has safe episodes and server is running.")
        return

    patch_array = np.stack(all_patch_ftles)  # (N_steps, 196)
    print(f"\nCollected {len(all_patch_ftles)} windows from safe episodes")

    p95  = np.percentile(patch_array, 95, axis=0)   # (196,)
    mean = np.mean(patch_array, axis=0)
    std  = np.std(patch_array, axis=0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, p95=p95, mean=mean, std=std)
    print(f"Saved patch stats to {out}")
    print(f"  P95 range: [{p95.min():.4f}, {p95.max():.4f}]")
    print(f"  Top-5 noisiest patches (by P95): {np.argsort(p95)[-5:][::-1].tolist()}")


if __name__ == "__main__":
    main()
