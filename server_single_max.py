import os
import zmq
import torch
import numpy as np
import hydra
import time
import logging
from pathlib import Path
from omegaconf import OmegaConf
from torchvision import transforms

# --- CONFIG ---
CHECKPOINT_PATH = "/home/sanger/wksp/dino_wm/outputs/model_latest_single.pth" 
PORT = 5556

ALL_MODEL_KEYS = [
    "encoder", "predictor", "decoder", 
    "proprio_encoder", "action_encoder",
]

def load_ckpt_payload(snapshot_path, device):
    print(f"Loading payload from: {snapshot_path}")
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    
    result = {k: v for k, v in payload.items() if k in ALL_MODEL_KEYS}
    result["epoch"] = payload.get("epoch", 0)
    return result

def load_model(model_ckpt, train_cfg, device):
    model_ckpt = Path(model_ckpt)
    
    def get_component(cfg_section=None, **kwargs):
        if cfg_section and hasattr(train_cfg, cfg_section):
            return hydra.utils.instantiate(getattr(train_cfg, cfg_section), **kwargs)
        return None

    instantiate_kwargs = {}
    instantiate_kwargs["encoder"] = get_component("encoder")
    encoder_emb_dim = getattr(instantiate_kwargs["encoder"], "emb_dim", 384)

    instantiate_kwargs["proprio_encoder"] = get_component("proprio_encoder", in_chans=4, emb_dim=train_cfg.proprio_emb_dim)
    instantiate_kwargs["action_encoder"] = get_component("action_encoder", in_chans=4, emb_dim=train_cfg.action_emb_dim)

    target_class = train_cfg.model._target_
    concat_dim = getattr(train_cfg, "concat_dim", 0)
    num_views = 1 
    
    patch_size = 16 
    patches_per_view = (getattr(train_cfg, "img_size", 224) // patch_size) ** 2
    predictor_num_patches = (num_views * patches_per_view) + (2 if concat_dim == 0 else 0)
    predictor_dim = encoder_emb_dim + (getattr(train_cfg, "action_emb_dim", 0) + getattr(train_cfg, "proprio_emb_dim", 0)) if concat_dim == 1 else encoder_emb_dim
    
    instantiate_kwargs["predictor"] = get_component("predictor", dim=predictor_dim, num_patches=predictor_num_patches, num_frames=train_cfg.num_hist)
    instantiate_kwargs["decoder"] = get_component("decoder", emb_dim=encoder_emb_dim)

    instantiate_kwargs.update({
        "proprio_dim": getattr(train_cfg, "proprio_emb_dim", 0),
        "action_dim": getattr(train_cfg, "action_emb_dim", 0),
        "concat_dim": concat_dim,
        "num_action_repeat": getattr(train_cfg, "num_action_repeat", 1),
        "num_proprio_repeat": getattr(train_cfg, "num_proprio_repeat", 1),
        "image_size": getattr(train_cfg, "img_size", 224),
        "num_hist": train_cfg.num_hist, "num_pred": train_cfg.num_pred
    })

    model = hydra.utils.instantiate(train_cfg.model, **instantiate_kwargs)
    model.to(device)

    if model_ckpt.exists():
        print(f"📂 Loading payload from: {model_ckpt}")
        payload = torch.load(model_ckpt, map_location=device, weights_only=False)
        
        component_map = {
            "encoder": model.encoder, 
            "proprio_encoder": model.proprio_encoder,
            "action_encoder": model.action_encoder, 
            "predictor": model.predictor,
            "decoder": model.decoder
        }

        for key, target_module in component_map.items():
            if key in payload and target_module is not None:
                saved_obj = payload[key]
                state_dict = saved_obj if isinstance(saved_obj, dict) else saved_obj.state_dict()
                target_module.load_state_dict(state_dict)
                print(f"  ✅ Loaded {key}")
            elif target_module is not None:
                print(f"  ⚠️ Warning: '{key}' not found in checkpoint!")

    else:
        print(f"🚨 FATAL: Checkpoint file not found at {model_ckpt}")

    model.eval()
    return model

def le_cos(z_noisy, z_orig, n_hist, T_span):
    cos_sim = torch.nn.functional.cosine_similarity(z_noisy, z_orig, dim=-1)
    patch_distances = 1 - cos_sim
    d_start = patch_distances[:, n_hist] + 1e-4
    d_end = patch_distances[:, -1] + 1e-4
    lyap_per_patch = (1.0 / T_span) * torch.log(d_end / d_start) 

    # Absolute Noise Floor
    significant_drift_mask = d_end > 1e-3 
    lyap_per_patch[~significant_drift_mask] = -float('inf')
    lyap_per_patch[:, :28] = -float('inf')

    max_lyap_vals, max_patch_indices = torch.max(lyap_per_patch, dim=-1)
    lyap_exp_np = max_lyap_vals.cpu().numpy()
    max_patch_idx_np = max_patch_indices.cpu().numpy()
    return lyap_exp_np, max_patch_idx_np

@hydra.main(version_base=None, config_path="conf/", config_name="train") 
def main(cfg: OmegaConf):
    print("=== HYDRA RUNTIME CONFIG ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.exists():
        ckpt_path = Path(cfg.ckpt_base_path) / "outputs" / "model_latest.pth"

    model = load_model(ckpt_path, cfg, device)
    TARGET_IMG_SIZE = getattr(cfg, "img_size", 224)

    stats_path = Path(cfg.ckpt_base_path) / "outputs" / "dataset_stats.pt"
    if stats_path.exists():
        print(f"Loading dynamic stats from {stats_path}")
        stats = torch.load(stats_path, map_location=device)
        ACTION_MEAN, ACTION_STD = stats["action_mean"], stats["action_std"]
        PROPRIO_MEAN, PROPRIO_STD = stats["proprio_mean"], stats["proprio_std"]
    else:
        print("Warning: dataset_stats.pt not found. Using hardcoded fallback stats.")
        ACTION_MEAN = torch.tensor([0.45678952, 0.00051019, 0.50954217, 0.21926114], device=device)
        ACTION_STD  = torch.tensor([0.03182372, 0.01151787, 0.03419121, 0.41397065], device=device)
        PROPRIO_MEAN = torch.tensor([0.4564166, 0.00056233, 0.50817657, 0.21921302], device=device)
        PROPRIO_STD = torch.tensor([0.03217997, 0.01056713, 0.0327194,  0.4139551 ], device=device)

    inference_transform = transforms.Compose([
        transforms.Resize(TARGET_IMG_SIZE),
        transforms.CenterCrop(TARGET_IMG_SIZE),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORT}")
    print(f"Live Rollout Server listening on tcp://*:{PORT}...")

    while True:
        try:
            message = socket.recv_pyobj()
            start_time = time.time()

            def to_tensor(arr):
                t = torch.from_numpy(arr).float().to(device)
                if arr.dtype == np.uint8:
                    t = t / 255.0 
                return t
            
            visual_t = to_tensor(message['visual'])
            proprio_t = to_tensor(message['proprio'])
            actions_t = to_tensor(message['actions'])
            
            if visual_t.ndim == 4: visual_t = visual_t.unsqueeze(0)
            if proprio_t.ndim == 2: proprio_t = proprio_t.unsqueeze(0)
            if actions_t.ndim == 2: actions_t = actions_t.unsqueeze(0)

            proprio_t = (proprio_t - PROPRIO_MEAN) / PROPRIO_STD
            actions_t = (actions_t - ACTION_MEAN) / ACTION_STD

            b, t, c, h, w = visual_t.shape
            visual_t = visual_t.view(b * t, c, h, w)
            visual_t = inference_transform(visual_t)
            visual_t = visual_t.view(b, t, c, TARGET_IMG_SIZE, TARGET_IMG_SIZE)

            obs_0 = {"visual": visual_t, "proprio": proprio_t}

            with torch.no_grad():
                z_obses, _ = model.rollout(obs_0, actions_t)
                
                b_size = actions_t.shape[0]
                n_hist = visual_t.shape[1]
                
                abs_max_le = 0.0
                abs_max_patch_idx = 0
                lyap_exp_np = np.zeros(max(1, b_size - 1))
                worst_traj_idx = 0
                
                if b_size > 1 and z_obses['visual'].shape[1] > n_hist:
                    z_visual = z_obses['visual'] 
                    z_orig = z_visual[0:1] # (1, T, 196, 384)
                    z_noisy = z_visual[1:] # (B-1, T, 196, 384)
                    T_span = z_visual.shape[1] - n_hist
                    
                    lyap_exp_np, max_patch_idx_np = le_cos(z_noisy, z_orig, n_hist, T_span)
                    
                    abs_max_le = float(np.max(lyap_exp_np))
                    worst_traj_idx = int(np.argmax(lyap_exp_np))
                    abs_max_patch_idx = int(max_patch_idx_np[worst_traj_idx])

                # --- DECODE ORIGINAL (0) AND WORST TRAJ (worst_traj_idx + 1) ---
                # Opt-in: the client only needs pixels for visualisation, but decoding runs
                # the VQVAE over 2 x T frames and ships ~3.6MB back on every request. Clients
                # that just want the score send return_states=False (or omit it).
                want_states = bool(message.get("return_states", False))
                if want_states and hasattr(model, "decoder") and model.decoder is not None:
                    if b_size > 1:
                        # Extract the original and the specific trajectory that failed the hardest
                        indices_to_decode = [0, worst_traj_idx + 1]
                        z_obses_subset = {k: v[indices_to_decode] for k, v in z_obses.items()}
                    else:
                        z_obses_subset = {k: v[0:1] for k, v in z_obses.items()}
                        
                    decoded_obs, _ = model.decode_obs(z_obses_subset)
                    pred_visual_np = (decoded_obs['visual'].cpu().numpy() + 1.0) / 2.0
                    # Shape will be (2, T, C, H, W)
                    decoded_images = np.clip(pred_visual_np * 255, 0, 255).astype(np.uint8) 
                else:
                    decoded_images = None

            # SEND CONDENSED RESPONSE
            socket.send_pyobj({
                'states': decoded_images, 
                'max_lyapunov': abs_max_le,  
                'max_patch_idx': abs_max_patch_idx,
                'all_lyapunovs': lyap_exp_np, # Sent back so client can count triggers
                'inference_time': time.time() - start_time
            })

        except Exception as e:
            print(f"❌ Error: {e}")
            try:
                socket.send_pyobj({'error': str(e)})
            except: pass

if __name__ == "__main__":
    main()