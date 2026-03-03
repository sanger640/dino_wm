import os
import zmq
import torch
import torch.nn.functional as F
import numpy as np
import hydra
import time
import logging
from pathlib import Path
from omegaconf import OmegaConf
from torchvision import transforms

# --- CONFIG ---
CHECKPOINT_PATH = "/home/sanger/dino_wm/outputs/model_latest.pth" 
PORT = 5556

ALL_MODEL_KEYS = [
    "encoder", "predictor", "decoder", "decoder_front", "decoder_wrist",
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
    ckpt_data = load_ckpt_payload(model_ckpt, device) if model_ckpt.exists() else {}

    def get_component(key, cfg_section=None, **kwargs):
        component = None
        if cfg_section and hasattr(train_cfg, cfg_section):
            component = hydra.utils.instantiate(getattr(train_cfg, cfg_section), **kwargs)
            component.to(device)
        
        if key in ckpt_data and component is not None:
            weights = ckpt_data[key]
            if isinstance(weights, dict): component.load_state_dict(weights)
            else: component.load_state_dict(weights.state_dict())
        return component

    instantiate_kwargs = {}
    instantiate_kwargs["encoder"] = get_component("encoder", "encoder")
    encoder_emb_dim = getattr(instantiate_kwargs["encoder"], "emb_dim", 384)

    instantiate_kwargs["proprio_encoder"] = get_component("proprio_encoder", "proprio_encoder", in_chans=4, emb_dim=train_cfg.proprio_emb_dim)
    instantiate_kwargs["action_encoder"] = get_component("action_encoder", "action_encoder", in_chans=4, emb_dim=train_cfg.action_emb_dim)

    target_class = train_cfg.model._target_
    is_dual = "dual" in target_class or "Dual" in target_class
    num_views = 2 if is_dual else 1
    concat_dim = getattr(train_cfg, "concat_dim", 0)
    
    patch_size = 16 
    patches_per_view = (getattr(train_cfg, "img_size", 224) // patch_size) ** 2
    predictor_num_patches = (num_views * patches_per_view) + (2 if concat_dim == 0 else 0)

    predictor_dim = encoder_emb_dim + (getattr(train_cfg, "action_emb_dim", 0) + getattr(train_cfg, "proprio_emb_dim", 0)) if concat_dim == 1 else encoder_emb_dim
    instantiate_kwargs["predictor"] = get_component("predictor", "predictor", dim=predictor_dim, num_patches=predictor_num_patches, num_frames=train_cfg.num_hist)
    
    if is_dual:
        instantiate_kwargs["decoder_front"] = get_component("decoder_front", "decoder", emb_dim=encoder_emb_dim)
        instantiate_kwargs["decoder_wrist"] = get_component("decoder_wrist", "decoder", emb_dim=encoder_emb_dim)
    else:
        instantiate_kwargs["decoder"] = get_component("decoder", "decoder", emb_dim=encoder_emb_dim)

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
    model.eval()
    return model

@hydra.main(version_base=None, config_path="conf/", config_name="train_dual")
def main(cfg: OmegaConf):
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
        ACTION_MEAN = torch.tensor([0.4472, 0.0025, 0.4921, 0.0202], device=device)
        ACTION_STD  = torch.tensor([0.0297, 0.0085, 0.0195, 0.1406], device=device)
        PROPRIO_MEAN, PROPRIO_STD = ACTION_MEAN, ACTION_STD

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

            orig_shape = visual_t.shape
            if visual_t.ndim == 6:
                b, t, v, c, h, w = orig_shape
                visual_t = visual_t.view(b * t * v, c, h, w)
                visual_t = inference_transform(visual_t)
                visual_t = visual_t.view(b, t, v, c, TARGET_IMG_SIZE, TARGET_IMG_SIZE)

            obs_0 = {"visual": visual_t, "proprio": proprio_t}

            with torch.no_grad():
                z_obses, _ = model.rollout(obs_0, actions_t)
                b_size = actions_t.shape[0]
                
                lyap_exp_np = None
                max_patch_idx_np = None
                
                # --- 1. DIVERGENCE METRIC (PER-PATCH LYAPUNOV) ---
                if b_size > 1:
                    z_visual = z_obses['visual'] 
                    n_hist = visual_t.shape[1]
                    
                    z_front = z_visual[:, :, 196:392, :] 
                    
                    z_orig = z_front[0:1] # Shape: (1, T, 196, 384)
                    z_noisy = z_front[1:] # Shape: (B-1, T, 196, 384)

                    # Calculate L2 distance per patch
                    patch_distances = torch.norm(z_noisy - z_orig, dim=-1) # Shape: (B-1, T, 196)

                    if patch_distances.shape[1] > n_hist:
                        d_start = patch_distances[:, n_hist] + 1e-8
                        d_end = patch_distances[:, -1] + 1e-8
                        T_span = patch_distances.shape[1] - n_hist
                        
                        # Calculate Lyapunov exponent individually for all 196 patches
                        lyap_per_patch = (1.0 / T_span) * torch.log(d_end / d_start) # Shape: (B-1, 196)
                        
                        # Extract the maximum exponent value and its corresponding patch index
                        max_lyap_vals, max_patch_indices = torch.max(lyap_per_patch, dim=-1)
                        
                        lyap_exp_np = max_lyap_vals.cpu().numpy()
                        max_patch_idx_np = max_patch_indices.cpu().numpy()
                    else:
                        lyap_exp_np = np.zeros(b_size - 1)
                        max_patch_idx_np = np.zeros(b_size - 1, dtype=int)

                # --- 2. DECODE IMAGES ---
                has_decoder = (hasattr(model, "decoder_front") and model.decoder_front is not None)
                
                if has_decoder:
                    decoded_obs, _ = model.decode_obs(z_obses)
                    pred_visual = decoded_obs['visual'] 

                    pred_visual_np = pred_visual.cpu().numpy()
                    pred_visual_np = (pred_visual_np + 1.0) / 2.0
                    result_images = np.clip(pred_visual_np * 255, 0, 255).astype(np.uint8)
                else:
                    result_images = None
            
            socket.send_pyobj({
                'states': result_images, 
                'lyapunov': lyap_exp_np,
                'max_patch_idx': max_patch_idx_np,  # Include the spatial coordinate index
                'inference_time': time.time() - start_time
            })

        except Exception as e:
            print(f"❌ Error: {e}")
            try:
                socket.send_pyobj({'error': str(e)})
            except: pass

if __name__ == "__main__":
    main()