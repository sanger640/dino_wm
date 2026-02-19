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

# --- CONFIG ---
# Path to your trained checkpoint
CHECKPOINT_PATH = "checkpoints/2026-02-09/model_latest.pth" 
PORT = 5556

# --- NORMALIZATION STATS (From Dataset) ---
# Must match training statistics exactly
ACTION_MEAN = torch.tensor([0.4472, 0.0025, 0.4921, 0.0202], device='cuda')
ACTION_STD  = torch.tensor([0.0297, 0.0085, 0.0195, 0.1406], device='cuda')

# Assuming Proprioception uses same stats (standard for this codebase)
PROPRIO_MEAN = ACTION_MEAN
PROPRIO_STD  = ACTION_STD
# ------------------------------------------

# Keys found in the checkpoint dictionary
ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",         # Single-view key
    "decoder_front",   # Dual-view key
    "decoder_wrist",   # Dual-view key
    "proprio_encoder",
    "action_encoder",
]

def load_ckpt_payload(snapshot_path, device):
    """Load model payload from checkpoint file (does not instantiate objects)."""
    print(f"Loading payload from: {snapshot_path}")
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            result[k] = v
    result["epoch"] = payload.get("epoch", 0)
    return result

def load_model(model_ckpt, train_cfg, device):
    """Reconstruct and load the VWorldModel (Dual or Single)."""
    model_ckpt = Path(model_ckpt)
    
    # 1. Load Checkpoint Data (Weights/Payload only)
    ckpt_data = {}
    if model_ckpt.exists():
        ckpt_data = load_ckpt_payload(model_ckpt, device)
    else:
        print(f"Warning: Checkpoint not found at {model_ckpt}. Initializing fresh model.")

    # 2. Helper to load components: Instantiate -> Load State
    def get_component(key, cfg_section=None, **kwargs):
        component = None
        
        # A. Instantiate from Config (Crucial: Always create fresh object)
        if cfg_section and hasattr(train_cfg, cfg_section):
            print(f"Instantiating {key} from config ({cfg_section})...")
            component = hydra.utils.instantiate(getattr(train_cfg, cfg_section), **kwargs)
            component.to(device)
        
        # B. Load Weights if available in checkpoint
        if key in ckpt_data and component is not None:
            print(f"Loading weights for {key}...")
            weights = ckpt_data[key]
            try:
                if isinstance(weights, dict):
                    component.load_state_dict(weights)
                elif isinstance(weights, torch.nn.Module):
                    component.load_state_dict(weights.state_dict())
            except Exception as e:
                print(f"Warning: Failed to load weights for {key}: {e}")
                
        elif key in ckpt_data and component is None:
             print(f"Warning: {key} found in checkpoint but not in config. Ignoring.")
             
        return component

    # 3. Prepare Constructor Arguments
    instantiate_kwargs = {}

    instantiate_kwargs["encoder"] = get_component("encoder", "encoder")
    if hasattr(instantiate_kwargs["encoder"], "emb_dim"):
        encoder_emb_dim = instantiate_kwargs["encoder"].emb_dim
    else:
        encoder_emb_dim = 384

    instantiate_kwargs["proprio_encoder"] = get_component(
        "proprio_encoder", "proprio_encoder", 
        in_chans=4, emb_dim=train_cfg.proprio_emb_dim
    )
    
    instantiate_kwargs["action_encoder"] = get_component(
        "action_encoder", "action_encoder", 
        in_chans=4, 
        emb_dim=train_cfg.action_emb_dim
    )

    # --- CALCULATE PREDICTOR ARGS ---
    target_class = train_cfg.model._target_
    is_dual = "dual" in target_class or "Dual" in target_class
    num_views = 2 if is_dual else 1
    img_size = getattr(train_cfg, "img_size", 224)
    concat_dim = getattr(train_cfg, "concat_dim", 0)
    num_hist = train_cfg.num_hist
    patch_size = getattr(instantiate_kwargs["encoder"], "patch_size", 14)

    patches_per_view = (img_size // patch_size) ** 2
    total_visual_patches = num_views * patches_per_view
    extra_tokens = 2 if concat_dim == 0 else 0
    predictor_num_patches = total_visual_patches + extra_tokens

    if concat_dim == 1:
        proprio_dim = getattr(train_cfg, "proprio_emb_dim", 0) * getattr(train_cfg, "num_proprio_repeat", 1)
        action_dim = getattr(train_cfg, "action_emb_dim", 0) * getattr(train_cfg, "num_action_repeat", 1)
        predictor_dim = encoder_emb_dim + action_dim + proprio_dim
    else:
        predictor_dim = encoder_emb_dim

    print(f"Initializing Predictor with: dim={predictor_dim}, num_patches={predictor_num_patches}, num_frames={num_hist}")

    instantiate_kwargs["predictor"] = get_component(
        "predictor", "predictor",
        dim=predictor_dim,
        num_patches=predictor_num_patches,
        num_frames=num_hist
    )

    if is_dual:
        instantiate_kwargs["decoder_front"] = get_component("decoder_front", "decoder", emb_dim=encoder_emb_dim)
        instantiate_kwargs["decoder_wrist"] = get_component("decoder_wrist", "decoder", emb_dim=encoder_emb_dim)
    else:
        instantiate_kwargs["decoder"] = get_component("decoder", "decoder", emb_dim=encoder_emb_dim)

    instantiate_kwargs["proprio_dim"] = getattr(train_cfg, "proprio_emb_dim", 0)
    instantiate_kwargs["action_dim"] = getattr(train_cfg, "action_emb_dim", 0)
    instantiate_kwargs["concat_dim"] = getattr(train_cfg, "concat_dim", 0)
    instantiate_kwargs["num_action_repeat"] = getattr(train_cfg, "num_action_repeat", 1)
    instantiate_kwargs["num_proprio_repeat"] = getattr(train_cfg, "num_proprio_repeat", 1)
    instantiate_kwargs["image_size"] = getattr(train_cfg, "img_size", 224)
    instantiate_kwargs["num_hist"] = train_cfg.num_hist
    instantiate_kwargs["num_pred"] = train_cfg.num_pred

    model = hydra.utils.instantiate(train_cfg.model, **instantiate_kwargs)
    model.to(device)
    model.eval()
    return model

@hydra.main(version_base=None, config_path="outputs/2026-02-09/19-18-51/.hydra/", config_name="config")
def main(cfg: OmegaConf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.exists():
        base = Path(cfg.ckpt_base_path)
        ckpt_path = base / "outputs" / "2026-02-09/19-18-51/checkpoints/model_latest.pth"
    
    print(f"Target Checkpoint: {ckpt_path}")

    model = load_model(ckpt_path, cfg, device)
    
    TARGET_IMG_SIZE = getattr(cfg, "img_size", 224)
    EXPECTED_ACTION_DIM = getattr(cfg, "action_emb_dim", 20)

    print(f"✅ Model Loaded.")
    print(f"   Image Size: {TARGET_IMG_SIZE}x{TARGET_IMG_SIZE}")
    print(f"   Action Dim: {EXPECTED_ACTION_DIM}")
    print(f"   Norm Stats Applied: Mean={ACTION_MEAN.cpu().numpy()}, Std={ACTION_STD.cpu().numpy()}")
    print("Starting ZMQ Server...")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORT}")
    print(f"Listening on tcp://*:{PORT}...")

    while True:
        try:
            message = socket.recv_pyobj()
            start_time = time.time()

            # --- Preprocessing ---
            def to_tensor(arr):
                t = torch.from_numpy(arr).float().to(device)
                if arr.dtype == np.uint8:
                    # Normalize Images to [-1, 1]
                    t = (t / 127.5) - 1.0 
                return t

            visual_t = to_tensor(message['visual'])
            proprio_t = to_tensor(message['proprio'])
            actions_t = to_tensor(message['actions'])

            # Ensure Batch Dims
            if visual_t.ndim == 4: visual_t = visual_t.unsqueeze(0)
            if proprio_t.ndim == 2: proprio_t = proprio_t.unsqueeze(0)
            if actions_t.ndim == 2: actions_t = actions_t.unsqueeze(0)
            print("ACTIONNNNNSS")
            print(actions_t)
            # --- 1. Normalize Actions & Proprioception ---
            # Input: (B, T, 4) -> Output: (B, T, 4) Normalized
            # # Note: We normalize BEFORE padding because the mean/std only apply to the active dimensions (4)
            # proprio_t = (proprio_t - PROPRIO_MEAN) / PROPRIO_STD
            # actions_t = (actions_t - ACTION_MEAN) / ACTION_STD

            # --- 2. Resize Images ---
            if visual_t.shape[-1] != TARGET_IMG_SIZE or visual_t.shape[-2] != TARGET_IMG_SIZE:
                orig_shape = visual_t.shape
                if visual_t.ndim == 6: # Dual view: (B, T, V, C, H, W)
                    b, t, v, c, h, w = orig_shape
                    visual_t = visual_t.view(b * t * v, c, h, w)
                    visual_t = F.interpolate(visual_t, size=(TARGET_IMG_SIZE, TARGET_IMG_SIZE), mode='bilinear', align_corners=False)
                    visual_t = visual_t.view(b, t, v, c, TARGET_IMG_SIZE, TARGET_IMG_SIZE)
                elif visual_t.ndim == 5: # Single view: (B, T, C, H, W)
                    b, t, c, h, w = orig_shape
                    visual_t = visual_t.view(b * t, c, h, w)
                    visual_t = F.interpolate(visual_t, size=(TARGET_IMG_SIZE, TARGET_IMG_SIZE), mode='bilinear', align_corners=False)
                    visual_t = visual_t.view(b, t, c, TARGET_IMG_SIZE, TARGET_IMG_SIZE)

            # --- 3. Pad Actions (4 -> 20) ---
            # curr_act_dim = actions_t.shape[-1]
            # if curr_act_dim < EXPECTED_ACTION_DIM:
            #     diff = EXPECTED_ACTION_DIM - curr_act_dim
            #     actions_t = F.pad(actions_t, (0, diff), "constant", 0)
            
            # --- 4. Inference ---
            obs_0 = {"visual": visual_t, "proprio": proprio_t}

            with torch.no_grad():
                z_obses, _ = model.rollout(obs_0, actions_t)
                
                has_decoder = (hasattr(model, "decoder") and model.decoder is not None) or \
                              (hasattr(model, "decoder_front") and model.decoder_front is not None)

                if has_decoder:
                    decoded_obs, _ = model.decode_obs(z_obses)
                    
                    full_visual_pred = decoded_obs['visual']
                    n_hist = visual_t.shape[1]
                    
                    if full_visual_pred.shape[1] > n_hist:
                        future_visual_pred = full_visual_pred[:, n_hist:]
                    else:
                        future_visual_pred = full_visual_pred 

                    # Denormalize Images: [-1, 1] -> [0, 255]
                    future_visual_np = future_visual_pred.cpu().numpy()
                    future_visual_np = (future_visual_np + 1.0) / 2.0
                    result_data = np.clip(future_visual_np * 255, 0, 255).astype(np.uint8)
                else:
                    result_data = z_obses['visual'].cpu().numpy()

            socket.send_pyobj({
                'states': result_data,
                'inference_time': time.time() - start_time
            })

        except Exception as e:
            print(f"❌ Error: {e}")
            try:
                socket.send_pyobj({'error': str(e)})
            except: pass

if __name__ == "__main__":
    main()