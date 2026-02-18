import os
import zmq
import torch
import numpy as np
import hydra
import time
import logging
from pathlib import Path
from omegaconf import OmegaConf

# --- CONFIG ---
# Path to your trained checkpoint
CHECKPOINT_PATH = "checkpoints/model_latest.pth" 
PORT = 5556
# --------------

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

def load_ckpt(snapshot_path, device):
    """Load model weights from checkpoint file."""
    print(f"Loading weights from: {snapshot_path}")
    with snapshot_path.open("rb") as f:
        # weights_only=False is required for PyTorch 2.6+ with this codebase
        payload = torch.load(f, map_location=device, weights_only=False)
    
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result

def load_model(model_ckpt, train_cfg, device):
    """Reconstruct and load the VWorldModel (Dual or Single)."""
    model_ckpt = Path(model_ckpt)
    
    # 1. Load Checkpoint Data
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
    else:
        print(f"Warning: Checkpoint not found at {model_ckpt}. Initializing fresh model.")
        result = {}

    # 2. Helper to load components
    def get_component(key, cfg_section=None, **kwargs):
        # Try loading from checkpoint first
        if key in result:
            return result[key]
        # Fallback: Instantiate from config
        if cfg_section and hasattr(train_cfg, cfg_section):
            print(f"Instantiating {key} from config ({cfg_section})...")
            return hydra.utils.instantiate(getattr(train_cfg, cfg_section), **kwargs)
        return None

    # 3. Prepare Constructor Arguments
    instantiate_kwargs = {}

    # Load Encoder first (needed for emb_dim)
    instantiate_kwargs["encoder"] = get_component("encoder", "encoder")
    
    # Get embedding dimension for decoder instantiation
    emb_dim = getattr(instantiate_kwargs["encoder"], "emb_dim", 384)

    # Load other standard components
    instantiate_kwargs["proprio_encoder"] = get_component(
        "proprio_encoder", "proprio_encoder", 
        in_chans=train_cfg.proprio_emb_dim, emb_dim=train_cfg.proprio_emb_dim
    )
    instantiate_kwargs["action_encoder"] = get_component(
        "action_encoder", "action_encoder", 
        in_chans=train_cfg.action_emb_dim, emb_dim=train_cfg.action_emb_dim
    )
    instantiate_kwargs["predictor"] = get_component("predictor", "predictor")

    # 4. Handle Single vs Dual View Logic
    target_class = train_cfg.model._target_
    is_dual = "dual" in target_class or "Dual" in target_class
    
    if is_dual:
        print(f"--> Detected Dual-View Model: {target_class}")
        # In train_dual.py, these keys are stored as 'decoder_front'/'decoder_wrist'
        # but configured using the 'decoder' config section.
        instantiate_kwargs["decoder_front"] = get_component("decoder_front", "decoder", emb_dim=emb_dim)
        instantiate_kwargs["decoder_wrist"] = get_component("decoder_wrist", "decoder", emb_dim=emb_dim)
    else:
        print(f"--> Detected Single-View Model: {target_class}")
        instantiate_kwargs["decoder"] = get_component("decoder", "decoder", emb_dim=emb_dim)

    # 5. Extract Dimensions & Params from Config
    instantiate_kwargs["proprio_dim"] = getattr(train_cfg, "proprio_emb_dim", 0)
    instantiate_kwargs["action_dim"] = getattr(train_cfg, "action_emb_dim", 0)
    instantiate_kwargs["concat_dim"] = getattr(train_cfg, "concat_dim", 0)
    instantiate_kwargs["num_action_repeat"] = getattr(train_cfg, "num_action_repeat", 1)
    instantiate_kwargs["num_proprio_repeat"] = getattr(train_cfg, "num_proprio_repeat", 1)
    instantiate_kwargs["image_size"] = getattr(train_cfg, "img_size", 224)
    instantiate_kwargs["num_hist"] = train_cfg.num_hist
    instantiate_kwargs["num_pred"] = train_cfg.num_pred

    # 6. Instantiate the Full Model
    print("Instantiating VWorldModel...")
    model = hydra.utils.instantiate(
        train_cfg.model,
        **instantiate_kwargs
    )
    
    model.to(device)
    model.eval()
    return model

@hydra.main(version_base=None, config_path="conf", config_name="train_dual")
def main(cfg: OmegaConf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine Checkpoint Path (Priority: Constant -> Hydra Config -> Default)
    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.exists():
        # Fallback to location defined in hydra config if constant is invalid
        base = Path(cfg.ckpt_base_path)
        ckpt_path = base / "outputs" / "model_latest.pth"
    
    print(f"Target Checkpoint: {ckpt_path}")

    # Load Model using the config passed by @hydra.main
    model = load_model(ckpt_path, cfg, device)
    print("✅ Model Loaded. Starting ZMQ Server...")

    # Setup ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORT}")
    print(f"Listening on tcp://*:{PORT}...")

    while True:
        try:
            # Receive Input
            message = socket.recv_pyobj()
            start_time = time.time()

            # --- Preprocessing ---
            # Inputs: 
            # visual: (B, T, V, C, H, W) [uint8 or float]
            # proprio: (B, T, D)
            # actions: (B, T, D)
            
            def to_tensor(arr):
                t = torch.from_numpy(arr).float().to(device)
                if arr.dtype == np.uint8:
                    t = t / 255.0
                return t

            visual_t = to_tensor(message['visual'])
            proprio_t = to_tensor(message['proprio'])
            actions_t = to_tensor(message['actions'])

            # Ensure Batch Dims
            if visual_t.ndim == 4: visual_t = visual_t.unsqueeze(0)
            if visual_t.ndim == 5 and visual_t.shape[2] != 3: # (B, T, V, C, H, W) check
                 pass # Assume correct if 5D or 6D. 
            if proprio_t.ndim == 2: proprio_t = proprio_t.unsqueeze(0)
            if actions_t.ndim == 2: actions_t = actions_t.unsqueeze(0)

            obs_0 = {"visual": visual_t, "proprio": proprio_t}

            # --- Inference (Rollout) ---
            with torch.no_grad():
                # rollout returns embeddings (z_obses) and combined z
                z_obses, _ = model.rollout(obs_0, actions_t)
                
                # --- Decoding ---
                # Check for decoders
                has_decoder = (hasattr(model, "decoder") and model.decoder is not None) or \
                              (hasattr(model, "decoder_front") and model.decoder_front is not None)

                if has_decoder:
                    decoded_obs, _ = model.decode_obs(z_obses)
                    
                    # 'visual' in decoded_obs is usually (B, T, C, H, W) or (B, T, V, C, H, W)
                    full_visual_pred = decoded_obs['visual']
                    
                    # Slice to return only the FUTURE predictions (remove history context)
                    n_hist = visual_t.shape[1]
                    if full_visual_pred.shape[1] > n_hist:
                        future_visual_pred = full_visual_pred[:, n_hist:]
                    else:
                        future_visual_pred = full_visual_pred # Fallback if no future gen

                    # Convert to Numpy uint8
                    future_visual_np = future_visual_pred.cpu().numpy()
                    result_data = np.clip(future_visual_np * 255, 0, 255).astype(np.uint8)
                else:
                    # Return embeddings if no decoder
                    result_data = z_obses['visual'].cpu().numpy()

            # Send Reply
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