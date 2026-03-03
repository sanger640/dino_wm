import os
import zmq
import torch
import torch.nn.functional as F
import numpy as np
import hydra
from pathlib import Path
from omegaconf import OmegaConf
from torchvision import transforms
from einops import rearrange

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
    
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            result[k] = v
    result["epoch"] = payload.get("epoch", 0)
    return result

def load_model(model_ckpt, train_cfg, device):
    """Reconstruct and load the full VWorldModel."""
    model_ckpt = Path(model_ckpt)
    ckpt_data = {}
    if model_ckpt.exists():
        ckpt_data = load_ckpt_payload(model_ckpt, device)

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
    total_visual_patches = num_views * patches_per_view
    extra_tokens = 2 if concat_dim == 0 else 0
    predictor_num_patches = total_visual_patches + extra_tokens

    if concat_dim == 1:
        proprio_dim = getattr(train_cfg, "proprio_emb_dim", 0) * getattr(train_cfg, "num_proprio_repeat", 1)
        action_dim = getattr(train_cfg, "action_emb_dim", 0) * getattr(train_cfg, "num_action_repeat", 1)
        predictor_dim = encoder_emb_dim + action_dim + proprio_dim
    else:
        predictor_dim = encoder_emb_dim

    instantiate_kwargs["predictor"] = get_component("predictor", "predictor", dim=predictor_dim, num_patches=predictor_num_patches, num_frames=train_cfg.num_hist)
    
    if is_dual:
        instantiate_kwargs["decoder_front"] = get_component("decoder_front", "decoder", emb_dim=encoder_emb_dim)
        instantiate_kwargs["decoder_wrist"] = get_component("decoder_wrist", "decoder", emb_dim=encoder_emb_dim)
    else:
        instantiate_kwargs["decoder"] = get_component("decoder", "decoder", emb_dim=encoder_emb_dim)

    instantiate_kwargs["proprio_dim"] = getattr(train_cfg, "proprio_emb_dim", 0)
    instantiate_kwargs["action_dim"] = getattr(train_cfg, "action_emb_dim", 0)
    instantiate_kwargs["concat_dim"] = concat_dim
    instantiate_kwargs["num_action_repeat"] = getattr(train_cfg, "num_action_repeat", 1)
    instantiate_kwargs["num_proprio_repeat"] = getattr(train_cfg, "num_proprio_repeat", 1)
    instantiate_kwargs["image_size"] = getattr(train_cfg, "img_size", 224)
    instantiate_kwargs["num_hist"] = train_cfg.num_hist
    instantiate_kwargs["num_pred"] = train_cfg.num_pred

    model = hydra.utils.instantiate(train_cfg.model, **instantiate_kwargs)
    model.to(device)
    model.eval()
    return model

@hydra.main(version_base=None, config_path="conf/", config_name="train_dual")
def main(cfg: OmegaConf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.exists():
        ckpt_path = Path(cfg.ckpt_base_path) / "outputs" / "model_latest.pth"

    model = load_model(ckpt_path, cfg, device)
    TARGET_IMG_SIZE = getattr(cfg, "img_size", 224)

    # --- INFERENCE TRANSFORM (Matches Training Exactly) ---
    inference_transform = transforms.Compose([
        transforms.Resize(TARGET_IMG_SIZE),
        transforms.CenterCrop(TARGET_IMG_SIZE),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORT}")
    print(f"DINO Evaluation Server listening on tcp://*:{PORT}...")

    while True:
        try:
            message = socket.recv_pyobj()
            
            # Expected shape from evaluate_folders.py: (2, 3, H, W) -> [Front, Wrist]
            img_arr = message['images'] 
            
            # 1. Convert to tensor and scale to [0.0, 1.0]
            t = torch.from_numpy(img_arr).float().to(device)
            if t.max() > 1.0:
                t = t / 255.0  
                
            # 2. Apply standard image transform -> (2, 3, 224, 224)
            t = inference_transform(t)
            
            # 3. Format into the exact 6D shape VWorldModel expects: (Batch, Time, View, C, H, W)
            # We have 1 batch, 1 timestep, 2 views
            t = t.view(1, 1, 2, 3, TARGET_IMG_SIZE, TARGET_IMG_SIZE)

            # 4. Dummy proprioception (encode_obs requires it, but we only care about visual output)
            dummy_proprio = torch.zeros((1, 1, 4), device=device)
            
            obs = {"visual": t, "proprio": dummy_proprio}

            with torch.no_grad():
                # 5. Let VWorldModel handle the encoder_transform and view concatenation
                z_dct = model.encode_obs(obs)
                
                # Output shape is (1, 1, 392, 384) -> (Batch, Time, Total_Patches, Dim)
                z_visual = z_dct['visual']
                
                # Reshape back to (2, 196, 384) for the client script to easily parse
                z_visual = z_visual.view(2, 196, 384)
            
            socket.send_pyobj({'z_visual': z_visual.cpu().numpy()})

        except Exception as e:
            print(f"❌ Error: {e}")
            try: socket.send_pyobj({'error': str(e)})
            except: pass

if __name__ == "__main__":
    main()