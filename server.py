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
CHECKPOINT_PATH = "/home/sanger/dino_wm/outputs/model_latest.pth" 
PORT = 5556

# --- NORMALIZATION STATS (From Dataset) ---
# These specific values are required for the model to understand the input 
ACTION_MEAN = torch.tensor([0.4472, 0.0025, 0.4921, 0.0202], device='cuda')
ACTION_STD  = torch.tensor([0.0297, 0.0085, 0.0195, 0.1406], device='cuda')

# Assuming Proprioception uses same stats
PROPRIO_MEAN = ACTION_MEAN
PROPRIO_STD  = ACTION_STD
# ------------------------------------------

ALL_MODEL_KEYS = [
    "encoder", "predictor", "decoder", "decoder_front", "decoder_wrist",
    "proprio_encoder", "action_encoder",
]

def load_ckpt_payload(snapshot_path, device):
    """Load model payload from checkpoint file."""
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
    """Reconstruct and load the VWorldModel."""
    model_ckpt = Path(model_ckpt)
    
    ckpt_data = {}
    if model_ckpt.exists():
        ckpt_data = load_ckpt_payload(model_ckpt, device)
    else:
        print(f"Warning: Checkpoint not found at {model_ckpt}. Initializing fresh model.")

    def get_component(key, cfg_section=None, **kwargs):
        component = None
        if cfg_section and hasattr(train_cfg, cfg_section):
            print(f"Instantiating {key} from config ({cfg_section})...")
            component = hydra.utils.instantiate(getattr(train_cfg, cfg_section), **kwargs)
            component.to(device)
        
        if key in ckpt_data and component is not None:
            print(f"Loading weights for {key}...")
            try:
                weights = ckpt_data[key]
                if isinstance(weights, dict):
                    component.load_state_dict(weights)
                elif isinstance(weights, torch.nn.Module):
                    component.load_state_dict(weights.state_dict())
            except Exception as e:
                print(f"Warning: Failed to load weights for {key}: {e}")
        return component

    instantiate_kwargs = {}
    
    # 1. Load Encoder
    # NOTE: If your checkpoint used patch_size=16, the encoder loaded here MUST match.
    # DINOv2 usually defaults to 14. If DINO weights fail to load later, we might need 
    # to force patch_size=16 in the hydra config for the encoder too.
    instantiate_kwargs["encoder"] = get_component("encoder", "encoder")
    
    if hasattr(instantiate_kwargs["encoder"], "emb_dim"):
        encoder_emb_dim = instantiate_kwargs["encoder"].emb_dim
    else:
        encoder_emb_dim = 384

    # 2. Proprio Encoder (Using 4 dims as input)
    instantiate_kwargs["proprio_encoder"] = get_component(
        "proprio_encoder", "proprio_encoder", 
        in_chans=4, emb_dim=train_cfg.proprio_emb_dim
    )
    
    # 3. Action Encoder (FORCE 20 DIMENSIONS TO MATCH CHECKPOINT)
    # The error "expected input[1, 20, 1] ... got [1, 4, 1]" means weights are 20-dim.
    instantiate_kwargs["action_encoder"] = get_component(
        "action_encoder", "action_encoder", 
        in_chans=4,  # <--- FORCE 20 HERE
        emb_dim=train_cfg.action_emb_dim
    )

    # 4. Predictor Configuration
    target_class = train_cfg.model._target_
    is_dual = "dual" in target_class or "Dual" in target_class
    num_views = 2 if is_dual else 1
    img_size = getattr(train_cfg, "img_size", 224)
    concat_dim = getattr(train_cfg, "concat_dim", 0)
    num_hist = train_cfg.num_hist
    
    # --- FORCE PATCH SIZE 16 TO MATCH CHECKPOINT ---
    # The error "1176 vs 1536" confirms checkpoint used 16, but code defaulted to 14.
    patch_size = 16 
    print(f"Forcing Patch Size to {patch_size} (Derived from Checkpoint Error)")
    # -----------------------------------------------

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

    # 5. Decoders
    if is_dual:
        instantiate_kwargs["decoder_front"] = get_component("decoder_front", "decoder", emb_dim=encoder_emb_dim)
        instantiate_kwargs["decoder_wrist"] = get_component("decoder_wrist", "decoder", emb_dim=encoder_emb_dim)
    else:
        instantiate_kwargs["decoder"] = get_component("decoder", "decoder", emb_dim=encoder_emb_dim)

    # 6. Other Args
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

@hydra.main(version_base=None, config_path="conf/", config_name="train_dual")
def main(cfg: OmegaConf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.exists():
        base = Path(cfg.ckpt_base_path)
        ckpt_path = base / "outputs" / "model_latest.pth"
    
    print(f"Target Checkpoint: {ckpt_path}")

    model = load_model(ckpt_path, cfg, device)
    
    TARGET_IMG_SIZE = getattr(cfg, "img_size", 224)
    EXPECTED_ACTION_DIM = 4 # We know this from the checkpoint error

    print(f"✅ Model Loaded.")
    print(f"   Image Size: {TARGET_IMG_SIZE}x{TARGET_IMG_SIZE}")
    print(f"   Action Dim: {EXPECTED_ACTION_DIM}")
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
                    # Normalize Images [-1, 1]
                    t = (t / 127.5) - 1.0 
                return t

            visual_t = to_tensor(message['visual'])
            proprio_t = to_tensor(message['proprio'])
            actions_t = to_tensor(message['actions'])

            if visual_t.ndim == 4: visual_t = visual_t.unsqueeze(0)
            if proprio_t.ndim == 2: proprio_t = proprio_t.unsqueeze(0)
            if actions_t.ndim == 2: actions_t = actions_t.unsqueeze(0)

            # 1. Normalize Actions & Proprio
            proprio_t = (proprio_t - PROPRIO_MEAN) / PROPRIO_STD
            actions_t = (actions_t - ACTION_MEAN) / ACTION_STD
            print("prop")
            print(proprio_t)
            print("act")
            print(actions_t)
            print(actions_t.shape)

            # 2. Resize Images
            if visual_t.shape[-1] != TARGET_IMG_SIZE or visual_t.shape[-2] != TARGET_IMG_SIZE:
                orig_shape = visual_t.shape
                if visual_t.ndim == 6:
                    b, t, v, c, h, w = orig_shape
                    visual_t = visual_t.view(b * t * v, c, h, w)
                    visual_t = F.interpolate(visual_t, size=(TARGET_IMG_SIZE, TARGET_IMG_SIZE), mode='bilinear', align_corners=False)
                    visual_t = visual_t.view(b, t, v, c, TARGET_IMG_SIZE, TARGET_IMG_SIZE)
                elif visual_t.ndim == 5:
                    b, t, c, h, w = orig_shape
                    visual_t = visual_t.view(b * t, c, h, w)
                    visual_t = F.interpolate(visual_t, size=(TARGET_IMG_SIZE, TARGET_IMG_SIZE), mode='bilinear', align_corners=False)
                    visual_t = visual_t.view(b, t, c, TARGET_IMG_SIZE, TARGET_IMG_SIZE)
            
            # 4. Inference
            obs_0 = {"visual": visual_t, "proprio": proprio_t}

            with torch.no_grad():
                z_obses, _ = model.rollout(obs_0, actions_t)
                # print(time.time() - start_time)

                b_size = actions_t.shape[0]
                lyap_exp_np = None
                
                if b_size > 1:
                    z_visual = z_obses['visual'] # Shape: (B, T, Tokens, Dim)
                    n_hist = visual_t.shape[1]
                    print("n history")
                    print(n_hist)
                    # Flatten spatial/feature dims to get a single vector per timestep: (B, T, Features)
                    # z_flat = z_visual.reshape(b_size, z_visual.shape[1], -1)
                    
                    # z_orig = z_flat[0:1]  # Original Trajectory (1, T, F)
                    # z_noisy = z_flat[1:]  # Noisy Trajectories  (B-1, T, F)
                    
                    # # Euclidean Distance per timestep
                    # distances = torch.norm(z_noisy - z_orig, dim=-1) # (B-1, T)

                    # z_orig = z_visual[0:1]
                    # z_noisy = z_visual[1:]

                    # # Calculate Euclidean distance on the feature dimension (-1) first
                    # # This yields a distance for every individual patch: Shape (B-1, T, Tokens)
                    # patch_distances = torch.norm(z_noisy - z_orig, dim=-1)

                    # # Get the MAXIMUM distance across all patches
                    # distances = patch_distances.amax(dim=-1) # Shape: (B-1, T)

                    # z_orig = z_visual[0:1]
                    # z_noisy = z_visual[1:]

                    # # 1. Calculate the absolute difference for every single feature
                    # # Shape: (B-1, T, Tokens, Dim)
                    # absolute_diff = torch.abs(z_noisy - z_orig)

                    # # 2. Apply L-infinity norm: Find the maximum feature change per patch
                    # # Shape: (B-1, T, Tokens)
                    # patch_distances_linf = torch.amax(absolute_diff, dim=-1)

                    # # 3. Find the single patch with the most extreme feature divergence
                    # # Shape: (B-1, T)
                    # distances = torch.amax(patch_distances_linf, dim=-1)
                    
                    K = 5
                    z_orig = z_visual[0:1]  # Shape: (1, T, 196, 384)
                    z_noisy = z_visual[1:]  # Shape: (B-1, T, 196, 384)
                    print("z_visual shape")
                    print(z_visual.shape)
                    # During your rollout evaluation script:
                    # z_orig shape: (1, T, 392, 384)  (196 wrist + 196 front)

                    # 1. Slice out ONLY the fixed front camera
                    # We take patches from index 196 to the end
                    z_front_orig = z_orig[:, :, 196:, :] # Shape: (1, T, 196, 384)
                    z_front_noisy = z_noisy[:, :, 196:, :] 

                    # 2. Run the standard Top-K metric on the stable camera
                    cos_sim = F.cosine_similarity(z_front_orig, z_front_noisy, dim=-1)
                    patch_distances = 1.0 - cos_sim
                    topk_distances = torch.topk(patch_distances, k=K, dim=-1)[0]
                    # print("z_orig")
                    # print(z_orig)
                    # print("z_noisy")
                    # print(z_noisy)
                    # 1. Calculate Cosine Similarity patch-by-patch (dim=-1)
                    # Shape: (B-1, T, 196)
                    # cos_sim = F.cosine_similarity(z_orig, z_noisy, dim=-1)

                    # # 2. Convert Similarity to Distance (1.0 - sim)
                    # # 0.0 means identical, 2.0 means exact opposite
                    # patch_distances = 1.0 - cos_sim

                    # # 3. Get the top K worst patches (highest distances)
                    # # topk returns (values, indices), we take [0] for values
                    # # Shape: (B-1, T, K)
                    # topk_distances = torch.topk(patch_distances, k=K, dim=-1)[0]

                    # # 4. Average only those Top-K worst patches
                    # # Shape: (B-1, T)
                    distances = topk_distances.mean(dim=-1)


                    # z_orig = z_visual[0:1]
                    # z_noisy = z_visual[1:]

                    # # Calculate Euclidean distance on the feature dimension (-1) first
                    # # This yields a distance for every individual patch: Shape (B-1, T, Tokens)
                    # patch_distances = torch.norm(z_noisy - z_orig, dim=-1)

                    # # Get the MAXIMUM distance across all patches
                    # distances = patch_distances.amax(dim=-1) # Shape: (B-1, T)

                    # (Optional: You can combine steps 2 and 3 into one line)
                    # distances = torch.amax(absolute_diff, dim=(-2, -1))

                    # 1. Separate the original and noisy trajectories
                    # z_orig = z_visual[0:1]  # Shape: (1, T, 196, 384)
                    # z_noisy = z_visual[1:]  # Shape: (B-1, T, 196, 384)

                    # # 2. Apply Global Average Pooling (GAP)
                    # # We average across the 'Tokens' dimension (dim=-2 or dim=2)
                    # # This collapses the 196 spatial patches into a single 384-D vector per timestep
                    # z_orig_pooled = z_orig.mean(dim=-2)   # Shape: (1, T, 384)
                    # z_noisy_pooled = z_noisy.mean(dim=-2) # Shape: (B-1, T, 384)

                    # # 3. Calculate the Euclidean distance on the pooled vectors
                    # # We calculate the norm across the final feature dimension (dim=-1)
                    # distances = torch.norm(z_noisy_pooled - z_orig_pooled, dim=-1) # Shape: (B-1, T)

                    # z_orig = z_visual[0:1]  # Shape: (1, T, 196, 384)
                    # z_noisy = z_visual[1:]  # Shape: (B-1, T, 196, 384)

                    # # 2. Calculate Cosine Similarity across the feature dimension (dim=-1)
                    # # This measures the angle between the 384-D vectors, ignoring their magnitudes.
                    # # Resulting shape: (B-1, T, 196)
                    # cos_sim = F.cosine_similarity(z_orig, z_noisy, dim=-1)

                    # # 3. Average the similarity across all 196 spatial patches
                    # # Resulting shape: (B-1, T)
                    # mean_cos_sim = cos_sim.mean(dim=-1)

                    # # 4. Convert Similarity to Distance
                    # # Cosine similarity ranges from 1.0 (identical) to -1.0 (exact opposite).
                    # # Subtracting from 1.0 gives a distance ranging from 0.0 (identical) to 2.0.
                    # distances = 1.0 - mean_cos_sim
                    print("distances shape")
                    print(distances.shape)                    
                    if distances.shape[1] > n_hist:
                        # d(0) is the distance at the first predicted step (after history)
                        # We add 1e-8 to prevent division by zero or ln(0) errors
                        d_start = distances[:, n_hist] + 1e-8
                        print("dstart")
                        print(d_start)

                        d_start = distances[:, n_hist+1] + 1e-8
                        print("dstart+1")
                        print(d_start)
                        # print("d start")
                        # print(d_start)
                        # d(T) is the distance at the final predicted step
                        d_end = distances[:, -1] + 1e-8
                        print("d_end")
                        print(d_end)
                        # print("d end")
                        # print(d_end)
                        
                        T_span = distances.shape[1] - n_hist
                        
                        # FTLE formula: (1 / T) * ln( d_end / d_start )
                        lyap_exp = (1.0 / T_span) * torch.log(d_end / d_start)
                        lyap_exp_np = lyap_exp.cpu().numpy()
                        print("lyap")
                        print(lyap_exp_np)
                    else:
                        lyap_exp_np = np.zeros(b_size - 1)
                
                has_decoder = (hasattr(model, "decoder") and model.decoder is not None) or \
                              (hasattr(model, "decoder_front") and model.decoder_front is not None)

                # if has_decoder:
                #     decoded_obs, _ = model.decode_obs(z_obses)
                    
                #     full_visual_pred = decoded_obs['visual']
                #     n_hist = visual_t.shape[1]
                    
                #     if full_visual_pred.shape[1] > n_hist:
                #         future_visual_pred = full_visual_pred[:, n_hist:]
                #     else:
                #         future_visual_pred = full_visual_pred 

                #     # Denormalize Images [-1, 1] -> [0, 255]
                #     future_visual_np = future_visual_pred.cpu().numpy()
                #     future_visual_np = (future_visual_np + 1.0) / 2.0
                #     result_data = np.clip(future_visual_np * 255, 0, 255).astype(np.uint8)
                # else:
                #     result_data = z_obses['visual'].cpu().numpy()

            
            socket.send_pyobj({
                'states': None,
                'lyapunov': lyap_exp_np,  # Send the exponents back!
                'inference_time': time.time() - start_time
            })
            

        except Exception as e:
            print(f"❌ Error: {e}")
            try:
                socket.send_pyobj({'error': str(e)})
            except: pass

if __name__ == "__main__":
    main()