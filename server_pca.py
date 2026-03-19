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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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
    is_dual = "dual" in target_class or "Dual" in target_class
    concat_dim = getattr(train_cfg, "concat_dim", 0)
    num_views = 2 if is_dual else 1
    
    patch_size = 16 
    patches_per_view = (getattr(train_cfg, "img_size", 224) // patch_size) ** 2
    predictor_num_patches = (num_views * patches_per_view) + (2 if concat_dim == 0 else 0)
    predictor_dim = encoder_emb_dim + (getattr(train_cfg, "action_emb_dim", 0) + getattr(train_cfg, "proprio_emb_dim", 0)) if concat_dim == 1 else encoder_emb_dim
    
    instantiate_kwargs["predictor"] = get_component("predictor", dim=predictor_dim, num_patches=predictor_num_patches, num_frames=train_cfg.num_hist)
    
    if is_dual:
        instantiate_kwargs["decoder_front"] = get_component("decoder", emb_dim=encoder_emb_dim)
        instantiate_kwargs["decoder_wrist"] = get_component("decoder", emb_dim=encoder_emb_dim)
    else:
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
            "encoder": model.encoder, "proprio_encoder": model.proprio_encoder,
            "action_encoder": model.action_encoder, "predictor": model.predictor,
            "decoder_front": model.decoder_front, "decoder_wrist": model.decoder_wrist,
            "wrist_head": model.wrist_head, "front_head": model.front_head, "proprio_head": model.proprio_head
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

@hydra.main(version_base=None, config_path="conf/", config_name="train_dual")
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

            b, t, v, c, h, w = visual_t.shape
            visual_t = visual_t.view(b * t * v, c, h, w)
            visual_t = inference_transform(visual_t)
            visual_t = visual_t.view(b, t, v, c, TARGET_IMG_SIZE, TARGET_IMG_SIZE)

            obs_0 = {"visual": visual_t, "proprio": proprio_t}

            with torch.no_grad():
                z_obses, _ = model.rollout(obs_0, actions_t)
                
                b_size = actions_t.shape[0]
                t_total = z_obses['visual'].shape[1]
                n_hist = visual_t.shape[1]
                
                lyap_exp_np = None
                max_patch_idx_np = None
                pca_images = None
                
                if b_size > 0:
                    z_front = z_obses['visual'][:, :, 196:392, :] 
                    z_orig = z_front[0:1] 
                    z_flat = z_orig.reshape(-1, z_front.shape[-1]).cpu().numpy()
                    num_flat_tokens = z_flat.shape[0]
                    
                    # --- 1. STAGE 1 PCA & BACKGROUND MASK ---
                    z_mean1 = z_flat.mean(axis=0)
                    z_std1 = z_flat.std(axis=0) + 1e-6
                    z_norm1 = (z_flat - z_mean1) / z_std1
                    
                    pca1 = PCA(n_components=1)
                    pc1 = pca1.fit_transform(z_norm1)[:, 0]
                    
                    # --- FIX 2: STRICTER STATISTICAL THRESHOLDING ---
                    threshold = np.mean(pc1) + 0.5 * np.std(pc1)
                    # if np.sum(pc1 > threshold) < np.sum(pc1 < -threshold):
                    #     fg_mask = pc1 > threshold
                    # else:
                    #     fg_mask = pc1 < -threshold
                    fg_mask = pc1 < threshold 
                    z_fg = z_flat[fg_mask]
                    canvas_features = np.zeros((num_flat_tokens, 3), dtype=np.float32)
                    blocks_mask = np.zeros(num_flat_tokens, dtype=bool) 
                    
                    if len(z_fg) > 3: 
                        # --- 2. STAGE 2 PCA (Object Coloring) ---
                        z_mean2 = z_fg.mean(axis=0)
                        z_std2 = z_fg.std(axis=0) + 1e-6
                        z_norm2 = (z_fg - z_mean2) / z_std2
                        
                        pca2 = PCA(n_components=3)
                        pca2_proj = pca2.fit_transform(z_norm2)
                        canvas_features[fg_mask] = pca2_proj

                        # --- 3. K-MEANS OBJECT SEPARATION ---
                        kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
                        cluster_labels = kmeans.fit_predict(z_fg)
                        
                        fg_indices = np.where(fg_mask)[0]
                        cluster_0_mask = np.zeros(num_flat_tokens, dtype=bool)
                        cluster_1_mask = np.zeros(num_flat_tokens, dtype=bool)
                        
                        cluster_0_mask[fg_indices[cluster_labels == 0]] = True
                        cluster_1_mask[fg_indices[cluster_labels == 1]] = True
                        
                        c0_y_avg = np.mean((fg_indices[cluster_labels == 0] % 196) // 14) if np.sum(cluster_labels == 0) > 0 else 0
                        c1_y_avg = np.mean((fg_indices[cluster_labels == 1] % 196) // 14) if np.sum(cluster_labels == 1) > 0 else 0
                        
                        if c0_y_avg > c1_y_avg:
                            blocks_mask = cluster_0_mask
                        else:
                            blocks_mask = cluster_1_mask
                    
                    # --- 4. BLOCKS-SPECIFIC LYAPUNOV ---
                    if b_size > 1:
                        z_noisy = z_front[1:]
                        cos_sim = torch.nn.functional.cosine_similarity(z_noisy, z_orig, dim=-1)
                        patch_distances = 1 - cos_sim

                        if patch_distances.shape[1] > n_hist:
                            d_start = patch_distances[:, n_hist] + 1e-8
                            d_end = patch_distances[:, -1] + 1e-8
                            T_span = patch_distances.shape[1] - n_hist
                            
                            lyap_per_patch = (1.0 / T_span) * torch.log(d_end / d_start) 

                            # 1. Apply K-Means Blocks Mask
                            last_frame_mask = torch.from_numpy(blocks_mask[-196:]).bool().to(device)
                            if last_frame_mask.sum() > 0:
                                lyap_per_patch[:, ~last_frame_mask] = -float('inf')

                            # --- FIX 1: ABSOLUTE NOISE FLOOR ---
                            # Prevent microscopic latent noise from generating huge LE spikes
                            significant_drift_mask = d_end > 1e-3 
                            lyap_per_patch[~significant_drift_mask] = -float('inf')

                            # Get max patch AFTER both masks are applied
                            max_lyap_vals, max_patch_indices = torch.max(lyap_per_patch, dim=-1)
                            lyap_exp_np = max_lyap_vals.cpu().numpy()
                            max_patch_idx_np = max_patch_indices.cpu().numpy()
                        else:
                            lyap_exp_np = np.zeros(b_size - 1)
                            max_patch_idx_np = np.zeros(b_size - 1, dtype=int)

                    # --- 5. PCA UPSAMPLING (BILINEAR) ---
                    canvas_tensor = torch.from_numpy(canvas_features).view(1, t_total, 14, 14, 3).permute(0, 1, 4, 2, 3)
                    fg_mask_tensor = torch.from_numpy(fg_mask).float().view(1, t_total, 1, 14, 14)
                    
                    upsampled_features = F.interpolate(
                        canvas_tensor.reshape(t_total, 3, 14, 14), 
                        size=(TARGET_IMG_SIZE, TARGET_IMG_SIZE), 
                        mode='bilinear', align_corners=False
                    ).view(1, t_total, 3, TARGET_IMG_SIZE, TARGET_IMG_SIZE)
                    
                    upsampled_mask = F.interpolate(
                        fg_mask_tensor.reshape(t_total, 1, 14, 14), 
                        size=(TARGET_IMG_SIZE, TARGET_IMG_SIZE), 
                        mode='bilinear', align_corners=False
                    ).view(1, t_total, 1, TARGET_IMG_SIZE, TARGET_IMG_SIZE)
                    
                    result_rgb = torch.zeros_like(upsampled_features)
                    for i in range(3):
                        channel_data = upsampled_features[0, :, i, :, :]
                        p_min, p_max = channel_data.min(), channel_data.max()
                        result_rgb[0, :, i, :, :] = (channel_data - p_min) / (p_max - p_min + 1e-6)

                    hard_mask = upsampled_mask > 0.5
                    result_rgb = result_rgb * hard_mask.float()
                    
                    result_rgb = result_rgb.repeat(b_size, 1, 1, 1, 1)
                    pca_dual_cam = torch.stack([result_rgb, result_rgb], dim=2)
                    pca_images = (pca_dual_cam.numpy() * 255).astype(np.uint8)

                # --- 6. DECODE STANDARD VQ-VAE IMAGES ---
                if hasattr(model, "decoder_front") and model.decoder_front is not None:
                    decoded_obs, _ = model.decode_obs(z_obses)
                    pred_visual_np = (decoded_obs['visual'].cpu().numpy() + 1.0) / 2.0
                    decoded_images = np.clip(pred_visual_np * 255, 0, 255).astype(np.uint8)
                else:
                    decoded_images = None

            # SEND RESPONSE
            socket.send_pyobj({
                'states': decoded_images, 
                'pca_mask': pca_images,
                'lyapunov': lyap_exp_np,
                'max_patch_idx': max_patch_idx_np,  
                'inference_time': time.time() - start_time
            })

        except Exception as e:
            print(f"❌ Error: {e}")
            try:
                socket.send_pyobj({'error': str(e)})
            except: pass

if __name__ == "__main__":
    main()