import os
import zmq
import torch
import torch.nn.functional as F
import numpy as np
import hydra
import time
from pathlib import Path
from omegaconf import OmegaConf
from torchvision import transforms

# --- CONFIG ---
CHECKPOINT_PATH = "/home/sanger/dino_wm/outputs/model_latest.pth" 
PORT = 5556

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
    model.eval()
    return model

@hydra.main(version_base=None, config_path="conf/", config_name="train_dual")
def main(cfg: OmegaConf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = Path(CHECKPOINT_PATH)
    model = load_model(ckpt_path, cfg, device)
    TARGET_IMG_SIZE = getattr(cfg, "img_size", 224)

    ACTION_MEAN = torch.tensor([0.45678952, 0.00051019, 0.50954217, 0.21926114], device=device)
    ACTION_STD  = torch.tensor([0.03182372, 0.01151787, 0.03419121, 0.41397065], device=device)
    PROPRIO_MEAN = torch.tensor([0.4564166, 0.00056233, 0.50817657, 0.21921302], device=device)
    PROPRIO_STD = torch.tensor([0.03217997, 0.01056713, 0.0327194,  0.4139551 ], device=device)

    inference_transform = transforms.Compose([
        transforms.Resize(TARGET_IMG_SIZE), transforms.CenterCrop(TARGET_IMG_SIZE),
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
            visual_t = torch.from_numpy(message['visual']).float().to(device) / 255.0
            proprio_t = torch.from_numpy(message['proprio']).float().to(device)
            actions_t = torch.from_numpy(message['actions']).float().to(device)

            proprio_t = (proprio_t - PROPRIO_MEAN) / PROPRIO_STD
            actions_t = (actions_t - ACTION_MEAN) / ACTION_STD

            b, t, v, c, h, w = visual_t.shape
            visual_t = visual_t.view(b * t * v, c, h, w)
            visual_t = inference_transform(visual_t)
            visual_t = visual_t.view(b, t, v, c, TARGET_IMG_SIZE, TARGET_IMG_SIZE)

            with torch.no_grad():
                z_obses, _ = model.rollout({"visual": visual_t, "proprio": proprio_t}, actions_t)
                
                lyap_exp_np = np.zeros(actions_t.shape[0] - 1)
                max_patch_idx_np = np.zeros((actions_t.shape[0] - 1, 5), dtype=int)

                if actions_t.shape[0] > 1:
                    z_front = z_obses['visual'][:, :, 196:392, :] 
                    z_orig, z_noisy = z_front[0:1], z_front[1:]
                    cos_sim = torch.nn.functional.cosine_similarity(z_noisy, z_orig, dim=-1)
                    dist = 1 - cos_sim 

                    if dist.shape[1] > t:
                        T_span = dist.shape[1] - t
                        lyap_1d = (1.0 / T_span) * torch.log((dist[:, -1] + 1e-8) / (dist[:, t] + 1e-8))
                        lyap_grid = lyap_1d.view(actions_t.shape[0] - 1, 14, 14)

                        for b_idx in range(actions_t.shape[0] - 1):
                            max_avg = -float('inf')
                            best_cluster = [0]*5
                            for r in range(1, 13):
                                for c in range(1, 13):
                                    center = r * 14 + c
                                    cluster = [center, (r-1)*14+c, (r+1)*14+c, r*14+(c-1), r*14+(c+1)]
                                    avg = sum([lyap_grid[b_idx, r_o, c_o] for r_o, c_o in [(r,c), (r-1,c), (r+1,c), (r,c-1), (r,c+1)]]) / 5.0
                                    if avg > max_avg:
                                        max_avg, best_cluster = avg, cluster
                            lyap_exp_np[b_idx] = max_avg.item()
                            max_patch_idx_np[b_idx] = best_cluster

                decoded, _ = model.decode_obs(z_obses)
                res_imgs = np.clip(((decoded['visual'].cpu().numpy() + 1.0) / 2.0) * 255, 0, 255).astype(np.uint8)
            
            socket.send_pyobj({'states': res_imgs, 'lyapunov': lyap_exp_np, 'max_patch_idx': max_patch_idx_np, 'inference_time': time.time() - start_time})
        except Exception as e:
            print(f"❌ Error: {e}")
            socket.send_pyobj({'error': str(e)})

if __name__ == "__main__":
    main()