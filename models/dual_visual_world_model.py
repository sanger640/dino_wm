import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from einops import rearrange, repeat

class VWorldModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder_front,
        decoder_wrist,
        predictor,
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        contrastive_weight=0.0,
        contrastive_eps=0.05,
        contrastive_gamma=0.9,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder_front = decoder_front  # decoder could be None
        self.decoder_wrist = decoder_wrist
        self.predictor = predictor  # The base ViT predictor
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.contrastive_weight = contrastive_weight
        self.contrastive_eps = contrastive_eps
        self.contrastive_gamma = contrastive_gamma
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        
        # Dimensions after potential repeating/tiling
        self.proprio_dim = proprio_dim * num_proprio_repeat 
        self.action_dim = action_dim * num_action_repeat 
        
        # Base embedding dimension from the DINO encoder (usually 384)
        self.base_emb_dim = self.encoder.emb_dim
        self.emb_dim = self.base_emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim)

        # --- LATENT SAFETY FILTER MLP HEADS ---
        # The paper specifies 3-layer MLPs with a hidden dim of 788.
        # We need independent heads for Wrist Cam, Front Cam, and Proprio.
        hidden_dim = 788
        
        # Note: If concat_dim == 1, the tokens fed into the MLP will have emb_dim.
        # The MLPs output the same dimension to maintain shape for the decoders.
        
        self.wrist_head = nn.Sequential(
            nn.Linear(self.emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.emb_dim)
        )
        
        self.front_head = nn.Sequential(
            nn.Linear(self.emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.emb_dim)
        )
        
        self.proprio_head = nn.Sequential(
            nn.Linear(self.emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.emb_dim) 
        )
        # ----------------------------------------

        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"num_proprio_repeat: {self.num_proprio_repeat}")
        print(f"proprio encoder: {proprio_encoder}")
        print(f"action encoder: {action_encoder}")
        print(f"proprio_dim: {proprio_dim}, after repeat: {self.proprio_dim}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"emb_dim: {self.emb_dim}")

        self.decoders = nn.ModuleList([self.decoder_wrist, self.decoder_front])

        self.concat_dim = concat_dim # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        print("Model emb_dim: ", self.emb_dim)

        if "dino" in self.encoder.name:
            decoder_scale = 16  # from vqvae
            num_side_patches = image_size // decoder_scale
            self.encoder_image_size = num_side_patches * encoder.patch_size
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(self.encoder_image_size)]
            )
        else:
            self.encoder_transform = lambda x: x

        self.decoder_criterion = nn.MSELoss()
        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss()

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
            self.wrist_head.train(mode)
            self.front_head.train(mode)
            self.proprio_head.train(mode)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        for d in self.decoders:
            if d is not None and self.train_decoder:
                d.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
            self.wrist_head.eval()
            self.front_head.eval()
            self.proprio_head.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()
        for d in self.decoders:
            if d is not None:
                d.eval()

    def encode(self, obs, act): 
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        
        if self.concat_dim == 0:
            z = torch.cat(
                    [z_dct['visual'], z_dct['proprio'].unsqueeze(2), act_emb.unsqueeze(2)], dim=2 
                )  
        
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat(
                [z_dct['visual'], proprio_repeated, act_repeated], dim=3
            )  
        return z
    
    def encode_act(self, act):
        act = self.action_encoder(act) 
        return act
    
    def encode_proprio(self, proprio):
        proprio = self.proprio_encoder(proprio)
        return proprio

    def encode_obs(self, obs):
        visual = obs['visual']
        if visual.ndim == 5: 
            visual = visual.unsqueeze(2)
            
        b, t, v, c, h, w = visual.shape
        
        visual = rearrange(visual, "b t v c h w -> (b t v) c h w")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder.forward(visual) 
        
        visual_embs = rearrange(visual_embs, "(b t v) p d -> b t (v p) d", b=b, t=t, v=v)

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)
        return {"visual": visual_embs, "proprio": proprio_emb}

    def predict(self, z):  
        """
        Passes the latent sequence through the ViT, then routes the hidden states
        through dedicated MLP heads based on token type.
        """
        B, T, num_tokens, D = z.shape
        
        # 1. Base ViT Prediction (Produces the unified "Hidden State")
        z_flat = rearrange(z, "b t p d -> b (t p) d")
        z_hidden_flat = self.predictor(z_flat)
        z_hidden = rearrange(z_hidden_flat, "b (t p) d -> b t p d", t=T)
        
        # 2. Route Hidden States to MLP Heads
        z_out = torch.zeros_like(z_hidden)
        
        if self.concat_dim == 0:
            # Tokens: [Wrist Patches, Front Patches, Proprio, Action]
            p_per_view = (num_tokens - 2) // 2 
            
            # Extract
            hidden_wrist = z_hidden[:, :, :p_per_view, :]
            hidden_front = z_hidden[:, :, p_per_view:2*p_per_view, :]
            hidden_proprio = z_hidden[:, :, -2, :]
            
            # Pass through MLP Heads
            out_wrist = self.wrist_head(hidden_wrist)
            out_front = self.front_head(hidden_front)
            out_proprio = self.proprio_head(hidden_proprio)
            
            # Reconstruct (Actions are passed through un-altered since we don't predict them)
            z_out[:, :, :p_per_view, :] = out_wrist
            z_out[:, :, p_per_view:2*p_per_view, :] = out_front
            z_out[:, :, -2, :] = out_proprio
            z_out[:, :, -1, :] = z_hidden[:, :, -1, :] # Keep action token
            
        elif self.concat_dim == 1:
            # Tokens: [Wrist Patches, Front Patches] (Proprio and Action are tiled onto the feature dimension)
            p_per_view = num_tokens // 2
            
            hidden_wrist = z_hidden[:, :, :p_per_view, :]
            hidden_front = z_hidden[:, :, p_per_view:, :]
            
            z_out[:, :, :p_per_view, :] = self.wrist_head(hidden_wrist)
            z_out[:, :, p_per_view:, :] = self.front_head(hidden_front)
            
        return z_out

    def decode(self, z):
        z_obs, z_act = self.separate_emb(z)
        obs, diff = self.decode_obs(z_obs)
        return obs, diff

    def decode_obs(self, z_obs):
        visual_z = z_obs["visual"] 
        b, t, vp, d = visual_z.shape
        
        num_views = 2 
        p = vp // num_views 

        visual_z = rearrange(visual_z, "b t (v p) d -> v b t p d", v=num_views, p=p)

        wrist_recon, wrist_diff = self.decoder_wrist(visual_z[0]) 
        front_recon, front_diff = self.decoder_front(visual_z[1])

        visual = torch.stack([front_recon, wrist_recon], dim=2) 
        visual = rearrange(visual, "(b t) c v h w -> b t v c h w", b=b, t=t)
        
        total_diff = wrist_diff + front_diff
        
        obs = {"visual": visual, "proprio": z_obs["proprio"]}
        return obs, total_diff
    
    def separate_emb(self, z):
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
                                         z[..., -(self.proprio_dim + self.action_dim) :-self.action_dim],  \
                                         z[..., -self.action_dim:]
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act

    def forward(self, obs, act):
        loss = 0
        loss_components = {}
        
        z = self.encode(obs, act)
        
        z_src = z[:, : self.num_hist, :, :]  
        z_tgt = z[:, self.num_pred :, :, :]  
        
        visual_src = obs['visual'][:, : self.num_hist, ...] 
        visual_tgt = obs['visual'][:, self.num_pred :, ...]
        
        if self.predictor is not None:
            z_pred = self.predict(z_src)

            # Contrastive stability loss: train the predictor to be contractive on the safe
            # training distribution. For small action perturbations, the distance between
            # predicted latents should shrink (or stay equal) over the prediction window.
            # This makes FTLE < 0 for stable dynamics → threshold δ = 0 is principled.
            # Gradients flow through predict() only; z_src_pert is computed with no_grad.
            if self.training and self.contrastive_weight > 0.0:
                with torch.no_grad():
                    act_src = act[:, :self.num_hist]                          # (B, T_hist, 4)
                    act_src_pert = act_src.clone()
                    noise = torch.randn_like(act_src_pert[:, :, :3]) * self.contrastive_eps
                    act_src_pert[:, :, :3] = act_src_pert[:, :, :3] + noise
                    z_src_pert = self.replace_actions_from_z(z_src.clone(), act_src_pert)

                z_pred_pert = self.predict(z_src_pert)

                # Extract visual patch tokens from predictions (exclude proprio/action tokens)
                if self.concat_dim == 0:
                    z_v      = z_pred[:, :, :-2, :]       # (B, T_hist, P, D)
                    z_v_pert = z_pred_pert[:, :, :-2, :]
                else:
                    n_extra  = self.proprio_dim + self.action_dim
                    z_v      = z_pred[..., :-n_extra]
                    z_v_pert = z_pred_pert[..., :-n_extra]

                # d_start: cosine distance at first predicted step (B,)
                # d_end:   cosine distance at last predicted step  (B,)
                d_start = (1 - F.cosine_similarity(z_v[:, 0], z_v_pert[:, 0], dim=-1)).mean(dim=-1)
                d_end   = (1 - F.cosine_similarity(z_v[:, -1], z_v_pert[:, -1], dim=-1)).mean(dim=-1)

                # Penalise when divergence grows: d_end > γ · d_start
                contrastive_loss = F.relu(d_end - self.contrastive_gamma * d_start).mean()
                loss = loss + self.contrastive_weight * contrastive_loss
                loss_components["contrastive_stability_loss"] = contrastive_loss

            if self.decoder_front is not None:
                obs_pred, diff_pred = self.decode(
                    z_pred.detach()
                )  
                
                visual_pred = obs_pred['visual'] 
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                
                decoder_loss_pred = (
                    recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            if self.concat_dim == 0:
                z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach())
                z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach())
                z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
            elif self.concat_dim == 1:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)], \
                    z_tgt[:, :, :, :-(self.proprio_dim + self.action_dim)].detach()
                )
                z_proprio_loss = self.emb_criterion(
                    z_pred[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim], 
                    z_tgt[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim].detach()
                )
                z_loss = self.emb_criterion(
                    z_pred[:, :, :, :-self.action_dim], 
                    z_tgt[:, :, :, :-self.action_dim].detach()
                )

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss
        else:
            visual_pred = None
            z_pred = None

        if self.decoder_front is not None:
            obs_reconstructed, diff_reconstructed = self.decode(
                z.detach()
            )  
            visual_reconstructed = obs_reconstructed["visual"]
            recon_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs['visual'])
            decoder_loss_reconstructed = (
                recon_loss_reconstructed
                + self.decoder_latent_loss_weight * diff_reconstructed
            )

            loss_components["decoder_recon_loss_reconstructed"] = (
                recon_loss_reconstructed
            )
            loss_components["decoder_vq_loss_reconstructed"] = diff_reconstructed
            loss_components["decoder_loss_reconstructed"] = (
                decoder_loss_reconstructed
            )
            loss = loss + decoder_loss_reconstructed
        else:
            visual_reconstructed = None
            
        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def replace_actions_from_z(self, z, act):
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z[:, :, -1, :] = act_emb
        elif self.concat_dim == 1:
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim:] = act_repeated
        return z

    def rollout(self, obs_0, act):
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        
        action = act[:, num_obs_init:] 
        z = self.encode(obs_0, act_0)
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self.predict(z[:, -self.num_hist :])
            z_new = z_pred[:, -inc:, ...]
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] 
        z = torch.cat([z, z_new], dim=1)
        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z