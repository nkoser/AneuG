"""
First-stage vessel-aware conditional aneurysm generation.

Trains a VesselAwareCVAE on GHD coefficients from the cap_v8 fitting,
conditioned on ostium geometry + local vessel surface points.

Usage:
    conda run --no-capture-output -n unified_env python first_stage_vessel_aware.py

Losses:
    1. MSE on GHD coefficients  (reconstruction)
    2. KL divergence             (regularisation)
    3. Vertex MSE                (mesh-space reconstruction via GHD_Reconstruct)
    4. Normal MSE                (mesh-space normal reconstruction)
    5. Ostium-plane loss         (opening ring lies on target plane)
    6. Vessel-penetration loss   (dome doesn't go below ostium plane)
    7. Laplacian + consistency   (mesh regularity)
"""
import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
import numpy as np

# project imports — avoid transitive relative-import chain by importing
# GHD base directly and constructing the lightweight reconstructor here
from models.vessel_aware_cvae import VesselAwareCVAE as VesselAwareCVAEV2
from models.vessel_aware_cvae_v8_resnet import VesselAwareCVAEV8ResNet
from models.vessel_conditioner import OstiumConditioner
from models.vessel_losses import IntrinsicPlaneLoss, IntrinsicPenetrationLoss, RingMatchLoss
from models.vae_datasets_vessel import VesselAwareGHDDataset
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


# ---- KL divergence (copied from vae_models to avoid import chain) ----
def KL_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def KL_divergence_terms(mu, logvar, prior_mu=None, prior_logvar=None,
                        free_bits: float = 0.0):
    """
    Return raw KL and train KL. free_bits is in nats per latent dimension.

    If a conditional prior is supplied, computes KL(q(z|x,c) || p(z|c));
    otherwise computes KL(q(z|x,c) || N(0,I)).
    """
    if prior_mu is None or prior_logvar is None:
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    else:
        prior_var = torch.exp(prior_logvar)
        post_var = torch.exp(logvar)
        kl_per_dim = 0.5 * (
            prior_logvar - logvar
            + (post_var + (mu - prior_mu).pow(2)) / (prior_var + 1e-8)
            - 1.0
        )
    kl_raw = kl_per_dim.mean()
    if free_bits > 0.0:
        kl_train = torch.clamp(kl_per_dim, min=free_bits).mean()
    else:
        kl_train = kl_raw
    return kl_raw, kl_train


# ---- Lightweight GHD Reconstructor ----
class LightGHDReconstruct:
    """
    Minimal GHD reconstructor that avoids the heavy relative-import chain in
    models/ghd_reconstruct.py.  Only provides what the training loop needs:
      - ghd_forward_as_Meshes(ghd_flat, mean, std) → Meshes
      - forward_verts(ghd_flat, mean, std)          → verts_padded [B,V,3]
    """
    def __init__(self, canonical_path: str, num_Basis: int, device: torch.device,
                 eigen_chk: str = None):
        from ghd.base.graph_harmonic_deformation import Graph_Harmonic_Deform
        canonical_Meshes = load_objs_as_meshes([canonical_path]).to(device)
        # Normalise using the SAME factor as GHD fitting (fitter.py line 709):
        #   norm = max_norm * 1.10
        # NOTE: models/ghd_reconstruct.py incorrectly uses * 1.10 * 2.50 — do NOT copy that.
        norm = torch.max(torch.norm(canonical_Meshes.verts_packed(), dim=-1)).item() * 1.10
        canonical_Meshes = canonical_Meshes.update_padded(canonical_Meshes.verts_padded() / norm)
        self.canonical = canonical_Meshes
        self.norm = norm
        # eigen_chk MUST match the one used in GHD fitting, otherwise the basis
        # vectors differ (eigsh is non-deterministic for sign/ordering of
        # degenerate eigenvalues) and the tokens decode to garbage.
        self.ghd = Graph_Harmonic_Deform(canonical_Meshes, num_Basis=num_Basis,
                                          eigen_chk=eigen_chk)
        self.eigvec = self.ghd.GBH_eigvec.to(device)  # [V, M]
        self.device = device

    def ghd_forward_as_Meshes(self, ghd_flat: torch.Tensor,
                               mean: torch.Tensor = None,
                               std: torch.Tensor = None) -> Meshes:
        """ghd_flat: [B, M*3]  → Meshes"""
        device = ghd_flat.device
        if mean is not None and std is not None:
            ghd_flat = ghd_flat * std.to(device) + mean.to(device)
        B = ghd_flat.shape[0]
        ghd = ghd_flat.reshape(B, -1, 3)
        verts_base = self.canonical.verts_padded().to(device)  # [1, V, 3]
        offset = torch.einsum('vm,bmc->bvc', self.eigvec.to(device), ghd)
        verts = verts_base + offset  # [B, V, 3]
        faces = self.canonical.faces_padded().to(device).expand(B, -1, -1)
        return Meshes(verts=verts, faces=faces)


def collate_fn(batch):
    """Custom collate for dict-based dataset."""
    return {
        'ghd':           torch.stack([b['ghd'] for b in batch]),
        'ostium_params': torch.stack([b['ostium_params'] for b in batch]),
        'vessel_pts':    torch.stack([b['vessel_pts'] for b in batch]),
    }


def parse_args():
    p = argparse.ArgumentParser("vessel_aware_cvae_training")
    # ---- paths ----
    p.add_argument('--ghd_chk_root', type=str,
                   default='/workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_roundrobin_v3')
    p.add_argument('--ghd_run', type=str, default='prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3')
    p.add_argument('--ghd_chk_name', type=str, default='ghb_fitting_checkpoint.pkl')
    p.add_argument('--canonical_mesh', type=str,
                   default='/workspace/AneuG/checkpoints/canonical_average/part_aligned.obj')
    p.add_argument('--eigen_chk', type=str,
                   default='/workspace/AneuG/checkpoints/canonical_average/eigen_chk_144.pkl',
                   help='Eigenvector checkpoint — MUST match the one used during GHD fitting.')
    p.add_argument('--opa_checkpoint', type=str,
                   default='/workspace/AneuG/checkpoints/canonical_average/opa_checkpoint_1op.pkl')
    p.add_argument('--data_root', type=str, default='/data/prepared_meshes_3',
                   help='Root of prepared_meshes_3 with vessel/ostium data')
    p.add_argument('--aligned_data_root', type=str,
                   default='/data/ghd_prepared_meshes_3_aneurysm_1op_new',
                   help='Root containing per-case prealign_transform.npy files')
    p.add_argument('--save_root', type=str,
                   default='./checkpoints/vessel_aware_cvae')
    p.add_argument('--meta', type=str, default='v5_cap_v6rr_v3_eigen_fixed')

    # ---- model ----
    p.add_argument('--num_Basis', type=int, default=144)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--latent_dim', type=int, default=64)
    p.add_argument('--vessel_cond_dim', type=int, default=32,
                   help='Output dim of OstiumConditioner')
    p.add_argument('--vessel_feat_dim', type=int, default=64,
                   help='PointNet feature dim for vessel points')
    p.add_argument('--num_vessel_pts', type=int, default=256)
    p.add_argument('--condition_space', type=str, default='ghd_local',
                   choices=['raw', 'ghd_local'],
                   help='Frame for vessel/ostium condition geometry')
    p.add_argument('--canonical_norm_factor', type=float, default=1.10,
                   help='Canonical mesh max-radius multiplier used by GHD fitting')
    p.add_argument('--dropout', type=float, default=0.02)
    p.add_argument('--model_type', type=str, default='v2',
                   choices=['v2', 'v8_resnet'],
                   help='v2 keeps the old checkpoint-compatible CVAE; v8_resnet uses deeper FiLM ResNet decoder')
    p.add_argument('--encoder_blocks', type=int, default=3,
                   help='Only used by --model_type v8_resnet')
    p.add_argument('--decoder_blocks', type=int, default=6,
                   help='Only used by --model_type v8_resnet')

    # ---- training ----
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--epochs', type=int, default=10000)
    p.add_argument('--batch_size', type=int, default=200,
                   help='>=num_samples to use full dataset per step')
    p.add_argument('--num_workers', type=int, default=2,
                   help='DataLoader workers; set 0 for sandbox/CPU smoke tests')
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--lr_step', type=int, default=2000)
    p.add_argument('--lr_gamma', type=float, default=0.5)
    p.add_argument('--grad_clip', type=float, default=1.0,
                   help='Max gradient norm (0 = disabled)')
    p.add_argument('--condition_dropout', type=float, default=0.0,
                   help='Per-sample probability to zero the condition during training')

    # ---- loss weights ----
    p.add_argument('--w_kl', type=float, default=0.00025,
                   help='Final KL weight (reached after warmup)')
    p.add_argument('--w_mse', type=float, default=1.0)
    p.add_argument('--w_vert', type=float, default=100.0)
    p.add_argument('--w_norm', type=float, default=10.0)
    p.add_argument('--w_plane', type=float, default=5000.0,
                   help='Weight for intrinsic ring-planarity loss')
    p.add_argument('--w_penetration', type=float, default=200.0,
                   help='Weight for intrinsic dome-penetration loss')
    p.add_argument('--w_ring', type=float, default=500.0,
                   help='Weight for ring-match loss (pred vs GT ring MSE)')
    p.add_argument('--w_lap', type=float, default=0.5)
    p.add_argument('--w_consistency', type=float, default=0.5)
    p.add_argument('--w_diversity', type=float, default=0.0,
                   help='Weight for diversity hinge loss on two stochastic decodes')
    p.add_argument('--diversity_target', type=float, default=0.35,
                   help='Target coeff-space RMSE between two samples under same condition')
    p.add_argument('--diversity_start', type=int, default=600,
                   help='Epoch when diversity hinge loss becomes active')

    # ---- scheduling ----
    p.add_argument('--ae_warmup', type=int, default=500,
                   help='Epochs of deterministic AE training (z=mu, no KL)')
    p.add_argument('--kl_warmup', type=int, default=2000,
                   help='Epochs of linear KL warmup after ae_warmup')
    p.add_argument('--kl_cap', type=float, default=20.0,
                   help='Cap raw KL loss at this value before weighting')
    p.add_argument('--free_bits', type=float, default=0.0,
                   help='Minimum KL in nats per latent dimension before weighting')
    p.add_argument('--use_conditional_prior', action='store_true',
                   help='Use p(z|condition) instead of standard normal prior when available')
    p.add_argument('--geo_phase_in', type=int, default=1000,
                   help='Epoch at which plane+penetration losses start')
    p.add_argument('--geo_ramp', type=int, default=1000,
                   help='Epochs to linearly ramp geo losses to full weight')

    # ---- misc ----
    p.add_argument('--log_freq', type=int, default=10)
    p.add_argument('--save_freq', type=int, default=1000)
    p.add_argument('--reload_epoch', type=int, default=None)
    p.add_argument('--use_wandb', action='store_true')
    p.add_argument('--wandb_project', type=str, default='vessel_aware_cvae')
    p.add_argument('--log_file', type=str, default=None,
                   help='Optional file to tee stdout/stderr into')

    return p.parse_args()


def main():
    args = parse_args()
    if args.log_file is not None:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_f = open(args.log_file, 'a', buffering=1)
        sys.stdout = TeeStream(sys.stdout, log_f)
        sys.stderr = TeeStream(sys.stderr, log_f)

    device = torch.device(args.device)
    log_path = os.path.join(args.save_root, args.meta)
    os.makedirs(log_path, exist_ok=True)

    # ---- wandb ----
    if args.use_wandb:
        import wandb
        wandb.login()
        wandb.init(project=args.wandb_project, name=args.meta, config=vars(args))

    # ================================================================
    # 1.  Data
    # ================================================================
    # Cases = every subdirectory in ghd_chk_root
    cases = [c for c in os.listdir(args.ghd_chk_root)
             if os.path.isdir(os.path.join(args.ghd_chk_root, c))]
    print(f"Found {len(cases)} GHD checkpoint directories")

    print("Building dataset …")
    dataset = VesselAwareGHDDataset(
        ghd_chk_root=args.ghd_chk_root,
        ghd_run=args.ghd_run,
        ghd_chk_name=args.ghd_chk_name,
        data_root=args.data_root,
        cases=cases,
        num_vessel_pts=args.num_vessel_pts,
        condition_space=args.condition_space,
        aligned_data_root=args.aligned_data_root,
        canonical_mesh=args.canonical_mesh,
        canonical_norm_factor=args.canonical_norm_factor,
        normalize=True,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, collate_fn=collate_fn,
                            num_workers=args.num_workers, drop_last=False)
    case_list_path = os.path.join(log_path, 'case_names.json')
    with open(case_list_path, 'w') as f:
        json.dump(dataset.case_names, f, indent=2)
    print(f"Saved case list → {case_list_path}")
    ghd_mean, ghd_std = dataset.get_mean_std()
    input_dim = dataset.get_dim()
    print(f"GHD input dim: {input_dim}  ({input_dim // 3} basis × 3)")

    # ---- GHD reconstructor (for mesh-space losses) ----
    print("Building LightGHDReconstruct …")
    ghd_reconstruct = LightGHDReconstruct(
        canonical_path=args.canonical_mesh,
        num_Basis=args.num_Basis,
        device=device,
        eigen_chk=args.eigen_chk,
    )

    # ---- Opening ring indices from canonical OPA checkpoint ----
    with open(args.opa_checkpoint, 'rb') as f:
        opa_chk = pickle.load(f)
    opening_ring_idx = torch.tensor(opa_chk['op_v_indices'][0], dtype=torch.long, device=device)
    print(f"Opening ring: {opening_ring_idx.shape[0]} vertices on canonical mesh")

    # ================================================================
    # 2.  Models
    # ================================================================
    conditioner = OstiumConditioner(
        vessel_feat_dim=args.vessel_feat_dim,
        ostium_plane_dim=8,
        ostium_feat_dim=16,
        cond_out_dim=args.vessel_cond_dim,
    ).to(device)

    if args.model_type == 'v8_resnet':
        generator = VesselAwareCVAEV8ResNet(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            vessel_cond_dim=args.vessel_cond_dim,
            extra_cond_dim=0,
            dropout=args.dropout,
            encoder_blocks=args.encoder_blocks,
            decoder_blocks=args.decoder_blocks,
        ).to(device)
    else:
        generator = VesselAwareCVAEV2(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            vessel_cond_dim=args.vessel_cond_dim,
            extra_cond_dim=0,
            dropout=args.dropout,
        ).to(device)

    # ---- losses (v2: intrinsic, no coordinate mismatch) ----
    plane_loss_fn = IntrinsicPlaneLoss().to(device)
    penetration_loss_fn = IntrinsicPenetrationLoss(margin=0.002).to(device)
    ring_match_fn = RingMatchLoss().to(device)

    # ---- optimiser ----
    all_params = list(generator.parameters()) + list(conditioner.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # ---- reload ----
    start_epoch = 0
    if args.reload_epoch is not None:
        chk_path = os.path.join(log_path, f'models_epoch_{args.reload_epoch}.pth')
        chk = torch.load(chk_path, map_location=device)
        generator.load_state_dict(chk['generator'])
        conditioner.load_state_dict(chk['conditioner'])
        optimizer.load_state_dict(chk['optimizer'])
        start_epoch = chk['epoch'] + 1
        print(f"Reloaded from epoch {chk['epoch']}")

    # ================================================================
    # 3.  Training loop
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs, {len(dataset)} samples")
    print(f"  Model {args.model_type} | hidden {args.hidden_dim} | latent {args.latent_dim}")
    print(f"  Condition space {args.condition_space} | vessel pts {args.num_vessel_pts}")
    print(f"  AE warmup {args.ae_warmup} ep | KL warmup {args.kl_warmup} ep | KL cap {args.kl_cap}")
    print(f"  Free bits {args.free_bits} | cond dropout {args.condition_dropout} | conditional prior {args.use_conditional_prior}")
    print(f"  Geo losses phase-in @ {args.geo_phase_in}, ramp {args.geo_ramp} ep")
    print(f"  Diversity @ {args.diversity_start}: w={args.w_diversity}, target_rmse={args.diversity_target}")
    print(f"  Grad clip {args.grad_clip}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        for batch in dataloader:
            ghd = batch['ghd'].to(device)                    # [B, D]
            ostium_params = batch['ostium_params'].to(device) # [B, 8]
            vessel_pts = batch['vessel_pts'].to(device)       # [B, N, 3]
            B = ghd.shape[0]

            optimizer.zero_grad()

            # ---- condition ----
            cond = conditioner(vessel_pts, ostium_params)     # [B, cond_dim]
            if args.condition_dropout > 0.0:
                keep = (torch.rand(B, 1, device=device) >= args.condition_dropout).float()
                cond = cond * keep

            # ---- forward ----
            use_det = (epoch < args.ae_warmup)  # AE warmup: z=mu
            ghd_recon, mu, logvar = generator(ghd, cond, deterministic=use_det)

            prior_mu, prior_logvar = None, None
            if args.use_conditional_prior:
                if not hasattr(generator, 'prior'):
                    raise ValueError("--use_conditional_prior requires a model with prior(cond)")
                prior_mu, prior_logvar = generator.prior(cond)

            # ---- diversity regularization (same condition, different z) ----
            # Encourage stochastic samples to spread under identical conditioning.
            if args.w_diversity > 0.0 and epoch >= args.diversity_start:
                if args.use_conditional_prior:
                    z1 = generator.reparameterize(prior_mu, prior_logvar)
                    z2 = generator.reparameterize(prior_mu, prior_logvar)
                else:
                    z1 = torch.randn(B, args.latent_dim, device=device)
                    z2 = torch.randn(B, args.latent_dim, device=device)
                ghd_s1 = generator.decode(z1, cond)
                ghd_s2 = generator.decode(z2, cond)
                pair_rmse = torch.sqrt(torch.mean((ghd_s1 - ghd_s2) ** 2, dim=1) + 1e-8)
                diversity_rmse = pair_rmse.mean()
                diversity_loss = F.relu(args.diversity_target - diversity_rmse)
            else:
                diversity_rmse = ghd.new_tensor(0.0)
                diversity_loss = ghd.new_tensor(0.0)

            # ---- GHD-space losses ----
            mse_loss = F.mse_loss(ghd_recon, ghd)
            kl_raw, kl_train = KL_divergence_terms(
                mu, logvar,
                prior_mu=prior_mu,
                prior_logvar=prior_logvar,
                free_bits=args.free_bits,
            )
            kl_loss = torch.clamp(kl_train, max=args.kl_cap)   # cap

            # ---- KL weight schedule ----
            # Phase 1: AE warmup (0 → ae_warmup) → kl_w = 0
            # Phase 2: linear ramp (ae_warmup → ae_warmup+kl_warmup) → 0 → w_kl
            # Phase 3: full weight
            if epoch < args.ae_warmup:
                kl_w = 0.0
            elif args.kl_warmup > 0 and epoch < args.ae_warmup + args.kl_warmup:
                kl_w = args.w_kl * ((epoch - args.ae_warmup) / args.kl_warmup)
            else:
                kl_w = args.w_kl

            # ---- mesh-space losses ----
            real_meshes  = ghd_reconstruct.ghd_forward_as_Meshes(ghd,      mean=ghd_mean, std=ghd_std)
            recon_meshes = ghd_reconstruct.ghd_forward_as_Meshes(ghd_recon, mean=ghd_mean, std=ghd_std)

            vert_loss = F.mse_loss(recon_meshes.verts_padded(), real_meshes.verts_padded())
            norm_loss = F.mse_loss(recon_meshes.verts_normals_padded(),
                                   real_meshes.verts_normals_padded())

            # ---- mesh regularisation ----
            loss_lap = mesh_laplacian_smoothing(recon_meshes, method="cot")
            loss_consistency = mesh_normal_consistency(recon_meshes)

            # ---- vessel-aware losses (intrinsic, phased in) ----
            pred_verts = recon_meshes.verts_padded()   # [B, V, 3]
            gt_verts   = real_meshes.verts_padded()    # [B, V, 3]

            loss_plane = plane_loss_fn(pred_verts, opening_ring_idx)
            loss_penetration = penetration_loss_fn(
                pred_verts, gt_verts, opening_ring_idx
            )
            loss_ring = ring_match_fn(pred_verts, gt_verts, opening_ring_idx)

            # Geo ramp: 0→1 over [geo_phase_in, geo_phase_in+geo_ramp]
            if epoch < args.geo_phase_in:
                geo_alpha = 0.0
            elif epoch < args.geo_phase_in + args.geo_ramp:
                geo_alpha = (epoch - args.geo_phase_in) / args.geo_ramp
            else:
                geo_alpha = 1.0

            # ---- total loss ----
            loss = (
                args.w_mse * mse_loss
                + kl_w * kl_loss
                + args.w_vert * vert_loss
                + args.w_norm * norm_loss
                + args.w_lap * loss_lap
                + args.w_consistency * loss_consistency
                + geo_alpha * args.w_plane * loss_plane
                + geo_alpha * args.w_penetration * loss_penetration
                + geo_alpha * args.w_ring * loss_ring
                + args.w_diversity * diversity_loss
            )

            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(all_params, args.grad_clip)
            optimizer.step()

        # ---- logging ----
        if epoch % args.log_freq == 0:
            log_dict = {
                "epoch": epoch,
                "total": loss.item(),
                "mse": mse_loss.item(),
                "kl_raw": kl_raw.item(),
                "kl_train": kl_train.item(),
                "kl_capped": kl_loss.item(),
                "kl_w": kl_w,
                "vert": vert_loss.item(),
                "norm": norm_loss.item(),
                "lap": loss_lap.item(),
                "consistency": loss_consistency.item(),
                "plane": loss_plane.item(),
                "penetration": loss_penetration.item(),
                "ring_match": loss_ring.item(),
                "div_rmse": diversity_rmse.item(),
                "div_hinge": diversity_loss.item(),
                "geo_alpha": geo_alpha,
                "lr": optimizer.param_groups[0]['lr'],
            }
            print(log_dict)
            if args.use_wandb:
                import wandb
                wandb.log({k: v for k, v in log_dict.items() if k != "epoch"}, step=epoch)

        # ---- save ----
        if (epoch % args.save_freq == 0 and epoch > 0) or epoch == args.epochs:
            save_path = os.path.join(log_path, f'models_epoch_{epoch}.pth')
            torch.save({
                'generator': generator.state_dict(),
                'conditioner': conditioner.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': vars(args),
                'ghd_mean': ghd_mean,
                'ghd_std': ghd_std,
                'ostium_mean': dataset.ostium_mean,
                'ostium_std': dataset.ostium_std,
                'vessel_center': dataset.vessel_center,
                'vessel_scale': dataset.vessel_scale,
                'case_names': dataset.case_names,
            }, save_path)
            print(f"Saved checkpoint → {save_path}")

        scheduler.step()

    print("\nTraining finished.")
    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
