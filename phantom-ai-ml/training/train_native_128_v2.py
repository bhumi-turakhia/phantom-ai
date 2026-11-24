"""
Phantom AI — Native 128x128 Conditional WGAN-GP, corrected training strategy (v2).

NEW FILE. Does not modify or overwrite any existing script or checkpoint.
All outputs go to phantom-ai-ml/runs/native128_v2/ (checkpoints, grids, metrics).

Fixes vs train_native_generator_128.py (root causes found in the quality audit):
  1. n_critic = 5 critic updates per generator update (was 1).
  2. Far longer training (default 30,000 generator steps, batch 64 = ~14.8 epochs of Split A1).
  3. Checkpoint selection by held-out C-FD on EMA generator (was abs(val W1) + forced last step).
  4. Generator EMA (0.999), evaluated and selected instead of the raw noisy generator.
  5. Upsample+conv generator option (removes ConvTranspose checkerboard), wider 128px stage.
  6. Fresh critic and generator (no warm start from earlier blurry runs).
  7. Regular visual grids: fixed-profile grid, one-profile-many-seeds grid, real reference grid.

Modes:
  --mode benchmark  : times a few steps, writes NOTHING to disk except stdout.
  --mode baseline   : computes C-FD / FAV of the existing native checkpoint with the SAME
                      protocol used here (read-only), plus a real-vs-real C-FD floor.
  --mode train      : full run (only when explicitly launched by the user).

Run from repo root:
  PYTHONPATH=phantom-ai-ml phantom-ai-ml/.venv/bin/python phantom-ai-ml/training/train_native_128_v2.py --mode benchmark
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from config.gan_config import get_device, set_reproducibility_seed
from data.celeba_dataset import CelebAIdentityDataset
from models.forensic_evaluator import ForensicFeatureCNN
from models.wgan_gp_model import (
    NativeConditionalWGANGPGenerator128,
    ProjectionCritic,
    weights_init_wgan,
)
from training.wgan_objectives import (
    compute_gradient_penalty,
    compute_masked_auxiliary_loss,
    compute_wasserstein_critic_loss,
    compute_wasserstein_generator_loss,
)
from training.train_native_generator_128 import create_fixed_evaluation_profiles
from utils.generative_eval_metrics import compute_cfd_from_features, extract_evaluator_features

IMAGE_DIR = "phantom-ai-ml/datasets/celeba/img_align_celeba"
EVALUATOR_PATH = "phantom-ai-ml/checkpoints/forensic_evaluator_split_b.pt"
OLD_NATIVE_PATH = "phantom-ai-ml/checkpoints/cgan_generator_128_native.pt"
RUN_DIR = "phantom-ai-ml/runs/native128_v2"


# --------------------------------------------------------------------------- config
@dataclass
class Config:
    arch: str = "upsample_conv"      # "upsample_conv" (recommended) | "convt" (existing architecture)
    total_g_steps: int = 30000
    batch_size: int = 64
    n_critic: int = 5
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    betas: Tuple[float, float] = (0.0, 0.9)
    lambda_gp: float = 10.0
    lambda_aux_d: float = 1.0
    lambda_aux_g: float = 0.5
    drift_eps: float = 1e-3
    ema_decay: float = 0.999
    attribute_dropout: float = 0.20
    latent_dim: int = 100
    gen_features: int = 64           # base width; upsample_conv uses 8x at 4x4
    grad_clip: float = 10.0
    num_workers: int = 0
    eval_interval: int = 1000        # C-FD / FAV evaluation + best-checkpoint decision
    grid_interval: int = 500         # visual grids
    snapshot_interval: int = 5000    # permanent EMA snapshots
    eval_samples: int = 512
    min_best_step: int = 3000        # best-checkpoint guard: ignore very early lucky evals
    max_hours: float = 0.0           # 0 = no wall-clock cap
    seed: int = 42


# --------------------------------------------------------------------------- generator
class UpBlock(nn.Module):
    """nearest-upsample -> conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> ReLU (no checkerboard)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class UpsampleConvGenerator128(nn.Module):
    """z(100)+cond(20) -> 4x4x(8f) -> 5 UpBlocks -> 128x128x3. Same call signature as the old generator."""

    def __init__(self, latent_dim=100, condition_dim=20, embed_dim=64, f=64):
        super().__init__()
        self.latent_dim, self.condition_dim = latent_dim, condition_dim
        self.condition_embed = nn.Sequential(nn.Linear(condition_dim, embed_dim), nn.LeakyReLU(0.2, True))
        self.fc = nn.Linear(latent_dim + embed_dim, 8 * f * 4 * 4, bias=False)
        self.bn0 = nn.BatchNorm2d(8 * f)
        self.f = f
        widths = [8 * f, 8 * f, 4 * f, 2 * f, f, f // 2 * 1]
        # 4->8->16->32->64->128 : five blocks
        self.blocks = nn.Sequential(*[UpBlock(widths[i], widths[i + 1]) for i in range(5)])
        self.head = nn.Sequential(nn.Conv2d(widths[-1], 3, 3, 1, 1), nn.Tanh())
        self.apply(weights_init_wgan)

    def forward(self, noise, condition, mask=None):
        c = condition * mask if mask is not None else condition
        h = torch.cat([noise, self.condition_embed(c)], dim=1)
        h = F.relu(self.bn0(self.fc(h).view(h.size(0), 8 * self.f, 4, 4)))
        return self.head(self.blocks(h))


def build_generator(cfg: Config) -> nn.Module:
    if cfg.arch == "upsample_conv":
        return UpsampleConvGenerator128(cfg.latent_dim, 20, 64, cfg.gen_features)
    return NativeConditionalWGANGPGenerator128(
        latent_dim=cfg.latent_dim, condition_dim=20, condition_embed_dim=64,
        feature_maps=cfg.gen_features, channels=3, image_size=128)


@torch.no_grad()
def ema_update(ema: nn.Module, model: nn.Module, decay: float):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


# --------------------------------------------------------------------------- helpers
def to_uint8_grid(x: torch.Tensor, cols: int) -> np.ndarray:
    a = np.clip((x.detach().cpu().numpy() + 1) * 127.5, 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)
    rows = [np.concatenate(list(a[i:i + cols]), axis=1) for i in range(0, len(a), cols)]
    return np.concatenate(rows, axis=0)


def save_grid(x: torch.Tensor, path: str, cols: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(to_uint8_grid(x, cols)).save(path)


def load_evaluator(device) -> ForensicFeatureCNN:
    ev = ForensicFeatureCNN().to(device)
    ck = torch.load(EVALUATOR_PATH, map_location=device, weights_only=False)
    ev.load_state_dict(ck.get("evaluator_state_dict") or ck["model_state_dict"], strict=True)
    return ev.eval()


def laplacian_hf(x: torch.Tensor) -> float:
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    return F.conv2d(x.mean(1, keepdim=True), k, padding=1).abs().mean().item()


class EvalSet:
    """Fixed held-out Split A2 subset: real images/conditions, cached real evaluator features."""

    def __init__(self, n: int, device, evaluator):
        ds = CelebAIdentityDataset(split="split_a2", image_dir=IMAGE_DIR, image_size=128,
                                   attribute_dropout=0.0, require_images=True)
        idx = np.random.RandomState(1234).choice(len(ds), size=n, replace=False)
        items = [ds[int(i)] for i in idx]
        self.real = torch.stack([it["image"] for it in items])
        self.cond = torch.stack([it["condition"] for it in items])
        self.mask = torch.stack([it["mask"] for it in items])
        self.evaluator, self.device = evaluator, device
        self.real_feats = extract_evaluator_features(evaluator, self.real, device=str(device))
        self.real_hf = laplacian_hf(self.real)
        self.z = torch.randn(n, 100, generator=torch.Generator().manual_seed(2024))

    @torch.no_grad()
    def evaluate(self, gen: nn.Module) -> Dict[str, float]:
        gen.eval()
        fakes = []
        for i in range(0, len(self.z), 128):
            fakes.append(gen(self.z[i:i + 128].to(self.device), self.cond[i:i + 128].to(self.device),
                             self.mask[i:i + 128].to(self.device)).cpu())
        fakes = torch.cat(fakes)
        feats = extract_evaluator_features(self.evaluator, fakes, device=str(self.device))
        cfd = compute_cfd_from_features(self.real_feats, feats)
        logits = []
        for i in range(0, len(fakes), 128):
            f64 = F.interpolate(fakes[i:i + 128].to(self.device), size=(64, 64), mode="bilinear", align_corners=False)
            logits.append(self.evaluator(f64)[0].cpu())
        logits = torch.cat(logits)
        obs = self.mask > 0
        fav = ((logits > 0) == (self.cond > 0))[obs].float().mean().item()
        return {
            "cfd": float(cfd), "fav_acc": float(fav),
            "hf_ratio": laplacian_hf(fakes) / max(self.real_hf, 1e-9),  # diagnostic only; ~1.0 is realistic
            "pixel_std": float(fakes.std()),
        }


def make_loader(cfg: Config) -> DataLoader:
    ds = CelebAIdentityDataset(split="split_a1", image_dir=IMAGE_DIR, image_size=128,
                               attribute_dropout=cfg.attribute_dropout, require_images=True)
    kw = dict(num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0) if cfg.num_workers else {}
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **kw)


class Cycler:
    def __init__(self, loader):
        self.loader, self.it = loader, iter(loader)

    def next(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)


def get_memory_mb(device: torch.device) -> float:
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        try:
            return float(torch.mps.current_allocated_memory() / (1024 * 1024))
        except Exception:
            return 0.0
    elif device.type == "cuda":
        return float(torch.cuda.memory_allocated(device) / (1024 * 1024))
    return 0.0


def critic_step(critic, gen, opt_d, batch, cfg, device):
    real = batch["image"].to(device)
    real = torch.where(torch.rand(real.size(0), 1, 1, 1, device=device) < 0.5, real.flip(3), real)  # h-flip aug
    c, m = batch["condition"].to(device), batch["mask"].to(device)
    with torch.no_grad():
        fake = gen(torch.randn(real.size(0), cfg.latent_dim, device=device), c, m)
    r, r_aux = critic(real, c, m, return_aux=True)
    f, _ = critic(fake, c, m)
    wass = compute_wasserstein_critic_loss(r, f)
    gp, gnorm = compute_gradient_penalty(critic, real, fake, c, m, lambda_gp=cfg.lambda_gp)
    aux = compute_masked_auxiliary_loss(r_aux, c, m)
    drift = cfg.drift_eps * (r ** 2).mean()
    loss = wass + gp + cfg.lambda_aux_d * aux + drift

    if torch.isnan(loss) or torch.isinf(loss):
        raise RuntimeError(f"NaN/Inf detected in Critic loss: wass={wass.item()}, gp={gp.item()}, aux={aux.item()}")

    opt_d.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.grad_clip)
    opt_d.step()
    return {"w1": float(-wass.detach()), "gp": float(gp.detach()), "gnorm": float(gnorm.detach()), "d_aux": float(aux.detach())}


def generator_step(critic, gen, opt_g, batch, cfg, device):
    c, m = batch["condition"].to(device), batch["mask"].to(device)
    fake = gen(torch.randn(c.size(0), cfg.latent_dim, device=device), c, m)
    f, f_aux = critic(fake, c, m, return_aux=True)
    loss = compute_wasserstein_generator_loss(f) + cfg.lambda_aux_g * compute_masked_auxiliary_loss(f_aux, c, m)

    if torch.isnan(loss) or torch.isinf(loss):
        raise RuntimeError(f"NaN/Inf detected in Generator loss: {loss.item()}")

    opt_g.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gen.parameters(), cfg.grad_clip)
    opt_g.step()
    return float(loss.detach())


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


# --------------------------------------------------------------------------- modes
def run_benchmark(cfg: Config, steps: int = 10):
    device = get_device("auto")
    set_reproducibility_seed(cfg.seed)
    gen, critic = build_generator(cfg).to(device), ProjectionCritic(3, 20, 64, 512, 128).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=cfg.lr_g, betas=cfg.betas)
    opt_d = torch.optim.Adam(critic.parameters(), lr=cfg.lr_d, betas=cfg.betas)
    cyc = Cycler(make_loader(cfg))
    n_params = sum(p.numel() for p in gen.parameters())
    print(f"device={device} arch={cfg.arch} gen_params={n_params:,} batch={cfg.batch_size} n_critic={cfg.n_critic}")
    times = []
    for s in range(steps + 2):
        sync(device); t = time.time()
        for _ in range(cfg.n_critic):
            critic_step(critic, gen, opt_d, cyc.next(), cfg, device)
        generator_step(critic, gen, opt_g, cyc.next(), cfg, device)
        sync(device)
        if s >= 2:  # discard 2 warm-up steps
            times.append(time.time() - t)
    sec = float(np.median(times))
    print(f"median sec / generator-step (incl. {cfg.n_critic} critic steps): {sec:.2f}s  "
          f"(min {min(times):.2f}, max {max(times):.2f})")
    for steps_total in (10000, 20000, 30000, 50000):
        print(f"  {steps_total:6d} G-steps -> {steps_total * sec / 3600:5.1f} h  "
              f"({steps_total * cfg.batch_size * cfg.n_critic / 130063:.1f} critic-epochs)")
    print("Benchmark wrote no files.")


def run_baseline(cfg: Config):
    device = get_device("auto")
    ev = load_evaluator(device)
    es = EvalSet(cfg.eval_samples, device, ev)
    gen = NativeConditionalWGANGPGenerator128().to(device)
    gen.load_state_dict(torch.load(OLD_NATIVE_PATH, map_location=device, weights_only=False)["generator_state_dict"])
    old = es.evaluate(gen)
    half = len(es.real) // 2
    floor = compute_cfd_from_features(es.real_feats[:half], es.real_feats[half:])
    print(json.dumps({"existing_native_checkpoint": old, "real_vs_real_cfd_floor_half_split": floor,
                      "n_eval": len(es.real), "protocol": "Split A2 subset, evaluator_split_b, 64px bilinear"}, indent=2))
    print("Baseline is read-only; wrote no files.")


def run_train(cfg: Config, resume: bool):
    device = get_device("auto")
    set_reproducibility_seed(cfg.seed)
    os.makedirs(RUN_DIR, exist_ok=True)
    for d in ("grids", "checkpoints"):
        os.makedirs(os.path.join(RUN_DIR, d), exist_ok=True)

    gen = build_generator(cfg).to(device)
    ema = copy.deepcopy(gen).eval()
    critic = ProjectionCritic(3, 20, 64, 512, 128).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=cfg.lr_g, betas=cfg.betas)
    opt_d = torch.optim.Adam(critic.parameters(), lr=cfg.lr_d, betas=cfg.betas)

    step, best_cfd, history = 0, float("inf"), []
    last_path = os.path.join(RUN_DIR, "checkpoints", "last.pt")
    if resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        gen.load_state_dict(ck["gen"]); ema.load_state_dict(ck["ema"]); critic.load_state_dict(ck["critic"])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
        step, best_cfd, history = ck["step"], ck["best_cfd"], ck["history"]
        print(f"[RESUME] step {step}, best C-FD {best_cfd:.3f}")

    evaluator = load_evaluator(device)
    es = EvalSet(cfg.eval_samples, device, evaluator)
    prof_c, prof_m, prof_labels = create_fixed_evaluation_profiles(device)
    prof_z = torch.randn(16, cfg.latent_dim, generator=torch.Generator().manual_seed(999)).to(device)
    seed_z = torch.randn(16, cfg.latent_dim, generator=torch.Generator().manual_seed(555)).to(device)
    save_grid(es.real[:16], os.path.join(RUN_DIR, "grids", "real_reference.png"), 8)
    with open(os.path.join(RUN_DIR, "config.json"), "w") as f:
        json.dump({**asdict(cfg), "profile_labels": prof_labels}, f, indent=2)

    cyc = Cycler(make_loader(cfg))
    t0 = time.time()
    worse_streak, last_cfd = 0, None
    step_times = []
    log_records = []
    print(f"[START] arch={cfg.arch} steps={cfg.total_g_steps} bs={cfg.batch_size} n_critic={cfg.n_critic} device={device}")

    while step < cfg.total_g_steps:
        t_step_start = time.time()
        gen.train(); critic.train()
        for _ in range(cfg.n_critic):
            dm = critic_step(critic, gen, opt_d, cyc.next(), cfg, device)
        g_loss = generator_step(critic, gen, opt_g, cyc.next(), cfg, device)
        ema_update(ema, gen, cfg.ema_decay)
        sync(device)
        step_dur = time.time() - t_step_start
        step_times.append(step_dur)
        mem_mb = get_memory_mb(device)
        step += 1

        # Check for NaN / Inf in outputs
        if math.isnan(g_loss) or math.isnan(dm["w1"]):
            raise RuntimeError(f"NaN loss encountered at step {step}!")

        log_entry = {
            "step": step,
            "w1": round(dm["w1"], 4),
            "gp": round(dm["gp"], 4),
            "gnorm": round(dm["gnorm"], 4),
            "g_loss": round(g_loss, 4),
            "d_aux": round(dm["d_aux"], 4),
            "step_time_s": round(step_dur, 3),
            "mps_mem_mb": round(mem_mb, 1),
            "elapsed_min": round((time.time() - t0) / 60, 2),
        }
        log_records.append(log_entry)

        if step % 25 == 0 or step == 1 or step == cfg.total_g_steps:
            el = time.time() - t0
            avg_step = sum(step_times[-25:]) / len(step_times[-25:])
            print(f"[{step:4d}/{cfg.total_g_steps}] W1 {dm['w1']:+.3f} | GP {dm['gp']:.3f} | |grad| {dm['gnorm']:.2f} | "
                  f"G {g_loss:+.3f} | auxD {dm['d_aux']:.3f} | Step: {step_dur:.2f}s (avg: {avg_step:.2f}s) | Mem: {mem_mb:.1f} MB | {el/60:.1f}m", flush=True)

        if step % cfg.grid_interval == 0 or step == 1 or step == cfg.total_g_steps:
            ema.eval()
            with torch.no_grad():
                prof_fakes = ema(prof_z, prof_c, prof_m)
                save_grid(prof_fakes, os.path.join(RUN_DIR, "grids", f"profiles_step_{step:06d}.png"), 4)
                one_c, one_m = prof_c[8:9].repeat(16, 1), prof_m[8:9].repeat(16, 1)
                seed_fakes = ema(seed_z, one_c, one_m)
                save_grid(seed_fakes, os.path.join(RUN_DIR, "grids", f"seeds_step_{step:06d}.png"), 8)

                # Save comparison grid against real held-out CelebA samples
                real_vs_fake = torch.cat([es.real[:8], prof_fakes[:8].cpu()], dim=0)
                save_grid(real_vs_fake, os.path.join(RUN_DIR, "grids", f"real_vs_fake_step_{step:06d}.png"), 8)

        if step % 50 == 0:
            payload = {"step": step, "gen": gen.state_dict(), "ema": ema.state_dict(),
                       "critic": critic.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                       "best_cfd": best_cfd, "history": history, "config": asdict(cfg), "arch": cfg.arch}
            torch.save(payload, last_path)

        hit_cap = cfg.max_hours > 0 and (time.time() - t0) / 3600 >= cfg.max_hours
        if step % cfg.eval_interval == 0 or step == cfg.total_g_steps or hit_cap:
            m = es.evaluate(ema)
            rec = {"step": step, "minutes": round((time.time() - t0) / 60, 1), "w1_train": dm["w1"], **m}
            history.append(rec)
            if last_cfd is not None:
                worse_streak = worse_streak + 1 if m["cfd"] > last_cfd else 0
            last_cfd = m["cfd"]
            is_best = step >= cfg.min_best_step and m["cfd"] < best_cfd
            if is_best:
                best_cfd = m["cfd"]
            payload = {"step": step, "gen": gen.state_dict(), "ema": ema.state_dict(),
                       "critic": critic.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                       "best_cfd": best_cfd, "history": history, "config": asdict(cfg), "arch": cfg.arch}
            torch.save(payload, last_path)
            slim = {"step": step, "arch": cfg.arch, "generator_state_dict": ema.state_dict(),
                    "cfd": m["cfd"], "fav_acc": m["fav_acc"], "config": asdict(cfg)}
            if is_best:
                torch.save(slim, os.path.join(RUN_DIR, "checkpoints", "best_ema.pt"))
            if step % cfg.snapshot_interval == 0 or step == cfg.total_g_steps:
                torch.save(slim, os.path.join(RUN_DIR, "checkpoints", f"ema_step_{step:06d}.pt"))

            pilot_summary = {
                "total_steps": step,
                "best_cfd": best_cfd,
                "final_eval": m,
                "history": history,
                "step_times": {
                    "median_s": round(float(np.median(step_times)), 3),
                    "mean_s": round(float(np.mean(step_times)), 3),
                    "min_s": round(float(np.min(step_times)), 3),
                    "max_s": round(float(np.max(step_times)), 3),
                },
                "peak_mps_mem_mb": max(r["mps_mem_mb"] for r in log_records) if log_records else 0.0,
                "log_records": log_records,
            }
            with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
                json.dump(pilot_summary, f, indent=2)

            flag = " <-- BEST" if is_best else ""
            warn = "  [WARN: C-FD rising 3 evals in a row]" if worse_streak >= 3 else ""
            print(f"[EVAL {step}] C-FD {m['cfd']:.2f} | FAV {m['fav_acc']*100:.1f}% | hf_ratio {m['hf_ratio']:.2f} | "
                  f"std {m['pixel_std']:.3f}{flag}{warn}", flush=True)
            if hit_cap:
                print("[STOP] wall-clock cap reached; resume with --resume"); break
    print(f"[DONE] 500-step pilot complete! Total time: {(time.time() - t0)/60:.2f} min. Final C-FD: {history[-1]['cfd'] if history else 'N/A'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["benchmark", "baseline", "train"], default="benchmark")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--bench_steps", type=int, default=10)
    for k, v in asdict(Config()).items():
        if isinstance(v, tuple):
            continue
        p.add_argument(f"--{k}", type=type(v), default=v)
    a = p.parse_args()
    cfg = Config(**{k: getattr(a, k) for k in asdict(Config()) if k != "betas"})
    if a.mode == "benchmark":
        run_benchmark(cfg, a.bench_steps)
    elif a.mode == "baseline":
        run_baseline(cfg)
    else:
        run_train(cfg, a.resume)


if __name__ == "__main__":
    main()

