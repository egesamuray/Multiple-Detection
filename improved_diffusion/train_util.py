# improved_diffusion/train_util.py
import os, time, copy
from typing import Dict, Any, Tuple, Optional
import torch, torch.nn as nn, torch.optim as optim

from . import dist_util as dist
from . import logger


class ModelEMA(nn.Module):
    """EMA wrapper for a model."""
    def __init__(self, model: nn.Module, decay: float = 0.9999, device=None):
        super().__init__()
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).to(device or next(model.parameters()).device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for p_ema, p in zip(self.ema.parameters(), model.parameters()):
            p_ema.data.mul_(d).add_(p.data, alpha=1.0 - d)
        for b_ema, b in zip(self.ema.buffers(), model.buffers()):
            b_ema.copy_(b)

    def state_dict(self, *a, **k):
        return self.ema.state_dict(*a, **k)

    def to(self, *a, **k):
        self.ema.to(*a, **k)
        return self


def _mean_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.float().mean()


def _extract_scalar_loss(loss_out: Any, device: torch.device) -> torch.Tensor:
    """
    Make a scalar loss from various outputs returned by diffusion.training_losses:
      - Tensor:                  return mean
      - dict with keys:          'loss', 'mse', 'loss_simple', 'total', 'vb', 'losses'
      - list/tuple of Tensors:   average their means
      - otherwise:               convert to tensor(float)
    """
    if torch.is_tensor(loss_out):
        return _mean_tensor(loss_out)

    if isinstance(loss_out, dict):
        # priority order of common keys
        for key in ("loss", "mse", "loss_simple", "total", "vb"):
            if key in loss_out and torch.is_tensor(loss_out[key]):
                return _mean_tensor(loss_out[key])
        # 'losses' could be a list/tuple of tensors
        if "losses" in loss_out and isinstance(loss_out["losses"], (list, tuple)):
            vals = [v for v in loss_out["losses"] if torch.is_tensor(v)]
            if vals:
                return torch.stack([_mean_tensor(v) for v in vals]).mean()
        # fallback: first tensor value in dict (even nested one level)
        for v in loss_out.values():
            if torch.is_tensor(v):
                return _mean_tensor(v)
            if isinstance(v, dict):
                for vv in v.values():
                    if torch.is_tensor(vv):
                        return _mean_tensor(vv)
        return torch.as_tensor(0.0, device=device)

    if isinstance(loss_out, (list, tuple)):
        vals = [torch.as_tensor(v, device=device) for v in loss_out]
        return torch.stack([_mean_tensor(v) for v in vals]).mean()

    try:
        return torch.as_tensor(float(loss_out), device=device)
    except Exception:
        return torch.as_tensor(0.0, device=device)


class TrainLoop:
    """
    Colab-friendly training loop:
      * no mandatory DDP; wraps in DDP only if a process group exists
      * robust AMP with torch.amp
      * robust loss extraction
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        diffusion: Any,
        data,
        batch_size: int,
        microbatch: int = -1,
        lr: float = 1e-4,
        ema_rate: Any = 0.9999,
        log_interval: int = 10,
        save_interval: int = 10000,
        resume_checkpoint: str = "",
        use_fp16: bool = False,
        fp16_scale_growth: float = 1e-3,
        weight_decay: float = 0.0,
        lr_anneal_steps: int = 0,
        max_training_steps: int = 0,
        **kwargs,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data

        self.batch_size = int(batch_size)
        self.microbatch = int(microbatch)
        self.lr = float(lr)

        self.log_interval = int(log_interval)
        self.save_interval = int(save_interval)
        self.resume_checkpoint = resume_checkpoint

        self.use_fp16 = bool(use_fp16)
        self.fp16_scale_growth = float(fp16_scale_growth)
        self.weight_decay = float(weight_decay)
        self.lr_anneal_steps = int(lr_anneal_steps)
        self.max_training_steps = int(max_training_steps) if int(max_training_steps) > 0 else None

        self.device = dist.dev()
        self.model.to(self.device)

        # Optional DDP if a process group is initialized
        self.ddp = False
        try:
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1:
                from torch.nn.parallel import DistributedDataParallel as DDP
                self.model = DDP(self.model, device_ids=[self.device.index] if self.device.type == "cuda" else None)
                self.ddp = True
        except Exception:
            self.ddp = False

        # Optimizer
        if self.weight_decay > 0.0:
            self.opt = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.opt = optim.Adam(self.model.parameters(), lr=self.lr)

        # AMP
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_fp16)

        # EMA on the (unwrapped) model copy
        self.ema = ModelEMA(self.model.module if self.ddp else self.model, decay=float(ema_rate), device=self.device)

        self.step = 0
        self.run_dir = logger.get_dir() or "."
        os.makedirs(self.run_dir, exist_ok=True)

        # Optional resume
        if self.resume_checkpoint and os.path.isfile(self.resume_checkpoint):
            try:
                state = torch.load(self.resume_checkpoint, map_location="cpu")
                (self.model.module if self.ddp else self.model).load_state_dict(state)
                logger.log(f"Resumed from {self.resume_checkpoint}")
            except Exception as e:
                logger.log(f"Resume failed: {e}")

    # ----- helpers -----
    def _next_batch(self):
        x, kw = next(self.data)
        x = x.to(self.device, non_blocking=True) if isinstance(x, torch.Tensor) else torch.as_tensor(x, device=self.device)
        mkw = {}
        if isinstance(kw, dict):
            for k, v in kw.items():
                mkw[k] = v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else torch.as_tensor(v, device=self.device)
        return x, mkw

    def _iter_micro(self, x, kw):
        B = x.shape[0]
        mb = self.batch_size if self.microbatch <= 0 else self.microbatch
        for s in range(0, B, mb):
            x_mb = x[s:s+mb]
            kw_mb = {k: (v[s:s+mb] if isinstance(v, torch.Tensor) and v.shape[:1] == (B,) else v) for k, v in kw.items()}
            yield x_mb, kw_mb

    def _lr_now(self) -> float:
        if self.lr_anneal_steps and self.step < self.lr_anneal_steps:
            return max(1e-8, self.lr * (1.0 - self.step / float(self.lr_anneal_steps)))
        return self.lr

    def _set_lr(self, v: float):
        for pg in self.opt.param_groups:
            pg["lr"] = v

    def _save_ckpt(self, tag: str):
        raw = (self.model.module if self.ddp else self.model).state_dict()
        ema = self.ema.state_dict()
        torch.save(raw, os.path.join(self.run_dir, f"model_{tag}.pt"))
        torch.save(ema, os.path.join(self.run_dir, f"ema_{tag}.pt"))
        logger.log(f"Saved: model_{tag}.pt, ema_{tag}.pt")

    # ----- main loop -----
    def run_loop(self):
        self.model.train()
        t0 = time.time()
        world = 1  # single GPU by default

        while True:
            if self.max_training_steps is not None and self.step >= self.max_training_steps:
                logger.log("Reached max_training_steps. Exiting.")
                break

            self._set_lr(self._lr_now())

            # Full batch
            x, kw = self._next_batch()
            num_timesteps = getattr(self.diffusion, "num_timesteps", 1000)

            self.opt.zero_grad(set_to_none=True)
            losses_collect = []

            # Microbatch over the batch
            for x_mb, kw_mb in self._iter_micro(x, kw):
                # Sample timesteps per microbatch
                t_mb = torch.randint(0, num_timesteps, (x_mb.shape[0],), device=self.device)

                with torch.amp.autocast("cuda", enabled=self.use_fp16):
                    L = self.diffusion.training_losses(self.model, x_mb, t_mb, model_kwargs=kw_mb)
                    loss_mb = _extract_scalar_loss(L, self.device)

                losses_collect.append(loss_mb.detach())

                if self.use_fp16:
                    self.scaler.scale(loss_mb).backward()
                else:
                    loss_mb.backward()

            # Optimizer step
            if self.use_fp16:
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()

            # EMA update
            self.ema.update(self.model.module if self.ddp else self.model)

            # Logging
            self.step += 1
            if self.step % self.log_interval == 0 or self.step == 1:
                loss_mean = torch.stack([l for l in losses_collect]).mean().item()
                elapsed = time.time() - t0
                logger.log(f"step {self.step} | loss {loss_mean:.6f} | lr {self.opt.param_groups[0]['lr']:.2e} | world {world} | time {elapsed:.1f}s")

            # Checkpointing
            if self.save_interval > 0 and self.step % self.save_interval == 0:
                self._save_ckpt(f"{self.step:09d}")

        # Final save
        self._save_ckpt(f"{self.step:09d}_final")
