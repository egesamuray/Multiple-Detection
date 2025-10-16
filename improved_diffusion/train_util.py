# improved_diffusion/train_util.py
import os, time, copy
from typing import Dict, Any, Tuple, Optional
import torch, torch.nn as nn, torch.optim as optim
from . import dist_util as dist
from . import logger

class ModelEMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.9999, device=None):
        super().__init__()
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).to(device or next(model.parameters()).device)
        for p in self.ema.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for p_ema, p in zip(self.ema.parameters(), model.parameters()):
            p_ema.data.mul_(d).add_(p.data, alpha=1.0 - d)
        for b_ema, b in zip(self.ema.buffers(), model.buffers()):
            b_ema.copy_(b)
    def state_dict(self, *a, **k): return self.ema.state_dict(*a, **k)
    def to(self, *a, **k): self.ema.to(*a, **k); return self

class TrainLoop:
    def __init__(self, *, model, diffusion, data, batch_size, microbatch=-1, lr=1e-4,
                 ema_rate=0.9999, log_interval=10, save_interval=10000, resume_checkpoint="",
                 use_fp16=False, fp16_scale_growth=1e-3, weight_decay=0.0, lr_anneal_steps=0,
                 max_training_steps=0, **_):
        self.model, self.diffusion, self.data = model, diffusion, data
        self.batch_size, self.microbatch = int(batch_size), int(microbatch)
        self.lr, self.log_interval, self.save_interval = float(lr), int(log_interval), int(save_interval)
        self.resume_checkpoint, self.use_fp16 = resume_checkpoint, bool(use_fp16)
        self.fp16_scale_growth, self.weight_decay = float(fp16_scale_growth), float(weight_decay)
        self.lr_anneal_steps = int(lr_anneal_steps)
        self.max_training_steps = int(max_training_steps) if int(max_training_steps)>0 else None
        self.device = dist.dev(); self.model.to(self.device)
        # optional DDP only if group exists
        self.ddp = False
        try:
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1:
                from torch.nn.parallel import DistributedDataParallel as DDP
                self.model = DDP(self.model, device_ids=[self.device.index] if self.device.type=="cuda" else None)
                self.ddp = True
        except Exception: pass
        self.opt = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay) \
                   if self.weight_decay>0 else optim.Adam(self.model.parameters(), lr=self.lr)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_fp16)
        self.ema = ModelEMA(self.model.module if self.ddp else self.model, decay=float(ema_rate), device=self.device)
        self.step = 0
        self.run_dir = logger.get_dir() or "."; os.makedirs(self.run_dir, exist_ok=True)
        if self.resume_checkpoint and os.path.isfile(self.resume_checkpoint):
            try:
                state = torch.load(self.resume_checkpoint, map_location="cpu")
                (self.model.module if self.ddp else self.model).load_state_dict(state)
                logger.log(f"Resumed from {self.resume_checkpoint}")
            except Exception as e:
                logger.log(f"Resume failed: {e}")

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
        mb = self.batch_size if self.microbatch<=0 else self.microbatch
        for s in range(0, B, mb):
            x_mb = x[s:s+mb]
            kw_mb = {k: (v[s:s+mb] if isinstance(v, torch.Tensor) and v.shape[:1] == (B,) else v) for k, v in kw.items()}
            yield x_mb, kw_mb

    def _lr_now(self):
        if self.lr_anneal_steps and self.step < self.lr_anneal_steps:
            return max(1e-8, self.lr * (1 - self.step/float(self.lr_anneal_steps)))
        return self.lr

    def _set_lr(self, val):
        for pg in self.opt.param_groups: pg["lr"] = val

    def _save_ckpt(self, tag):
        raw = (self.model.module if self.ddp else self.model).state_dict()
        ema = self.ema.state_dict()
        torch.save(raw, os.path.join(self.run_dir, f"model_{tag}.pt"))
        torch.save(ema, os.path.join(self.run_dir, f"ema_{tag}.pt"))
        logger.log(f"Saved: model_{tag}.pt, ema_{tag}.pt")

    def run_loop(self):
        self.model.train()
        t0 = time.time()
        world = 1
        while True:
            if self.max_training_steps is not None and self.step >= self.max_training_steps:
                logger.log("Reached max_training_steps."); break
            self._set_lr(self._lr_now())
            x, kw = self._next_batch()
            num_timesteps = getattr(self.diffusion, "num_timesteps", 1000)
            t = torch.randint(0, num_timesteps, (x.shape[0],), device=self.device)

            self.opt.zero_grad(set_to_none=True)
            losses = []
            for x_mb, kw_mb in self._iter_micro(x, kw):
                with torch.amp.autocast("cuda", enabled=self.use_fp16):
                    L = self.diffusion.training_losses(self.model, x_mb, t[:x_mb.shape[0]], model_kwargs=kw_mb)
                    if isinstance(L, dict):
                        loss_mb = (L.get("loss", None) or L.get("mse", None))
                        loss_mb = loss_mb.mean() if loss_mb is not None else torch.as_tensor(0.0, device=self.device)
                    else:
                        loss_mb = torch.as_tensor(L, device=self.device).mean()
                losses.append(loss_mb.detach())
                if self.use_fp16: self.scaler.scale(loss_mb).backward()
                else: loss_mb.backward()
            if self.use_fp16:
                self.scaler.step(self.opt); self.scaler.update()
            else:
                self.opt.step()
            self.ema.update(self.model.module if self.ddp else self.model)

            self.step += 1
            if self.step % 10 == 0:
                loss_mean = torch.stack(losses).mean().item()
                logger.log(f"step {self.step} | loss {loss_mean:.6f} | lr {self.opt.param_groups[0]['lr']:.2e} | world {world} | time {time.time()-t0:.1f}s")
            if self.save_interval>0 and self.step % self.save_interval == 0:
                self._save_ckpt(f"{self.step:09d}")
        self._save_ckpt(f"{self.step:09d}_final")
