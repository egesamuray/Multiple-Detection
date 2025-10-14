# improved_diffusion/paired_curvelet_datasets.py
import os, math, h5py
from typing import Iterable, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .curvelet_ops import fdct2, pack_highfreq   # repo's ops (fdct2/pack) :contentReference[oaicite:2]{index=2}
from .curvelet_datasets import _angles_parse     # reuse repo helpers :contentReference[oaicite:3]{index=3}

def _to_tensor_1ch(arr: np.ndarray, image_size: Optional[int]) -> torch.Tensor:
    """
    arr: (H,W) float32 array. Normalize to [-1,1], return tensor (1,H,W).
    """
    arr = np.asarray(arr, dtype=np.float32)
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    arr = 2.0 * arr - 1.0  # [-1,1]
    x = torch.from_numpy(arr)[None, :, :]  # (1,H,W)
    if image_size is not None and (x.shape[-2] != image_size or x.shape[-1] != image_size):
        x = F.interpolate(x.unsqueeze(0), size=(image_size, image_size),
                          mode="bilinear", align_corners=False).squeeze(0)
    return x

class PairedCurveletH5(Dataset):
    """
    Returns (X, KW) where:
      X  = packed wedges of TARGET (e.g., prim) at scale j, whitened using provided stats
      KW = {"conditional": coarse of INPUT (e.g., srmemult)}, resized to Nj x Nj
    """
    def __init__(
        self,
        h5_path: str,
        input_key: str = "srmemult",
        target_key: str = "prim",
        j: int = 1,
        image_size: Optional[int] = 256,
        angles_per_scale: Optional[Iterable[int] or str] = None,
        stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        limit: Optional[int] = None,
    ):
        super().__init__()
        self.h5_path = os.path.expanduser(h5_path)
        self.input_key = input_key
        self.target_key = target_key
        self.j = int(j)
        self.image_size = int(image_size) if image_size is not None else None
        self.angles = _angles_parse(angles_per_scale)  # e.g. "8,16,32,32" → [8,16,32,32]
        self.C = 1  # seismic grayscale

        with h5py.File(self.h5_path, "r") as f:
            assert self.input_key in f and self.target_key in f, f"Keys not found in {self.h5_path}"
            self.N = int(f[self.input_key].shape[0])
            # optional limit
            if limit is not None:
                self.N = min(self.N, int(limit))

        self.mean, self.std = (stats if stats is not None else (None, None))

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        with h5py.File(self.h5_path, "r") as f:
            inp = f[self.input_key][idx]   # (H,W) float32
            tgt = f[self.target_key][idx]  # (H,W) float32

        x_inp = _to_tensor_1ch(inp, self.image_size).unsqueeze(0)  # (1,1,H,W)
        x_tgt = _to_tensor_1ch(tgt, self.image_size).unsqueeze(0)  # (1,1,H,W)

        # Curvelet transforms (repo’s fdct2 uses angles/J like in single-image loader) :contentReference[oaicite:4]{index=4}
        J = (len(self.angles) if self.angles else max(self.j, 3))
        coeffs_inp = fdct2(x_inp, J=J, angles_per_scale=self.angles)
        coeffs_tgt = fdct2(x_tgt, J=J, angles_per_scale=self.angles)

        coarse_inp = coeffs_inp["coarse"]           # (1,1,Hc,Wc)
        packed_tgt = pack_highfreq(coeffs_tgt, self.j)  # (1, 1*Wj, Nj, Nj)

        # Match conditioning coarse size to Nj x Nj (same as pack size)
        if coarse_inp.shape[-2:] != packed_tgt.shape[-2:]:
            coarse_inp = F.interpolate(coarse_inp, size=packed_tgt.shape[-2:], mode="bilinear", align_corners=False)

        X = packed_tgt[0]              # (Wj, Nj, Nj)
        cond = coarse_inp[0]           # (1,  Nj, Nj)

        # Whitening wedges using dataset stats (mean/std are [C + C*Wj] long; C=1) :contentReference[oaicite:5]{index=5}
        if self.mean is not None and self.std is not None:
            Wj = X.shape[0]  # since C=1, packed channels == Wj
            mean_w = self.mean[self.C : self.C + self.C * Wj].view(-1, 1, 1)
            std_w  = self.std[self.C : self.C + self.C * Wj].clamp_min(1e-6).view(-1, 1, 1)
            X = (X - mean_w) / std_w

        return X, {"conditional": cond}

def load_data_curvelet_paired_h5(
    h5_path: str,
    input_key: str,
    target_key: str,
    batch_size: int,
    j: int,
    image_size: int,
    angles_per_scale: Optional[Iterable[int] or str] = None,
    stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    deterministic: bool = False,
    num_workers: int = 0,
    limit: Optional[int] = None,
):
    ds = PairedCurveletH5(
        h5_path=h5_path, input_key=input_key, target_key=target_key,
        j=j, image_size=image_size, angles_per_scale=angles_per_scale,
        stats=stats, limit=limit,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=not deterministic,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True
    )

    def _gen():
        while True:
            for X, KW in loader:
                yield X, {k: v for k, v in KW.items()}

    return _gen()

@torch.no_grad()
def curvelet_stats_hdf5(
    h5_path: str,
    target_key: str = "prim",
    j: int = 1,
    angles_per_scale: Optional[Iterable[int] or str] = None,
    image_size: Optional[int] = 256,
    limit: Optional[int] = None,
):
    """
    Compute mean/std over channel axis of [coarse ⊕ packed_wedges] for TARGET in HDF5.
    Returns (mean, std) as 1D tensors of length (1 + 1*Wj).
    """
    angles = _angles_parse(angles_per_scale)
    with h5py.File(h5_path, "r") as f:
        data = f[target_key]
        N = data.shape[0] if limit is None else min(int(limit), int(data.shape[0]))

    sum_c, sumsq_c, count = None, None, 0
    for i in range(N):
        with h5py.File(h5_path, "r") as f:
            arr = f[target_key][i]  # (H,W)

        x = _to_tensor_1ch(arr, image_size).unsqueeze(0)  # (1,1,H,W)
        J = (len(angles) if angles else max(j, 3))
        coeffs = fdct2(x, J=J, angles_per_scale=angles)
        coarse = coeffs["coarse"]
        packed = pack_highfreq(coeffs, j)

        if coarse.shape[-2:] != packed.shape[-2:]:
            coarse = F.interpolate(coarse, size=packed.shape[-2:], mode="bilinear", align_corners=False)

        combo = torch.cat([coarse, packed], dim=1)  # (1, 1 + 1*Wj, Nj, Nj)
        Ctot = combo.size(1)
        flat = combo.view(Ctot, -1)

        if sum_c is None:
            sum_c = flat.sum(dim=1)
            sumsq_c = (flat ** 2).sum(dim=1)
        else:
            sum_c += flat.sum(dim=1)
            sumsq_c += (flat ** 2).sum(dim=1)
        count += flat.size(1)

    mean = sum_c / count
    var  = sumsq_c / count - mean ** 2
    std  = torch.sqrt(var.clamp_min(1e-12))
    return mean.cpu(), std.cpu()
