# improved_diffusion/paired_curvelet_datasets.py
import os, h5py, numpy as np, torch, torch.nn.functional as F
from typing import Iterable, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from .curvelet_ops import fdct2, pack_highfreq

try:
    from .curvelet_datasets import _angles_parse
except:
    def _angles_parse(x):
        if x is None: return None
        if isinstance(x, str): return [int(t) for t in x.split(",") if t.strip()]
        return list(x)

def _to_tensor_1ch(arr: np.ndarray, image_size: Optional[int]) -> torch.Tensor:
    a = np.asarray(arr, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(a, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-6:
        lo, hi = float(a.min()), float(a.max())
        if hi - lo < 1e-6: x = np.zeros_like(a, dtype=np.float32)
        else: x = (a - lo) / (hi - lo)
    else:
        x = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    x = (2.0 * x - 1.0).astype(np.float32)
    t = torch.from_numpy(x)[None, :, :]  # (1,H,W)
    if image_size is not None and (t.shape[-2] != image_size or t.shape[-1] != image_size):
        t = F.interpolate(t.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)
    return t

def _parse_input_spec(h5f, idx: int, input_spec: str):
    """
    input_spec examples:
      - "dataset:srmemult"          -> use f["srmemult"][idx]
      - "sum:prim+mul"              -> use f["prim"][idx] + f["mul"][idx]
    """
    if input_spec.startswith("dataset:"):
        key = input_spec.split(":",1)[1]
        return h5f[key][idx]
    if input_spec.startswith("sum:"):
        keys = input_spec.split(":",1)[1].split("+")
        assert len(keys)>=2, "sum: needs at least two keys, e.g. sum:prim+mul"
        arr = np.zeros_like(h5f[keys[0]][idx], dtype=np.float32)
        for k in keys: arr = arr + h5f[k][idx]
        return arr
    raise ValueError(f"Unsupported input_spec: {input_spec}")

class PairedCurveletH5(Dataset):
    """
    Returns (X, KW) where:
      X  = packed wedges of TARGET (e.g., prim) at scale j, optionally whitened
      KW = {"conditional": coarse(input_spec)} resized to Nj x Nj
    """
    def __init__(self, h5_path: str, input_spec: str = "dataset:srmemult", target_key: str = "prim",
                 j: int = 1, image_size: Optional[int] = 256, angles_per_scale=None,
                 stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, limit: Optional[int] = None):
        super().__init__()
        self.h5_path, self.input_spec, self.target_key = os.path.expanduser(h5_path), input_spec, target_key
        self.j = int(j); self.image_size = int(image_size) if image_size is not None else None
        self.angles = _angles_parse(angles_per_scale); self.C = 1
        with h5py.File(self.h5_path, "r") as f:
            self.N = int(f[self.target_key].shape[0])
            if limit is not None: self.N = min(self.N, int(limit))
        self.mean, self.std = (stats if stats is not None else (None, None))

    def __len__(self): return self.N

    def __getitem__(self, idx: int):
        with h5py.File(self.h5_path, "r") as f:
            inp = _parse_input_spec(f, idx, self.input_spec)   # (H,W)
            tgt = f[self.target_key][idx]                      # (H,W)

        x_inp = _to_tensor_1ch(inp, self.image_size).unsqueeze(0)  # (1,1,S,S)
        x_tgt = _to_tensor_1ch(tgt, self.image_size).unsqueeze(0)  # (1,1,S,S)

        J = (len(self.angles) if self.angles else max(self.j, 3))
        coeffs_inp = fdct2(x_inp, J=J, angles_per_scale=self.angles)
        coeffs_tgt = fdct2(x_tgt, J=J, angles_per_scale=self.angles)

        coarse_inp = coeffs_inp["coarse"]
        packed_tgt = pack_highfreq(coeffs_tgt, self.j)

        if coarse_inp.shape[-2:] != packed_tgt.shape[-2:]:
            coarse_inp = F.interpolate(coarse_inp, size=packed_tgt.shape[-2:], mode="bilinear", align_corners=False)

        X = packed_tgt[0]      # (Wj,Nj,Nj)
        cond = coarse_inp[0]   # (1,Nj,Nj)

        if self.mean is not None and self.std is not None:
            Wj = X.shape[0]
            mean_w = self.mean[1:1+Wj].view(-1,1,1)
            std_w  = self.std [1:1+Wj].clamp_min(1e-6).view(-1,1,1)
            X = (X - mean_w) / std_w

        return X, {"conditional": cond}

def load_data_curvelet_paired_h5(h5_path: str, input_spec: str, target_key: str, batch_size: int,
                                 j: int, image_size: int, angles_per_scale=None, stats=None,
                                 deterministic=False, num_workers: int=0, limit: Optional[int]=None):
    ds = PairedCurveletH5(h5_path=h5_path, input_spec=input_spec, target_key=target_key,
                          j=j, image_size=image_size, angles_per_scale=angles_per_scale,
                          stats=stats, limit=limit)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=not deterministic,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True)
    def _gen():
        while True:
            for X, KW in loader: yield X, {k: v for k, v in KW.items()}
    return _gen()

@torch.no_grad()
def curvelet_stats_hdf5(h5_path: str, target_key: str="prim", j: int=1, angles_per_scale=None,
                        image_size: Optional[int]=256, limit: Optional[int]=None):
    angles = _angles_parse(angles_per_scale)
    with h5py.File(h5_path, "r") as f:
        N = f[target_key].shape[0] if limit is None else min(int(limit), int(f[target_key].shape[0]))
    import torch
    sum_c = sumsq_c = None; count = 0
    for i in range(N):
        with h5py.File(h5_path, "r") as f:
            arr = f[target_key][i]
        x = _to_tensor_1ch(arr, image_size).unsqueeze(0)
        J = (len(angles) if angles else max(j,3))
        coeffs = fdct2(x, J=J, angles_per_scale=angles)
        coarse = coeffs["coarse"]; packed = pack_highfreq(coeffs, j)
        if coarse.shape[-2:] != packed.shape[-2:]:
            coarse = F.interpolate(coarse, size=packed.shape[-2:], mode="bilinear", align_corners=False)
        combo = torch.cat([coarse, packed], dim=1)  # (1,1+Wj,Nj,Nj)
        flat = combo.view(combo.size(1), -1)
        sc = flat.sum(dim=1); ss = (flat**2).sum(dim=1)
        sum_c = sc if sum_c is None else (sum_c + sc)
        sumsq_c = ss if sumsq_c is None else (sumsq_c + ss)
        count += flat.size(1)
    mean = sum_c / count
    std  = torch.sqrt((sumsq_c / count) - mean**2).clamp_min(1e-12)
    return mean.cpu(), std.cpu()
