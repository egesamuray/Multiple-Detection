# scripts/curvelet_stats_hdf5.py
import argparse, os, numpy as np, torch
from improved_diffusion.paired_curvelet_datasets import curvelet_stats_hdf5

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5_path", required=True, type=str)
    p.add_argument("--target_key", default="prim", type=str)
    p.add_argument("--j", type=int, default=1)
    p.add_argument("--angles_per_scale", type=str, default=None)  # e.g. "8,16,32,32"
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out_npz", type=str, required=True)
    args = p.parse_args()

    mean, std = curvelet_stats_hdf5(
        h5_path=args.h5_path,
        target_key=args.target_key,
        j=args.j,
        angles_per_scale=args.angles_per_scale,
        image_size=args.image_size,
        limit=args.limit,
    )
    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez(args.out_npz, mean=mean.numpy(), std=std.numpy())
    print("Saved:", args.out_npz)

if __name__ == "__main__":
    main()
