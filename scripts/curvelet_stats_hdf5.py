# scripts/curvelet_stats_hdf5.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse, numpy as np
from improved_diffusion.paired_curvelet_datasets import curvelet_stats_hdf5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_path", required=True)
    ap.add_argument("--target_key", default="prim")
    ap.add_argument("--j", type=int, default=1)
    ap.add_argument("--angles_per_scale", type=str, default="8,16,32,32")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--out_npz", required=True)
    args = ap.parse_args()

    mean, std = curvelet_stats_hdf5(
        h5_path=args.h5_path, target_key=args.target_key, j=args.j,
        angles_per_scale=args.angles_per_scale, image_size=args.image_size,
        limit=(None if args.limit < 0 else args.limit)
    )
    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez(args.out_npz, mean=mean.numpy(), std=std.numpy())
    print("Saved:", args.out_npz)

if __name__ == "__main__":
    main()
