# scripts/curvelet_cond_predict.py
import os, argparse, h5py, numpy as np, torch, torch.nn.functional as F
from improved_diffusion import script_util, curvelet_ops
from improved_diffusion.paired_curvelet_datasets import _angles_parse, _to_tensor_1ch

def _load_stats(npz_path, C, Wj, device):
    z = np.load(npz_path)
    mean = torch.from_numpy(z["mean"]).float().to(device)
    std  = torch.from_numpy(z["std"]).float().clamp_min(1e-6).to(device)
    coarse_mean = mean[:C].view(1, C, 1, 1)
    coarse_std  = std[:C].view(1, C, 1, 1)
    wedge_mean  = mean[C : C + C*Wj].view(1, C*Wj, 1, 1)
    wedge_std   = std [C : C + C*Wj].view(1, C*Wj, 1, 1).clamp_min(1e-6)
    return coarse_mean, coarse_std, wedge_mean, wedge_std

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_path", required=True, type=str)
    ap.add_argument("--input_key", default="srmemult", type=str)
    ap.add_argument("--out_h5", required=True, type=str)
    ap.add_argument("--cond_model_path", required=True, type=str)
    ap.add_argument("--stats_npz", required=True, type=str)
    ap.add_argument("--j", type=int, default=1)
    ap.add_argument("--angles_per_scale", type=str, default=None)   # "8,16,32,32"
    ap.add_argument("--large_size", type=int, default=256)          # training image_size
    ap.add_argument("--color_channels", type=int, default=1)
    ap.add_argument("--diffusion_steps", type=int, default=256)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    angles = _angles_parse(args.angles_per_scale)
    C = int(args.color_channels)

    # Build conditional curvelet model
    params = script_util.model_and_diffusion_defaults(task="curvelet")  # repo helper :contentReference[oaicite:8]{index=8}
    params.update(dict(
        j=args.j, conditional=True, angles_per_scale=angles,
        large_size=args.large_size, small_size=args.large_size,
        diffusion_steps=args.diffusion_steps, color_channels=C,
    ))
    model, diff = script_util.create_model_and_diffusion(task="curvelet",
        **script_util.args_to_dict(argparse.Namespace(**params), params.keys()))
    model.load_state_dict(torch.load(args.cond_model_path, map_location="cpu"))
    model.to(device).eval()

    # Determine W_j from angles/j (repo helper) :contentReference[oaicite:9]{index=9}
    W_j = max(1, script_util._wedges_at_scale(args.j if args.j else 1, angles))
    c_mean, c_std, w_mean, w_std = _load_stats(args.stats_npz, C, W_j, device)

    # Open input and output HDF5
    with h5py.File(args.h5_path, "r") as f_in:
        dset = f_in[args.input_key]
        N = dset.shape[0]
        start = max(0, args.start)
        end = N if args.end < 0 else min(N, args.end)

        # Create output file and dataset
        os.makedirs(os.path.dirname(args.out_h5), exist_ok=True)
        with h5py.File(args.out_h5, "w") as f_out:
            # We'll save full-res predictions (match input H,W)
            H, W = dset.shape[1], dset.shape[2]
            out = f_out.create_dataset("pred", shape=(end - start, H, W), dtype="float32")

            for idx in range(start, end):
                arr = dset[idx]  # (H,W)
                x = _to_tensor_1ch(arr, args.large_size).unsqueeze(0).to(device)  # (1,1,S,S)

                # Compute coarse of INPUT and whiten for conditional
                coeffs = curvelet_ops.fdct2(x, J=(len(angles) if angles else max(args.j, 3)), angles_per_scale=angles)
                coarse = coeffs["coarse"]                                  # (1,1,Hc,Wc)
                # Resize coarse to Nj x Nj (same as pack size); approximate via W_j
                # (We can infer Nj from pack_highfreq on INPUT for j)
                packed_dummy = curvelet_ops.pack_highfreq(coeffs, args.j)   # (1, C*Wj, Nj, Nj)
                if coarse.shape[-2:] != packed_dummy.shape[-2:]:
                    coarse = F.interpolate(coarse, size=packed_dummy.shape[-2:], mode="bilinear", align_corners=False)

                cond_in = (coarse - c_mean) / c_std

                shape = (1, C * W_j, cond_in.shape[-2], cond_in.shape[-1])
                with torch.no_grad():
                    wedges_white = diff.p_sample_loop(model, shape, model_kwargs={"conditional": cond_in}, device=device)
                wedges = wedges_white * w_std + w_mean

                wedges_list = curvelet_ops.unpack_highfreq(
                    wedges, j=args.j, meta={"angles_per_scale": angles or [], "color_channels": C}
                )
                recon = curvelet_ops.ifdct2({"coarse": coarse, "bands": [wedges_list]}, output_size=args.large_size)  # (1,1,S,S)

                # Back to full resolution to save (H,W)
                y = F.interpolate(recon, size=(H, W), mode="bilinear", align_corners=False)
                out[idx - start] = y[0, 0].clamp(-1, 1).detach().cpu().numpy()

    print("Saved predictions to", args.out_h5)

if __name__ == "__main__":
    main()
