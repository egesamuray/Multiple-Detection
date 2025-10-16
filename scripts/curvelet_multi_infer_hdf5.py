import os, glob, json, argparse, h5py, numpy as np, torch, torch.nn.functional as F
from improved_diffusion import script_util, curvelet_ops
from improved_diffusion.paired_curvelet_datasets import _to_tensor_1ch, _angles_parse

def latest_ckpt(d): 
    c = sorted(glob.glob(os.path.join(d, "*.pt"))); 
    if not c: raise FileNotFoundError(f"No ckpt in {d}"); 
    # prefer EMA if both exist
    c_ema = [x for x in c if os.path.basename(x).startswith("ema_")]
    return (c_ema[-1] if c_ema else c[-1])

def load_args_json(logdir):
    p = os.path.join(logdir, "args.json")
    if not os.path.isfile(p): raise FileNotFoundError(f"args.json missing in {logdir}")
    with open(p, "r") as f: return json.load(f)

def build_model_from_args(args_j):
    params = script_util.model_and_diffusion_defaults(task="curvelet")
    params.update(dict(
        j=int(args_j["j"]), conditional=True, angles_per_scale=_angles_parse(args_j["angles_per_scale"]),
        large_size=int(args_j["large_size"]), small_size=int(args_j["large_size"]),
        diffusion_steps=int(args_j["diffusion_steps"]), color_channels=int(args_j["color_channels"]),
    ))
    model, diff = script_util.create_model_and_diffusion(task="curvelet",
        **script_util.args_to_dict(argparse.Namespace(**params), params.keys()))
    return model, diff

def parse_input(h5f, idx, spec):
    if spec.startswith("dataset:"):
        key = spec.split(":",1)[1]; return h5f[key][idx]
    if spec.startswith("sum:"):
        keys = spec.split(":",1)[1].split("+")
        arr = np.zeros_like(h5f[keys[0]][idx], dtype=np.float32)
        for k in keys: arr = arr + h5f[k][idx]
        return arr
    raise ValueError("Unsupported input_spec")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_path", required=True)
    ap.add_argument("--input_spec", default="sum:prim+mul")
    ap.add_argument("--out_h5", required=True)
    ap.add_argument("--j_list", type=str, default="1,2,3,4,5,6")
    ap.add_argument("--logroot", type=str, default="results")
    ap.add_argument("--angles_per_scale", type=str, default="8,16,32,32,32,32")
    ap.add_argument("--large_size", type=int, default=512)
    ap.add_argument("--color_channels", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    J_LIST = [int(t) for t in args.j_list.split(",") if t.strip()]

    # build per-j models from their saved args.json
    models, diffs, stats, angles_j = {}, {}, {}, {}
    for j in J_LIST:
        logdir = os.path.join(args.logroot, f"curvelet_J{j}", "cond")
        aj = load_args_json(logdir)
        model, diff = build_model_from_args(aj)
        ckpt = latest_ckpt(logdir); state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state)
        model.to(device).eval()
        models[j], diffs[j] = model, diff
        z = np.load(os.path.join(args.logroot, f"curvelet_J{j}", f"curvelet_stats_j{j}_prim.npz"))
        stats[j]  = (torch.from_numpy(z["mean"]).float().to(device),
                     torch.from_numpy(z["std"]).float().clamp_min(1e-6).to(device))
        angles_j[j] = _angles_parse(aj["angles_per_scale"])

    os.makedirs(os.path.dirname(args.out_h5), exist_ok=True)
    with h5py.File(args.h5_path, "r") as f_in, h5py.File(args.out_h5, "w") as f_out:
        N,H,W = f_in["prim"].shape
        out = f_out.create_dataset("pred", shape=(N,H,W), dtype="float32")
        for i in range(N):
            x_arr = parse_input(f_in, i, args.input_spec)
            x_t = _to_tensor_1ch(x_arr, args.large_size).unsqueeze(0).to(device)
            # use angles of the largest J to compute coarse
            ANG_BIG = _angles_parse(args.angles_per_scale)
            coeffs = curvelet_ops.fdct2(x_t, J=len(ANG_BIG), angles_per_scale=ANG_BIG)
            coarse = coeffs["coarse"]
            bands = []
            for j in J_LIST:
                ang = angles_j[j]
                # repack under this j's angles: recompute coeffs if angles differ
                if len(ang) != len(ANG_BIG) or ang != ANG_BIG:
                    coeffs_j = curvelet_ops.fdct2(x_t, J=len(ang), angles_per_scale=ang)
                    coarse_j = coeffs_j["coarse"]
                    packed = curvelet_ops.pack_highfreq(coeffs_j, j)
                    if coarse_j.shape[-2:] != packed.shape[-2:]:
                        coarse_j = F.interpolate(coarse_j, size=packed.shape[-2:], mode="bilinear", align_corners=False)
                    coarse_use = coarse_j
                else:
                    packed = curvelet_ops.pack_highfreq(coeffs, j)
                    coarse_use = coarse
                    if coarse_use.shape[-2:] != packed.shape[-2:]:
                        coarse_use = F.interpolate(coarse_use, size=packed.shape[-2:], mode="bilinear", align_corners=False)

                mean_all, std_all = stats[j]
                C = int(args.color_channels)
                Wj = packed.shape[1] // C
                wmean = mean_all[C : C+C*Wj].view(1, C*Wj, 1, 1)
                wstd  = std_all [C : C+C*Wj].view(1, C*Wj, 1, 1)

                shape = (1, C*Wj, packed.shape[-2], packed.shape[-1])
                with torch.no_grad():
                    w_white = diffs[j].p_sample_loop(models[j], shape, model_kwargs={"conditional": coarse_use}, device=device)
                w = w_white * wstd + wmean
                bands.append(curvelet_ops.unpack_highfreq(w, j=j, meta={"angles_per_scale":ang, "color_channels":C}))

            recon = curvelet_ops.ifdct2({"coarse": coarse, "bands": bands}, output_size=args.large_size)
            y = F.interpolate(recon, size=(H, W), mode="bilinear", align_corners=False)
            out[i] = y[0,0].clamp(-1,1).detach().cpu().numpy()
    print("Saved:", args.out_h5)

if __name__ == "__main__":
    main()
