# scripts/train_curvelet_paired_h5.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os, json, argparse, numpy as np, torch
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict, add_dict_to_argparser
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.paired_curvelet_datasets import load_data_curvelet_paired_h5

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    os.makedirs(args.logdir, exist_ok=True)
    logger.configure(dir=args.logdir)
    with open(os.path.join(args.logdir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    model, diffusion = create_model_and_diffusion(task="curvelet",
        **args_to_dict(args, model_and_diffusion_defaults(task="curvelet").keys()))
    model.to(dist_util.dev())

    stats = None
    if args.stats_npz and os.path.isfile(args.stats_npz):
        z = np.load(args.stats_npz)
        stats = (torch.from_numpy(z["mean"]).float(), torch.from_numpy(z["std"]).float())

    data = load_data_curvelet_paired_h5(
        h5_path=args.h5_path, input_spec=args.input_spec, target_key=args.target_key,
        batch_size=args.batch_size, j=args.j, image_size=args.large_size,
        angles_per_scale=args.angles_per_scale, stats=stats,
        deterministic=False, num_workers=0, limit=(None if args.limit<0 else args.limit),
    )

    TrainLoop(model=model, diffusion=diffusion, data=data, batch_size=args.batch_size,
              microbatch=args.microbatch, lr=args.lr, ema_rate=args.ema_rate, log_interval=args.log_interval,
              save_interval=args.save_interval, resume_checkpoint=args.resume_checkpoint,
              use_fp16=args.use_fp16, fp16_scale_growth=args.fp16_scale_growth, weight_decay=args.weight_decay,
              lr_anneal_steps=args.lr_anneal_steps, max_training_steps=args.max_training_steps).run_loop()

def create_argparser():
    defaults = dict(
        # data/model
        h5_path="", input_spec="sum:prim+mul", target_key="prim",
        j=1, angles_per_scale="8,16,32,32,32,32", large_size=512, color_channels=1,
        diffusion_steps=256,
        # training
        batch_size=8, lr=1e-4, weight_decay=0.0, lr_anneal_steps=0, max_training_steps=32000,
        microbatch=-1, ema_rate="0.9999", log_interval=10, save_interval=8000,
        resume_checkpoint="", use_fp16=False, fp16_scale_growth=1e-3,
        # misc
        logdir="results/curvelet_J/cond", stats_npz="", limit=-1,
    )
    defaults.update(model_and_diffusion_defaults(task="curvelet"))
    p = argparse.ArgumentParser()
    add_dict_to_argparser(p, defaults)
    return p

if __name__ == "__main__":
    main()
