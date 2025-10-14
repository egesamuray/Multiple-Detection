# scripts/image_train.py  (FULL FILE with paired-HDF5 support)
import argparse
import json
import os

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.wavelet_datasets import load_data_wavelet
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.curvelet_datasets import load_data_curvelet
from improved_diffusion.paired_curvelet_datasets import load_data_curvelet_paired_h5
import numpy as np
import torch

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    with open(f"{logger.get_dir()}/args.json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        task=args.task, **args_to_dict(args, model_and_diffusion_defaults(task=args.task).keys())
    )
    model.to(dist_util.dev())

    logger.log("creating data loader...")
    if args.task == "standard":
        data = load_data(args.data_dir, args.batch_size, args.large_size, args.class_cond)
    elif args.task == "super_res":
        data = load_data(args.data_dir, args.batch_size, args.large_size, args.class_cond)
    elif args.task == "wavelet":
        data = load_data_wavelet(args.data_dir, args.batch_size, args.j, args.conditional)
    elif args.task == "curvelet":
        if bool(args.paired):
            # optional stats for whitening wedges (recommended)
            stats = None
            if args.stats_npz and os.path.isfile(args.stats_npz):
                z = np.load(args.stats_npz)
                stats = (torch.from_numpy(z["mean"]).float(), torch.from_numpy(z["std"]).float())
            data = load_data_curvelet_paired_h5(
                h5_path=args.h5_path,
                input_key=args.input_key,
                target_key=args.target_key,
                batch_size=args.batch_size,
                j=args.j,
                image_size=args.large_size,
                angles_per_scale=args.angles_per_scale,
                stats=stats,
                deterministic=False,
                num_workers=0,
                limit=args.limit if args.limit and int(args.limit) > 0 else None,
            )
        else:
            data = load_data_curvelet(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                j=args.j,
                conditional=args.conditional,
                angles_per_scale=args.angles_per_scale,
                image_size=args.large_size,
                color_channels=args.color_channels,
            )
    else:
        raise ValueError("unsupported task")

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=diffusion.schedule,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        max_training_steps=args.max_training_steps,
    ).run_loop()

def create_argparser():
    defaults = dict(
        task="standard",
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        max_training_steps=500000,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        # Paired HDF5 options:
        paired=False,
        h5_path="",
        input_key="srmemult",
        target_key="prim",
        stats_npz="",
        limit=-1,
    )
    defaults.update(model_and_diffusion_defaults(task=defaults["task"]))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
