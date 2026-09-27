"""
Train a polar diffusion model on ImageNet-64 (improved-diffusion backbone).
Launch with torchrun; --batch_size is per GPU.
"""
import argparse
import os
import os.path as op

from polar_diffusion.improved_diffusion.image_datasets import load_data, load_data_web
from polar_diffusion.improved_diffusion.resample import create_named_schedule_sampler
from polar_diffusion.improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from polar_diffusion.improved_diffusion.train_util import TrainLoop
from polar_diffusion.utils import ddp_cleanup, setup_torchrun

SEED = 1234


def latest_checkpoint(logdir):
    ckpts = [f for f in os.listdir(logdir) if f.startswith("model") and f.endswith(".pt")]
    ckpts = sorted(ckpts, key=lambda x: int(x[len("model"):-len(".pt")]))
    return op.join(logdir, ckpts[-1]) if ckpts else ""


@ddp_cleanup
def main():
    args = create_argparser().parse_args()

    assert args.logdir is not None, "--logdir is required"
    rank, world_size, device, logger = setup_torchrun(SEED, op.join(args.logdir, "logs"))

    logger.info("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(device)
    logger.info(f"model params: {sum(i.numel() for i in model.parameters())}")
    logger.info(f"radius normalizer: {model.radius_norm}")
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.info("creating data loader...")
    loader = load_data if op.exists(args.data_dir) else load_data_web  # otherwise a WebDataset shard pattern
    data = loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        num_workers=args.num_workers,
    )

    if args.resume_checkpoint == "" and args.resume:
        args.resume_checkpoint = latest_checkpoint(args.logdir)
        if args.resume_checkpoint:
            logger.info(f"resuming from {args.resume_checkpoint}")

    logger.info("training...")
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
        sample_interval=args.sample_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        total_step=args.total_step,
        device=device,
        logger=logger,
        logdir=args.logdir,
        image_size=args.image_size,
        sample_size=args.sample_size,
        skip_first_sample=args.skip_first_sample,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        logdir=None,
        resume=True,  # resume from the latest checkpoint in logdir
        num_workers=4,
        sample_interval=10000,  # save a grid of samples every N steps
        sample_size=16,
        skip_first_sample=False,
        total_step=200000,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=20000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
