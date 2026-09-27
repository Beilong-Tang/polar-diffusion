"""
Generate samples from a trained ImageNet-64 polar diffusion model.

Launch with torchrun; --batch_size is the total over all GPUs. Images are saved
as <outdir>/imgs/<index>.png; re-running the same command resumes an
interrupted run.
"""
import argparse
import functools
import os
import os.path as op

import torch as th
import tqdm
from PIL import Image

from polar_diffusion.improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from polar_diffusion.utils import ddp_cleanup, pending_sample_indices, setup_torchrun

SEED = 42


@ddp_cleanup
def main():
    args = create_argparser().parse_args()

    assert args.outdir is not None, "--outdir is required"
    img_dir = op.join(args.outdir, "imgs")
    os.makedirs(img_dir, exist_ok=True)

    # offset the seed by the number of saved images so a resumed run draws fresh noise
    seed = SEED + len(os.listdir(img_dir))
    rank, world_size, device, logger = setup_torchrun(
        seed, op.join(args.outdir, "logs"), init_process=False)

    logger.info("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path, map_location="cpu"))
    model.to(device)
    model.eval()
    logger.info(f"radius normalizer: {model.radius_norm}")

    if args.use_ddim:
        sample_fn = diffusion.ddim_sample_loop
    else:
        sample_fn = functools.partial(
            diffusion.p_sample_loop,
            use_ddpm_sample=args.sample_method == "ddpm",
            polar_beta_threshold=args.polar_beta_threshold,
        )
    logger.info(f"sampling with {'ddim' if args.use_ddim else args.sample_method}, "
                f"polar_beta_threshold={args.polar_beta_threshold}")

    batch_size = args.batch_size // world_size  # per rank
    img_gen_idxs = pending_sample_indices(args.num_samples, rank, world_size, img_dir)
    logger.info(f"rank {rank}: {len(img_gen_idxs)} images to generate", all=True)

    pbar = tqdm.tqdm(total=len(img_gen_idxs), disable=rank != 0)
    for idx in range(0, len(img_gen_idxs), batch_size):
        _img_idxs = img_gen_idxs[idx:idx + batch_size]
        B = len(_img_idxs)

        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(low=0, high=NUM_CLASSES, size=(B,), device=device)
            model_kwargs["y"] = classes
            save_paths = [op.join(img_dir, f"{i:05d}_{c:04d}.png") for i, c in zip(_img_idxs, classes.tolist())]
        else:
            save_paths = [op.join(img_dir, f"{i:05d}.png") for i in _img_idxs]

        sample = sample_fn(
            model,
            (B, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous().cpu()  # [N, H, W, C]
        for img, path in zip(sample, save_paths):
            Image.fromarray(img.numpy()).save(path)
        pbar.update(B)
    pbar.close()
    logger.info("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,  # total over all ranks
        use_ddim=False,
        model_path="",
        outdir=None,  # images go to <outdir>/imgs
        sample_method="polar",  # "polar" (reverse SDE in polar coordinates) or "ddpm"
        polar_beta_threshold=-1.,  # DDPM step where beta_t exceeds this; <= 0 means polar everywhere
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
