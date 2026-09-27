"""
Generate samples from a trained CIFAR-10 polar diffusion model.

Work is split across GPUs with one process per GPU. Images are saved as
<save_dir>/imgs/<index>.png; re-running the same command resumes an
interrupted run.
"""
import json
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm

from polar_diffusion.ddpm_torch import DATASET_DICT, DATASET_INFO, GaussianDiffusion, seed_all
from polar_diffusion.ddpm_torch.build import (
    RADIUS_STATS_FILE, add_polar_args, build_diffusion, build_model, radius_normalizer, resolve_polar_config)
from polar_diffusion.ddpm_torch.ddim import DDIM, get_selection_schedule
from polar_diffusion.utils import pending_sample_indices


def load_weights(model, chkpt_path, use_ema, device):
    state_dict = torch.load(chkpt_path, map_location=device)
    if "model" in state_dict:  # a training checkpoint
        state_dict = state_dict["ema"]["shadow"] if use_ema else state_dict["model"]
    for k in list(state_dict.keys()):
        if k.startswith("module."):  # state_dict of DDP
            state_dict[k.split(".", maxsplit=1)[1]] = state_dict.pop(k)
    model.load_state_dict(state_dict)


def generate(rank, args, world_size):
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    is_leader = rank == 0

    with open(args.config_path, "r") as f:
        meta_config = json.load(f)
    dataset = meta_config.get("dataset", args.dataset)
    in_channels = DATASET_INFO[dataset]["channels"]
    image_res = DATASET_INFO[dataset]["resolution"][0]

    polar = resolve_polar_config(meta_config, args)
    if is_leader:
        print(f"polar config: {polar}, sample_method: {args.sample_method}")

    diffusion_config = meta_config["diffusion"]
    sample_kwargs = dict(sample_method=args.sample_method, polar_beta_threshold=args.polar_beta_threshold)
    if args.use_ddim:
        subsequence = get_selection_schedule(
            args.skip_schedule, size=args.subseq_size, timesteps=diffusion_config["timesteps"])
        diffusion = build_diffusion(
            dict(diffusion_config, model_var_type="fixed-small"), polar, cls=DDIM,
            eta=args.eta, subsequence=subsequence, **sample_kwargs)
    else:
        diffusion = build_diffusion(diffusion_config, polar, cls=GaussianDiffusion, **sample_kwargs)

    model_config = dict(meta_config["model"], in_channels=in_channels, out_channels=in_channels)
    model = build_model(model_config, polar, img_size=image_res)
    # the radius statistics are cached next to the checkpoint by train.py
    stats_path = os.path.join(os.path.dirname(args.chkpt_path), RADIUS_STATS_FILE)
    model.radius_norm = radius_normalizer(polar, stats_path, dataset=dataset, root=os.path.expanduser(args.root))
    use_ema = meta_config["train"].get("use_ema", args.use_ema)
    load_weights(model, args.chkpt_path, use_ema, device)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if args.save_dir is None:
        folder_name = os.path.basename(args.chkpt_path)[:-3] + args.suffix
        exp_name = os.path.basename(args.config_path)[:-5]
        args.save_dir = os.path.join("./output/eval", exp_name, folder_name)
    img_dir = os.path.join(args.save_dir, "imgs")
    os.makedirs(img_dir, exist_ok=True)

    if torch.backends.cudnn.is_available():  # noqa
        torch.backends.cudnn.benchmark = True  # noqa

    batch_size = args.batch_size // world_size  # per process
    idxs = pending_sample_indices(args.total_size, rank, world_size, img_dir)
    print(f"rank {rank}: {len(idxs)} images to generate")
    # offset the seed by the first pending index so a resumed run draws fresh noise
    seed_all(args.seed + (idxs[0] if idxs else 0))

    stacked = []
    for i in tqdm(range(0, len(idxs), batch_size), disable=not is_leader):
        batch_idxs = idxs[i:i + batch_size]
        shape = (len(batch_idxs), in_channels, image_res, image_res)
        x = diffusion.p_sample(model, shape=shape, device=device, noise=torch.randn(shape, device=device)).cpu()
        x = (x * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()  # [B, H, W, C]
        if args.save_mode == "png":
            for img, idx in zip(x, batch_idxs):
                Image.fromarray(img, mode="RGB").save(os.path.join(img_dir, f"{idx:06d}.png"))
        else:
            stacked.append(x)
    if args.save_mode == "npy" and stacked:
        images = np.concatenate(stacked, axis=0)
        np.save(os.path.join(img_dir, f"rank_{rank}_samples-{'x'.join(map(str, images.shape))}.npy"), images)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config-path", type=str, default="./configs/cifar10.json")
    parser.add_argument("--dataset", choices=DATASET_DICT.keys(), default="cifar10")
    parser.add_argument("--root", default="~/datasets", type=str,
                        help="root directory of datasets (only needed if the radius statistics are not cached)")
    parser.add_argument("--chkpt-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default=None,
                        help="images go to <save-dir>/imgs; defaults to ./output/eval/<config>/<chkpt><suffix>")
    parser.add_argument("--suffix", default="", type=str)
    parser.add_argument("--batch-size", default=128, type=int, help="total batch size across all GPUs")
    parser.add_argument("--total-size", default=50000, type=int)
    parser.add_argument("--num-gpus", default=None, type=int, help="defaults to all visible GPUs")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--save-mode", choices=["png", "npy"], default="png")
    parser.add_argument("--use-ema", action="store_true")
    # samplers
    parser.add_argument("--sample-method", choices=["ddpm", "polar"], default="polar")
    parser.add_argument("--polar-beta-threshold", type=float, default=-1.,
                        help="use a DDPM step where beta_t exceeds this; <= 0 means polar at every step")
    parser.add_argument("--use-ddim", action="store_true")
    parser.add_argument("--eta", default=0., type=float)
    parser.add_argument("--skip-schedule", default="linear", type=str)
    parser.add_argument("--subseq-size", default=50, type=int)
    add_polar_args(parser)
    args = parser.parse_args()

    world_size = args.num_gpus or max(torch.cuda.device_count(), 1)
    if world_size > 1:
        mp.spawn(generate, args=(args, world_size), nprocs=world_size, join=True)
    else:
        generate(0, args, 1)


if __name__ == "__main__":
    main()
