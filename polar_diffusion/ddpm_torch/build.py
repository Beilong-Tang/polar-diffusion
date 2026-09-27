"""
Build the polar diffusion process and UNet from a JSON config (configs/cifar10.json)
plus command-line overrides. Shared by scripts/cifar10/{train,generate}.py.
"""
import os

from .datasets import get_dataloader
from .diffusion import GaussianDiffusion, get_beta_schedule
from .models import UNet
from ..polar import RadiusNormalizer, compute_radius_stats, load_radius_stats, save_radius_stats

# Defaults of the "polar" config block; these are the settings used in the paper.
POLAR_DEFAULTS = dict(
    wx=1.0,                 # weight of the tangent loss
    wr=0.05,                # weight of the radial loss (scaled by 1/n)
    proj_inf=True,          # project the tangent prediction when sampling
    std_r=True,             # standardize the radius input with data statistics
    field="r",              # radius input: "r" or "rho" = log r
    mlp_pool_hidden=False,  # hidden layer in the radial read-out head
    condition="concat",     # how the radius embedding joins the time embedding
    scale_x=1.0,            # scale of the direction input
    input_field="x",        # network input: direction "x" or raw sample "y"
)

RADIUS_STATS_FILE = "std.json"


def add_polar_args(parser):
    """CLI overrides for the "polar" config block (None = keep the config value)."""
    from ..utils import str2bool
    g = parser.add_argument_group("polar (override the config's \"polar\" block)")
    g.add_argument("--wx", type=float, default=None)
    g.add_argument("--wr", type=float, default=None)
    g.add_argument("--proj-inf", type=str2bool, default=None)
    g.add_argument("--std-r", type=str2bool, default=None)
    g.add_argument("--field", choices=["r", "rho"], default=None)
    g.add_argument("--mlp-pool-hidden", type=str2bool, default=None)
    g.add_argument("--condition", choices=["concat", "add"], default=None)
    g.add_argument("--scale-x", type=float, default=None)
    g.add_argument("--input-field", choices=["x", "y"], default=None)


def resolve_polar_config(meta_config, args):
    polar = dict(POLAR_DEFAULTS)
    # older configs keep wx/wr in the "diffusion" block
    for k in ("wx", "wr"):
        if k in meta_config.get("diffusion", {}):
            polar[k] = meta_config["diffusion"][k]
    polar.update(meta_config.get("polar", {}))
    for k in POLAR_DEFAULTS:
        v = getattr(args, k, None)
        if v is not None:
            polar[k] = v
    return polar


def build_diffusion(diffusion_config, polar, cls=GaussianDiffusion, **kwargs):
    """diffusion_config: the config's "diffusion" block; kwargs go to `cls`."""
    betas = get_beta_schedule(
        diffusion_config["beta_schedule"], beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"], timesteps=diffusion_config["timesteps"])
    return cls(
        betas=betas,
        model_mean_type=diffusion_config["model_mean_type"],
        model_var_type=diffusion_config["model_var_type"],
        loss_type=diffusion_config["loss_type"],
        wx=polar["wx"], wr=polar["wr"], proj_inf=polar["proj_inf"],
        **kwargs)


def build_model(model_config, polar, img_size):
    """model_config: the config's "model" block."""
    model_config = {k: v for k, v in model_config.items() if k != "block_size"}
    model_config.setdefault("out_channels", model_config["in_channels"])
    return UNet(
        **model_config,
        img_size=img_size,
        condition=polar["condition"],
        mlp_pool_hidden=polar["mlp_pool_hidden"],
        radius_field=polar["field"],
        scale_x=polar["scale_x"],
        input_field=polar["input_field"],
    )


def radius_normalizer(polar, stats_path, dataset=None, root=None, compute=True):
    """
    The radius normalizer for this run. With std_r, the statistics are read from
    stats_path, or (if compute) computed over the training set and cached there.
    """
    if not polar["std_r"]:
        return RadiusNormalizer(polar["field"])
    if not os.path.exists(stats_path):
        if not compute:
            raise FileNotFoundError(f"radius statistics not found at {stats_path}")
        split = "all" if dataset == "celeba" else "train"
        loader, _ = get_dataloader(
            dataset, batch_size=256, split=split, root=root, num_workers=4, distributed=False)
        save_radius_stats(stats_path, compute_radius_stats(loader, field=polar["field"]))
    stats = load_radius_stats(stats_path, field=polar["field"])
    return RadiusNormalizer(polar["field"], stats["mean"], stats["std"])
