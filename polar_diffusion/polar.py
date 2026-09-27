"""
Polar diffusion: the pieces shared by every backbone.

A noisy sample y_t is written as y_t = r_t * x_t, with radius r_t = ||y_t|| and
direction x_t on the unit sphere. The network predicts the noise eps split into
  - a tangent (direction) part  eps_x = eps - <x_t, eps> x_t
  - a radial part               eps_r = <x_t, eps>
so eps = eps_r * x_t + eps_x. Training regresses both parts
(`polar_loss`). Sampling either recombines them and uses the usual DDPM
update (`compose_eps`) or integrates the reverse SDE in polar coordinates
(rho = log r on the line, x on the sphere) (`polar_reverse_step`).

Networks take y_t and receive the direction x_t as the image input and a
normalized radius (`RadiusNormalizer`) through `RadiusEmbedding`, which
conditions every residual block along with the time embedding. The radial
prediction is read out of the bottleneck features with `MLPPool`.
"""
import json
import os

import torch
import torch.nn as nn

from .geometry import normalize, proj, exp_map


def _bcast(v, like):
    """Reshape a [B] tensor so that it broadcasts against `like` ([B, ...])."""
    return v.view((-1,) + (1,) * (like.ndim - 1))


def split_eps(x_t_unit, eps):
    """Split eps into its tangent part at x_t_unit and its radial coefficient."""
    eps_r = torch.sum((x_t_unit * eps).flatten(1), -1)  # [B]
    eps_x = eps - _bcast(eps_r, eps) * x_t_unit
    return eps_x, eps_r


def compose_eps(y_t, out_x, out_r, project=True):
    """
    Recombine the network's polar outputs into a full eps prediction.

    project: project out_x onto the tangent space first (the network output is
        not constrained to be tangent).
    """
    x_t_unit = normalize(y_t)
    if project:
        out_x = proj(x_t_unit, out_x)
    return _bcast(out_r, y_t) * x_t_unit + out_x


def polar_loss(y_t, eps, out_x, out_r, wx=1.0, wr=0.05):
    """
    Polar eps-prediction loss.

    loss = wx * mean_i (eps_x - out_x)_i^2 + wr / n * (eps_r - out_r)^2,
    where n = dim(y_t); the 1/n puts both terms on a per-dimension scale.

    returns (loss, loss_x, loss_r), each of shape [B]
    """
    n = y_t[0].numel() * 1.
    target_x, target_r = split_eps(normalize(y_t), eps)
    loss_x = (target_x - out_x).pow(2).mean(dim=list(range(1, y_t.ndim)))
    loss_r = (target_r - out_r).pow(2)
    loss = wx * loss_x + wr * 1. / n * loss_r
    return loss, loss_x, loss_r


def polar_reverse_step(y_t, out_x, out_r, beta_t, sigma_t, add_noise, project=True):
    """
    One Euler-Maruyama step of the reverse-time VP SDE in polar coordinates.

    The radius evolves as rho = log r on the real line and the direction by the
    exponential map on the sphere, so samples never leave the polar chart.

    y_t:      [B, ...] current sample
    out_x:    [B, ...] predicted tangent part of eps
    out_r:    [B]      predicted radial part of eps
    beta_t:   [B]      discrete beta at step t
    sigma_t:  [B]      sqrt(1 - alpha_bar_t)
    add_noise: False at the final step (t == 0)
    project:  project out_x onto the tangent space before use
    returns y_{t-1}
    """
    n = y_t[0].numel() * 1.
    x_t, r_t = normalize(y_t, return_norm=True)
    rho_t = torch.log(r_t)
    if project:
        out_x = proj(x_t, out_x)

    score_x = -1 * _bcast(r_t, y_t) / _bcast(sigma_t, y_t) * out_x
    score_rho = n - r_t / sigma_t * out_r

    exp_minus_rho_t = torch.exp(-1 * rho_t)
    exp_minus_rho_t_square = torch.square(exp_minus_rho_t)
    drift_rho = (-0.5 * beta_t + (n + 2.) / 2. * beta_t * exp_minus_rho_t_square
                 - beta_t * exp_minus_rho_t_square * score_rho)
    noise_rho = torch.randn_like(rho_t) if add_noise else torch.zeros_like(rho_t)
    diff_rho = torch.sqrt(beta_t) * exp_minus_rho_t * noise_rho
    rho_t_prev = rho_t - drift_rho + diff_rho

    drift_x = _bcast(beta_t * exp_minus_rho_t_square, y_t) * score_x
    noise_x = torch.randn_like(x_t) if add_noise else torch.zeros_like(x_t)
    noise_x = proj(x_t, noise_x)
    diff_x = _bcast(torch.sqrt(beta_t) * exp_minus_rho_t, y_t) * noise_x
    x_t_prev = exp_map(x_t, drift_x + diff_x)

    return _bcast(torch.exp(rho_t_prev), y_t) * x_t_prev


# ---------------------------------------------------------------------------
# Network components
# ---------------------------------------------------------------------------

class RadiusEmbedding(nn.Module):
    """MLP embedding of the (normalized) radius, used like a time embedding."""

    def __init__(self, dim, linear=nn.Linear):
        super().__init__()
        self.net = nn.Sequential(
            linear(1, dim),
            nn.SiLU(),
            linear(dim, dim),
        )

    def forward(self, r):
        # r: [B] or [B, 1]
        if r.dim() == 1:
            r = r[:, None]
        return self.net(r)


class MLPPool(nn.Module):
    """Read a scalar (the radial eps prediction) out of a [B, C, W, W] feature map."""

    def __init__(self, in_ch, width, hidden=False, hidden_dim=2048, linear=nn.Linear):
        super().__init__()
        input_dim = in_ch * width * width
        if hidden:
            self.module = nn.Sequential(
                nn.Flatten(),
                linear(input_dim, hidden_dim),
                nn.SiLU(),
                linear(hidden_dim, 1),
            )
        else:
            self.module = nn.Sequential(
                nn.Flatten(),
                nn.SiLU(),
                linear(input_dim, 1),
            )

    def forward(self, x):
        return self.module(x).squeeze(-1)


class RadiusNormalizer(nn.Module):
    """
    Map the raw radius r to the network's radius input.

    field: "r" uses r itself, "rho" uses log r.
    mean, std: statistics of that field over the training data (see
        `compute_radius_stats`); the defaults leave it unstandardized.

    Holds no parameters or buffers, so checkpoints do not depend on it.
    """

    def __init__(self, field="r", mean=0.0, std=1.0):
        super().__init__()
        assert field in ("r", "rho"), f"field {field} not supported"
        self.field, self.mean, self.std = field, float(mean), float(std)

    def forward(self, r):
        if self.field == "rho":
            r = r.log()
        return (r - self.mean) / self.std

    def extra_repr(self):
        return f"field={self.field}, mean={self.mean:.4g}, std={self.std:.4g}"


# ---------------------------------------------------------------------------
# Radius statistics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_radius_stats(batches, field="r"):
    """
    Mean and std of r = ||x|| (field="r") or rho = log r (field="rho") over an
    iterable of image batches in [-1, 1]. Batches may be (images, ...) tuples.
    """
    assert field in ("r", "rho")
    res = []
    for d in batches:
        if isinstance(d, (list, tuple)):
            d = d[0]
        _, r = normalize(torch.as_tensor(d).float(), return_norm=True)
        res.append(torch.log(r) if field == "rho" else r)
    res = torch.cat(res)
    return {"mean": torch.mean(res).item(), "std": torch.std(res).item(), "field": field}


def load_radius_stats(path, field=None):
    """Load {"mean", "std"} from a JSON written by `save_radius_stats`."""
    with open(path, "r") as f:
        stats = json.load(f)
    # older stats files do not record the field
    if field is not None and stats.get("field", field) != field:
        raise ValueError(f"{path} holds stats for field={stats['field']}, not {field}")
    return stats


def save_radius_stats(path, stats):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f)
