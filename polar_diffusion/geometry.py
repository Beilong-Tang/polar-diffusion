"""
Geometry on the unit hypersphere S^{n-1}, where n = C * H * W.

A sample y is written in polar form y = r * x with r = ||y|| and x = y / ||y||.
All functions treat everything except the batch dimension as one flat vector.
"""
import torch


def normalize(x, eps=1e-9, return_norm=False):
    """
    Project x onto the unit sphere.

    x: [B, ...]
    returns x / ||x|| with the same shape (and ||x|| of shape [B] if return_norm)
    """
    x_shape = x.shape
    x_flat = torch.flatten(x, start_dim=1)
    norm = x_flat.norm(dim=1, keepdim=True).clamp_min(eps)
    x_flat = x_flat / norm
    if return_norm:
        return x_flat.view(x_shape), norm.squeeze(-1)
    return x_flat.view(x_shape)


def exp_map(p, v):
    """
    Exponential map on the unit sphere at point p along tangent vector v.

    p, v: [B, ...] with the same shape
    """
    p_shape = p.shape
    bb = p_shape[0]
    p = p.reshape(bb, -1)
    v = v.reshape(bb, -1)

    theta = torch.norm(v, dim=-1, keepdim=True)  # ||v||
    res = torch.cos(theta) * p + torch.sin(theta) * (v / (theta + 1e-8))
    res = normalize(res)
    return res.reshape(p_shape)


def proj(p, w):
    """
    Project w onto the tangent space of the unit sphere at p: w - <p, w> p.

    p, w: [B, ...] with the same shape
    """
    p_shape = p.shape
    p = p.reshape(p.size(0), -1)
    w = w.reshape(w.size(0), -1)

    inner = (p * w).sum(dim=-1, keepdim=True)
    res = w - inner * p
    return res.reshape(p_shape)
