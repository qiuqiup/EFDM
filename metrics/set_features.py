"""Per-set spatial features used by the paper's Wasserstein-1 scores."""

from typing import List, Optional

import torch


@torch.no_grad()
def compute_feature(
    sets: List[torch.Tensor],
    feature_name: str,
    *,
    eps: float = 1e-8,
    unbiased_var: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Return [number of sets, feature dimension] for mean/var/skew/kurt/nn1."""
    if len(sets) == 0:
        return torch.empty(0, 1)
    if device is None:
        device = sets[0].device
    if dtype is None:
        dtype = sets[0].dtype
    name = feature_name.lower().strip()
    if name not in ("mean", "var", "skew", "kurt", "nn1"):
        raise ValueError(f"Unknown feature_name={feature_name}")

    d = None
    for points in sets:
        if points is not None and points.ndim == 2 and points.shape[0] > 0:
            d = int(points.shape[1])
            break
    if d is None:
        if sets[0] is not None and sets[0].ndim == 2:
            d = int(sets[0].shape[1])
        else:
            raise ValueError("Cannot infer point dimension from empty or invalid inputs")

    features = torch.zeros(len(sets), 1 if name == "nn1" else d, device=device, dtype=dtype)
    for i, points in enumerate(sets):
        if points is None or points.numel() == 0 or points.ndim != 2 or points.shape[0] == 0:
            continue
        points = points.to(device=device, dtype=dtype)
        if name == "nn1":
            if len(points) >= 2:
                dist = torch.cdist(points, points, p=2)
                dist.fill_diagonal_(float("inf"))
                features[i, 0] = dist.min(dim=1).values.mean()
            continue

        mean = points.mean(dim=0)
        if name == "mean":
            features[i] = mean
            continue
        var = points.var(dim=0, unbiased=unbiased_var)
        if name == "var":
            features[i] = var
            continue
        std = torch.sqrt(torch.clamp(var, min=0.0))
        z = torch.where(std < eps, torch.zeros_like(points - mean),
                        (points - mean) / (std + eps))
        if name == "skew":
            features[i] = (z ** 3).mean(dim=0)
        else:
            features[i] = (z ** 4).mean(dim=0) - 3.0
    return features
