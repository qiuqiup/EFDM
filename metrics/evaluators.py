"""Count and per-set feature Wasserstein-1 metrics."""

from typing import Any, Dict, List, Optional

import torch
from scipy.stats import wasserstein_distance

from .set_features import compute_feature


def _assign_bins_by_n(ns: torch.Tensor, bin_edges: List[int]) -> torch.Tensor:
    edges = torch.tensor(bin_edges, device=ns.device)
    return (torch.searchsorted(edges, ns, right=False) - 1).clamp(0, len(bin_edges) - 2)


@torch.no_grad()
def eval_feature_w1_binned(
    gen_sets: List[torch.Tensor],
    real_sets: List[torch.Tensor],
    feature_name: str,
    bin_edges: List[int],
    *,
    eps: float = 1e-8,
    min_count_per_bin: int = 1,
    fallback: str = "skip",
) -> Dict[str, Any]:
    """Average per-coordinate W1 of features, weighting bins by generated sets."""
    device = gen_sets[0].device if gen_sets else (real_sets[0].device if real_sets else torch.device("cpu"))
    gen_n = torch.tensor([int(s.shape[0]) for s in gen_sets], device=device, dtype=torch.long)
    real_n = torch.tensor([int(s.shape[0]) for s in real_sets], device=device, dtype=torch.long)
    num_bins = len(bin_edges) - 1
    gen_bin = _assign_bins_by_n(gen_n, bin_edges)
    real_bin = _assign_bins_by_n(real_n, bin_edges)
    gen_feat = compute_feature(gen_sets, feature_name, eps=eps)
    real_feat = compute_feature(real_sets, feature_name, eps=eps)
    dim = gen_feat.shape[1] if gen_feat.numel() > 0 else real_feat.shape[1]
    per_bin = [None] * num_bins
    weights = torch.zeros(num_bins, device=device, dtype=torch.float32)
    used = 0

    def nearest_real_bin(index: int) -> Optional[int]:
        candidates = [j for j in range(num_bins)
                      if (real_bin == j).sum().item() >= min_count_per_bin]
        if not candidates:
            return None
        candidates = torch.tensor(candidates, device=device)
        return int(candidates[torch.argmin(torch.abs(candidates - index))].item())

    for index in range(num_bins):
        gmask = gen_bin == index
        rmask = real_bin == index
        ng = int(gmask.sum().item())
        nr = int(rmask.sum().item())
        if ng < min_count_per_bin:
            continue
        if nr < min_count_per_bin:
            if fallback == "skip":
                continue
            if fallback != "nearest_bin":
                raise ValueError(f"Unknown fallback={fallback}")
            nearest = nearest_real_bin(index)
            if nearest is None:
                continue
            rmask = real_bin == nearest
            if int(rmask.sum().item()) < min_count_per_bin:
                continue
        a = gen_feat[gmask]
        b = real_feat[rmask]
        values = [wasserstein_distance(a[:, k].cpu().numpy(), b[:, k].cpu().numpy())
                  for k in range(dim)]
        per_bin[index] = torch.tensor(values, device=device, dtype=torch.float32)
        weights[index] = float(ng)
        used += 1

    total = weights.sum().clamp(min=1.0)
    overall = torch.zeros(dim, device=device, dtype=torch.float32)
    for index, value in enumerate(per_bin):
        if value is not None:
            overall += (weights[index] / total) * value.to(torch.float32)
    return {
        "overall_w1_per_dim": overall,
        "overall_w1": float(overall.mean().item()),
        "per_bin_w1_per_dim": per_bin,
        "bin_weights": weights / total,
        "used_bins": used,
        "num_bins": num_bins,
    }


@torch.no_grad()
def w1_cardinality(gen_data: torch.Tensor, real_data: torch.Tensor) -> torch.Tensor:
    """W1 between integer set counts; accepts either sign of infinite padding."""
    gen_count = (~torch.isinf(gen_data[:, :, 0])).sum(dim=1).float()
    real_count = (~torch.isinf(real_data[:, :, 0])).sum(dim=1).float()
    value = wasserstein_distance(gen_count.cpu().numpy(), real_count.cpu().numpy())
    return torch.tensor(value, device=gen_count.device, dtype=torch.float32)
