"""Exact CPU/CUDA JMMD² and ICMMD² using shared sliced-Wasserstein projections."""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


N_PROJECTIONS = 50
PAIR_BLOCK_SIZE = 256


def _group_sets(sets: Sequence[np.ndarray]) -> Dict[int, List[np.ndarray]]:
    groups: Dict[int, List[np.ndarray]] = {}
    for points in sets:
        points = np.asarray(points, dtype=np.float64)
        if points.size == 0 and points.ndim == 1:
            points = points.reshape(0, 0)
        if points.ndim != 2 or (len(points) and points.shape[1] == 0):
            raise ValueError("Every point set must have shape (n_points, dimension)")
        if not np.isfinite(points).all():
            raise ValueError("Point sets must be finite and have padding removed")
        groups.setdefault(len(points), []).append(points)
    return groups


def _squared_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Direct FP64 distances avoid cancellation at near-identical sets."""
    output = torch.empty((len(a), len(b)), dtype=torch.float64, device=a.device)
    for first in range(0, len(a), PAIR_BLOCK_SIZE):
        for second in range(0, len(b), PAIR_BLOCK_SIZE):
            block = torch.cdist(
                a[first:first + PAIR_BLOCK_SIZE],
                b[second:second + PAIR_BLOCK_SIZE],
                p=2, compute_mode="donot_use_mm_for_euclid_dist")
            output[first:first + len(block), second:second + block.shape[1]] = block.square()
    return output


@torch.no_grad()
def evaluate_joint_conditional(
    gen_sets: Sequence[np.ndarray], real_sets: Sequence[np.ndarray],
    sigmas: Sequence[float], seed: int = 0, device: str = "cuda",
) -> Tuple[List[float], List[Dict[str, float]]]:
    """Return JMMD² values and ICMMD dictionaries in input bandwidth order."""
    sigmas = [float(sigma) for sigma in sigmas]
    if any(not np.isfinite(sigma) or sigma <= 0 for sigma in sigmas):
        raise ValueError("sigmas must be finite and strictly positive")
    if not sigmas:
        return [], []
    gen = _group_sets(gen_sets)
    real = _group_sets(real_sets)
    dimensions = {points.shape[1] for groups in (gen, real)
                  for count, group in groups.items() if count
                  for points in group}
    if len(dimensions) > 1:
        raise ValueError("Nonempty point sets must have a common dimension")
    dimension = next(iter(dimensions), 1)
    directions = np.random.RandomState(seed).randn(dimension, N_PROJECTIONS)
    directions /= np.sqrt(np.sum(directions ** 2, axis=0, keepdims=True))
    device = torch.device(device)
    directions = torch.as_tensor(directions, dtype=torch.float64, device=device)

    def embed(group, count):
        if not group or count == 0:
            return torch.zeros((len(group), max(count * N_PROJECTIONS, 1)),
                               dtype=torch.float64, device=device)
        points = torch.as_tensor(np.stack(group), dtype=torch.float64, device=device)
        projected = torch.matmul(points, directions).sort(dim=1).values
        return projected.reshape(len(group), -1) / np.sqrt(count * N_PROJECTIONS)

    sums = {}
    for count in dict.fromkeys((*gen, *real)):
        generated = embed(gen.get(count, []), count)
        reference = embed(real.get(count, []), count)
        dxx = _squared_distances(generated, generated)
        dyy = _squared_distances(reference, reference)
        dxy = _squared_distances(generated, reference)
        dxx.fill_diagonal_(float("inf"))
        dyy.fill_diagonal_(float("inf"))
        by_sigma = []
        for sigma in sigmas:
            denom = 2 * sigma ** 2
            by_sigma.append(tuple(float(torch.exp(-distances / denom).sum().item())
                                  for distances in (dxx, dyy, dxy)))
        sums[count] = by_sigma
        del generated, reference, dxx, dyy, dxy

    n_gen, n_real = len(gen_sets), len(real_sets)
    eligible = {count: group for count, group in real.items() if len(group) >= 2}
    eligible_mass = sum(len(group) for group in eligible.values()) / max(1, n_real)
    weights = {count: len(group) / n_real for count, group in eligible.items()}
    total_weight = sum(weights.values())
    jmmd, icmmd = [], []
    for index in range(len(sigmas)):
        kxx = sum(sums[count][index][0] for count in gen)
        kyy = sum(sums[count][index][1] for count in real)
        kxy = sum(sums[count][index][2] for count in gen)
        jmmd.append(float(kxx / max(1, n_gen * (n_gen - 1))
                          + kyy / max(1, n_real * (n_real - 1))
                          - 2 * kxy / max(1, n_gen * n_real)))
        acc = covered_acc = covered_weight = 0.0
        for count, weight in weights.items():
            ng, nr = len(gen.get(count, [])), len(real[count])
            sxx, syy, sxy = sums[count][index]
            eyy = syy / (nr * (nr - 1))
            if ng >= 2:
                conditional = sxx / (ng * (ng - 1)) + eyy - 2 * sxy / (ng * nr)
                covered_acc += weight * conditional
                covered_weight += weight
            else:
                conditional = 1 + eyy
            acc += weight * conditional
        icmmd.append({
            "icmmd2": acc / max(total_weight, 1e-12),
            "icmmd2_covered": covered_acc / covered_weight if covered_weight > 0 else float("nan"),
            "coverage": covered_weight / max(total_weight, 1e-12),
            "eligible_mass": eligible_mass,
            "n_eligible": len(eligible),
        })
    return jmmd, icmmd
