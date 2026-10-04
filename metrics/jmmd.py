"""Signed unbiased joint MMD² on cardinality and set shape.

The kernel is 1[N=N'] exp(-SW²/(2σ²)), using 50 fixed projections.
Calibrate σ on equal-count training pairs with ``median_sw_sigma_equal``.
Empty-set pairs have SW=0.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from ot import sliced_wasserstein_distance

N_PROJECTIONS = 50


def _sw(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    if len(x) == 0 and len(y) == 0:
        return 0.0
    return float(sliced_wasserstein_distance(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64),
        n_projections=N_PROJECTIONS, seed=seed))


def median_sw_sigma(ref_sets: List[np.ndarray], max_pairs: int = 20000,
                    seed: int = 0) -> float:
    """Median SW over (subsampled) ALL pairs within a reference collection."""
    rng = np.random.default_rng(seed)
    m = len(ref_sets)
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    if len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]
    d = [_sw(ref_sets[i], ref_sets[j], seed) for i, j in pairs]
    return float(np.median(d))


def median_sw_sigma_equal(ref_sets: List[np.ndarray], max_pairs: int = 20000,
                          seed: int = 0) -> float:
    """Median SW over (subsampled) EQUAL-COUNT pairs within a reference
    collection — the population the indicator-product kernel actually sees."""
    rng = np.random.default_rng(seed)
    byn: Dict[int, List[int]] = {}
    for i, s in enumerate(ref_sets):
        byn.setdefault(len(s), []).append(i)
    pairs = [(i, j) for idx in byn.values()
             for a, i in enumerate(idx) for j in idx[a + 1:]]
    if not pairs:
        raise ValueError("no equal-count pairs in reference collection")
    if len(pairs) > max_pairs:
        sel = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[k] for k in sel]
    d = [_sw(ref_sets[i], ref_sets[j], seed) for i, j in pairs]
    return float(np.median(d))


def _equal_n_pair_dists(A: List[np.ndarray], B: List[np.ndarray], seed: int,
                        exclude_diagonal: bool) -> Tuple[np.ndarray, int]:
    """SW distances over all equal-count (a, b) pairs, plus the FULL grid pair
    count (including indicator-zero pairs, which contribute k = 0)."""
    byn_a: Dict[int, List[int]] = {}
    for i, s in enumerate(A):
        byn_a.setdefault(len(s), []).append(i)
    byn_b: Dict[int, List[int]] = {}
    for j, s in enumerate(B):
        byn_b.setdefault(len(s), []).append(j)
    ds = []
    for n, ia in byn_a.items():
        jb = byn_b.get(n)
        if not jb:
            continue
        for i in ia:
            for j in jb:
                if exclude_diagonal and i == j:
                    continue
                ds.append(_sw(A[i], B[j], seed))
    total = len(A) * len(B) - (len(A) if exclude_diagonal else 0)
    return np.asarray(ds, dtype=np.float64), total


def jmmd_multi(gen_sets: List[np.ndarray], real_sets: List[np.ndarray],
               sigmas: Sequence[float], seed: int = 0) -> List[float]:
    """Signed unbiased JMMD^2 at several bandwidths, sharing one distance pass."""
    dxx, nxx = _equal_n_pair_dists(gen_sets, gen_sets, seed, exclude_diagonal=True)
    dyy, nyy = _equal_n_pair_dists(real_sets, real_sets, seed, exclude_diagonal=True)
    dxy, nxy = _equal_n_pair_dists(gen_sets, real_sets, seed, exclude_diagonal=False)
    out = []
    for sig in sigmas:
        s2 = float(sig) ** 2
        kxx = np.exp(-(dxx ** 2) / (2.0 * s2)).sum()
        kyy = np.exp(-(dyy ** 2) / (2.0 * s2)).sum()
        kxy = np.exp(-(dxy ** 2) / (2.0 * s2)).sum()
        out.append(float(kxx / max(1, nxx) + kyy / max(1, nyy)
                         - 2.0 * kxy / max(1, nxy)))
    return out


def jmmd_sets(gen_sets: List[np.ndarray], real_sets: List[np.ndarray],
              sigma: Optional[float] = None, seed: int = 0) -> Tuple[float, float]:
    """Single-bandwidth convenience wrapper. Returns (jmmd2, sigma)."""
    if sigma is None:
        sigma = median_sw_sigma(real_sets, seed=seed)
    return jmmd_multi(gen_sets, real_sets, [sigma], seed=seed)[0], float(sigma)
