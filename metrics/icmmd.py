"""Conditional MMD² by cardinality, weighted by the test count marginal.

Only test counts with at least two sets are eligible. Missing generated count
groups receive penalty 1+E[k(Y,Y')]; coverage reports supported test mass.
"""
from typing import Dict, List, Sequence

import numpy as np

from .jmmd import _sw


def _pair_dists(A: List[np.ndarray], B: List[np.ndarray], seed: int,
                distinct: bool) -> np.ndarray:
    ds = []
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            if distinct and i >= j:
                continue
            ds.append(_sw(a, b, seed))
    return np.asarray(ds, dtype=np.float64)


def icmmd_multi(gen_sets: List[np.ndarray], real_sets: List[np.ndarray],
                sigmas: Sequence[float], seed: int = 0,
                min_test_group: int = 2) -> List[Dict[str, float]]:
    """ICMMD^2 at several bandwidths from one distance pass.

    Returns one dict per sigma with keys:
      icmmd2, icmmd2_covered, coverage, eligible_mass, n_eligible
    (coverage/eligible_mass/n_eligible are bandwidth-independent).
    """
    by_n_real: Dict[int, List[np.ndarray]] = {}
    for s in real_sets:
        by_n_real.setdefault(len(s), []).append(s)
    by_n_gen: Dict[int, List[np.ndarray]] = {}
    for s in gen_sets:
        by_n_gen.setdefault(len(s), []).append(s)

    n_test = len(real_sets)
    eligible = {n: T for n, T in by_n_real.items() if len(T) >= min_test_group}
    elig_mass = sum(len(T) for T in eligible.values()) / max(1, n_test)

    groups = []
    for n, T in eligible.items():
        w = len(T) / n_test
        dyy = _pair_dists(T, T, seed, distinct=True)
        G = by_n_gen.get(n, [])
        if len(G) >= 2:
            dxx = _pair_dists(G, G, seed, distinct=True)
            dxy = _pair_dists(G, T, seed, distinct=False)
            groups.append((w, dyy, dxx, dxy))
        else:
            groups.append((w, dyy, None, None))

    out = []
    tot_w = sum(w for w, *_ in groups)
    for sig in sigmas:
        s2 = float(sig) ** 2
        acc = acc_cov = w_cov = 0.0
        for w, dyy, dxx, dxy in groups:
            e_yy = float(np.mean(np.exp(-(dyy ** 2) / (2 * s2)))) if len(dyy) else float("nan")
            if dxx is not None:
                e_xx = float(np.mean(np.exp(-(dxx ** 2) / (2 * s2))))
                e_xy = float(np.mean(np.exp(-(dxy ** 2) / (2 * s2))))
                mmd2_n = e_xx + e_yy - 2.0 * e_xy
                acc_cov += w * mmd2_n
                w_cov += w
            else:
                mmd2_n = 1.0 + e_yy
            acc += w * mmd2_n
        out.append({
            "icmmd2": acc / max(tot_w, 1e-12),
            "icmmd2_covered": (acc_cov / w_cov) if w_cov > 0 else float("nan"),
            "coverage": w_cov / max(tot_w, 1e-12),
            "eligible_mass": elig_mass,
            "n_eligible": len(eligible),
        })
    return out


def icmmd_sets(gen_sets: List[np.ndarray], real_sets: List[np.ndarray],
               sigma: float, seed: int = 0,
               min_test_group: int = 2) -> Dict[str, float]:
    """Single-bandwidth convenience wrapper."""
    return icmmd_multi(gen_sets, real_sets, [sigma], seed=seed,
                       min_test_group=min_test_group)[0]
