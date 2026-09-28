"""Full-grid sliced-W2 MMD used in the paper's spatial experiments.

Each ordered pair draws 50 fresh projections from one seeded NumPy RNG.
GG, TT, and GT include every pair, including self pairs.
"""
from dataclasses import dataclass
import time

import numpy as np
import torch


@dataclass
class PackedSets:
    points: torch.Tensor
    counts: np.ndarray

    @classmethod
    def from_sets(cls, sets, device="cpu"):
        arrays = [np.asarray(s, dtype=np.float64) for s in sets]
        if not arrays or any(s.ndim != 2 or not len(s) or not np.isfinite(s).all() for s in arrays):
            raise ValueError("SW requires finite nonempty sets; no samples are dropped")
        if len({s.shape[1] for s in arrays}) != 1:
            raise ValueError("Inconsistent coordinate dimensions")
        counts = np.array([len(s) for s in arrays], dtype=np.int64)
        points = np.zeros((len(arrays), int(counts.max()), arrays[0].shape[1]), dtype=np.float64)
        for i, s in enumerate(arrays):
            points[i, :len(s)] = s
        return cls(torch.as_tensor(points, device=device), counts)


@torch.no_grad()
def pair_distances(a, b, ii, jj, rng, batch_size=256, n_projections=50, progress=None):
    """Exact empirical 1D transport integration over random directions."""
    ii, jj = np.asarray(ii), np.asarray(jj)
    if len(ii) != len(jj) or batch_size < 1:
        raise ValueError("Invalid pair indices or batch size")
    device = a.points.device
    if b.points.device != device:
        raise ValueError("Both collections must use the same device")
    result = np.empty(len(ii), dtype=np.float64)
    last_log = time.monotonic()
    for start in range(0, len(ii), batch_size):
        stop = min(start + batch_size, len(ii))
        ia, ib = ii[start:stop], jj[start:stop]
        na, nb = a.counts[ia], b.counts[ib]
        ka, kb = int(na.max()), int(nb.max())
        ca = torch.as_tensor(na, device=device)
        cb = torch.as_tensor(nb, device=device)
        xa = a.points[torch.as_tensor(ia, device=device), :ka]
        xb = b.points[torch.as_tensor(ib, device=device), :kb]
        directions = rng.randn(stop-start, xa.shape[-1], n_projections)
        directions /= np.sqrt(np.sum(directions**2, axis=1, keepdims=True))
        projections = torch.as_tensor(directions, device=device).transpose(1, 2)
        pa = torch.bmm(projections, xa.transpose(1, 2))
        pb = torch.bmm(projections, xb.transpose(1, 2))
        ma = torch.arange(ka, device=device)[None, :] >= ca[:, None]
        mb = torch.arange(kb, device=device)[None, :] >= cb[:, None]
        pa = pa.masked_fill(ma[:, None, :], torch.inf).sort(dim=2).values
        pb = pb.masked_fill(mb[:, None, :], torch.inf).sort(dim=2).values
        if np.array_equal(na, nb):
            delta = pa.masked_fill(ma[:, None, :], 0) - pb.masked_fill(mb[:, None, :], 0)
            squared = delta.square().sum(dim=(1, 2)) / (ca * n_projections)
        else:
            den = ca * cb
            ea = torch.arange(1, ka+1, device=device)[None, :] * cb[:, None]
            eb = torch.arange(1, kb+1, device=device)[None, :] * ca[:, None]
            ends = torch.cat((torch.minimum(ea, den[:, None]),
                              torch.minimum(eb, den[:, None])), dim=1).sort(dim=1).values
            widths = torch.diff(ends, prepend=torch.zeros_like(ends[:, :1]), dim=1).to(torch.float64) / den[:, None]
            qa = torch.div(ends + cb[:, None] - 1, cb[:, None], rounding_mode="floor") - 1
            qb = torch.div(ends + ca[:, None] - 1, ca[:, None], rounding_mode="floor") - 1
            va = torch.gather(pa, 2, qa[:, None, :].expand(-1, n_projections, -1))
            vb = torch.gather(pb, 2, qb[:, None, :].expand(-1, n_projections, -1))
            squared = ((va-vb).square() * widths[:, None, :]).sum(dim=(1, 2)) / n_projections
        result[start:stop] = squared.sqrt().cpu().numpy()
        if progress and (stop == len(ii) or time.monotonic() - last_log > 30):
            progress(stop, len(ii))
            last_log = time.monotonic()
    return result


def distance_grids(generated, reference, device="cpu", seed=0, batch_size=256, progress=None):
    a, b = PackedSets.from_sets(generated, device), PackedSets.from_sets(reference, device)
    rng = np.random.RandomState(seed)
    grids = {}
    for name, x, y in (("gg", a, a), ("tt", b, b), ("gt", a, b)):
        nx, ny = len(x.counts), len(y.counts)
        ii, jj = np.repeat(np.arange(nx), ny), np.tile(np.arange(ny), nx)
        callback = None if progress is None else lambda done, total: progress(name, done, total)
        grids[name] = pair_distances(x, y, ii, jj, rng, batch_size, progress=callback).reshape(nx, ny)
    return grids


def kernel_summary(grids, h):
    """Return the square root of biased MMD², including diagonal terms."""
    if not np.isfinite(h) or h <= 0:
        raise ValueError("Bandwidth h must be finite and positive")
    kernels = {key: np.exp(-value/(2*h)) for key, value in grids.items()}
    terms = {key: float(value.mean()) for key, value in kernels.items()}
    squared = terms['gg'] - 2*terms['gt'] + terms['tt']
    n, m = kernels['gt'].shape
    return dict(mmd=float(np.sqrt(squared)) if squared >= 0 else None, mmd2=float(squared),
                h=float(h), sigma=float(np.sqrt(h)), kernel_means=terms,
                mean_offdiag_gg=float((kernels['gg'].sum()-np.trace(kernels['gg']))/(n*(n-1))) if n > 1 else None,
                mean_offdiag_tt=float((kernels['tt'].sum()-np.trace(kernels['tt']))/(m*(m-1))) if m > 1 else None,
                unavailable_reason=None if squared >= 0 else "Negative biased MMD2; not clipped")


def calibrate_train(sets, device="cpu", batch_size=256, progress=None):
    """Median of every ordered train/train distance, including the diagonal."""
    packed = PackedSets.from_sets(sets, device)
    n = len(sets)
    ii, jj = np.repeat(np.arange(n), n), np.tile(np.arange(n), n)
    distances = pair_distances(packed, packed, ii, jj, np.random.RandomState(0),
                              batch_size, progress=progress).reshape(n, n)
    median = float(np.median(distances))
    if not np.isfinite(median) or median <= 0:
        raise ValueError("Training distance median must be finite and positive")
    return median, distances


def evaluate_grids(grids, train_m):
    """Score original PSD-style MMD and the paper's train-scale MMD.

    Original MMD uses σ_eval = median of flattened GG, GT, TT and
    exp(-D/(2σ_eval²)). Train-scale MMD uses exp(-D/(2 train_m)); its
    sensitivity scores multiply σ=sqrt(train_m) by 0.5 and 2.
    """
    if set(grids) != {"gg", "tt", "gt"}:
        raise ValueError("Need complete GG, TT, GT grids")
    gg, tt, gt = (np.asarray(grids[name], dtype=np.float64)
                  for name in ("gg", "tt", "gt"))
    n, m = gt.shape
    if min(n, m) < 2 or gg.shape != (n, n) or tt.shape != (m, m):
        raise ValueError("Incomplete distance grids or fewer than two sets")
    if not all(np.isfinite(values).all() and (values >= 0).all()
               for values in (gg, tt, gt)):
        raise ValueError("Distance grids must be finite and nonnegative")
    if not np.isfinite(train_m) or train_m <= 0:
        raise ValueError("Training distance median must be finite and positive")

    sigma_eval = float(np.median(np.concatenate([gg.ravel(), gt.ravel(), tt.ravel()])))
    original = kernel_summary(grids, sigma_eval ** 2)
    train_scale = kernel_summary(grids, train_m)
    half = kernel_summary(grids, train_m * 0.5 ** 2)
    twice = kernel_summary(grids, train_m * 2.0 ** 2)
    return {
        "mmd": original["mmd"],
        "mmd_train_scale": train_scale["mmd"],
        "mmd_train_scale_bw05": half["mmd"],
        "mmd_train_scale_bw20": twice["mmd"],
        "mmd2_train_scale": train_scale["mmd2"],
        "sigma_eval": sigma_eval,
        "train_m": float(train_m),
        "original": original,
        "train_scale": train_scale,
    }
