from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from networks.egnn_backbone.egnn_core import polynomial_schedule

DEFAULT_PRECISION = 1e-5
DEFAULT_VP_SDE_BETA_MIN = 0.1
DEFAULT_VP_SDE_BETA_MAX = 20.0

def schedule_name_from_cfg(cfg) -> str:
    """Schedule requested by a config. Absent key => "linear" (legacy behavior)."""
    return str(getattr(cfg, "noise_schedule", "linear")).strip().lower()

def is_vp_sde_cfg(cfg) -> bool:
    """Whether ``cfg`` requests the continuous variance-preserving SDE."""
    return schedule_name_from_cfg(cfg) in ("vp_sde", "vpsde", "vp-sde")

def vp_sde_beta(
    t: torch.Tensor,
    *,
    beta_min: float = DEFAULT_VP_SDE_BETA_MIN,
    beta_max: float = DEFAULT_VP_SDE_BETA_MAX,
) -> torch.Tensor:
    """Instantaneous VP-SDE beta(t), for t in [0, 1]."""
    beta_min = float(beta_min)
    beta_max = float(beta_max)
    if beta_min <= 0.0 or beta_max < beta_min:
        raise ValueError(
            f"Expected 0 < beta_min <= beta_max, got {beta_min}, {beta_max}."
        )
    return beta_min + (beta_max - beta_min) * t

def vp_sde_marginal(
    t: torch.Tensor,
    *,
    beta_min: float = DEFAULT_VP_SDE_BETA_MIN,
    beta_max: float = DEFAULT_VP_SDE_BETA_MAX,
) -> Tuple[torch.Tensor, torch.Tensor]:

    vp_sde_beta(t, beta_min=beta_min, beta_max=beta_max)
    beta_min = float(beta_min)
    beta_max = float(beta_max)
    log_signal = (
        -0.5 * beta_min * t
        -0.25 * (beta_max - beta_min) * t.square()
    )
    signal = torch.exp(log_signal)

    std = torch.sqrt((1.0 - signal.square()).clamp_min(0.0))
    return signal, std

def build_ddpm_schedule(
    name: str,
    T: int,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
    precision: float | None = None,
    vp_sde_beta_min: float = DEFAULT_VP_SDE_BETA_MIN,
    vp_sde_beta_max: float = DEFAULT_VP_SDE_BETA_MAX,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    T = int(T)
    if T < 2:
        raise ValueError(f"T must be >= 2, got {T}")
    name = str(name).strip().lower()

    if name in ("linear", "ddpm", "ddpm_linear"):
        beta = torch.linspace(0.0001, 0.02, T, device=device, dtype=dtype)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        return beta, alpha, alpha_bar

    if name.startswith("polynomial"):
        parts = name.split("_")
        if len(parts) != 2:
            raise ValueError(
                f"polynomial schedule must be 'polynomial_<power>', got {name!r}"
            )
        power = float(parts[1])
        s = DEFAULT_PRECISION if precision is None else float(precision)

        alphas2 = polynomial_schedule(T, s=s, power=power)
        if len(alphas2) != T + 1:
            raise RuntimeError(
                f"polynomial_schedule returned {len(alphas2)} entries, expected {T+1}"
            )

        alpha_bar = torch.from_numpy(np.asarray(alphas2[1:], dtype=np.float64))
        alpha_bar = alpha_bar.to(device=device, dtype=dtype)

        alpha = torch.cat([alpha_bar[:1], alpha_bar[1:] / alpha_bar[:-1]], dim=0)
        beta = 1.0 - alpha
        return beta, alpha, alpha_bar

    if name in ("vp_sde", "vpsde", "vp-sde"):

        times = torch.arange(1, T + 1, device=device, dtype=dtype) / float(T)
        signal, _ = vp_sde_marginal(
            times,
            beta_min=vp_sde_beta_min,
            beta_max=vp_sde_beta_max,
        )
        alpha_bar = signal.square()
        alpha = torch.cat(
            [alpha_bar[:1], alpha_bar[1:] / alpha_bar[:-1]], dim=0
        )
        beta = 1.0 - alpha
        return beta, alpha, alpha_bar

    if name == "cosine":
        raise NotImplementedError(
            "cosine is available in egnn_core.cosine_beta_schedule but is not wired "
            "up here; use 'linear' or 'polynomial_2'."
        )

    raise ValueError(
        f"Unknown noise_schedule {name!r}. Supported: 'linear', "
        "'polynomial_<power>', 'vp_sde'."
    )

def build_from_cfg(cfg, T=None, *, device=None, dtype: torch.dtype = torch.float32):
    """Convenience wrapper: read the schedule name and precision off a config."""
    if T is None:
        T = cfg.T
    return build_ddpm_schedule(
        schedule_name_from_cfg(cfg),
        T,
        device=device,
        dtype=dtype,
        precision=getattr(cfg, "noise_precision", None),
        vp_sde_beta_min=getattr(
            cfg, "vp_sde_beta_min", DEFAULT_VP_SDE_BETA_MIN
        ),
        vp_sde_beta_max=getattr(
            cfg, "vp_sde_beta_max", DEFAULT_VP_SDE_BETA_MAX
        ),
    )
