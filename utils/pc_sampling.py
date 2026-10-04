"""Shared policy helpers for discrete DDPM predictor/corrector samplers.

The policy helpers cover the uniform, adjacent-step DDPM samplers used by
``MaskMethod`` and ``NoGroupMethod``; the guarded epsilon-to-score Langevin step
is also shared by ``JumpMethod``.  Model invocation and state constraints
(padding masks, dynamic support, zero-CoM projection, and so on) remain owned by
the individual methods.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Optional
from numbers import Integral, Real

import torch


@dataclass(frozen=True)
class DiscretePCModeSpec:
    """Complete policy for one uniform discrete-DDPM sampling mode."""

    name: str
    required_T: Optional[int]
    corrector_steps_per_level: int
    corrector_snr: Optional[float]
    corrector_start_time: Optional[float]
    corrector_finish_time: Optional[float]
    expected_corrector_nfe: int


_DISCRETE_PC_MODE_SPECS = {
    "none": DiscretePCModeSpec(
        name="none",
        required_T=None,
        corrector_steps_per_level=0,
        corrector_snr=None,
        corrector_start_time=None,
        corrector_finish_time=None,
        expected_corrector_nfe=0,
    ),
    "1000p485c": DiscretePCModeSpec(
        name="1000p485c",
        required_T=1000,
        corrector_steps_per_level=5,
        corrector_snr=0.3,
        corrector_start_time=0.1,
        corrector_finish_time=0.003,
        expected_corrector_nfe=485,
    ),
}


def resolve_discrete_pc_mode(
    pc_mode: str, T: int, *, corrector_snr: float | None = None,
    corrector_steps: int | None = None,
) -> DiscretePCModeSpec:
    """Resolve a named uniform discrete-DDPM PC policy."""

    normalized = str(pc_mode).strip().lower()
    try:
        spec = _DISCRETE_PC_MODE_SPECS[normalized]
    except KeyError as exc:
        choices = ", ".join(repr(name) for name in _DISCRETE_PC_MODE_SPECS)
        raise ValueError(
            f"Unknown pc_mode {pc_mode!r}; expected one of {choices}."
        ) from exc

    if spec.required_T is not None and int(T) != spec.required_T:
        raise ValueError(
            f"pc_mode={spec.name!r} requires T={spec.required_T}, got T={T}."
        )
    if corrector_snr is not None or corrector_steps is not None:
        if spec.name == "none":
            raise ValueError("Corrector overrides require pc_mode with a corrector.")
        if corrector_snr is not None:
            if (isinstance(corrector_snr, bool) or not isinstance(corrector_snr, Real)
                    or not math.isfinite(corrector_snr) or corrector_snr < 0):
                raise ValueError("corrector_snr must be a finite non-negative real number.")
            spec = replace(spec, corrector_snr=float(corrector_snr))
        if corrector_steps is not None:
            if (isinstance(corrector_steps, bool) or not isinstance(corrector_steps, Integral)
                    or corrector_steps <= 0):
                raise ValueError("corrector_steps must be a positive integer.")
            spec = replace(spec, corrector_steps_per_level=int(corrector_steps),
                           expected_corrector_nfe=97 * int(corrector_steps))
    return spec


def discrete_corrector_target(
    spec: DiscretePCModeSpec,
    *,
    source_index: int,
    T: int,
) -> tuple[int, float] | None:
    """Return ``(target_index, target_time)`` when this stage is corrected.

    The open window is evaluated at the predictor's source time.  The corrector
    itself is evaluated after the adjacent predictor, at ``source_index - 1``.
    With T=1000 and the (.003, .1) window this selects source indices 3..99,
    i.e. 97 levels and exactly 485 model evaluations at five steps per level.
    """

    if spec.corrector_steps_per_level == 0:
        return None
    if T < 2:
        raise ValueError(f"T must be at least 2, got {T}.")

    target_index = int(source_index) - 1
    if target_index < 0:
        return None

    source_time = float(source_index) / float(T - 1)
    assert spec.corrector_finish_time is not None
    assert spec.corrector_start_time is not None
    if not (
        spec.corrector_finish_time
        < source_time
        < spec.corrector_start_time
    ):
        return None
    return target_index, float(target_index) / float(T - 1)


def snr_langevin_step_size(
    score: torch.Tensor,
    noise: torch.Tensor,
    *,
    alpha: torch.Tensor,
    snr: float,
) -> torch.Tensor:
    """Return the batch-mean SNR-calibrated Langevin step size.

    The first dimension is treated as the sampling batch.  Callers must zero
    invalid/padded entries before invoking this helper.  A flat or non-finite
    score maps to a zero step instead of producing NaN/Inf.
    """

    if score.shape != noise.shape:
        raise ValueError(
            "score and noise must have identical shapes, got "
            f"{tuple(score.shape)} and {tuple(noise.shape)}."
        )
    if score.ndim < 2:
        raise ValueError(
            f"score and noise must include batch and feature axes, got ndim={score.ndim}."
        )
    if score.shape[0] == 0:
        raise ValueError("score and noise must have a non-empty batch axis.")
    if not score.is_floating_point() or not noise.is_floating_point():
        raise TypeError("score and noise must be floating-point tensors.")
    if score.dtype != noise.dtype or score.device != noise.device:
        raise ValueError("score and noise must have the same dtype and device.")
    snr = float(snr)
    if not math.isfinite(snr) or snr < 0.0:
        raise ValueError(f"snr must be finite and non-negative, got {snr}.")

    alpha_tensor = torch.as_tensor(alpha, device=score.device, dtype=score.dtype)
    if alpha_tensor.numel() != 1:
        raise ValueError(
            f"alpha must be scalar, got shape {tuple(alpha_tensor.shape)}."
        )
    alpha_tensor = alpha_tensor.reshape(())

    grad_norm = torch.linalg.vector_norm(score.reshape(score.shape[0], -1), dim=1).mean()
    noise_norm = torch.linalg.vector_norm(noise.reshape(noise.shape[0], -1), dim=1).mean()
    norm_eps = torch.finfo(score.dtype).eps
    safe_alpha = torch.nan_to_num(
        alpha_tensor, nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    step_size = (
        2.0
        * safe_alpha
        * (snr * noise_norm / grad_norm.clamp_min(norm_eps)).square()
    )
    valid_norms = (
        torch.isfinite(grad_norm)
        & torch.isfinite(noise_norm)
        & torch.isfinite(alpha_tensor)
        & (alpha_tensor >= 0.0)
        & (grad_norm > norm_eps)
    )
    return torch.where(
        valid_norms,
        torch.nan_to_num(step_size, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0),
        torch.zeros_like(step_size),
    )


def epsilon_langevin_corrector_step(
    state: torch.Tensor,
    epsilon: torch.Tensor,
    *,
    marginal_std: torch.Tensor,
    alpha: torch.Tensor,
    snr: float,
    epsilon_clip: float,
    state_clip: float,
    active_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply one guarded Langevin step using an epsilon-parameterized model.

    ``active_mask`` follows point-set convention (True means active) and must
    match every state axis except the final feature axis.  Inactive entries are
    excluded from the norm and returned as exact zeros.
    """

    if state.shape != epsilon.shape:
        raise ValueError(
            "state and epsilon must have identical shapes, got "
            f"{tuple(state.shape)} and {tuple(epsilon.shape)}."
        )
    if state.shape[0] == 0:
        raise ValueError("state and epsilon must have a non-empty batch axis.")
    if not state.is_floating_point() or not epsilon.is_floating_point():
        raise TypeError("state and epsilon must be floating-point tensors.")
    if state.dtype != epsilon.dtype or state.device != epsilon.device:
        raise ValueError("state and epsilon must have the same dtype and device.")
    epsilon_clip = float(epsilon_clip)
    state_clip = float(state_clip)
    for name, value in (
        ("epsilon_clip", epsilon_clip),
        ("state_clip", state_clip),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}.")

    epsilon = torch.nan_to_num(
        epsilon, nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(-epsilon_clip, epsilon_clip)
    active = None
    if active_mask is not None:
        if tuple(active_mask.shape) != tuple(state.shape[:-1]):
            raise ValueError(
                "active_mask must match state.shape[:-1], got "
                f"{tuple(active_mask.shape)} and {tuple(state.shape[:-1])}."
            )
        active = active_mask.to(device=state.device, dtype=torch.bool).unsqueeze(-1)
        epsilon = epsilon.masked_fill(~active, 0.0)

    marginal_std_tensor = torch.as_tensor(
        marginal_std, device=state.device, dtype=state.dtype
    )
    if marginal_std_tensor.numel() != 1:
        raise ValueError(
            "marginal_std must be scalar, got shape "
            f"{tuple(marginal_std_tensor.shape)}."
        )
    marginal_std_tensor = marginal_std_tensor.reshape(())
    norm_eps = torch.finfo(state.dtype).eps
    valid_std = (
        torch.isfinite(marginal_std_tensor)
        & (marginal_std_tensor > norm_eps)
    )
    safe_std = torch.where(
        valid_std, marginal_std_tensor, torch.ones_like(marginal_std_tensor)
    )
    score = torch.where(valid_std, -epsilon / safe_std, torch.zeros_like(epsilon))
    noise = (torch.randn_like(state) if generator is None else
             torch.randn(state.shape, device=state.device, dtype=state.dtype, generator=generator))
    if active is not None:
        noise = noise.masked_fill(~active, 0.0)
    step_size = snr_langevin_step_size(
        score, noise, alpha=alpha, snr=snr
    )
    noise_scale = torch.sqrt(2.0 * step_size)
    state = state + step_size * score + noise_scale * noise
    state = torch.nan_to_num(
        state, nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(-state_clip, state_clip)
    if active is not None:
        state = state.masked_fill(~active, 0.0)
    return state


def build_discrete_sampling_stats(
    spec: DiscretePCModeSpec,
    *,
    predictor_nfe: int,
    corrector_nfe: int,
    noise_schedule: str,
) -> dict:
    """Build the common sampler accounting payload."""

    corrector_steps = spec.corrector_steps_per_level
    return {
        "pc_mode": spec.name,
        "sampler_mode": spec.name,
        "predictor_nfe": int(predictor_nfe),
        "corrector_nfe": int(corrector_nfe),
        "total": int(predictor_nfe + corrector_nfe),
        "total_nfe": int(predictor_nfe + corrector_nfe),
        "corrector_steps": int(corrector_steps),
        "corrector_levels": (
            int(corrector_nfe // corrector_steps) if corrector_steps else 0
        ),
        "corrector_snr": spec.corrector_snr,
        "corrector_start_time": spec.corrector_start_time,
        "corrector_finish_time": spec.corrector_finish_time,
        "predictor_schedule": "uniform",
        "predictor_solver": "legacy_ddpm_adjacent_fixed_large",
        "corrector_window_basis": "raw_time" if corrector_steps else None,
        "noise_schedule": str(noise_schedule),
    }
