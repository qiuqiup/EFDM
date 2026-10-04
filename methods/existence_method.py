import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional

import torch

from methods.base_method import BaseMethod
from utils.nonempty_sampling import reject_empty_sets
from utils.noise_schedule import build_from_cfg as _build_noise_schedule
from utils.noise_schedule import is_vp_sde_cfg as _is_vp_sde_cfg
from utils.noise_schedule import vp_sde_beta as _vp_sde_beta
from utils.noise_schedule import vp_sde_marginal as _vp_sde_marginal
from networks.existence_nets import ExistenceNet, ExistenceNetSoftCount
from networks.existence_nets_with_egnn import ExistenceNetWithEGNN

@dataclass(frozen=True)
class _PCModeSpec:
    """Sampling preset selected by the public ``pc_mode`` argument."""

    name: str
    predictor_schedule: str
    required_T: Optional[int]
    expected_predictor_nfe: Optional[int]
    corrector_steps_per_level: int
    corrector_snr: Optional[float]
    corrector_start_time: Optional[float]
    corrector_finish_time: Optional[float]
    expected_corrector_levels: int

_PC_MODE_SPECS = {
    "none": _PCModeSpec(
        name="none",
        predictor_schedule="uniform",
        required_T=None,
        expected_predictor_nfe=None,
        corrector_steps_per_level=0,
        corrector_snr=None,
        corrector_start_time=None,
        corrector_finish_time=None,
        expected_corrector_levels=0,
    ),
    "510p485c": _PCModeSpec(
        name="510p485c",
        predictor_schedule="nfe_matched_analytic",
        required_T=1000,
        expected_predictor_nfe=510,
        corrector_steps_per_level=5,
        corrector_snr=0.3,
        corrector_start_time=0.1,
        corrector_finish_time=0.003,
        expected_corrector_levels=97,
    ),
    "1000p485c": _PCModeSpec(
        name="1000p485c",
        predictor_schedule="uniform",
        required_T=1000,
        expected_predictor_nfe=1000,
        corrector_steps_per_level=5,
        corrector_snr=0.3,
        corrector_start_time=0.1,
        corrector_finish_time=0.003,
        expected_corrector_levels=97,
    ),
}

def _resolve_pc_mode(pc_mode: str, T: int) -> _PCModeSpec:
    """Resolve one named sampler policy without guessing from NFE totals."""
    normalized = str(pc_mode).strip().lower()
    try:
        spec = _PC_MODE_SPECS[normalized]
    except KeyError as exc:
        choices = ", ".join(repr(name) for name in _PC_MODE_SPECS)
        raise ValueError(
            f"Unknown pc_mode {pc_mode!r}; expected one of {choices}."
        ) from exc

    if spec.required_T is not None and int(T) != spec.required_T:
        raise ValueError(
            f"pc_mode={spec.name!r} requires T={spec.required_T}, got T={T}."
        )
    return spec

def _build_nfe_matched_transitions(T: int) -> tuple[tuple[int, int], ...]:

    T = int(T)
    if T != 1000:
        raise ValueError(f"510p485c requires T=1000, got T={T}.")

    source_indices = list(range(T - 1, 499, -50)) + list(range(499, -1, -1))
    target_indices = source_indices[1:] + [-1]
    return tuple(zip(source_indices, target_indices))

class ExistenceMethod(BaseMethod):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)


        dataset_type = getattr(cfg, 'dataset_type', 'trip')
        existence_keep_mode = getattr(cfg, "existence_keep_mode", "bernoulli")
        if existence_keep_mode not in ("bernoulli", "threshold"):
            raise ValueError(
                "existence_keep_mode must be 'bernoulli' or 'threshold'."
            )
        if dataset_type == "molecule" and existence_keep_mode == "threshold":
            raise ValueError(
                "existence_keep_mode='threshold' is only supported for non-molecule datasets."
            )

        if dataset_type == 'molecule':

            self.model = ExistenceNetWithEGNN(cfg).to(device)
        else:

            net_arch = getattr(cfg, 'existence_net_arch', 'legacy')
            if net_arch == 'soft_count':
                self.model = ExistenceNetSoftCount(cfg).to(device)
            elif net_arch == 'legacy':
                self.model = ExistenceNet(cfg).to(device)
            else:
                raise ValueError(f"Unsupported existence_net_arch={net_arch!r}")

        self.cfg = cfg
        self.K_max = cfg.K_max
        self.pad_mode = cfg.pad_mode
        self.existence_eps = cfg.existence_eps
        self.dataset_type = dataset_type
        self.existence_keep_mode = existence_keep_mode

        self.shuffle_points_each_epoch = bool(getattr(cfg, "shuffle_points_each_epoch", True))
        self.existence_prepad_to_kmax = bool(getattr(cfg, "existence_prepad_to_kmax", False))
        self._cached_train_pad: tuple[torch.Tensor, torch.Tensor] | None = None
        self._cached_val_pad: tuple[torch.Tensor, torch.Tensor] | None = None

        self.debug_print = bool(getattr(cfg, "debug_print", False))
        self.debug_print_every = int(getattr(cfg, "debug_print_every", 50))
        self.debug_raise_on_nan = bool(getattr(cfg, "debug_raise_on_nan", False))
        self.molecule_com0 = bool(getattr(cfg, "molecule_com0", True))

        self.use_vp_sde = _is_vp_sde_cfg(cfg)
        self.vp_sde_beta_min = float(getattr(cfg, "vp_sde_beta_min", 0.1))
        self.vp_sde_beta_max = float(getattr(cfg, "vp_sde_beta_max", 20.0))
        self.vp_sde_min_t = float(getattr(cfg, "vp_sde_min_t", 0.001))
        if self.use_vp_sde:
            if not (0.0 < self.vp_sde_min_t <= 1.0):
                raise ValueError(
                    "vp_sde_min_t must be in (0, 1], got "
                    f"{self.vp_sde_min_t}."
                )

            _vp_sde_beta(
                torch.tensor(0.0),
                beta_min=self.vp_sde_beta_min,
                beta_max=self.vp_sde_beta_max,
            )

    @staticmethod
    def _tensor_debug_stats(x: torch.Tensor) -> dict:
        """Return quick stats, robust to NaN/Inf."""
        if not torch.is_tensor(x):
            return {"not_tensor": True}
        nan_n = int(torch.isnan(x).sum().item())
        inf_n = int(torch.isinf(x).sum().item())
        x_safe = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)

        if x_safe.numel() == 0:
            return {"shape": tuple(x.shape), "nan": nan_n, "inf": inf_n, "empty": True}
        return {
            "shape": tuple(x.shape),
            "nan": nan_n,
            "inf": inf_n,
            "min": float(x_safe.min().item()),
            "max": float(x_safe.max().item()),
            "mean": float(x_safe.mean().item()),
            "std": float(x_safe.std(unbiased=False).item()),
            "absmax": float(x_safe.abs().max().item()),
        }

    def _maybe_print_debug(self, tag: str, t: int | None = None, **tensors: torch.Tensor):
        t_str = "" if t is None else f" t={t}"
        parts = [f"[existence/debug]{t_str} {tag}"]
        for name, ten in tensors.items():
            st = self._tensor_debug_stats(ten)
            parts.append(f"{name}={st}")
        print(" | ".join(parts))

    @staticmethod
    def _safe_type_probs(type_logits: torch.Tensor) -> torch.Tensor:
        """Return safe probabilities over [dummy, atom_types...] with dummy logit fixed to 0."""
        B, K, _ = type_logits.shape
        dummy_logit = torch.zeros(B, K, 1, device=type_logits.device, dtype=type_logits.dtype)
        logits_all = torch.cat([dummy_logit, type_logits], dim=-1)
        safe_logits = torch.nan_to_num(logits_all, nan=-1e9, posinf=1e9, neginf=-1e9).clamp(min=-80.0, max=80.0)
        probs = torch.softmax(safe_logits, dim=-1)
        return torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _remove_mean_with_weights(x: torch.Tensor, weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:

        if weights.dim() == 2:
            weights = weights.unsqueeze(-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        denom = weights.sum(dim=1, keepdim=True).clamp(min=eps)
        mean = (x * weights).sum(dim=1, keepdim=True) / denom
        return x - mean


    @staticmethod
    def _pad_or_cut_to_Kmax(
        arr,
        K_max,
        device,
        pad_mode="duplicate",
        shuffle_points=False,
    ):
        arr = arr.to(device)
        N, K0, d = arr.shape

        out = torch.empty(N, K_max, d, device=device, dtype=arr.dtype)
        m0  = torch.zeros(N, K_max, device=device, dtype=torch.float32)

        for i in range(N):
            mask_i = ~torch.isinf(arr[i, :, 0])
            x_valid = arr[i, mask_i]
            n_i = x_valid.size(0)

            if n_i == 0:
                if pad_mode == "randn":
                    out[i] = torch.randn(K_max, d, device=device, dtype=arr.dtype)
                elif pad_mode in ("zeros", "duplicate"):
                    out[i] = torch.zeros(K_max, d, device=device, dtype=arr.dtype)
                else:
                    raise ValueError(f"Unknown pad_mode={pad_mode}")
                continue

            if n_i >= K_max:
                out[i, :, :] = x_valid[:K_max]
                m0[i, :] = 1.0
            else:
                out[i, :n_i, :] = x_valid
                m0[i, :n_i] = 1.0

                pad_len = K_max - n_i
                if pad_mode == "duplicate":
                    idx = torch.randint(0, n_i, (pad_len,), device=device)
                    pad = x_valid[idx]
                elif pad_mode == "randn":
                    pad = torch.randn(pad_len, d, device=device, dtype=arr.dtype)
                elif pad_mode == "zeros":
                    pad = torch.zeros(pad_len, d, device=device, dtype=arr.dtype)
                else:
                    raise ValueError(f"Unknown pad_mode={pad_mode}")

                out[i, n_i:, :] = pad

        if shuffle_points:
            perm = torch.randperm(K_max, device=device)
            out = out[:, perm, :]
            m0  = m0[:, perm]

        return out, m0

    @staticmethod
    def _pad_or_cut_to_Kmax_fast(arr, K_max, device, pad_mode="duplicate", shuffle_points=False):
        arr = arr.to(device)
        N, K0, d = arr.shape

        valid = ~torch.isinf(arr[..., 0])
        n = valid.sum(dim=1).clamp(min=0)
        n_safe = n.clamp(min=1)

        ar = torch.arange(K_max, device=device)[None, :]
        m0 = (ar < n[:, None]).to(torch.float32)

        idx = ar.expand(N, K_max).clone()

        if pad_mode == "duplicate":

            dup = (torch.rand(N, K_max, device=device) * n_safe[:, None]).to(torch.long)
            idx = torch.where(ar < n[:, None], idx, dup)

            out = arr.gather(1, idx[:, :, None].expand(N, K_max, d))

            zero_mask = (n == 0)
            if zero_mask.any():
                out[zero_mask] = 0.0

        elif pad_mode == "randn":
            out = torch.randn(N, K_max, d, device=device, dtype=arr.dtype)

            take = min(K0, K_max)
            out[:, :take, :] = arr[:, :take, :]

            out = torch.where(m0[:, :, None].bool(), out, out)

            bad = torch.isinf(out[..., 0])
            if bad.any():
                out[bad] = torch.randn_like(out[bad])

        elif pad_mode == "zeros":
            out = torch.zeros(N, K_max, d, device=device, dtype=arr.dtype)
            take = min(K0, K_max)
            out[:, :take, :] = arr[:, :take, :]
            bad = torch.isinf(out[..., 0])
            if bad.any():
                out[bad] = 0.0
        else:
            raise ValueError(f"Unknown pad_mode={pad_mode}")

        if shuffle_points:
            perm = torch.randperm(K_max, device=device)
            out = out[:, perm, :]
            m0  = m0[:, perm]

        return out, m0

    def prepare_train_epoch(self, train_data_raw):
        if not self.existence_prepad_to_kmax:
            train_x, train_m0 = self._pad_or_cut_to_Kmax_fast(
                train_data_raw,
                self.K_max,
                self.device,
                pad_mode=self.pad_mode,
                shuffle_points=self.shuffle_points_each_epoch,
            )
            return {"x0": train_x, "m0": train_m0}

        if self._cached_train_pad is None:
            train_x_base, train_m0_base = self._pad_or_cut_to_Kmax_fast(
                train_data_raw,
                self.K_max,
                self.device,
                pad_mode=self.pad_mode,
                shuffle_points=False,
            )
            self._cached_train_pad = (train_x_base, train_m0_base)

        train_x, train_m0 = self._cached_train_pad
        if self.shuffle_points_each_epoch:
            perm = torch.randperm(self.K_max, device=self.device)
            train_x = train_x[:, perm, :]
            train_m0 = train_m0[:, perm]
        return {"x0": train_x, "m0": train_m0}

    def prepare_val_data(self, val_data_raw):
        if not self.existence_prepad_to_kmax:
            val_x, val_m0 = self._pad_or_cut_to_Kmax_fast(
                val_data_raw,
                self.K_max,
                self.device,
                pad_mode=self.pad_mode,
                shuffle_points=False,
            )
            return {"x0": val_x, "m0": val_m0}

        if self._cached_val_pad is None:
            val_x, val_m0 = self._pad_or_cut_to_Kmax_fast(
                val_data_raw,
                self.K_max,
                self.device,
                pad_mode=self.pad_mode,
                shuffle_points=False,
            )
            self._cached_val_pad = (val_x, val_m0)

        val_x, val_m0 = self._cached_val_pad
        return {"x0": val_x, "m0": val_m0}

    def prepare_batch(self, batch_raw: torch.Tensor, *, split: str = "train") -> dict:

        batch_raw = batch_raw.to(self.device, non_blocking=True)
        shuffle_points = (split == "train") and self.shuffle_points_each_epoch
        x, m0 = self._pad_or_cut_to_Kmax_fast(
            batch_raw,
            self.K_max,
            self.device,
            pad_mode=self.pad_mode,
            shuffle_points=shuffle_points,
        )
        return {"x0": x, "m0": m0}


    def _forward_loss(self, batch, generator=None):
        x0 = batch["x0"].to(self.device)
        m0 = batch["m0"].to(self.device)

        B, K, d = x0.shape
        T = self.cfg.T
        if self.use_vp_sde:

            if generator is None:
                t_uniform = torch.rand(B, device=self.device)
            else:
                t_uniform = torch.rand(B, device=self.device, generator=generator)
            t_norm = self.vp_sde_min_t + (1.0 - self.vp_sde_min_t) * t_uniform
            signal, std = _vp_sde_marginal(
                t_norm,
                beta_min=self.vp_sde_beta_min,
                beta_max=self.vp_sde_beta_max,
            )
            vp_signal = signal.view(B, 1, 1)
            vp_std = std.view(B, 1, 1)

            t_int = torch.round(t_norm * (T - 1)).long()
        else:
            beta, alpha, alpha_bar = _build_noise_schedule(
                self.cfg, T, device=self.device
            )
            if generator is None:
                t_int = torch.randint(0, T, (B,), device=self.device)
            else:
                t_int = torch.randint(
                    0, T, (B,), device=self.device, generator=generator
                )
            a_bar = alpha_bar[t_int].view(B, 1, 1)
            t_norm = t_int.float() / (T - 1)

        if self.dataset_type == 'molecule':

            atom_mask = (m0 > 0.5)
            atom_mask_f = atom_mask.float()

            node_mask_all = torch.ones(B, K, device=self.device, dtype=x0.dtype)

            num_atom_types = int(getattr(self.cfg, "num_atom_types", 5))
            assert d >= 3 + num_atom_types, f"Expected x0 dim >= 3+num_atom_types, got d={d}, num_atom_types={num_atom_types}"

            x0_pos = x0[:, :, 0:3]
            x0_type = x0[:, :, 3:3 + num_atom_types]
            if self.molecule_com0:
                x0_pos = self._remove_mean_with_weights(x0_pos, node_mask_all.unsqueeze(-1))

            type_scale = float(getattr(self.cfg, "type_scale", 1.0))
            neg_type_logit = float(getattr(self.cfg, "neg_type_logit", -type_scale))
            pos_type_logit = float(getattr(self.cfg, "pos_type_logit", type_scale))

            x0_type_scaled = torch.full((B, K, num_atom_types), neg_type_logit, device=self.device, dtype=x0_pos.dtype)

            true_type_idx = torch.argmax(x0_type, dim=-1)
            x0_type_scaled[atom_mask, true_type_idx[atom_mask]] = pos_type_logit

            if generator is None:
                eps_pos = torch.randn(B, K, 3, device=self.device)
                eps_type = torch.randn(B, K, num_atom_types, device=self.device)
            else:
                eps_pos = torch.randn(B, K, 3, device=self.device, generator=generator)
                eps_type = torch.randn(B, K, num_atom_types, device=self.device, generator=generator)
            if self.molecule_com0:
                eps_pos = self._remove_mean_with_weights(eps_pos, node_mask_all.unsqueeze(-1))

            if self.use_vp_sde:
                x_t_pos = vp_signal * x0_pos + vp_std * eps_pos
                x_t_type = vp_signal * x0_type_scaled + vp_std * eps_type
            else:
                x_t_pos = torch.sqrt(a_bar) * x0_pos + torch.sqrt(1.0 - a_bar) * eps_pos
                x_t_type = torch.sqrt(a_bar) * x0_type_scaled + torch.sqrt(1.0 - a_bar) * eps_type
            if self.molecule_com0:
                x_t_pos = self._remove_mean_with_weights(x_t_pos, node_mask_all.unsqueeze(-1))

            if self.debug_print:
                self._maybe_print_debug(
                    "forward_diffuse",
                    t=int(t_int[0].item()) if t_int.numel() > 0 else None,
                    x_t_pos=x_t_pos,
                    x_t_type=x_t_type,
                )
                if self.debug_raise_on_nan and (torch.isnan(x_t_pos).any() or torch.isnan(x_t_type).any() or torch.isinf(x_t_pos).any() or torch.isinf(x_t_type).any()):
                    raise RuntimeError("NaN/Inf detected right after forward diffuse (x_t_pos/x_t_type).")

            type_clip = float(getattr(self.cfg, "type_clip", 10.0))
            x_t_type = torch.clamp(x_t_type, min=-type_clip, max=type_clip)
            x_in = torch.cat([x_t_pos, x_t_type], dim=2)

            eps_pos_pred, eps_type_pred = self.model(x_in, t_norm)

            if self.debug_print:
                self._maybe_print_debug(
                    "eps_pred",
                    t=int(t_int[0].item()) if t_int.numel() > 0 else None,
                    eps_pos_pred=eps_pos_pred,
                    eps_type_pred=eps_type_pred,
                )
                if self.debug_raise_on_nan and (torch.isnan(eps_pos_pred).any() or torch.isnan(eps_type_pred).any() or torch.isinf(eps_pos_pred).any() or torch.isinf(eps_type_pred).any()):
                    raise RuntimeError("NaN/Inf detected in model eps prediction.")

            pos_err = (eps_pos_pred - eps_pos) ** 2
            type_err = (eps_type_pred - eps_type) ** 2

            if getattr(self.cfg, "weight_pos_by_exist", False):
                dummy_logit_t = torch.zeros(B, K, 1, device=self.device, dtype=x_t_type.dtype)
                logits_all_t = torch.cat([dummy_logit_t, x_t_type], dim=-1)
                p_type_dummy_t = torch.softmax(logits_all_t, dim=-1)
                p_exist_t = 1.0 - p_type_dummy_t[:, :, 0]
                pos_err = pos_err * p_exist_t.unsqueeze(-1)
            pos_loss = pos_err.mean()
            type_loss = type_err.mean()
            type_w = float(getattr(self.cfg, "type_loss_weight", 1.0))
            loss = pos_loss + type_w * type_loss

            with torch.no_grad():

                if self.use_vp_sde:
                    x0_hat_type = (x_t_type - vp_std * eps_type_pred) / vp_signal
                else:
                    x0_hat_type = (x_t_type - torch.sqrt(1.0 - a_bar) * eps_type_pred) / torch.sqrt(a_bar)

                dummy_logit = torch.zeros(B, K, 1, device=self.device, dtype=x0_hat_type.dtype)
                logits_all = torch.cat([dummy_logit, x0_hat_type], dim=-1)
                p_type_dummy = torch.softmax(logits_all, dim=-1)

                pred_type = torch.argmax(p_type_dummy[:, :, 1:], dim=-1)
                true_type = torch.argmax(x0_type, dim=-1)
                type_acc = (pred_type[atom_mask] == true_type[atom_mask]).float().mean().item() if atom_mask.any() else 0.0
                dummy_prob_mean = p_type_dummy[:, :, 0][~atom_mask].mean().item() if (~atom_mask).any() else 0.0

            log = {
                "loss": loss.item(),
                "pos_loss": pos_loss.item(),
                "type_loss": type_loss.item(),
                "type_acc": type_acc,
                "dummy_prob_mean_on_pad": dummy_prob_mean,
            }
            return loss, log

        t_tile = t_norm.unsqueeze(1).expand(-1, K)

        m0_clamped = torch.clamp(m0, self.cfg.existence_eps, 1 - self.cfg.existence_eps)
        u0 = torch.log(m0_clamped) - torch.log1p(-m0_clamped)
        u0 = u0.unsqueeze(-1)

        if generator is None:
            eps_x = torch.randn(B, K, d, device=self.device)
            eps_u = torch.randn(B, K, 1, device=self.device)
        else:
            eps_x = torch.randn(B, K, d, device=self.device, generator=generator)
            eps_u = torch.randn(B, K, 1, device=self.device, generator=generator)

        if self.use_vp_sde:
            x_t = vp_signal * x0 + vp_std * eps_x
            u_t = vp_signal * u0 + vp_std * eps_u
        else:
            x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * eps_x
            u_t = torch.sqrt(a_bar) * u0 + torch.sqrt(1.0 - a_bar) * eps_u

        eps_x_pred, eps_u_pred = self.model(x_t, u_t.squeeze(-1), t_tile)

        m_t_prob = torch.sigmoid(u_t)

        loss_x_elem = (eps_x_pred - eps_x) ** 2
        if self.cfg.weight_pos_by_exist:
            loss_x_elem = loss_x_elem * m_t_prob
        loss_x = loss_x_elem.mean()
        loss_u = ((eps_u_pred - eps_u) ** 2).mean()
        loss = loss_x + loss_u

        log = {
            "loss": loss.item(),
            "loss_x": loss_x.item(),
            "loss_u": loss_u.item(),
        }
        return loss, log

    def training_step(self, batch):
        self.model.train()
        loss, log = self._forward_loss(batch, generator=None)
        return loss, log

    @torch.no_grad()
    def eval_step(self, batch, generator=None):
        self.model.eval()
        loss, log = self._forward_loss(batch, generator=generator)
        return loss, log

    @torch.no_grad()
    @reject_empty_sets
    def sample_sets(
        self,
        num_samples: int,
        device: Optional[torch.device] = None,
        val_n: torch.Tensor | None = None,
        pc_mode: str = "none",
        terminal_variance_scale: float = 1.0,
        corrector_snr: float | None = None,
        corrector_steps: int | None = None,
        predictor_generator: torch.Generator | None = None,
        corrector_generator: torch.Generator | None = None,
    ):

        if device is None:
            device = self.device
        terminal_variance_scale = float(terminal_variance_scale)
        if (
            not math.isfinite(terminal_variance_scale)
            or terminal_variance_scale < 0.0
        ):
            raise ValueError(
                "terminal_variance_scale must be finite and non-negative."
            )

        B = num_samples
        K = self.cfg.K_max
        T = self.cfg.T
        pc_spec = _resolve_pc_mode(pc_mode, T)
        pc_mode = pc_spec.name
        corrector_overrides = {}
        if corrector_snr is not None or corrector_steps is not None:
            if pc_mode == "none":
                raise ValueError("Corrector overrides require pc_mode with a corrector.")
            if self.dataset_type == "molecule":
                raise ValueError("Corrector overrides are only supported for non-molecule datasets.")
            if corrector_snr is not None:
                if (
                    isinstance(corrector_snr, bool)
                    or not isinstance(corrector_snr, Real)
                    or not math.isfinite(corrector_snr)
                    or corrector_snr < 0.0
                ):
                    raise ValueError("corrector_snr must be a finite non-negative real number.")
                corrector_overrides["corrector_snr"] = float(corrector_snr)
            if corrector_steps is not None:
                if (
                    isinstance(corrector_steps, bool)
                    or not isinstance(corrector_steps, Integral)
                    or corrector_steps <= 0
                ):
                    raise ValueError("corrector_steps must be a positive integer.")
                corrector_overrides["corrector_steps"] = int(corrector_steps)
        if self.dataset_type == "molecule" and (
            predictor_generator is not None or corrector_generator is not None
        ):
            raise ValueError("Sampling generators are only supported for non-molecule datasets.")
        corrector_steps = corrector_overrides.get(
            "corrector_steps", pc_spec.corrector_steps_per_level
        )
        corrector_snr = corrector_overrides.get("corrector_snr", pc_spec.corrector_snr)
        expected_corrector_nfe = pc_spec.expected_corrector_levels * corrector_steps

        def draw_normal_like(state, generator):
            if generator is None:
                return torch.randn_like(state)
            return torch.randn(
                state.shape, device=state.device, dtype=state.dtype, generator=generator
            )

        if self.use_vp_sde:

            vp_dt = 1.0 / float(T)
            if self.vp_sde_beta_max * vp_dt >= 1.0:
                raise ValueError(
                    "VP-SDE Euler sampler requires vp_sde_beta_max / T < 1; "
                    f"got beta_max={self.vp_sde_beta_max}, T={T}."
                )
            vp_no_noise_final_step = bool(
                getattr(self.cfg, "vp_sde_no_noise_final_step", False)
            )
        else:
            beta, alpha, alpha_bar = _build_noise_schedule(
                self.cfg, T, device=self.device
            )

        base_model = self.model.module if hasattr(self.model, "module") else self.model
        model = base_model.to(device)

        model.eval()

        predictor_schedule = pc_spec.predictor_schedule
        predictor_transitions = None
        if predictor_schedule == "nfe_matched_analytic":
            predictor_transitions = _build_nfe_matched_transitions(T)
        corrector_start_time = pc_spec.corrector_start_time
        corrector_finish_time = pc_spec.corrector_finish_time
        predictor_nfe = 0
        corrector_nfe = 0

        if self.dataset_type == 'molecule':
            num_atom_types = int(getattr(self.cfg, "num_atom_types", 5))
            is_egnn_model = bool(getattr(model, "use_egnn_backbone", False))
            sample_com0_default = True if is_egnn_model else getattr(self.cfg, "qm9_preprocess_com0", False)
            sample_com0 = bool(getattr(self.cfg, "sample_com0", sample_com0_default))
            sample_com0 = sample_com0 and is_egnn_model
            sample_com0_weight_floor = float(getattr(self.cfg, "sample_com0_weight_floor", 1e-3))

            x_t_pos = torch.randn(B, K, 3, device=device)
            x_t_type = torch.randn(B, K, num_atom_types, device=device)
            if self.molecule_com0:
                x_t_pos = self._remove_mean_with_weights(
                    x_t_pos,
                    torch.ones(B, K, 1, device=device, dtype=x_t_pos.dtype),
                )
            elif sample_com0:
                x_t_pos = self._remove_mean_with_weights(
                    x_t_pos,
                    torch.ones(B, K, 1, device=device, dtype=x_t_pos.dtype),
                )
        else:
            d = self.cfg.point_dim
            if predictor_generator is None:
                xt = torch.randn(B, K, d, device=device)
                ut = torch.randn(B, K, 1, device=device)
            else:
                xt = torch.randn(B, K, d, device=device, generator=predictor_generator)
                ut = torch.randn(B, K, 1, device=device, generator=predictor_generator)

        vp_corrector_window_clock = None
        if (
            self.use_vp_sde
            and corrector_steps > 0
        ):
            vp_corrector_window_clock = torch.tensor(1.0, dtype=torch.float32)

        if predictor_transitions is None:
            predictor_iterator = (
                (reverse_index, t, None)
                for reverse_index, t in enumerate(reversed(range(T)))
            )
        else:
            predictor_iterator = (
                (reverse_index, source_index, target_index)
                for reverse_index, (source_index, target_index) in enumerate(
                    predictor_transitions
                )
            )

        for reverse_index, t, predictor_target_index in predictor_iterator:
            use_nfe_matched_transition = predictor_target_index is not None
            if self.use_vp_sde:
                if use_nfe_matched_transition:
                    continuous_t = float(t + 1) / float(T)
                    predictor_target_time = (
                        float(predictor_target_index + 1) / float(T)
                        if predictor_target_index >= 0
                        else 0.0
                    )
                    predictor_clock_dt = (
                        continuous_t - predictor_target_time
                    )
                    t_norm = torch.full((B,), continuous_t, device=device)
                    vp_signal_t, vp_std_t = _vp_sde_marginal(
                        torch.as_tensor(continuous_t, device=device),
                        beta_min=self.vp_sde_beta_min,
                        beta_max=self.vp_sde_beta_max,
                    )
                    vp_signal_target, _ = _vp_sde_marginal(
                        torch.as_tensor(predictor_target_time, device=device),
                        beta_min=self.vp_sde_beta_min,
                        beta_max=self.vp_sde_beta_max,
                    )
                    analytic_ab_t = vp_signal_t.square()
                    analytic_ab_target = vp_signal_target.square()
                    add_reverse_noise = predictor_target_index >= 0
                else:
                    continuous_t = 1.0 - reverse_index * vp_dt
                    t_norm = torch.full((B,), continuous_t, device=device)
                    beta_t = _vp_sde_beta(
                        torch.as_tensor(continuous_t, device=device),
                        beta_min=self.vp_sde_beta_min,
                        beta_max=self.vp_sde_beta_max,
                    )
                    _, vp_std_t = _vp_sde_marginal(
                        torch.as_tensor(continuous_t, device=device),
                        beta_min=self.vp_sde_beta_min,
                        beta_max=self.vp_sde_beta_max,
                    )

                    vp_state_scale = 2.0 - torch.sqrt(1.0 - beta_t * vp_dt)
                    vp_score_scale = beta_t * vp_dt / vp_std_t.clamp_min(0.001)
                    vp_noise_scale = torch.sqrt(beta_t * vp_dt)
                    add_reverse_noise = not (
                        vp_no_noise_final_step and reverse_index == T - 1
                    )
            else:
                t_norm = torch.full((B,), float(t) / (T - 1), device=device)
                if use_nfe_matched_transition:
                    analytic_ab_t = alpha_bar[t]
                    analytic_ab_target = (
                        alpha_bar[predictor_target_index]
                        if predictor_target_index >= 0
                        else torch.ones_like(analytic_ab_t)
                    )
                    add_reverse_noise = predictor_target_index >= 0
                else:
                    a_t = alpha[t]
                    ab_t = alpha_bar[t]
                    b_t = beta[t]
                    add_reverse_noise = t > 0

            if use_nfe_matched_transition:

                analytic_interval_alpha = (
                    analytic_ab_t / analytic_ab_target
                ).clamp(min=torch.finfo(analytic_ab_t.dtype).eps, max=1.0)
                analytic_interval_beta = (
                    1.0 - analytic_interval_alpha
                ).clamp_min(0.0)
                analytic_source_variance = (
                    1.0 - analytic_ab_t
                ).clamp_min(torch.finfo(analytic_ab_t.dtype).eps)
                analytic_state_scale = torch.rsqrt(analytic_interval_alpha)
                analytic_eps_scale = (
                    analytic_interval_beta
                    * analytic_state_scale
                    / torch.sqrt(analytic_source_variance)
                )
                analytic_posterior_variance = (
                    (1.0 - analytic_ab_target).clamp_min(0.0)
                    * analytic_interval_beta
                    / analytic_source_variance
                ).clamp_min(0.0)
                analytic_noise_scale = torch.sqrt(
                    analytic_posterior_variance
                )

            do_corrector = False
            if corrector_steps > 0:
                if self.use_vp_sde:
                    corrector_source_time = float(
                        vp_corrector_window_clock.item()
                    )
                    if use_nfe_matched_transition:
                        corrector_clock_dt = predictor_clock_dt
                        corrector_time = predictor_target_time
                        corrector_index = predictor_target_index
                    else:
                        corrector_clock_dt = vp_dt
                        corrector_time = max(continuous_t - vp_dt, 0.0)
                        corrector_index = t - 1

                    vp_corrector_window_clock.sub_(corrector_clock_dt)
                else:
                    corrector_source_time = float(t) / float(T - 1)
                    corrector_index = (
                        predictor_target_index
                        if use_nfe_matched_transition
                        else t - 1
                    )
                    corrector_time = (
                        float(corrector_index) / float(T - 1)
                        if corrector_index >= 0
                        else 0.0
                    )

                do_corrector = corrector_index >= 0 and (
                    corrector_finish_time
                    < corrector_source_time
                    < corrector_start_time
                )
                if do_corrector:
                    corrector_t_norm = torch.full(
                        (B,), corrector_time, device=device
                    )
                    if self.use_vp_sde:
                        corrector_beta = _vp_sde_beta(
                            torch.as_tensor(corrector_time, device=device),
                            beta_min=self.vp_sde_beta_min,
                            beta_max=self.vp_sde_beta_max,
                        )
                        _, corrector_std = _vp_sde_marginal(
                            torch.as_tensor(corrector_time, device=device),
                            beta_min=self.vp_sde_beta_min,
                            beta_max=self.vp_sde_beta_max,
                        )
                        corrector_alpha = (
                            1.0 - corrector_clock_dt * corrector_beta
                        ).clamp_min(torch.finfo(corrector_beta.dtype).eps)
                    else:
                        corrector_std = torch.sqrt(
                            (1.0 - alpha_bar[corrector_index]).clamp_min(0.0)
                        )
                        corrector_alpha = alpha[corrector_index]
                    corrector_score_scale = corrector_std.clamp_min(0.001)

            if self.dataset_type == 'molecule':

                type_clip = float(getattr(self.cfg, "type_clip", 10.0))
                x_t_type = x_t_type.clamp(-type_clip, type_clip)
                x_in = torch.cat([x_t_pos, x_t_type], dim=2)
                eps_pos_pred, eps_type_pred = model(x_in, t_norm)
                predictor_nfe += 1

                eps_clip = float(getattr(self.cfg, "sample_eps_clip", 1e3))
                eps_pos_pred = torch.nan_to_num(eps_pos_pred, nan=0.0, posinf=0.0, neginf=0.0).clamp(-eps_clip, eps_clip)
                eps_type_pred = torch.nan_to_num(eps_type_pred, nan=0.0, posinf=0.0, neginf=0.0).clamp(-eps_clip, eps_clip)
                if self.molecule_com0:
                    eps_pos_pred = self._remove_mean_with_weights(eps_pos_pred, torch.ones(B, K, 1, device=device, dtype=eps_pos_pred.dtype))

                do_periodic = self.debug_print and (t == T - 1 or t == 0 or (self.debug_print_every > 0 and (t % self.debug_print_every == 0)))
                has_bad = (torch.isnan(eps_pos_pred).any() or torch.isnan(eps_type_pred).any() or torch.isinf(eps_pos_pred).any() or torch.isinf(eps_type_pred).any() or
                           torch.isnan(x_t_pos).any() or torch.isnan(x_t_type).any() or torch.isinf(x_t_pos).any() or torch.isinf(x_t_type).any())
                if do_periodic or (self.debug_print and has_bad):
                    self._maybe_print_debug(
                        "sample_step",
                        t=t,
                        x_t_pos=x_t_pos,
                        x_t_type=x_t_type,
                        eps_pos_pred=eps_pos_pred,
                        eps_type_pred=eps_type_pred,
                    )
                if self.debug_raise_on_nan and has_bad:
                    raise RuntimeError(f"NaN/Inf detected during sampling at t={t}. See debug prints above.")

                z_pos = torch.randn_like(x_t_pos) if add_reverse_noise else torch.zeros_like(x_t_pos)
                z_type = torch.randn_like(x_t_type) if add_reverse_noise else torch.zeros_like(x_t_type)

                if use_nfe_matched_transition:
                    x_t_pos = (
                        analytic_state_scale * x_t_pos
                        - analytic_eps_scale * eps_pos_pred
                        + analytic_noise_scale * z_pos
                    )
                    x_t_type = (
                        analytic_state_scale * x_t_type
                        - analytic_eps_scale * eps_type_pred
                        + analytic_noise_scale * z_type
                    )
                elif self.use_vp_sde:

                    x_t_pos = (
                        vp_state_scale * x_t_pos
                        - vp_score_scale * eps_pos_pred
                        + vp_noise_scale * z_pos
                    )
                    x_t_type = (
                        vp_state_scale * x_t_type
                        - vp_score_scale * eps_type_pred
                        + vp_noise_scale * z_type
                    )
                else:
                    x_t_pos = (
                        (x_t_pos - (1.0 - a_t) * eps_pos_pred / torch.sqrt(1.0 - ab_t))
                        / torch.sqrt(a_t)
                        + torch.sqrt(b_t) * z_pos
                    )
                    x_t_type = (
                        (x_t_type - (1.0 - a_t) * eps_type_pred / torch.sqrt(1.0 - ab_t))
                        / torch.sqrt(a_t)
                        + torch.sqrt(b_t) * z_type
                    )

                coord_clip = float(getattr(self.cfg, "sample_coord_clip", 1e3))
                x_t_pos = torch.nan_to_num(x_t_pos, nan=0.0, posinf=0.0, neginf=0.0).clamp(-coord_clip, coord_clip)
                x_t_type = torch.nan_to_num(x_t_type, nan=0.0, posinf=0.0, neginf=0.0).clamp(-type_clip, type_clip)
                if self.molecule_com0:
                    x_t_pos = self._remove_mean_with_weights(x_t_pos, torch.ones(B, K, 1, device=device, dtype=x_t_pos.dtype))
                elif sample_com0:
                    p_type_dummy_t = self._safe_type_probs(x_t_type)
                    p_exist_t = 1.0 - p_type_dummy_t[:, :, 0:1]
                    weights = p_exist_t.clamp(min=sample_com0_weight_floor)
                    x_t_pos = self._remove_mean_with_weights(x_t_pos, weights)

                if corrector_steps > 0:
                    if do_corrector:
                        for _ in range(corrector_steps):
                            x_t_type = x_t_type.clamp(-type_clip, type_clip)
                            corrector_input = torch.cat(
                                [x_t_pos, x_t_type], dim=2
                            )
                            corr_eps_pos, corr_eps_type = model(
                                corrector_input, corrector_t_norm
                            )
                            corrector_nfe += 1

                            corr_eps_pos = torch.nan_to_num(
                                corr_eps_pos,
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            ).clamp(-eps_clip, eps_clip)
                            corr_eps_type = torch.nan_to_num(
                                corr_eps_type,
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            ).clamp(-eps_clip, eps_clip)
                            if self.molecule_com0:
                                corr_eps_pos = self._remove_mean_with_weights(
                                    corr_eps_pos,
                                    torch.ones(
                                        B,
                                        K,
                                        1,
                                        device=device,
                                        dtype=corr_eps_pos.dtype,
                                    ),
                                )

                            score_pos = -corr_eps_pos / corrector_score_scale
                            score_type = -corr_eps_type / corrector_score_scale

                            noise_pos = torch.randn_like(x_t_pos)
                            noise_type = torch.randn_like(x_t_type)
                            if self.molecule_com0:
                                noise_pos = self._remove_mean_with_weights(
                                    noise_pos,
                                    torch.ones(
                                        B,
                                        K,
                                        1,
                                        device=device,
                                        dtype=noise_pos.dtype,
                                    ),
                                )

                            score_joint = torch.cat(
                                [
                                    score_pos.reshape(B, -1),
                                    score_type.reshape(B, -1),
                                ],
                                dim=1,
                            )
                            noise_joint = torch.cat(
                                [
                                    noise_pos.reshape(B, -1),
                                    noise_type.reshape(B, -1),
                                ],
                                dim=1,
                            )
                            grad_norm = torch.linalg.vector_norm(
                                score_joint, dim=1
                            ).mean()
                            noise_norm = torch.linalg.vector_norm(
                                noise_joint, dim=1
                            ).mean()

                            norm_eps = torch.finfo(x_t_pos.dtype).eps
                            step_size = (
                                2.0
                                * corrector_alpha
                                * (
                                    corrector_snr
                                    * noise_norm
                                    / grad_norm.clamp_min(norm_eps)
                                ).square()
                            )
                            valid_norms = (
                                torch.isfinite(grad_norm)
                                & torch.isfinite(noise_norm)
                                & (grad_norm > norm_eps)
                            )
                            step_size = torch.where(
                                valid_norms,
                                torch.nan_to_num(
                                    step_size,
                                    nan=0.0,
                                    posinf=0.0,
                                    neginf=0.0,
                                ).clamp_min(0.0),
                                torch.zeros_like(step_size),
                            )

                            langevin_noise_scale = torch.sqrt(2.0 * step_size)
                            x_t_pos = (
                                x_t_pos
                                + step_size * score_pos
                                + langevin_noise_scale * noise_pos
                            )
                            x_t_type = (
                                x_t_type
                                + step_size * score_type
                                + langevin_noise_scale * noise_type
                            )

                            x_t_pos = torch.nan_to_num(
                                x_t_pos,
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            ).clamp(-coord_clip, coord_clip)
                            x_t_type = torch.nan_to_num(
                                x_t_type,
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            ).clamp(-type_clip, type_clip)
                            if self.molecule_com0:
                                x_t_pos = self._remove_mean_with_weights(
                                    x_t_pos,
                                    torch.ones(
                                        B,
                                        K,
                                        1,
                                        device=device,
                                        dtype=x_t_pos.dtype,
                                    ),
                                )
                            elif sample_com0:
                                p_type_dummy_t = self._safe_type_probs(x_t_type)
                                p_exist_t = 1.0 - p_type_dummy_t[:, :, 0:1]
                                weights = p_exist_t.clamp(
                                    min=sample_com0_weight_floor
                                )
                                x_t_pos = self._remove_mean_with_weights(
                                    x_t_pos, weights
                                )
            else:
                t_tile = t_norm.unsqueeze(1).expand(-1, K)
                eps_x_pred, eps_u_pred = model(xt, ut.squeeze(-1), t_tile)
                predictor_nfe += 1

                add_terminal_noise = bool(
                    not self.use_vp_sde
                    and t == 0
                    and terminal_variance_scale > 0.0
                )
                draw_reverse_noise = add_reverse_noise or add_terminal_noise
                z_x = (
                    draw_normal_like(xt, predictor_generator)
                    if draw_reverse_noise
                    else torch.zeros_like(xt)
                )
                z_u = (
                    draw_normal_like(ut, predictor_generator)
                    if draw_reverse_noise
                    else torch.zeros_like(ut)
                )
                if add_terminal_noise:
                    terminal_std_multiplier = math.sqrt(
                        terminal_variance_scale
                    )
                    z_x = z_x * terminal_std_multiplier
                    z_u = z_u * terminal_std_multiplier

                if use_nfe_matched_transition:

                    analytic_nonmol_noise_scale = analytic_noise_scale
                    if add_terminal_noise:
                        analytic_nonmol_noise_scale = torch.sqrt(beta[0])
                    xt = (
                        analytic_state_scale * xt
                        - analytic_eps_scale * eps_x_pred
                        + analytic_nonmol_noise_scale * z_x
                    )
                    ut = (
                        analytic_state_scale * ut
                        - analytic_eps_scale * eps_u_pred
                        + analytic_nonmol_noise_scale * z_u
                    )
                elif self.use_vp_sde:
                    xt = (
                        vp_state_scale * xt
                        - vp_score_scale * eps_x_pred
                        + vp_noise_scale * z_x
                    )
                    ut = (
                        vp_state_scale * ut
                        - vp_score_scale * eps_u_pred
                        + vp_noise_scale * z_u
                    )
                else:
                    xt = (
                        (xt - (1.0 - a_t) * eps_x_pred / torch.sqrt(1.0 - ab_t))
                        / torch.sqrt(a_t)
                        + torch.sqrt(b_t) * z_x
                    )
                    ut = (
                        (ut - (1.0 - a_t) * eps_u_pred / torch.sqrt(1.0 - ab_t))
                        / torch.sqrt(a_t)
                        + torch.sqrt(b_t) * z_u
                    )

                if do_corrector:
                    corrector_t_tile = corrector_t_norm.unsqueeze(1).expand(-1, K)
                    eps_clip = float(getattr(self.cfg, "sample_eps_clip", 1e3))
                    state_clip = float(
                        getattr(self.cfg, "sample_coord_clip", 1e3)
                    )
                    u_clip = float(getattr(self.cfg, "sample_u_clip", 1e3))
                    for _ in range(corrector_steps):
                        corr_eps_x, corr_eps_u = model(
                            xt, ut.squeeze(-1), corrector_t_tile
                        )
                        corrector_nfe += 1
                        corr_eps_x = torch.nan_to_num(
                            corr_eps_x, nan=0.0, posinf=0.0, neginf=0.0
                        ).clamp(-eps_clip, eps_clip)
                        corr_eps_u = torch.nan_to_num(
                            corr_eps_u, nan=0.0, posinf=0.0, neginf=0.0
                        ).clamp(-eps_clip, eps_clip)

                        score_x = -corr_eps_x / corrector_score_scale
                        score_u = -corr_eps_u / corrector_score_scale
                        noise_x = draw_normal_like(xt, corrector_generator)
                        noise_u = draw_normal_like(ut, corrector_generator)

                        score_joint = torch.cat(
                            [score_x.reshape(B, -1), score_u.reshape(B, -1)],
                            dim=1,
                        )
                        noise_joint = torch.cat(
                            [noise_x.reshape(B, -1), noise_u.reshape(B, -1)],
                            dim=1,
                        )
                        grad_norm = torch.linalg.vector_norm(
                            score_joint, dim=1
                        ).mean()
                        noise_norm = torch.linalg.vector_norm(
                            noise_joint, dim=1
                        ).mean()
                        norm_eps = torch.finfo(xt.dtype).eps
                        step_size = (
                            2.0
                            * corrector_alpha
                            * (
                                corrector_snr
                                * noise_norm
                                / grad_norm.clamp_min(norm_eps)
                            ).square()
                        )
                        valid_norms = (
                            torch.isfinite(grad_norm)
                            & torch.isfinite(noise_norm)
                            & (grad_norm > norm_eps)
                        )
                        step_size = torch.where(
                            valid_norms,
                            torch.nan_to_num(
                                step_size,
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            ).clamp_min(0.0),
                            torch.zeros_like(step_size),
                        )
                        noise_scale = torch.sqrt(2.0 * step_size)
                        xt = xt + step_size * score_x + noise_scale * noise_x
                        ut = ut + step_size * score_u + noise_scale * noise_u
                        xt = torch.nan_to_num(
                            xt, nan=0.0, posinf=0.0, neginf=0.0
                        ).clamp(-state_clip, state_clip)
                        ut = torch.nan_to_num(
                            ut, nan=0.0, posinf=0.0, neginf=0.0
                        ).clamp(-u_clip, u_clip)

        expected_predictor_nfe = (
            T
            if pc_spec.expected_predictor_nfe is None
            else pc_spec.expected_predictor_nfe
        )
        if predictor_nfe != expected_predictor_nfe:
            raise RuntimeError(
                f"pc_mode={pc_mode!r} produced {predictor_nfe} predictor NFEs; "
                f"expected {expected_predictor_nfe}."
            )
        if corrector_nfe != expected_corrector_nfe:
            raise RuntimeError(
                f"pc_mode={pc_mode!r} produced {corrector_nfe} corrector NFEs; "
                f"expected {expected_corrector_nfe}."
            )

        if predictor_schedule == "nfe_matched_analytic":
            predictor_solver = "analytic_respaced_posterior"
        elif self.use_vp_sde:
            predictor_solver = "reverse_vp_sde_euler"
        else:
            predictor_solver = "legacy_ddpm_adjacent_fixed_large"

        self.last_sampling_stats = {
            "pc_mode": pc_mode,

            "sampler_mode": pc_mode,
            "predictor_nfe": predictor_nfe,
            "corrector_nfe": corrector_nfe,
            "total": predictor_nfe + corrector_nfe,
            "total_nfe": predictor_nfe + corrector_nfe,
            "corrector_steps": corrector_steps,
            "corrector_levels": (
                corrector_nfe // corrector_steps if corrector_steps else 0
            ),
            "corrector_snr": corrector_snr,
            "corrector_start_time": corrector_start_time,
            "corrector_finish_time": corrector_finish_time,
            "predictor_schedule": predictor_schedule,
            "predictor_solver": predictor_solver,
            "corrector_window_basis": (
                "raw_time" if corrector_steps else None
            ),
            "noise_schedule": str(
                getattr(self.cfg, "noise_schedule", "linear")
            ),
            "existence_keep_mode": (
                "argmax" if self.dataset_type == "molecule" else self.existence_keep_mode
            ),
        }
        if corrector_overrides:
            self.last_sampling_stats["corrector_overrides"] = corrector_overrides

        if self.dataset_type == 'molecule':

            x0_hat_pos = x_t_pos

            x0_hat_type = x_t_type

            dummy_logit = torch.zeros(B, K, 1, device=device, dtype=x0_hat_type.dtype)
            logits_all = torch.cat([dummy_logit, x0_hat_type], dim=-1)

            safe_logits = torch.nan_to_num(logits_all, nan=-1e9, posinf=1e9, neginf=-1e9).clamp(min=-80.0, max=80.0)
            p_type_dummy = torch.softmax(safe_logits, dim=-1)
            p_type_dummy = torch.nan_to_num(p_type_dummy, nan=0.0, posinf=0.0, neginf=0.0)
            dummy_prob = p_type_dummy[:, :, 0:1]

            if self.debug_print:
                self._maybe_print_debug(
                    "sample_done",
                    t=0,
                    x0_hat_pos=x0_hat_pos,
                    x0_hat_type=x0_hat_type,
                    dummy_prob=dummy_prob,
                )

            pred_class = torch.argmax(safe_logits, dim=-1)

            keep = pred_class != 0
            if self.molecule_com0:
                x0_hat_pos = self._remove_mean_with_weights(x0_hat_pos, torch.ones(B, K, 1, device=device, dtype=x0_hat_pos.dtype))
            elif sample_com0:
                x0_hat_pos = self._remove_mean_with_weights(x0_hat_pos, keep.unsqueeze(-1).float())

            pred_type = (pred_class - 1).clamp(min=0)
            one_hot = torch.nn.functional.one_hot(pred_type, num_classes=num_atom_types).float()
            one_hot = one_hot * keep.unsqueeze(-1).float()

            out = torch.cat([x0_hat_pos, one_hot], dim=-1)
            out = out.clone()
            out[~keep] = float("-inf")
            return out, dummy_prob, keep

        m_prob = torch.sigmoid(ut)
        decoder_uniform = (
            torch.rand_like(m_prob)
            if predictor_generator is None
            else torch.rand(
                m_prob.shape,
                device=m_prob.device,
                dtype=m_prob.dtype,
                generator=predictor_generator,
            )
        )
        if self.existence_keep_mode == "threshold":
            keep = (m_prob > 0.5).squeeze(-1)
        else:
            keep = (decoder_uniform < m_prob).squeeze(-1)

        xt_final = xt.clone()
        xt_final[~keep] = float("-inf")

        return xt_final, m_prob, keep
