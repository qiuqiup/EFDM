import math
from typing import Optional

import torch
import torch.nn.functional as F

from methods.base_method import BaseMethod
from utils.nonempty_sampling import reject_empty_sets
from networks.psd_nets import PSDNet


class PSDMethod(BaseMethod):

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.cfg = cfg
        self.device = device

        self.point_dim = int(cfg.point_dim)
        self.T = int(cfg.T)
        self.max_num_points = int(cfg.max_num_points)

        self.bce_weight = float(getattr(cfg, "bce_weight", 1.0))
        self.nll_weight = float(getattr(cfg, "nll_weight", 1.0))

        self.domain_margin = float(getattr(cfg, "domain_margin", 0.05))

        self.model = PSDNet(cfg).to(device)

        schedule = str(getattr(cfg, "alpha_schedule", "cosine"))
        t_frac = torch.arange(self.T + 1, dtype=torch.float64) / self.T
        if schedule == "cosine":
            a_bar = torch.cos(t_frac * math.pi / 2.0) ** 2
        elif schedule == "linear":
            a_bar = 1.0 - t_frac
        else:
            raise ValueError(f"Unknown alpha_schedule: {schedule}")
        a_bar[0] = 1.0
        a_bar[-1] = 0.0
        self.register_buffer("alpha_bar", a_bar.float().to(device))

        d = self.point_dim
        self.register_buffer("domain_min", torch.zeros(d, device=device))
        self.register_buffer("domain_max", torch.ones(d, device=device))
        self.register_buffer("noise_expected_n", torch.tensor(0.0, device=device))
        self.register_buffer("data_n_max", torch.tensor(0.0, device=device))
        self.register_buffer("stats_fitted", torch.tensor(0, dtype=torch.long, device=device))

    @torch.no_grad()
    def fit_data_statistics(self, train_data: torch.Tensor) -> None:
        """
        Fit the noise process from training data: bounding box of the domain
        (with a small margin), the expected cardinality E[|X_0|], and the max
        cardinality (loss scale).

        Args:
            train_data: [N, K, d] tensor padded with +/-inf rows.
        """
        x = train_data.to(self.device)
        valid = ~torch.isinf(x[..., 0])
        pts = x[valid]
        if pts.numel() == 0:
            raise ValueError("PSDMethod.fit_data_statistics: no valid points in train_data")

        mn = pts.min(dim=0).values
        mx = pts.max(dim=0).values
        margin = (mx - mn).clamp(min=1e-6) * self.domain_margin
        self.domain_min.copy_(mn - margin)
        self.domain_max.copy_(mx + margin)
        counts = valid.sum(dim=1)
        self.noise_expected_n.fill_(counts.float().mean().item())
        self.data_n_max.fill_(float(counts.max().item()))
        self.stats_fitted.fill_(1)

    def _check_fitted(self):
        if int(self.stats_fitted.item()) != 1:
            raise RuntimeError(
                "PSDMethod: dataset statistics not fitted. Call "
                "fit_data_statistics(train_data) before training, or load a "
                "checkpoint that contains the fitted buffers."
            )

    def _loss_n_max(self) -> float:
        n = float(self.data_n_max.item())
        return n if n > 0 else float(self.max_num_points)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        span = (self.domain_max - self.domain_min).clamp(min=1e-6)
        return 2.0 * (x - self.domain_min) / span - 1.0

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        span = (self.domain_max - self.domain_min).clamp(min=1e-6)
        return (x + 1.0) / 2.0 * span + self.domain_min

    @staticmethod
    def _compact(x: torch.Tensor, valid: torch.Tensor, labels: torch.Tensor | None = None,
                 max_cols: int | None = None):
        """
        Reorder each set so valid points come first, then trim trailing all-pad
        columns (and optionally cap the number of points at max_cols; overflow
        points are dropped). Point order inside a set carries no meaning.
        """
        B, K = valid.shape
        order = torch.argsort((~valid).to(torch.int8), dim=1)
        valid = torch.gather(valid, 1, order)
        x = torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        if labels is not None:
            labels = torch.gather(labels, 1, order)

        k_new = max(int(valid.sum(dim=1).max().item()), 1)
        if max_cols is not None:
            k_new = min(k_new, int(max_cols))
        x = x[:, :k_new]
        valid = valid[:, :k_new]
        if labels is not None:
            labels = labels[:, :k_new]
        return x, valid, labels

    def _rand(self, shape, generator=None):
        if generator is None:
            return torch.rand(*shape, device=self.device)
        return torch.rand(*shape, device=self.device, generator=generator)

    def _randn(self, shape, generator=None):
        if generator is None:
            return torch.randn(*shape, device=self.device)
        return torch.randn(*shape, device=self.device, generator=generator)

    @staticmethod
    def _intensity_params(mix_w_raw: torch.Tensor, n_xt: torch.Tensor):
        """
        Official parameterization: w = softplus(raw) plays a double role.

        Returns:
            log_w: [B, M] normalized mixture log-weights
            lam:   [B]    cumulative intensity Lambda = sum(w) * (|X_t| + 1)
        """
        w_pos = F.softplus(mix_w_raw).clamp_min(1e-12)
        lam = w_pos.sum(dim=-1) * (n_xt.float() + 1.0)
        log_w = torch.log(w_pos) - torch.log(w_pos.sum(dim=-1, keepdim=True))
        return log_w, lam

    def _mixture_log_prob(self, x, log_w, mix_mean, mix_var):
        diff = x.unsqueeze(2) - mix_mean.unsqueeze(1)
        var = mix_var.unsqueeze(1)
        log_comp = (
            -0.5 * diff.pow(2) / var
            - 0.5 * torch.log(var)
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)
        return torch.logsumexp(log_w.unsqueeze(1) + log_comp, dim=-1)

    @staticmethod
    def _domain_mass(log_w, mix_mean, mix_var):
        std = mix_var.sqrt()
        zu = (1.0 - mix_mean) / std
        zl = (-1.0 - mix_mean) / std
        phi_u = torch.erfc(-zu * 0.7071067811865475) * 0.5
        phi_l = torch.erfc(-zl * 0.7071067811865475) * 0.5
        comp_mass = phi_u.prod(dim=-1) - phi_l.prod(dim=-1)
        mass = (log_w.exp() * comp_mass).sum(dim=-1)
        return mass.clamp(min=1e-8, max=1.0)

    def _sample_truncated_mixture(self, log_w, mix_mean, mix_var, counts, device):
        """
        Rejection-sample `counts[b]` points per batch row from the Gaussian
        mixture truncated to [-1, 1]^d (official `MixtureIntensity.sample`).

        Returns:
            pts:   [B, a_max, d]
            valid: [B, a_max] bool
        """
        B, M, d = mix_mean.shape
        a_max = max(int(counts.max().item()), 1)
        pts = torch.zeros(B, a_max, d, device=device)
        filled = torch.zeros(B, dtype=torch.long, device=device)
        probs = log_w.exp()
        b_idx = torch.arange(B, device=device)

        for attempt in range(64):
            remaining = (counts - filled).clamp(min=0)
            if int(remaining.max().item()) == 0:
                break
            m = int(remaining.max().item()) * (2 + attempt) + 8
            comp = torch.multinomial(probs, m, replacement=True)
            mu = torch.gather(mix_mean, 1, comp.unsqueeze(-1).expand(-1, -1, d))
            sd = torch.gather(mix_var, 1, comp.unsqueeze(-1).expand(-1, -1, d)).sqrt()
            cand = mu + sd * torch.randn_like(mu)
            inside = (cand.abs() <= 1.0).all(dim=-1)

            slot = filled.unsqueeze(1) + inside.cumsum(dim=1) - 1
            place = inside & (slot < counts.unsqueeze(1)) & (slot < a_max)
            rows = b_idx.unsqueeze(1).expand(-1, m)
            pts[rows[place], slot[place]] = cand[place]
            filled = torch.minimum(filled + inside.sum(dim=1), counts)
        else:
            remaining = (counts - filled).clamp(min=0)
            if int(remaining.max().item()) > 0:
                m = int(remaining.max().item())
                comp = torch.multinomial(probs, m, replacement=True)
                mu = torch.gather(mix_mean, 1, comp.unsqueeze(-1).expand(-1, -1, d))
                sd = torch.gather(mix_var, 1, comp.unsqueeze(-1).expand(-1, -1, d)).sqrt()
                cand = (mu + sd * torch.randn_like(mu)).clamp(-1.0, 1.0)
                for j in range(m):
                    sel = filled + j < counts
                    pts[b_idx[sel], (filled + j)[sel]] = cand[sel, j]

        valid = torch.arange(a_max, device=device).unsqueeze(0) < counts.unsqueeze(1)
        return pts, valid

    def _forward_loss(self, batch, generator=None):
        self._check_fitted()
        x0 = batch["x0"].to(self.device)
        B, K, d = x0.shape
        T = self.T

        valid0 = ~torch.isinf(x0[..., 0])
        x0n = self._normalize(torch.where(valid0.unsqueeze(-1), x0, torch.zeros_like(x0)))
        x0n = torch.where(valid0.unsqueeze(-1), x0n, torch.zeros_like(x0n))

        if generator is None:
            t_int = torch.randint(1, T + 1, (B,), device=self.device)
        else:
            t_int = torch.randint(1, T + 1, (B,), device=self.device, generator=generator)
        a_bar = self.alpha_bar[t_int]
        b_bar = 1.0 - a_bar

        keep = valid0 & (self._rand((B, K), generator) < a_bar.unsqueeze(1))
        removed = valid0 & ~keep

        lam_noise = b_bar * self.noise_expected_n
        m = torch.poisson(lam_noise, generator=generator).long().clamp(max=self.max_num_points)
        m_max = max(int(m.max().item()), 1)
        noise = self._rand((B, m_max, d), generator) * 2.0 - 1.0
        noise_valid = torch.arange(m_max, device=self.device).unsqueeze(0) < m.unsqueeze(1)

        xt = torch.cat([x0n, noise], dim=1)
        xt_valid = torch.cat([keep, noise_valid], dim=1)
        labels = torch.cat(
            [torch.ones(B, K, device=self.device), torch.zeros(B, m_max, device=self.device)],
            dim=1,
        )
        xt, xt_valid, labels = self._compact(xt, xt_valid, labels)
        xt = torch.where(xt_valid.unsqueeze(-1), xt, torch.zeros_like(xt))
        pad_mask = ~xt_valid

        t_norm = t_int.float() / T
        t_tile = t_norm.unsqueeze(1).expand(-1, xt.shape[1])

        keep_logits, mix_w_raw, mix_mean, mix_var = self.model(xt, t_tile, pad_mask)

        bce_el = F.binary_cross_entropy_with_logits(keep_logits, labels, reduction="none")
        bce = (bce_el * xt_valid.float()).sum() / B

        n_xt = xt_valid.sum(dim=1)
        log_w, lam = self._intensity_params(mix_w_raw, n_xt)
        log_p = self._mixture_log_prob(x0n, log_w, mix_mean, mix_var)
        mass = self._domain_mass(log_w, mix_mean, mix_var)
        ll = ((log_p + torch.log(lam).unsqueeze(1)) * removed.float()).sum(dim=1) \
            - lam * mass
        nll = -ll.mean()

        n_max_loss = self._loss_n_max()
        loss = (self.bce_weight * bce + self.nll_weight * nll) / n_max_loss
        log = {
            "loss": loss.item(),
            "bce": bce.item(),
            "nll": nll.item(),
            "lam_mean": (lam * mass).mean().item(),
        }
        return loss, log

    def training_step(self, batch):
        self.model.train()
        return self._forward_loss(batch, generator=None)

    @torch.no_grad()
    def eval_step(self, batch, generator=None):
        self.model.eval()
        return self._forward_loss(batch, generator=generator)

    @torch.no_grad()
    @reject_empty_sets
    def sample_sets(self, num_samples: int, device: Optional[torch.device] = None,
                    val_n: torch.Tensor | None = None):
        """
        Unconditional sampling. PSD models the cardinality itself, so `val_n`
        is ignored (kept for interface compatibility with the other methods).

        Returns:
            samples: [B, K, d] tensor in data space, padded with +inf rows.
        """
        self._check_fitted()
        if device is None:
            device = self.device

        base_model = self.model.module if hasattr(self.model, "module") else self.model
        model = base_model.to(device)
        model.eval()

        B = int(num_samples)
        d = self.point_dim
        T = self.T

        lam0 = self.noise_expected_n.to(device).expand(B)
        m = torch.poisson(lam0).long().clamp(min=0, max=self.max_num_points)
        K = max(int(m.max().item()), 1)
        x = torch.rand(B, K, d, device=device) * 2.0 - 1.0
        valid = torch.arange(K, device=device).unsqueeze(0) < m.unsqueeze(1)

        for t in range(T, 0, -1):
            a_prev = float(self.alpha_bar[t - 1].item())
            a_cur = float(self.alpha_bar[t].item())
            p_add = (a_prev - a_cur) / max(1.0 - a_cur, 1e-8)
            p_noise = (1.0 - a_prev) / max(1.0 - a_cur, 1e-8)

            x_in = torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))
            t_tile = torch.full((B, x.shape[1]), t / T, device=device)
            keep_logits, mix_w_raw, mix_mean, mix_var = model(x_in, t_tile, ~valid)

            p_keep = torch.sigmoid(keep_logits)
            is_data = valid & (torch.rand_like(p_keep) < p_keep)
            is_noise = valid & ~is_data

            n_xt = valid.sum(dim=1)
            log_w, lam = self._intensity_params(mix_w_raw, n_xt)
            mass = self._domain_mass(log_w, mix_mean, mix_var)
            n_add = torch.round(lam * mass).long().clamp(min=0)
            room = (self.max_num_points - is_data.sum(dim=1)).clamp(min=0)
            n_add = torch.minimum(n_add, room)

            add_pts, in_range = self._sample_truncated_mixture(
                log_w, mix_mean, mix_var, n_add, device
            )
            add_valid = in_range & (torch.rand(in_range.shape, device=device) < p_add)

            noise_kept = is_noise & (torch.rand(B, x.shape[1], device=device) < p_noise)

            x = torch.cat([x, add_pts], dim=1)
            valid = torch.cat([is_data | noise_kept, add_valid], dim=1)
            x, valid, _ = self._compact(x, valid, max_cols=self.max_num_points)

        dmin = self.domain_min.to(device)
        span = (self.domain_max.to(device) - dmin).clamp(min=1e-6)
        out = (x + 1.0) / 2.0 * span + dmin
        out = out.masked_fill(~valid.unsqueeze(-1), float("inf"))
        self.last_sampling_stats = {
            "pc_mode": "none",
            "sampler_mode": "native",
            "predictor_nfe": T,
            "corrector_nfe": 0,
            "total": T,
            "total_nfe": T,
        }
        return out
