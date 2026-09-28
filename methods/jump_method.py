import torch
import torch.nn.functional as F
from typing import Optional

from methods.base_method import BaseMethod
from utils.nonempty_sampling import reject_empty_sets
from networks.jump_nets import JumpNet, JumpNetDiT, resolve_jump_net_arch
from utils.pc_sampling import epsilon_langevin_corrector_step, resolve_discrete_pc_mode


def make_lambda_schedule(T: int, constant_lambda: float = 0.5, cutoff_ratio: float = 0.1) -> torch.Tensor:
    """
    Create lambda schedule with cutoff at both ends.
    Both start and end have cutoff_ratio * T steps with zero lambda (no deletion).
    """
    schedule = torch.zeros(T, dtype=torch.float32)
    start = int(cutoff_ratio * T)
    end = int((1 - cutoff_ratio) * T)
    schedule[start:end] = constant_lambda
    return schedule


_JUMP_PC_MODES = ("none", "1000p485c")


def _resolve_jump_pc_mode(pc_mode: str, T: int) -> str:
    """Resolve the complete continuous predictor/corrector sampling policy."""
    normalized = str(pc_mode).strip().lower()
    if normalized not in _JUMP_PC_MODES:
        choices = ", ".join(repr(name) for name in _JUMP_PC_MODES)
        raise ValueError(
            f"Unknown pc_mode {pc_mode!r}; expected one of {choices}."
        )
    if normalized == "1000p485c" and int(T) != 1000:
        raise ValueError(
            f"pc_mode='1000p485c' requires T=1000, got T={T}."
        )
    return normalized


def _jump_langevin_corrector_step(
    state: torch.Tensor,
    padding_mask: torch.Tensor,
    epsilon: torch.Tensor,
    *,
    marginal_std: torch.Tensor,
    alpha: torch.Tensor,
    snr: float,
    epsilon_clip: float,
    state_clip: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply one guarded, batch-SNR-calibrated step on active points."""
    return epsilon_langevin_corrector_step(
        state,
        epsilon,
        marginal_std=marginal_std,
        alpha=alpha,
        snr=snr,
        epsilon_clip=epsilon_clip,
        state_clip=state_clip,
        active_mask=~padding_mask,
        generator=generator,
    )


class JumpMethod(BaseMethod):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.cfg = cfg
        self.device = device

        self.dataset_type = getattr(cfg, "dataset_type", "trip")
        self.point_dim = cfg.point_dim
        if self.dataset_type == "molecule":
            self.atom_type_dim = getattr(cfg, "atom_type_dim", 5)
            self.charge_dim = getattr(cfg, "charge_dim", 1)
            self.total_dim = self.point_dim + self.atom_type_dim + self.charge_dim
        else:
            self.atom_type_dim = 0
            self.charge_dim = 0
            self.total_dim = self.point_dim

        if resolve_jump_net_arch(cfg) == "dit":
            self.model = JumpNetDiT(cfg).to(device)
        else:
            self.model = JumpNet(cfg).to(device)

        self.T = cfg.T
        beta = torch.linspace(1e-4, 2e-2, self.T, device=device)

        self.beta = beta
        self.alpha = (1.0 - beta)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        cutoff_ratio    = getattr(cfg, "cutoff_ratio", 0.1)
        adaptive        = getattr(cfg, "lambda_adaptive", False)

        self.adaptive = adaptive

        if not adaptive:
            max_n = getattr(cfg, "max_n", None)
            if max_n is None:
                max_n = getattr(cfg, "K_max", None) or getattr(cfg, "max_total_n", 1000)

            start = int(cutoff_ratio * self.T)
            end = int((1 - cutoff_ratio) * self.T)
            duration = max(end - start, 1)

            constant_lambda = (max_n - 1) / duration if duration > 0 else 0.0
            constant_lambda = max(constant_lambda, 0.0)

            lambda_schedule = make_lambda_schedule(
                T=self.T,
                constant_lambda=constant_lambda,
                cutoff_ratio=cutoff_ratio,
            )
            self.lambda_schedule = lambda_schedule.to(device)
            self.lambda_cumsum = torch.cumsum(self.lambda_schedule, dim=0)
        else:
            self.lambda_schedule = None
            self.lambda_cumsum = None
            self.cutoff_ratio = cutoff_ratio

        self.gamma_jump_rate = getattr(cfg, "gamma_jump_rate", 0.2)


    def _compute_n0(self, x0: torch.Tensor) -> torch.Tensor:
        valid = ~torch.isinf(x0[..., 0])
        return valid.sum(dim=1).long()

    def _forward_loss(self, batch, generator=None):

        x0 = batch["x0"].to(self.device)
        B, K, d = x0.shape

        n0 = self._compute_n0(x0)
        max_n = K

        if generator is None:
            t_int = torch.randint(0, self.T, (B,), device=self.device)
        else:
            t_int = torch.randint(0, self.T, (B,), device=self.device, generator=generator)
        t_norm = t_int.float() / (self.T - 1)

        a_bar = self.alpha_bar[t_int].view(B, 1, 1)

        if self.adaptive:
            lambda_schedule = torch.zeros(B, self.T, device=self.device)

            start = int(self.cutoff_ratio * self.T)
            end = int((1 - self.cutoff_ratio) * self.T)
            duration = max(end - start, 1)

            for i in range(B):
                n_init = n0[i].item()
                delta = max(n_init - 1, 0)

                schedule = torch.zeros(self.T, device=self.device)
                if duration > 0:
                    lambda_const = delta / duration
                    schedule[start:end] = lambda_const

                lambda_schedule[i] = schedule

            lambda_cumsum_batch = torch.cumsum(lambda_schedule, dim=1)
            lambda_cum = lambda_cumsum_batch[torch.arange(B, device=self.device), t_int]
        else:
            lambda_cum = self.lambda_cumsum[t_int]

        n_deleted = torch.poisson(lambda_cum)
        n_t = (n0 - n_deleted).clamp(min=1)

        if self.adaptive:
            lambda_t = lambda_schedule[torch.arange(B, device=self.device), t_int]
        else:
            lambda_t = self.lambda_schedule[t_int]

        padding_mask = torch.arange(max_n, device=self.device).expand(B, -1) >= n0.unsqueeze(-1)

        M_t = torch.zeros(B, max_n, dtype=torch.bool, device=self.device)
        for i in range(B):
            n0_i = int(n0[i].item())
            nt_i = int(n_t[i].item())
            indices = torch.randperm(n0_i)[:nt_i]
            M_t[i, indices] = True

        mask = padding_mask | (~M_t)

        if generator is None:
            eps = torch.randn_like(x0)
        else:
            eps = torch.randn_like(x0, generator=generator)

        x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * eps
        x_t = x_t.masked_fill(mask.unsqueeze(-1), 0.0)


        t_tile = t_norm.unsqueeze(-1).expand(-1, max_n)

        model_output = self.model(x_t, t_tile, mask)
        score_pred, predicted_log_n0, n0_logits, nearest_atom_logits, add_weights_or_insert_mu, add_feature_mean_or_insert_log_std, add_feature_log_std_or_none = model_output

        if self.dataset_type == "molecule":
            add_weights = add_weights_or_insert_mu
            add_feature_mean = add_feature_mean_or_insert_log_std
            add_feature_log_std = add_feature_log_std_or_none

            add_weights_probs = F.softmax(add_weights.masked_fill(mask, float('-inf')), dim=-1)
            add_weights_probs = add_weights_probs.unsqueeze(-1)

            feature_mean_agg = (add_feature_mean * add_weights_probs).sum(dim=1)
            feature_log_std_agg = (add_feature_log_std * add_weights_probs).sum(dim=1)

            insert_mu = feature_mean_agg[:, :self.point_dim]
            insert_log_std = feature_log_std_agg[:, :self.point_dim]

            atom_type_mean = feature_mean_agg[:, self.point_dim:self.point_dim + self.atom_type_dim]
            atom_logits = atom_type_mean

            charge_mu = feature_mean_agg[:, self.point_dim + self.atom_type_dim:]
            charge_log_std = feature_log_std_agg[:, self.point_dim + self.atom_type_dim:]
        else:
            insert_mu = add_weights_or_insert_mu
            insert_log_std = add_feature_mean_or_insert_log_std
            atom_logits = None
            charge_mu, charge_log_std = None, None

        expected_n0 = torch.exp(predicted_log_n0.clamp(max=11.5))

        if self.dataset_type == "molecule":
            score_pred_coords = score_pred[:, :, :self.point_dim]
            score_pred_atoms = score_pred[:, :, self.point_dim:self.point_dim + self.atom_type_dim]
            score_pred_charges = score_pred[:, :, self.point_dim + self.atom_type_dim:]
            eps_coords = eps[:, :, :self.point_dim]
            eps_atoms = eps[:, :, self.point_dim:self.point_dim + self.atom_type_dim]
            eps_charges = eps[:, :, self.point_dim + self.atom_type_dim:]

            coord_error = ((score_pred_coords - eps_coords) ** 2).sum(dim=-1)
            coord_error = coord_error.masked_fill(mask, 0.0)
            coord_loss = 0.5 * coord_error.sum(dim=1).mean()

            atom_error = ((score_pred_atoms - eps_atoms) ** 2).sum(dim=-1)
            atom_error = atom_error.masked_fill(mask, 0.0)
            atom_loss = 0.5 * atom_error.sum(dim=1).mean()

            charge_error = ((score_pred_charges - eps_charges) ** 2).sum(dim=-1)
            charge_error = charge_error.masked_fill(mask, 0.0)
            charge_loss = 0.5 * charge_error.sum(dim=1).mean()

            denoise_loss = coord_loss + atom_loss + charge_loss
        else:
            squared_error = ((score_pred - eps) ** 2).sum(dim=-1)
            squared_error = squared_error.masked_fill(mask, 0.0)
            denoise_loss = 0.5*squared_error.sum(dim=1).mean()

        reverse_jump_rate = lambda_t * F.relu(expected_n0 - n_t) / (lambda_cum + 1e-6)
        reverse_jump_rate_loss = reverse_jump_rate.mean()

        reverse_jump_rate_y = lambda_t * F.relu(expected_n0 - n_t + 1) / (lambda_cum + 1e-6)
        reverse_jump_penalty =  torch.log(reverse_jump_rate_y.clamp(min=1e-6))

        if insert_log_std is None:
            raise ValueError("insert_log_std is None. Model output may not match expected format.")
        insert_log_std = insert_log_std.clamp(min=-6.0, max=4.0)
        insert_std = torch.exp(insert_log_std)
        insert_target = torch.zeros(B, max_n, x0.shape[-1], device=self.device)
        insert_valid_mask = torch.zeros(B, max_n, dtype=torch.bool, device=self.device)
        for i in range(B):
            deleted = (~M_t[i]) & (~padding_mask[i])
            insert_valid_mask[i, deleted] = True
            insert_target[i, deleted] = (
                torch.sqrt(a_bar[i]) * x0[i, deleted] +
                torch.sqrt(1 - a_bar[i]) * eps[i, deleted]
            )

        if self.dataset_type == "molecule" and atom_logits is not None:
            insert_target_coords = insert_target[:, :, :self.point_dim]
            insert_target_charges = insert_target[:, :, self.point_dim + self.atom_type_dim:]

            insert_mu_coords_expanded = insert_mu[:, None, :].expand(-1, max_n, -1)
            insert_std_coords_expanded = insert_std[:, None, :].expand(-1, max_n, -1)
            insert_log_std_coords_expanded = insert_log_std[:, None, :].expand(-1, max_n, -1)

            nll_coords = (
                0.5 * ((insert_target_coords - insert_mu_coords_expanded) ** 2) / (insert_std_coords_expanded ** 2) +
                insert_log_std_coords_expanded
            )

            if charge_mu is not None and charge_log_std is not None:
                charge_std_expanded = torch.exp(charge_log_std[:, None, :]).expand(-1, max_n, -1)
                charge_log_std_expanded = charge_log_std[:, None, :].expand(-1, max_n, -1)
                charge_mu_expanded = charge_mu[:, None, :].expand(-1, max_n, -1)

                nll_charges = (
                    0.5 * ((insert_target_charges - charge_mu_expanded) ** 2) / (charge_std_expanded ** 2) +
                    charge_log_std_expanded
                )
                nll_charges_masked = nll_charges.sum(dim=-1)
                nll_charges_masked = nll_charges_masked.masked_fill(~insert_valid_mask, 0.0)
            else:
                nll_charges_masked = torch.zeros(B, max_n, device=self.device)

            atom_logits_expanded = atom_logits[:, None, :].expand(-1, max_n, -1)
            true_atom_indices = torch.zeros(B, max_n, dtype=torch.long, device=self.device)
            for i in range(B):
                deleted = (~M_t[i]) & (~padding_mask[i])
                if deleted.any():
                    x0_atoms_deleted = x0[i, deleted, self.point_dim:self.point_dim + self.atom_type_dim]
                    true_atom_indices[i, deleted] = x0_atoms_deleted.argmax(dim=-1)

            atom_ce = F.cross_entropy(
                atom_logits_expanded.view(-1, self.atom_type_dim),
                true_atom_indices.view(-1),
                reduction='none'
            ).view(B, max_n)

            nll_coords_masked = nll_coords.sum(dim=-1)
            nll_coords_masked = nll_coords_masked.masked_fill(~insert_valid_mask, 0.0)
            atom_ce_masked = atom_ce.masked_fill(~insert_valid_mask, 0.0)

            insert_loss = (nll_coords_masked.sum() + atom_ce_masked.sum() + nll_charges_masked.sum()) / insert_valid_mask.sum().clamp(min=1)
        else:
            insert_mu_expanded = insert_mu[:, None, :].expand(-1, max_n, -1)
            insert_std_expanded = insert_std[:, None, :].expand(-1, max_n, -1)
            insert_log_std_expanded = insert_log_std[:, None, :].expand(-1, max_n, -1)

            nll = (
                0.5 * ((insert_target - insert_mu_expanded) ** 2) / (insert_std_expanded ** 2) +
                insert_log_std_expanded
            )

            insert_loss = nll[insert_valid_mask].sum(dim=-1).sum() / insert_valid_mask.sum().clamp(min=1)

        jump_rate_loss = F.l1_loss(torch.log1p(expected_n0), torch.log1p(n0.float()))


        penalty_weight = (1 - ((t_int - 1) / t_int).clamp(min=1e-6).pow((n0 - n_t).clamp(min=0)))
        penalty_weight_expanded = penalty_weight.view(-1, 1)

        masked_insert_loss = insert_loss * penalty_weight_expanded
        masked_reverse_penalty = reverse_jump_penalty * penalty_weight


        loss = (
            denoise_loss
            + reverse_jump_rate_loss
            - masked_reverse_penalty.mean()
            + masked_insert_loss.mean()
            + self.gamma_jump_rate * jump_rate_loss
        )

        log = {
            "loss": float(loss.item()),
            "denoise_loss": float(denoise_loss.item()),
            "reverse_jump_rate_loss": float(reverse_jump_rate_loss.item()),
            "reverse_jump_penalty": float(masked_reverse_penalty.mean().item()),
            "insert_loss": float(insert_loss.item()),
            "jump_rate_loss": float(jump_rate_loss.item()),
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
    @reject_empty_sets(batch_parameters=("val_n",))
    def sample_sets(
        self,
        num_samples: int,
        device: Optional[torch.device] = None,
        val_n: Optional[torch.Tensor] = None,
        fixed_n0: bool = False,
        pc_mode: str = "none",
        corrector_snr: float | None = None,
        corrector_steps: int | None = None,
        corrector_generator: torch.Generator | None = None,
    ):
        """Sample sets with optional post-jump continuous correction.

        ``pc_mode='none'`` preserves the legacy sampler. ``'1000p485c'`` keeps
        its 1000 adjacent predictor steps and adds five Langevin evaluations at
        each of the 97 low-noise levels in the open ``(.003, .1)`` window.
        """
        if device is None:
            device = self.device
        model = self.model.to(device)
        model.eval()

        B = num_samples
        d = self.total_dim
        T = self.T
        pc_mode = _resolve_jump_pc_mode(pc_mode, T)
        pc_spec = resolve_discrete_pc_mode(
            pc_mode, T, corrector_snr=corrector_snr, corrector_steps=corrector_steps
        )
        corrector_steps = pc_spec.corrector_steps_per_level
        corrector_snr = pc_spec.corrector_snr
        corrector_start_time = pc_spec.corrector_start_time
        corrector_finish_time = pc_spec.corrector_finish_time
        expected_corrector_nfe = pc_spec.expected_corrector_nfe
        initialization_nfe = 0
        predictor_nfe = 0
        corrector_nfe = 0
        corrector_levels = 0
        alpha = self.alpha
        beta = self.beta
        alpha_bar = self.alpha_bar

        if not self.adaptive:
            lambda_schedule = self.lambda_schedule
            lambda_cumsum = self.lambda_cumsum
        else:
            lambda_schedule = None
            lambda_cumsum = None
            start = int(self.cutoff_ratio * T)
            end = int((1 - self.cutoff_ratio) * T)
            duration = max(end - start, 1)

        max_total_n = int(getattr(self.cfg, "max_total_n", 1000))

        init_n_cfg = int(getattr(self.cfg, "init_n", 1))
        if isinstance(init_n_cfg, int):
            init_n = torch.full((B,), init_n_cfg, dtype=torch.long, device=device)
        elif torch.is_tensor(init_n_cfg):
            init_n = init_n_cfg.to(device).long()
            assert init_n.shape[0] == B
        else:
            raise ValueError("cfg.init_n must be int or torch.Tensor")

        max_init_n = int(init_n.max().item())
        X = torch.full((B, max_init_n, d), float('inf'), device=device)
        mask = torch.ones(B, max_init_n, dtype=torch.bool, device=device)

        for i in range(B):
            n = int(init_n[i].item())
            if n > 0:
                X[i, :n] = torch.randn(n, d, device=device)
                mask[i, :n] = False

        sample_fixed_n0 = bool(fixed_n0) or (val_n is not None)
        expected_n0_fixed = None
        if sample_fixed_n0:
            if val_n is not None:
                val_n_t = val_n.to(device).float()
                if val_n_t.numel() == 1:
                    expected_n0_fixed = val_n_t.view(1).repeat(B)
                else:
                    expected_n0_fixed = val_n_t.view(-1)
                    assert expected_n0_fixed.shape[0] == B, "val_n must be scalar or have shape [B]"
            else:
                B0, L0, _ = X.shape
                if T > 1:
                    t0 = torch.full((B0, L0), 1.0, device=device)
                else:
                    t0 = torch.zeros((B0, L0), device=device)
                X0_in = X.masked_fill(mask.unsqueeze(-1), 0.0)
                model_output0 = model(X0_in, t0, mask)
                initialization_nfe += 1
                _, predicted_log_n0_0, _, _, _, _, _ = model_output0
                expected_n0_fixed = torch.exp(predicted_log_n0_0)

            expected_n0_fixed = expected_n0_fixed.clamp(min=1.0, max=float(max_total_n))

            if self.adaptive:
                lambda_schedule_batch = torch.zeros(B, T, device=device)
                if duration > 0 and end > start:
                    lambda_const = (expected_n0_fixed - 1.0).clamp(min=0.0) / float(duration)
                    lambda_schedule_batch[:, start:end] = lambda_const.unsqueeze(1)
                lambda_cumsum_batch = torch.cumsum(lambda_schedule_batch, dim=1)

        delta_t = 1.0
        eps = 1e-6

        for t in reversed(range(T)):
            Bc, L, _ = X.shape
            t_tensor = torch.full((Bc, L), float(t) / (T - 1), device=device)

            X_in = X.masked_fill(mask.unsqueeze(-1), 0.0)

            model_output = model(X_in, t_tensor, mask)
            predictor_nfe += 1
            score, predicted_log_n0, n0_logits, nearest_atom_logits, add_weights_or_insert_mu, add_feature_mean_or_insert_log_std, add_feature_log_std_or_none = model_output

            if self.dataset_type == "molecule":
                add_weights = add_weights_or_insert_mu
                add_feature_mean = add_feature_mean_or_insert_log_std
                add_feature_log_std = add_feature_log_std_or_none

                add_weights_probs = F.softmax(add_weights.masked_fill(mask, float('-inf')), dim=-1)
                add_weights_probs = add_weights_probs.unsqueeze(-1)

                feature_mean_agg = (add_feature_mean * add_weights_probs).sum(dim=1)
                feature_log_std_agg = (add_feature_log_std * add_weights_probs).sum(dim=1)

                insert_mu = feature_mean_agg[:, :self.point_dim]
                insert_log_std = feature_log_std_agg[:, :self.point_dim]

                atom_type_mean = feature_mean_agg[:, self.point_dim:self.point_dim + self.atom_type_dim]
                atom_logits = atom_type_mean

                charge_mu = feature_mean_agg[:, self.point_dim + self.atom_type_dim:]
                charge_log_std = feature_log_std_agg[:, self.point_dim + self.atom_type_dim:]
            else:
                insert_mu = add_weights_or_insert_mu
                insert_log_std = add_feature_mean_or_insert_log_std
                atom_logits = None
                charge_mu, charge_log_std = None, None

            if sample_fixed_n0:
                expected_n0 = expected_n0_fixed
            else:
                expected_n0 = torch.exp(predicted_log_n0)
            if insert_log_std is None:
                raise ValueError("insert_log_std is None. Model output may not match expected format.")
            insert_std = torch.exp(insert_log_std)

            a_t = alpha[t]
            b_t = beta[t]
            ab_t = alpha_bar[t]
            noise = torch.randn_like(X_in) if t > 0 else torch.zeros_like(X_in)

            X = (
                (X_in - (1.0 - a_t) * score / torch.sqrt(1.0 - ab_t + 1e-12))
                / torch.sqrt(a_t + 1e-12)
                + torch.sqrt(b_t) * noise
            )
            X = X.masked_fill(mask.unsqueeze(-1), 0.0)

            n_t = (~mask).sum(dim=1)

            if self.adaptive:
                if sample_fixed_n0:
                    lam_t = lambda_schedule_batch[:, t]
                    lambda_cum = lambda_cumsum_batch[:, t]
                else:
                    lambda_schedule_batch = torch.zeros(Bc, T, device=device)
                    for b in range(Bc):
                        n_init = max(expected_n0[b].item(), 1.0)
                        delta = max(n_init - 1.0, 0.0)

                        schedule = torch.zeros(T, device=device)
                        if duration > 0 and delta > 0:
                            lambda_const = delta / duration
                            schedule[start:end] = lambda_const
                        lambda_schedule_batch[b] = schedule

                    lambda_cumsum_batch = torch.cumsum(lambda_schedule_batch, dim=1)
                    lam_t = lambda_schedule_batch[:, t]
                    lambda_cum = lambda_cumsum_batch[:, t]
            else:
                lam_t = lambda_schedule[t].item()
                lambda_cum = lambda_cumsum[t].item()
                lam_t = torch.full((Bc,), lam_t, device=device)
                lambda_cum = torch.full((Bc,), lambda_cum, device=device)

            jump_rate = lam_t * (expected_n0 - n_t.float()) / (lambda_cum + eps)
            jump_rate = torch.clamp(jump_rate, min=0.0)
            jump_prob = 1.0 - torch.exp(-jump_rate * delta_t)
            jump_prob = torch.clamp(jump_prob, 0.0, 1.0)

            if self.adaptive and getattr(self.cfg, "sample_debug", False) and (t % 100 == 0 or t == 0):
                print(f"[t={t:>3}] Adaptive mode debug:")
                print(f"  expected_n0: {expected_n0.cpu().numpy()}")
                print(f"  n_t: {n_t.cpu().numpy()}")
                print(f"  lam_t: {lam_t.cpu().numpy()}")
                print(f"  lambda_cum: {lambda_cum.cpu().numpy()}")
                print(f"  jump_rate: {jump_rate.cpu().numpy()}")
                print(f"  jump_prob: {jump_prob.cpu().numpy()}")
                if Bc > 0:
                    first_schedule = lambda_schedule_batch[0].cpu().numpy()
                    first_n_init = max(expected_n0[0].item(), 1.0)
                    first_delta = max(first_n_init - 1.0, 0.0)
                    print(f"  lambda_schedule[0] (first sample): n_init={first_n_init:.2f}, delta={first_delta:.2f}, max={first_schedule.max():.6f}, mean={first_schedule[first_schedule>0].mean() if (first_schedule>0).any() else 0:.6f}")
                    print(f"  lambda_schedule[0] at t={t}: {first_schedule[t]:.6f}")

            new_X = []
            new_mask = []
            num_insertions = 0
            for b in range(Bc):
                u = torch.rand((), device=device).item()
                if u < float(jump_prob[b].item()) and X.shape[1] < max_total_n:
                    num_insertions += 1
                    if self.dataset_type == "molecule" and atom_logits is not None:
                        coord_sample = insert_mu[b] + insert_std[b] * torch.randn_like(insert_mu[b])

                        atom_probs = F.softmax(atom_logits[b], dim=-1)
                        atom_type_idx = torch.multinomial(atom_probs, 1).item()
                        atom_onehot = torch.zeros(self.atom_type_dim, device=device)
                        atom_onehot[atom_type_idx] = 1.0

                        if charge_mu is not None and charge_log_std is not None:
                            charge_std = torch.exp(charge_log_std[b])
                            charge_sample = charge_mu[b] + charge_std * torch.randn_like(charge_mu[b])
                        else:
                            charge_sample = torch.zeros(self.charge_dim, device=device)

                        new_pt = torch.cat([coord_sample, atom_onehot, charge_sample], dim=0)
                    else:
                        new_pt = insert_mu[b] + insert_std[b] * torch.randn_like(insert_mu[b])

                    x_b = torch.cat([X[b], new_pt.unsqueeze(0)], dim=0)
                    m_b = torch.cat([mask[b], torch.tensor([False], device=device)], dim=0)
                else:
                    x_b = X[b]
                    m_b = mask[b]
                new_X.append(x_b)
                new_mask.append(m_b)

            max_len = max(x.shape[0] for x in new_X)
            padded_X = []
            padded_mask = []
            for x, m in zip(new_X, new_mask):
                pad_len = max_len - x.shape[0]
                x_pad = F.pad(x, (0, 0, 0, pad_len), value=0.0)
                m_pad = F.pad(m, (0, pad_len), value=True)
                padded_X.append(x_pad)
                padded_mask.append(m_pad)

            X = torch.stack(padded_X, dim=0)
            mask = torch.stack(padded_mask, dim=0)

            corrector_index = t - 1
            corrector_source_time = float(t) / float(T - 1)
            do_corrector = (
                corrector_steps > 0
                and corrector_index >= 0
                and corrector_finish_time
                < corrector_source_time
                < corrector_start_time
            )
            if do_corrector:
                corrector_levels += 1
                corrector_time = float(corrector_index) / float(T - 1)
                corrector_t_tensor = torch.full(
                    (Bc, X.shape[1]), corrector_time, device=device
                )
                corrector_std = torch.sqrt(
                    (1.0 - alpha_bar[corrector_index]).clamp_min(0.0)
                ).clamp_min(0.001)
                corrector_alpha = alpha[corrector_index]
                eps_clip = float(getattr(self.cfg, "sample_eps_clip", 1e3))
                state_clip = float(getattr(self.cfg, "sample_coord_clip", 1e3))

                for _ in range(corrector_steps):
                    corrector_input = X.masked_fill(mask.unsqueeze(-1), 0.0)
                    corr_eps = model(
                        corrector_input, corrector_t_tensor, mask
                    )[0]
                    corrector_nfe += 1

                    X = _jump_langevin_corrector_step(
                        X,
                        mask,
                        corr_eps,
                        marginal_std=corrector_std,
                        alpha=corrector_alpha,
                        snr=corrector_snr,
                        generator=corrector_generator,
                        epsilon_clip=eps_clip,
                        state_clip=state_clip,
                    )

            if getattr(self.cfg, "sample_debug", False) and (t % 100 == 0 or t == 0):
                if (~mask).any():
                    mn = X[~mask].min().item()
                    mx = X[~mask].max().item()
                else:
                    mn, mx = 0.0, 0.0
                cnt = (~mask).sum(dim=1)
                print(f"[t={t:>3}] shape={X.shape}, range=({mn:.3f},{mx:.3f}), "
                    f"valid=(max:{cnt.max().item()}, min:{cnt.min().item()})")
                if self.adaptive:
                    print(f"  Insertions in this step: {num_insertions}/{Bc}")

        if predictor_nfe != T:
            raise RuntimeError(
                f"pc_mode={pc_mode!r} produced {predictor_nfe} predictor NFEs; "
                f"expected {T}."
            )
        if corrector_nfe != expected_corrector_nfe:
            raise RuntimeError(
                f"pc_mode={pc_mode!r} produced {corrector_nfe} corrector NFEs; "
                f"expected {expected_corrector_nfe}."
            )

        total_nfe = initialization_nfe + predictor_nfe + corrector_nfe
        self.last_sampling_stats = {
            "pc_mode": pc_mode,
            "sampler_mode": pc_mode,
            "initialization_nfe": initialization_nfe,
            "predictor_nfe": predictor_nfe,
            "corrector_nfe": corrector_nfe,
            "total": total_nfe,
            "total_nfe": total_nfe,
            "corrector_steps": corrector_steps,
            "corrector_levels": corrector_levels,
            "corrector_snr": corrector_snr,
            "corrector_start_time": corrector_start_time,
            "corrector_finish_time": corrector_finish_time,
            "predictor_schedule": "uniform",
            "predictor_solver": "legacy_ddpm_adjacent_fixed_large",
            "corrector_window_basis": (
                "raw_time" if corrector_steps else None
            ),
            "corrector_support": (
                "post_jump_active" if corrector_steps else None
            ),
        }

        final_points = []
        for x, m in zip(X, mask):
            final_points.append(x[~m])

        max_len = max(p.shape[0] for p in final_points) if len(final_points) > 0 else 1

        out = []
        for pts in final_points:
            pad_len = max_len - pts.shape[0]
            pts_pad = F.pad(pts, (0, 0, 0, pad_len), value=float("-inf"))
            out.append(pts_pad)

        X_out = torch.stack(out, dim=0)

        if self.dataset_type == "molecule":
            coord_part = X_out[:, :, :self.point_dim]
            atom_part = X_out[:, :, self.point_dim:self.point_dim + self.atom_type_dim]
            charge_part = X_out[:, :, self.point_dim + self.atom_type_dim:]

            sampling_method = getattr(self.cfg, "atom_type_sampling", "multinomial")

            valid_mask = ~torch.isinf(X_out[:, :, 0])

            if sampling_method == "argmax":
                atom_indices = atom_part.argmax(dim=-1)
                atom_onehot = F.one_hot(atom_indices, num_classes=self.atom_type_dim).float()
            else:
                atom_probs = F.softmax(atom_part, dim=-1)

                atom_probs_flat = atom_probs.view(-1, self.atom_type_dim)
                valid_mask_flat = valid_mask.view(-1)

                atom_indices_flat = torch.multinomial(atom_probs_flat, 1).squeeze(-1)

                atom_onehot_flat = F.one_hot(atom_indices_flat, num_classes=self.atom_type_dim).float()

                atom_onehot_flat = atom_onehot_flat * valid_mask_flat.unsqueeze(-1)

                atom_onehot = atom_onehot_flat.view(atom_part.shape[0], atom_part.shape[1], self.atom_type_dim)

            X_out = torch.cat([coord_part, atom_onehot, charge_part], dim=-1)

        return X_out
