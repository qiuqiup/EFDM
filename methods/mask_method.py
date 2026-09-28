import torch
import torch.nn.functional as F
from typing import Optional


from methods.base_method import BaseMethod
from utils.nonempty_sampling import reject_empty_sets
from utils.noise_schedule import build_from_cfg as _build_noise_schedule
from utils.pc_sampling import (
    build_discrete_sampling_stats,
    discrete_corrector_target,
    epsilon_langevin_corrector_step,
    resolve_discrete_pc_mode,
)
from networks.mask_nets import MaskNet, MaskNetDiT
from networks.mask_nets_with_egnn import MaskNetWithEGNN
from utils.sampling import sample_n_from_val_n
from utils.n_distribution import NDistributionMixture

class MaskMethod(BaseMethod):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.cfg = cfg
        self.device = device

        self.dataset_type = getattr(cfg, "dataset_type", "trip")
        self.point_dim = int(cfg.point_dim)
        if self.dataset_type == "molecule":
            if hasattr(cfg, "num_atom_types"):
                self.atom_type_dim = int(cfg.num_atom_types)
                self.coord_dim = 3
                self.total_dim = self.coord_dim + self.atom_type_dim
            elif self.point_dim > 3:
                self.coord_dim = 3
                self.atom_type_dim = int(self.point_dim - 3)
                self.total_dim = int(self.point_dim)
            else:
                self.coord_dim = int(self.point_dim)
                self.atom_type_dim = int(getattr(cfg, "atom_type_dim", 5))
                self.total_dim = self.coord_dim + self.atom_type_dim
        else:
            self.coord_dim = int(self.point_dim)
            self.atom_type_dim = 0
            self.total_dim = int(self.point_dim)

        net_arch = getattr(cfg, "mask_net_arch", "legacy")
        if net_arch == "egnn":
            if self.dataset_type != "molecule":
                raise ValueError(
                    "mask_net_arch='egnn' requires dataset_type='molecule'; "
                    f"got {self.dataset_type!r}."
                )
            self.model = MaskNetWithEGNN(cfg).to(device)
        elif net_arch == "dit":
            self.model = MaskNetDiT(cfg).to(device)
        else:
            self.model = MaskNet(cfg).to(device)

        self.molecule_com0 = bool(getattr(cfg, "molecule_com0", False))

        n_components = getattr(cfg, "n_distribution_components", 5)
        self.n_distribution = NDistributionMixture(
            n_components=n_components,
            random_state=None
        )
        self.n_distribution_fitted = False

    @staticmethod
    def _remove_mean_with_mask(x: torch.Tensor, node_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Subtract the mean over valid atoms only; padded rows are left at 0.

        node_mask: [B,K] or [B,K,1] float, 1 = valid.
        """
        if node_mask.dim() == 2:
            node_mask = node_mask.unsqueeze(-1)
        denom = node_mask.sum(dim=1, keepdim=True).clamp(min=eps)
        mean = (x * node_mask).sum(dim=1, keepdim=True) / denom
        return (x - mean) * node_mask

    def _forward_loss(self, batch, generator=None):
        x0 = batch["x0"].to(self.device)

        B, K, d = x0.shape
        T = self.cfg.T
        beta, alpha, alpha_bar = _build_noise_schedule(self.cfg, T, device=self.device)

        mask_x = torch.isinf(x0[..., 0])
        mask_valid = ~mask_x
        mask_valid_f = mask_valid.unsqueeze(-1).float()
        x0_clean = torch.where(mask_valid_f.bool(), x0, torch.zeros_like(x0))

        if generator is None:
            t_int = torch.randint(0, T, (B,), device=self.device)
        else:
            t_int = torch.randint(0, T, (B,), device=self.device, generator=generator)

        a_bar = alpha_bar[t_int].view(B, 1, 1)
        t_norm = t_int.float() / (T - 1)
        t_tile = t_norm.unsqueeze(1).expand(-1, K)


        if self.dataset_type == "molecule":
            if d != self.total_dim:
                raise ValueError(f"MaskMethod(molecule): expected x0 dim {self.total_dim}, got d={d}")

            x0_pos = x0_clean[:, :, :self.coord_dim]
            x0_type = x0_clean[:, :, self.coord_dim:self.coord_dim + self.atom_type_dim]

            if generator is None:
                eps_pos = torch.randn(B, K, self.coord_dim, device=self.device)
                eps_type = torch.randn(B, K, self.atom_type_dim, device=self.device)
            else:
                eps_pos = torch.randn(B, K, self.coord_dim, device=self.device, generator=generator)
                eps_type = torch.randn(B, K, self.atom_type_dim, device=self.device, generator=generator)

            if self.molecule_com0:
                x0_pos = self._remove_mean_with_mask(x0_pos, mask_valid_f)
                eps_pos = self._remove_mean_with_mask(eps_pos, mask_valid_f)

            x_t_pos = torch.sqrt(a_bar) * x0_pos + torch.sqrt(1.0 - a_bar) * eps_pos
            x_t_type = torch.sqrt(a_bar) * x0_type + torch.sqrt(1.0 - a_bar) * eps_type
            if self.molecule_com0:
                x_t_pos = self._remove_mean_with_mask(x_t_pos, mask_valid_f)
            x_t = torch.cat([x_t_pos, x_t_type], dim=-1)

            eps_pos_pred, eps_type_pred = self.model(x_t, t_tile, mask_x)
            pos_loss = ((eps_pos_pred - eps_pos)[mask_valid] ** 2).mean()
            type_loss = ((eps_type_pred - eps_type)[mask_valid] ** 2).mean()
            loss = pos_loss + type_loss
            log = {
                "loss": loss.item(),
                "pos_loss": pos_loss.item(),
                "type_loss": type_loss.item(),
            }
        else:
            if generator is None:
                eps_x = torch.randn(B, K, d, device=self.device)
            else:
                eps_x = torch.randn(B, K, d, device=self.device, generator=generator)
            x_t = torch.sqrt(a_bar) * x0_clean + torch.sqrt(1.0 - a_bar) * eps_x
            eps_pred = self.model(x_t, t_tile, mask_x)
            loss = ((eps_pred - eps_x)[mask_valid].pow(2)).mean()
            log = {"loss": loss.item()}
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

    def fit_n_distribution(self, train_n: torch.Tensor) -> None:
        """
        Fit the mixture model to learn the distribution of n from training data.

        Args:
            train_n: Tensor of shape [N] containing number of points for each training sample
        """
        self.n_distribution.fit(train_n)
        self.n_distribution_fitted = True

    @torch.no_grad()
    @reject_empty_sets
    def sample_sets(
        self,
        num_samples: int,
        device: Optional[torch.device] = None,
        val_n: torch.Tensor | None = None,
        pc_mode: str = "none",
        corrector_snr: float | None = None,
        corrector_steps: int | None = None,
        corrector_generator: torch.Generator | None = None,
    ):
        T = int(self.cfg.T)
        pc_spec = resolve_discrete_pc_mode(
            pc_mode, T, corrector_snr=corrector_snr, corrector_steps=corrector_steps
        )
        if self.dataset_type == "molecule" and pc_spec.name != "none":
            raise ValueError(
                "MaskMethod corrector sampling is currently supported only for "
                "non-molecule datasets."
            )

        if self.n_distribution_fitted:
            n_samples = self.n_distribution.sample(num_samples, device)
        else:
            assert val_n is not None, "MaskMethod.sample_sets needs either fitted n_distribution or val_n"
            assert val_n.numel() > 0, "val_n must not be empty when n_distribution is not fitted"
            n_samples = sample_n_from_val_n(val_n, num_samples, device)

        if device is None:
            device = self.device

        B = int(num_samples)
        d = int(self.total_dim)

        max_len = int(n_samples.max().item())

        beta, alpha, alpha_bar = _build_noise_schedule(self.cfg, T, device=device)

        base_model = self.model.module if hasattr(self.model, "module") else self.model
        model = base_model.to(device)
        model.eval()

        xt = torch.full((B, max_len, d), float("-inf"), device=device)

        for i in range(B):
            n = int(n_samples[i].item())
            xt[i, :n] = torch.randn(n, d, device=device)

        padding_mask = torch.isinf(xt[..., 0])
        xt = xt.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        com0 = bool(self.molecule_com0) and self.dataset_type == "molecule"
        valid_f = (~padding_mask).unsqueeze(-1).to(xt.dtype)
        if com0:
            xt[:, :, :self.coord_dim] = self._remove_mean_with_mask(
                xt[:, :, :self.coord_dim], valid_f
            )

        predictor_nfe = 0
        corrector_nfe = 0
        for t in reversed(range(T)):
            t_norm = t / (T - 1)
            t_tile = torch.full((B, max_len), t_norm, device=device)

            if self.dataset_type == "molecule":
                eps_pos_pred, eps_type_pred = model(xt, t_tile, padding_mask)
                if com0:
                    eps_pos_pred = self._remove_mean_with_mask(eps_pos_pred, valid_f)
                eps_pred = torch.cat([eps_pos_pred, eps_type_pred], dim=-1)
            else:
                eps_pred = model(xt, t_tile, padding_mask)
            predictor_nfe += 1

            z = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)
            if com0 and t > 0:
                z[:, :, :self.coord_dim] = self._remove_mean_with_mask(
                    z[:, :, :self.coord_dim], valid_f
                )
            a_t = alpha[t]
            ab_t = alpha_bar[t]
            b_t = beta[t]

            xt = (
                (xt - (1.0 - a_t) * eps_pred / torch.sqrt(1.0 - ab_t + 1e-12))
                / torch.sqrt(a_t + 1e-12)
                + torch.sqrt(b_t) * z
            )

            xt = xt.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            if com0:
                xt[:, :, :self.coord_dim] = self._remove_mean_with_mask(
                    xt[:, :, :self.coord_dim], valid_f
                )

            corrector_target = discrete_corrector_target(
                pc_spec, source_index=t, T=T
            )
            if corrector_target is not None:
                corrector_index, corrector_time = corrector_target
                corrector_t_tile = torch.full(
                    (B, max_len), corrector_time, device=device
                )
                corrector_std = torch.sqrt(
                    (1.0 - alpha_bar[corrector_index]).clamp_min(0.0)
                ).clamp_min(0.001)
                corrector_alpha = alpha[corrector_index]
                eps_clip = float(getattr(self.cfg, "sample_eps_clip", 1e3))
                state_clip = float(getattr(self.cfg, "sample_coord_clip", 1e3))

                for _ in range(pc_spec.corrector_steps_per_level):
                    corr_eps = model(xt, corrector_t_tile, padding_mask)
                    corrector_nfe += 1
                    xt = epsilon_langevin_corrector_step(
                        xt,
                        corr_eps,
                        marginal_std=corrector_std,
                        alpha=corrector_alpha,
                        snr=pc_spec.corrector_snr,
                        generator=corrector_generator,
                        epsilon_clip=eps_clip,
                        state_clip=state_clip,
                        active_mask=~padding_mask,
                    )

        if predictor_nfe != T:
            raise RuntimeError(
                f"pc_mode={pc_spec.name!r} produced {predictor_nfe} predictor "
                f"NFEs; expected {T}."
            )
        if corrector_nfe != pc_spec.expected_corrector_nfe:
            raise RuntimeError(
                f"pc_mode={pc_spec.name!r} produced {corrector_nfe} corrector "
                f"NFEs; expected {pc_spec.expected_corrector_nfe}."
            )
        self.last_sampling_stats = build_discrete_sampling_stats(
            pc_spec,
            predictor_nfe=predictor_nfe,
            corrector_nfe=corrector_nfe,
            noise_schedule=getattr(self.cfg, "noise_schedule", "linear"),
        )

        xt = xt.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))

        if self.dataset_type == "molecule":
            valid_mask = ~padding_mask
            coord_part = xt[:, :, :self.coord_dim]
            type_logits = xt[:, :, self.coord_dim:self.coord_dim + self.atom_type_dim]
            type_idx = torch.argmax(type_logits, dim=-1)
            type_onehot = F.one_hot(type_idx, num_classes=self.atom_type_dim).float()
            type_onehot = type_onehot * valid_mask.unsqueeze(-1).float()
            xt = torch.cat([coord_part, type_onehot], dim=-1)
            xt = xt.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))

        return xt
