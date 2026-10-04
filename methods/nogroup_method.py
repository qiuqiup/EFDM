import torch
import torch.nn.functional as F
from typing import Optional


from methods.base_method import BaseMethod
from utils.nonempty_sampling import reject_empty_sets
from networks.nogroup_nets import NoGroupNet
from utils.pc_sampling import (
    build_discrete_sampling_stats,
    discrete_corrector_target,
    epsilon_langevin_corrector_step,
    resolve_discrete_pc_mode,
)
from utils.sampling import sample_n_from_val_n
from utils.n_distribution import NDistributionMixture


class NoGroupMethod(BaseMethod):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.cfg = cfg
        self.device = device

        self.dataset_type = getattr(cfg, "dataset_type", "trip")
        self.point_dim = cfg.point_dim
        if self.dataset_type == "molecule":
            self.atom_type_dim = getattr(cfg, "atom_type_dim", 5)
            self.total_dim = self.point_dim + self.atom_type_dim
        else:
            self.atom_type_dim = 0
            self.total_dim = self.point_dim

        self.model = NoGroupNet(cfg).to(device)

        n_components = getattr(cfg, "n_distribution_components", 5)
        self.n_distribution = NDistributionMixture(
            n_components=n_components,
            random_state=None
        )
        self.n_distribution_fitted = False


    def _flatten_data(self, data_raw: torch.Tensor) -> torch.Tensor:
        if data_raw.ndim == 2:
            return data_raw.to(self.device)

        mask = ~torch.isinf(data_raw[..., 0])
        pts  = data_raw[mask]
        pts  = pts.view(-1, self.total_dim)
        return pts.to(self.device)

    def prepare_train_epoch(self, train_data_raw):
        train_x_flat = self._flatten_data(train_data_raw)
        return {"x0": train_x_flat}

    def prepare_val_data(self, val_data_raw):
        val_x_flat = self._flatten_data(val_data_raw)
        return {"x0": val_x_flat}

    def _forward_loss(self, batch, generator=None):
        x0 = batch["x0"].to(self.device)

        B, d = x0.shape
        T = self.cfg.T
        beta = torch.linspace(0.0001, 0.02, T, device=self.device)
        alpha = 1-beta
        alpha_bar = torch.cumprod(alpha, dim=0).to(self.device)

        if generator is None:
            t_int = torch.randint(0, T, (B,), device=self.device)
        else:
            t_int = torch.randint(0, T, (B,), device=self.device, generator=generator)

        a_bar = alpha_bar[t_int].view(B, 1)
        if generator is None:
            eps_x = torch.randn(B, d, device=self.device)
        else:
            eps_x = torch.randn(B, d, device=self.device, generator=generator)

        x_t    = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * eps_x

        t_norm = t_int.float() / (T - 1)
        eps_x_pred = self.model(x_t, t_norm)

        if self.dataset_type == "molecule":
            eps_coords = eps_x[:, :self.point_dim]
            eps_coords_pred = eps_x_pred[:, :self.point_dim]
            eps_atoms_pred = eps_x_pred[:, self.point_dim:]

            coord_loss = (eps_coords_pred - eps_coords).pow(2).mean()

            x0_atoms = x0[:, self.point_dim:]
            true_atom_indices = x0_atoms.argmax(dim=-1)
            atom_loss = F.cross_entropy(eps_atoms_pred, true_atom_indices)

            loss = coord_loss + atom_loss
            log = {
                "loss": loss.item(),
                "coord_loss": coord_loss.item(),
                "atom_loss": atom_loss.item()
            }
        else:
            loss = (eps_x_pred - eps_x).pow(2).mean()
            log = {
                "loss": loss.item()
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
                "NoGroupMethod corrector sampling is currently supported only "
                "for non-molecule datasets."
            )

        if self.n_distribution_fitted:
            n_samples = self.n_distribution.sample(num_samples, device)
        else:
            assert val_n is not None, "NoGroupMethod.sample_sets needs either fitted n_distribution or val_n"
            n_samples = sample_n_from_val_n(val_n, num_samples, device)

        if device is None:
            device = self.device

        B = int(num_samples)
        d = int(self.total_dim)

        max_len = int(n_samples.max().item())
        total_n = int(n_samples.sum().item())

        out = torch.full((B, max_len, d), float("-inf"), device=device)
        if total_n == 0:
            self.last_sampling_stats = build_discrete_sampling_stats(
                pc_spec,
                predictor_nfe=0,
                corrector_nfe=0,
                noise_schedule="linear",
            )
            return out

        xt = torch.randn(total_n, d, device=device)

        beta = torch.linspace(0.0001, 0.02, T, device=device)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        model = self.model.to(device)
        model.eval()

        predictor_nfe = 0
        corrector_nfe = 0
        for t in reversed(range(T)):
            t_norm = float(t) / (T - 1)
            t_vec = torch.full((total_n,), t_norm, device=device)

            eps_pred = model(xt, t_vec)
            predictor_nfe += 1

            a_t = alpha[t]
            ab_t = alpha_bar[t]
            b_t = beta[t]
            z = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)

            xt = (
                (xt - (1.0 - a_t) * eps_pred / torch.sqrt(1.0 - ab_t + 1e-12))
                / torch.sqrt(a_t + 1e-12)
                + torch.sqrt(b_t) * z
            )

            corrector_target = discrete_corrector_target(
                pc_spec, source_index=t, T=T
            )
            if corrector_target is not None:
                corrector_index, corrector_time = corrector_target
                corrector_t_vec = torch.full(
                    (total_n,), corrector_time, device=device
                )
                corrector_std = torch.sqrt(
                    (1.0 - alpha_bar[corrector_index]).clamp_min(0.0)
                ).clamp_min(0.001)
                corrector_alpha = alpha[corrector_index]
                eps_clip = float(getattr(self.cfg, "sample_eps_clip", 1e3))
                state_clip = float(getattr(self.cfg, "sample_coord_clip", 1e3))

                for _ in range(pc_spec.corrector_steps_per_level):
                    corr_eps = model(xt, corrector_t_vec)
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
            noise_schedule="linear",
        )

        if self.dataset_type == "molecule":
            coord_part = xt[:, :self.point_dim]
            atom_part = xt[:, self.point_dim:]

            sampling_method = getattr(self.cfg, "atom_type_sampling", "multinomial")

            if sampling_method == "argmax":
                atom_indices = atom_part.argmax(dim=-1)
            else:
                atom_probs = F.softmax(atom_part, dim=-1)
                atom_indices = torch.multinomial(atom_probs, 1).squeeze(-1)

            atom_onehot = F.one_hot(atom_indices, num_classes=self.atom_type_dim).float()

            xt = torch.cat([coord_part, atom_onehot], dim=-1)

        offset = 0
        for i in range(B):
            n = int(n_samples[i].item())
            if n > 0:
                out[i, :n] = xt[offset: offset + n]
                offset += n

        return out
