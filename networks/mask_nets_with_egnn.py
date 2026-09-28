
import torch
import torch.nn as nn

from networks.common_blocks import SinusoidalTimestep, SinusoidalTimestepLegacy
from networks.egnn_backbone import EGNNAtomTypeBackbone


class MaskNetWithEGNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset_type = getattr(cfg, "dataset_type", "trip")
        if self.dataset_type != "molecule":
            raise ValueError(
                "MaskNetWithEGNN is molecule-only; use MaskNet/MaskNetDiT for "
                f"dataset_type={self.dataset_type!r}."
            )

        self.embedding_dim = int(cfg.embedding_dim)
        self.nheads = int(cfg.nheads)
        self.dim_feedforward = int(cfg.dim_feedforward)
        self.num_layers = int(cfg.num_layers)

        self.point_dim = int(cfg.point_dim)
        self.coord_dim = 3
        if hasattr(cfg, "num_atom_types"):
            self.num_atom_types = int(cfg.num_atom_types)
        else:
            self.num_atom_types = int(self.point_dim - 3)
        self.total_input_dim = self.coord_dim + self.num_atom_types

        self.molecule_com0 = bool(getattr(cfg, "molecule_com0", False))
        self.project_pos_eps_com0 = bool(
            getattr(cfg, "project_pos_eps_com0", self.molecule_com0)
        )

        self.egnn_backbone = EGNNAtomTypeBackbone(
            num_atom_types=self.num_atom_types,
            hidden_nf=self.embedding_dim,
            n_layers=getattr(cfg, "egnn_n_layers", 9),
            attention=getattr(cfg, "egnn_attention", False),
            tanh=bool(getattr(cfg, "egnn_tanh", False)),
            norm_constant=getattr(cfg, "egnn_norm_constant", 0),
            normalization_factor=getattr(cfg, "egnn_normalization_factor", 100),
            aggregation_method=getattr(cfg, "egnn_aggregation_method", "sum"),
            condition_time=True,
            CoM0=self.molecule_com0,
            return_hidden=True,
            return_pos_pred=True,
            coord_clip=getattr(cfg, "egnn_coord_clip", 50.0),
            feat_clip=getattr(cfg, "egnn_feat_clip", 10.0),
            center_coords=getattr(cfg, "egnn_center_coords", True),
            coord_norm_scale=getattr(cfg, "egnn_pos_norm", 1.0),
            embedding_out_mode=int(getattr(cfg, "egnn_embedding_out_mode", 1)),
        )

        if bool(getattr(cfg, "time_embedder_legacy", False)):
            self.time_embedder = SinusoidalTimestepLegacy(self.embedding_dim)
        else:
            self.time_embedder = SinusoidalTimestep(self.embedding_dim)

        self.fuse = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.SiLU(),
        )
        self.input_norm = nn.LayerNorm(self.embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.nheads,
            dropout=0.0,
            batch_first=True,
            dim_feedforward=self.dim_feedforward,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.head_type_eps = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.num_atom_types),
        )

    @staticmethod
    def _remove_mean_with_mask(x: torch.Tensor, node_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Subtract the mean over valid nodes only, leaving padded rows at 0."""
        if node_mask.dim() == 2:
            node_mask = node_mask.unsqueeze(-1)
        denom = node_mask.sum(dim=1, keepdim=True).clamp(min=eps)
        mean = (x * node_mask).sum(dim=1, keepdim=True) / denom
        return (x - mean) * node_mask

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            x:    [B, K, 3 + num_atom_types]
            t:    [B, K] normalized time in [0,1]
            mask: [B, K] bool, True = padding
        Returns:
            eps_pos:  [B, K, 3]
            eps_type: [B, K, num_atom_types]
        """
        B, K, d = x.shape
        if d != self.total_input_dim:
            raise ValueError(
                f"Expected x dim {self.total_input_dim} (3 coords + "
                f"{self.num_atom_types} atom types), got {d}"
            )
        if t.dim() != 2 or t.shape[0] != B or t.shape[1] != K:
            raise ValueError(f"Expected t shape [B,K]=({B},{K}), got {tuple(t.shape)}")
        if mask.shape != (B, K):
            raise ValueError(f"Expected mask shape ({B},{K}), got {tuple(mask.shape)}")

        node_mask = (~mask).to(dtype=x.dtype)
        t_norm = t[:, 0]

        _, egnn_hidden, eps_pos = self.egnn_backbone(x, node_mask, t_norm)

        t_emb = self.time_embedder(t)
        h = torch.cat([egnn_hidden, t_emb], dim=-1)
        h = self.fuse(h)
        h = self.input_norm(h)

        all_pad = mask.all(dim=1)
        if bool(all_pad.any()):
            safe_mask = mask.clone()
            safe_mask[all_pad, 0] = False
        else:
            safe_mask = mask
        h = self.encoder(h, src_key_padding_mask=safe_mask)

        eps_type = self.head_type_eps(h)

        if self.project_pos_eps_com0:
            eps_pos = self._remove_mean_with_mask(eps_pos, node_mask)

        eps_type = eps_type * node_mask.unsqueeze(-1)
        return eps_pos, eps_type
