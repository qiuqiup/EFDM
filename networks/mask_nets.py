import torch
import torch.nn as nn

from networks.common_blocks import (
    SinusoidalTimestep, SinusoidalTimestepLegacy, AdaLNBlock, AdaLNFinalLayer,
)
from networks.egnn_backbone import EGNNAtomTypeBackbone


class MaskNetDiT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding_dim = cfg.embedding_dim
        self.nheads = cfg.nheads
        self.dim_feedforward = cfg.dim_feedforward
        self.num_layers = cfg.num_layers

        self.dataset_type = getattr(cfg, "dataset_type", "trip")
        self.use_egnn_backbone = bool(getattr(cfg, "use_egnn_backbone", False))

        self.point_dim = int(cfg.point_dim)
        if self.dataset_type == "molecule":
            if hasattr(cfg, "num_atom_types"):
                self.atom_type_dim = int(cfg.num_atom_types)
                self.coord_dim = 3
                self.total_input_dim = self.coord_dim + self.atom_type_dim
            elif self.point_dim > 3:
                self.coord_dim = 3
                self.atom_type_dim = int(self.point_dim - 3)
                self.total_input_dim = int(self.point_dim)
            else:
                self.coord_dim = int(self.point_dim)
                self.atom_type_dim = int(getattr(cfg, "atom_type_dim", 5))
                self.total_input_dim = self.coord_dim + self.atom_type_dim
        else:
            self.coord_dim = int(self.point_dim)
            self.atom_type_dim = 0
            self.total_input_dim = int(self.point_dim)

        self.time_embedder = SinusoidalTimestep(self.embedding_dim)

        if self.dataset_type == "molecule" and self.use_egnn_backbone:
            self.egnn_backbone = EGNNAtomTypeBackbone(
                num_atom_types=self.atom_type_dim,
                hidden_nf=self.embedding_dim,
                n_layers=getattr(cfg, "egnn_n_layers", 4),
                attention=getattr(cfg, "egnn_attention", False),
                normalization_factor=getattr(cfg, "egnn_normalization_factor", 100),
                aggregation_method=getattr(cfg, "egnn_aggregation_method", "sum"),
                condition_time=True,
                CoM0=True,
                return_hidden=True,
                embedding_out_mode=int(getattr(cfg, "egnn_embedding_out_mode", 1)),
            )
        else:
            self.coord_embedding = nn.Linear(self.total_input_dim, self.embedding_dim, bias=False)

        self.blocks = nn.ModuleList([
            AdaLNBlock(self.embedding_dim, self.nheads, self.dim_feedforward)
            for _ in range(self.num_layers)
        ])
        out_dim = self.coord_dim + (self.atom_type_dim if self.dataset_type == "molecule" else 0)
        self.final_layer = AdaLNFinalLayer(self.embedding_dim, out_dim)

    def forward(self, x, t, mask):
        B, K, d = x.shape
        if self.dataset_type == "molecule" and self.use_egnn_backbone:
            node_mask = (~mask).float()
            if t.dim() != 2 or t.shape[0] != B:
                raise ValueError(f"Expected t shape [B,K], got {tuple(t.shape)}")
            t_norm = t[:, 0]
            _, egnn_hidden = self.egnn_backbone(x, node_mask, t_norm)
            h = egnn_hidden
        else:
            h = self.coord_embedding(x)

        c = self.time_embedder(t)
        for block in self.blocks:
            h = block(h, c, key_padding_mask=mask)
        out = self.final_layer(h, c)

        eps_coords = out[..., :self.coord_dim]
        if self.dataset_type == "molecule":
            eps_type = out[..., self.coord_dim:]
            return eps_coords, eps_type
        return eps_coords


class MaskNet(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.embedding_dim = cfg.embedding_dim
        self.nheads = cfg.nheads
        self.dim_feedforward = cfg.dim_feedforward
        self.num_layers = cfg.num_layers

        self.dataset_type = getattr(cfg, "dataset_type", "trip")
        self.use_egnn_backbone = bool(getattr(cfg, "use_egnn_backbone", False))

        self.point_dim = int(cfg.point_dim)
        if self.dataset_type == "molecule":
            if hasattr(cfg, "num_atom_types"):
                self.atom_type_dim = int(cfg.num_atom_types)
                self.coord_dim = 3
                self.total_input_dim = self.coord_dim + self.atom_type_dim
            elif self.point_dim > 3:
                self.coord_dim = 3
                self.atom_type_dim = int(self.point_dim - 3)
                self.total_input_dim = int(self.point_dim)
            else:
                self.coord_dim = int(self.point_dim)
                self.atom_type_dim = int(getattr(cfg, "atom_type_dim", 5))
                self.total_input_dim = self.coord_dim + self.atom_type_dim
        else:
            self.coord_dim = int(self.point_dim)
            self.atom_type_dim = 0
            self.total_input_dim = int(self.point_dim)

        if bool(getattr(cfg, "time_embedder_legacy", False)):
            self.time_embedder = SinusoidalTimestepLegacy(self.embedding_dim)
        else:
            self.time_embedder = SinusoidalTimestep(self.embedding_dim)
        self.input_norm      = nn.LayerNorm(self.embedding_dim)

        if self.dataset_type == "molecule" and self.use_egnn_backbone:
            self.egnn_backbone = EGNNAtomTypeBackbone(
                num_atom_types=self.atom_type_dim,
                hidden_nf=self.embedding_dim,
                n_layers=getattr(cfg, "egnn_n_layers", 4),
                attention=getattr(cfg, "egnn_attention", False),
                normalization_factor=getattr(cfg, "egnn_normalization_factor", 100),
                aggregation_method=getattr(cfg, "egnn_aggregation_method", "sum"),
                condition_time=True,
                CoM0=True,
                return_hidden=True,
                embedding_out_mode=int(getattr(cfg, "egnn_embedding_out_mode", 1)),
            )
            x_emb_dim = self.embedding_dim
        else:
            self.coord_embedding = nn.Linear(self.total_input_dim, self.embedding_dim, bias=False)
            x_emb_dim = self.embedding_dim

        self.f1 = nn.Sequential(
            nn.Linear(x_emb_dim * 2, self.embedding_dim),
            nn.SiLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.nheads,
            dropout=0.0,
            batch_first=True,
            dim_feedforward=self.dim_feedforward,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.head_pos = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.coord_dim)
        )

        if self.dataset_type == "molecule":
            self.head_type = nn.Sequential(
                nn.LayerNorm(self.embedding_dim),
                nn.Linear(self.embedding_dim, self.atom_type_dim)
            )

    def forward(self, x, t, mask):
        B, K, d = x.shape
        if self.dataset_type == "molecule" and self.use_egnn_backbone:
            node_mask = (~mask).float()
            if t.dim() != 2 or t.shape[0] != B:
                raise ValueError(f"Expected t shape [B,K], got {tuple(t.shape)}")
            t_norm = t[:, 0]
            _, egnn_hidden = self.egnn_backbone(x, node_mask, t_norm)
            x_emb = egnn_hidden
        else:
            x_emb = self.coord_embedding(x)

        t_emb = self.time_embedder(t)

        h = torch.cat([x_emb, t_emb], dim=-1)
        h = self.f1(h)
        h = self.input_norm(h)
        h = self.encoder(h, src_key_padding_mask=mask)

        eps_coords = self.head_pos(h)

        if self.dataset_type == "molecule":
            eps_type = self.head_type(h)
            return eps_coords, eps_type
        else:
            return eps_coords
