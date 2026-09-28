import torch
import torch.nn as nn

from networks.common_blocks import SinusoidalTimestep, SinusoidalTimestepLegacy


class NoGroupNet(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.embedding_dim = cfg.embedding_dim
        self.point_dim = cfg.point_dim
        self.dim_feedforward = cfg.dim_feedforward
        self.num_layers = cfg.num_layers
        self.activation = cfg.activation

        self.dataset_type = getattr(cfg, "dataset_type", "trip")
        self.point_dim = cfg.point_dim
        if self.dataset_type == "molecule":
            self.atom_type_dim = getattr(cfg, "atom_type_dim", 5)
            self.total_input_dim = self.point_dim + self.atom_type_dim
        else:
            self.atom_type_dim = 0
            self.total_input_dim = self.point_dim

        self.coord_embedding = nn.Linear(self.total_input_dim, self.embedding_dim, bias=False)
        if bool(getattr(cfg, "time_embedder_legacy", False)):
            self.time_embedder = SinusoidalTimestepLegacy(self.embedding_dim)
        else:
            self.time_embedder = SinusoidalTimestep(self.embedding_dim)
        self.input_norm      = nn.LayerNorm(self.embedding_dim)

        self.f1 = nn.Sequential(
            nn.Linear(self.embedding_dim*2, self.embedding_dim),
            nn.SiLU()
        )

        act_layer = nn.Tanh if self.activation.lower() == "tanh" else nn.ReLU

        layers = []

        layers.append(nn.Linear(self.embedding_dim, self.dim_feedforward))
        layers.append(act_layer())

        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.dim_feedforward, self.dim_feedforward))
            layers.append(act_layer())

        layers.append(nn.Linear(self.dim_feedforward, self.total_input_dim))

        self.ff = nn.Sequential(*layers)

    def forward(self, x, t):
        x_emb = self.coord_embedding(x)
        t_emb = self.time_embedder(t)

        h = torch.cat([x_emb, t_emb], dim=-1)
        h = self.f1(h)
        h = self.input_norm(h)
        h = self.ff(h)
        return h
