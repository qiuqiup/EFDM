import torch
import torch.nn as nn

from networks.common_blocks import SinusoidalTimestep


class PSDNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.embedding_dim = int(cfg.embedding_dim)
        self.nheads = int(cfg.nheads)
        self.dim_feedforward = int(cfg.dim_feedforward)
        self.num_layers = int(cfg.num_layers)
        self.point_dim = int(cfg.point_dim)

        self.max_num_points = int(cfg.max_num_points)
        self.num_mixture_components = int(getattr(cfg, "num_mixture_components", 16))

        self.coord_embedding = nn.Linear(self.point_dim, self.embedding_dim, bias=False)
        self.time_embedder = SinusoidalTimestep(self.embedding_dim)
        self.card_embedder = SinusoidalTimestep(self.embedding_dim)

        self.f1 = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, self.embedding_dim),
            nn.SiLU(),
        )
        self.input_norm = nn.LayerNorm(self.embedding_dim)

        self.f_cls = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
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

        self.head_cls = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1),
        )

        M, d = self.num_mixture_components, self.point_dim
        self.head_mix = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, M * (1 + 2 * d)),
        )

    def forward(self, x, t, mask):
        B, K, d = x.shape
        M = self.num_mixture_components

        x_emb = self.coord_embedding(x)
        t_emb = self.time_embedder(t)

        n_valid = (~mask).sum(dim=1).float()
        n_norm = n_valid / float(self.max_num_points)
        n_emb_tok = self.card_embedder(n_norm.unsqueeze(1)).expand(-1, K, -1)
        h = self.f1(torch.cat([x_emb, t_emb, n_emb_tok], dim=-1))
        h = self.input_norm(h)

        t_global = t[:, :1]
        t_emb_g = self.time_embedder(t_global)
        n_emb_g = self.card_embedder(n_norm.unsqueeze(1))
        cls_tok = self.f_cls(torch.cat([t_emb_g, n_emb_g], dim=-1))

        tokens = torch.cat([cls_tok, h], dim=1)
        pad = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1
        )
        enc = self.encoder(tokens, src_key_padding_mask=pad)

        cls_out = enc[:, 0]
        point_out = enc[:, 1:]

        keep_logits = self.head_cls(point_out).squeeze(-1)

        mix_raw = self.head_mix(cls_out).view(B, M, 1 + 2 * d)
        mix_w_raw = mix_raw[:, :, 0]
        mix_mean = torch.tanh(mix_raw[:, :, 1:1 + d])
        mix_var = torch.exp(-torch.abs(mix_raw[:, :, 1 + d:])) + 1e-3

        return keep_logits, mix_w_raw, mix_mean, mix_var
