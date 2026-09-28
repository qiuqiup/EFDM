import torch
import torch.nn as nn
from networks.common_blocks import SinusoidalTimestep, UEmbed

class ExistenceNet(nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.embedding_dim = cfg.embedding_dim
        self.point_dim = cfg.point_dim
        self.nheads = cfg.nheads
        self.dim_feedforward = cfg.dim_feedforward
        self.num_layers = cfg.num_layers
        self.exist_embedder_type = cfg.exist_embedder_type

        self.coord_embedding = nn.Linear(self.point_dim, self.embedding_dim, bias=False)

        if bool(getattr(cfg, "time_embedder_legacy", False)):
            raise ValueError("The historical time embedder is not part of this supplement")
        self.time_embedder = SinusoidalTimestep(self.embedding_dim)
        if self.exist_embedder_type == 'Sinusoidal':
            self.exist_embedder = SinusoidalTimestep(self.embedding_dim)
        elif cfg.exist_embedder_type == 'MLP':
            self.exist_embedder = UEmbed(self.embedding_dim)

        self.f1 = nn.Sequential(
            nn.Linear(self.embedding_dim*4, self.embedding_dim),
            nn.SiLU()
        )
        self.f2 = nn.Linear(self.embedding_dim, 1)
        self.input_norm = nn.LayerNorm(self.embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.nheads,
            dropout=0.0,
            batch_first=True,
            dim_feedforward=self.dim_feedforward,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.head_pos = nn.Sequential(nn.LayerNorm(self.embedding_dim), nn.Linear(self.embedding_dim, self.point_dim))
        self.head_exist = nn.Sequential(nn.LayerNorm(self.embedding_dim), nn.Linear(self.embedding_dim, 1))

    def _fuse_embeddings(self, x_emb, u_emb, t_emb, cross, u_prob):
        return self.f1(torch.cat([x_emb, u_emb, t_emb, cross], dim=-1))

    def forward(self, x, u_logit, t):
        u_logit = u_logit.float()
        x_emb = self.coord_embedding(x)
        t_emb = self.time_embedder(t)

        u_prob  = torch.sigmoid(u_logit)

        if self.exist_embedder_type == 'Sinusoidal':
            u_emb = self.exist_embedder(u_prob)
        elif self.exist_embedder_type == 'MLP':
            u_emb = self.exist_embedder(u_logit)

        cross = x_emb * u_prob.unsqueeze(-1)

        h = self._fuse_embeddings(x_emb, u_emb, t_emb, cross, u_prob)
        h = self.input_norm(h)
        h = self.encoder(h)

        eps_x = self.head_pos(h)
        eps_u = self.head_exist(h)
        return eps_x, eps_u


class ExistenceNetSoftCount(ExistenceNet):
    """ExistenceNet with a global soft-cardinality input stream."""

    def __init__(self, cfg):
        super().__init__(cfg)
        dim = self.embedding_dim
        self.count_embedder = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim),
        )
        self.f1 = nn.Sequential(nn.Linear(5 * dim, dim), nn.SiLU())

    def _fuse_embeddings(self, x_emb, u_emb, t_emb, cross, u_prob):
        centered_count = 2.0 * u_prob.mean(dim=1, keepdim=True) - 1.0
        count_emb = self.count_embedder(centered_count).unsqueeze(1)
        count_emb = count_emb.expand(-1, x_emb.shape[1], -1)
        return self.f1(torch.cat([x_emb, u_emb, t_emb, cross, count_emb], dim=-1))
