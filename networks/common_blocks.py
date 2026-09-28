import torch
import torch.nn as nn

class SinusoidalTimestep(nn.Module):

    def __init__(self, embed_dim, max_period=10000.0, scale=1000.0):
        super().__init__()
        self.half = embed_dim // 2
        self.max_period = max_period
        self.scale = scale
        self.proj = nn.Sequential(
            nn.Linear(2*self.half, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    def forward(self, t):
        t = t.float() * self.scale
        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period, device=t.device))
            * torch.arange(self.half, device=t.device).float() / (self.half - 1)
        )
        ang = t.unsqueeze(-1) * freqs
        fea = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.proj(fea)

class UEmbed(nn.Module):

    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    def forward(self, u_logit):
        return self.mlp(u_logit.unsqueeze(-1))


class SinusoidalTimestepLegacy(nn.Module):
    def __init__(self, embed_dim, num_freqs=32):
        super().__init__()
        self.num_freqs = num_freqs
        self.proj = nn.Sequential(
            nn.Linear(2*num_freqs, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    def forward(self, t):
        freqs = 2.0 ** torch.arange(self.num_freqs, device=t.device).float()
        ang = 2 * torch.pi * t.unsqueeze(-1) * freqs
        fea = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.proj(fea)


class AdaLNBlock(nn.Module):
    def __init__(self, dim, nheads, dim_feedforward):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, nheads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, dim),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, h, c, key_padding_mask=None):
        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = self.norm1(h) * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn(x, x, x, need_weights=False,
                                key_padding_mask=key_padding_mask)
        if key_padding_mask is not None:
            empty = key_padding_mask.all(dim=-1)
            if empty.any():
                attn_out = attn_out.masked_fill(empty.view(-1, 1, 1), 0.0)
        h = h + gate_msa * attn_out
        x = self.norm2(h) * (1 + scale_mlp) + shift_mlp
        h = h + gate_mlp * self.mlp(x)
        return h


class AdaLNFinalLayer(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.linear = nn.Linear(dim, out_dim)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(self.norm(h) * (1 + scale) + shift)


class TimeAwareCrossGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, u_logit, u_emb, t_emb):
        delta = self.net(
            torch.cat([u_emb, t_emb], dim=-1)
        )
        gate_logit = u_logit.unsqueeze(-1) + delta
        return torch.sigmoid(gate_logit)
