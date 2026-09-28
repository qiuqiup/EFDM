import torch
import torch.nn as nn
from networks.egnn_backbone import EGNNAtomTypeBackbone
from networks.common_blocks import SinusoidalTimestep

class ExistenceNetWithEGNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if not bool(getattr(cfg, "use_egnn_backbone", False)):
            raise ValueError("Molecule EFDM requires use_egnn_backbone: true")
        self.cfg = cfg
        self.dataset_type = "molecule"
        self.use_egnn_backbone = True
        self._build_egnn_network(cfg)

    def _build_egnn_network(self, cfg):

        self.embedding_dim = cfg.embedding_dim
        self.point_dim = cfg.point_dim


        if hasattr(cfg, 'num_atom_types'):
            self.num_atom_types = int(cfg.num_atom_types)
        else:

            self.num_atom_types = int(self.point_dim - 3)


        self.egnn_hidden_nf = int(getattr(cfg, "egnn_hidden_nf", cfg.embedding_dim))

        self.egnn_backbone = EGNNAtomTypeBackbone(
            num_atom_types=self.num_atom_types,
            hidden_nf=self.egnn_hidden_nf,
            n_layers=getattr(cfg, 'egnn_n_layers', 4),
            attention=getattr(cfg, 'egnn_attention', False),

            inv_sublayers=int(getattr(cfg, 'egnn_inv_sublayers', 2)),

            tanh=bool(getattr(cfg, 'egnn_tanh', False)),
            norm_constant=getattr(cfg, 'egnn_norm_constant', 0),

            normalization_factor=getattr(cfg, 'egnn_normalization_factor', 100),
            aggregation_method=getattr(cfg, 'egnn_aggregation_method', 'sum'),
            condition_time=True,
            CoM0=True,
            return_hidden=True,
            return_pos_pred=True,
            coord_clip=getattr(cfg, "egnn_coord_clip", 50.0),
            feat_clip=getattr(cfg, "egnn_feat_clip", 10.0),
            center_coords=getattr(cfg, "egnn_center_coords", True),
            coord_norm_scale=getattr(cfg, "egnn_pos_norm", 1.0),

            embedding_out_mode=int(getattr(cfg, "egnn_embedding_out_mode", 1)),
        )

        self.project_pos_eps_com0 = bool(getattr(cfg, "project_pos_eps_com0", True))

        self.egnn_p_exist_input = bool(getattr(cfg, "egnn_p_exist_input", False))

        self.egnn_feat_gate = bool(getattr(cfg, "egnn_feat_gate", False))
        if self.egnn_feat_gate:
            self.feat_gate_mlp = nn.Sequential(
                nn.Linear(1 + self.embedding_dim, self.embedding_dim),
                nn.SiLU(),
                nn.Linear(self.embedding_dim, 1),
            )
            nn.init.zeros_(self.feat_gate_mlp[-1].weight)
            nn.init.zeros_(self.feat_gate_mlp[-1].bias)


        self.nheads = cfg.nheads
        self.dim_feedforward = cfg.dim_feedforward
        self.num_layers = cfg.num_layers


        if bool(getattr(cfg, "time_embedder_legacy", False)):
            raise ValueError("The historical time embedder is not part of this supplement")
        self.time_embedder = SinusoidalTimestep(self.embedding_dim)

        self.use_p_exist_in_fuse = bool(getattr(cfg, "use_p_exist_in_fuse", False))
        self.p_exist_embed_dim = int(getattr(cfg, "p_exist_embed_dim", self.embedding_dim))

        self.p_exist_time_aware = bool(getattr(cfg, "p_exist_time_aware", False))
        if self.use_p_exist_in_fuse:
            self.p_exist_embed = nn.Sequential(
                nn.Linear(1, self.p_exist_embed_dim),
                nn.SiLU(),
            )
            if self.p_exist_time_aware:
                self.p_exist_tmod = nn.Linear(
                    self.p_exist_embed_dim + self.embedding_dim,
                    self.p_exist_embed_dim,
                )
                nn.init.zeros_(self.p_exist_tmod.weight)
                nn.init.zeros_(self.p_exist_tmod.bias)

            fuse_in_dim = self.egnn_hidden_nf + self.embedding_dim + self.p_exist_embed_dim
        else:
            self.p_exist_embed = None
            self.p_exist_time_aware = False
            fuse_in_dim = self.egnn_hidden_nf + self.embedding_dim

        self.fuse = nn.Sequential(
            nn.Linear(fuse_in_dim, self.embedding_dim),
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
        if self.egnn_p_exist_input:

            self.egnn_p_exist_input_weight = nn.Parameter(
                torch.zeros(self.egnn_hidden_nf, 1)
            )

    def _p_exist_from_type_logits(self, x: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """Existence probability from fixed dummy logit and atom-type logits."""
        B, K, _ = x.shape
        type_logits = x[:, :, 3:3 + self.num_atom_types]
        type_logits = torch.nan_to_num(type_logits, nan=-80.0, posinf=80.0, neginf=-80.0)
        type_logits = type_logits.clamp(min=-80.0, max=80.0)

        dummy_logit = torch.zeros(B, K, 1, device=x.device, dtype=type_logits.dtype)
        logits_all = torch.cat([dummy_logit, type_logits], dim=-1)
        probs_all = torch.softmax(logits_all, dim=-1)
        return probs_all[:, :, 1:].sum(dim=-1, keepdim=keepdim).to(dtype=x.dtype)

    @staticmethod
    def _remove_mean(x: torch.Tensor) -> torch.Tensor:
        """Project fixed-size existence slots onto the zero-CoM subspace."""
        return x - x.mean(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        B, K, _ = x.shape
        if t.dim() != 1 or t.shape[0] != B:
            raise ValueError(f"Expected t shape [B], got {tuple(t.shape)}")
        node_mask = torch.ones(B, K, device=x.device, dtype=x.dtype)
        feat_gate = None
        if self.egnn_feat_gate:
            t_emb_gate = self.time_embedder(t.view(B, 1).expand(B, K))
            p_exist_gate = self._p_exist_from_type_logits(x, keepdim=True)
            delta = self.feat_gate_mlp(torch.cat([p_exist_gate, t_emb_gate], dim=-1))
            feat_gate = 2.0 * torch.sigmoid(delta)
        p_exist_input_add = None
        if self.egnn_p_exist_input:
            p_exist_input = self._p_exist_from_type_logits(x, keepdim=True)
            p_exist_input_add = torch.matmul(
                p_exist_input, self.egnn_p_exist_input_weight.transpose(0, 1)
            )
        _, egnn_hidden, eps_pos_egnn = self.egnn_backbone(
            x, node_mask, t, feat_gate=feat_gate,
            input_embedding_add=p_exist_input_add,
        )
        t_tile = t.view(B, 1).expand(B, K)
        t_emb = self.time_embedder(t_tile)
        if self.use_p_exist_in_fuse:
            p_exist = self._p_exist_from_type_logits(x, keepdim=True)
            p_exist_emb = self.p_exist_embed(p_exist)
            if self.p_exist_time_aware:
                delta = self.p_exist_tmod(torch.cat([p_exist_emb, t_emb], dim=-1))
                p_exist_emb = p_exist_emb * (1.0 + delta)
            h = torch.cat([egnn_hidden, t_emb, p_exist_emb], dim=-1)
        else:
            h = torch.cat([egnn_hidden, t_emb], dim=-1)
        h = self.encoder(self.input_norm(self.fuse(h)))
        eps_type = self.head_type_eps(h)
        eps_pos = eps_pos_egnn
        if self.project_pos_eps_com0:
            eps_pos = self._remove_mean(eps_pos)
        return eps_pos, eps_type
