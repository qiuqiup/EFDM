import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.common_blocks import (
    SinusoidalTimestep, SinusoidalTimestepLegacy, AdaLNBlock, AdaLNFinalLayer,
)


def resolve_jump_net_arch(cfg):
    """Resolve the architecture, accepting the old name in saved configurations."""
    arch = getattr(cfg, "jump_net_arch", "fused")
    if arch == "legacy":
        arch = "nonfused"
    if arch not in ("fused", "nonfused", "dit"):
        raise ValueError(f"Unknown jump_net_arch {arch!r}; use fused, nonfused, or dit")
    return arch


class JumpNetDiT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding_dim = cfg.embedding_dim
        self.point_dim = cfg.point_dim
        self.nheads = cfg.nheads
        self.dim_feedforward = cfg.dim_feedforward
        self.num_layers = cfg.num_layers

        self.dataset_type = getattr(cfg, "dataset_type", "trip")
        if self.dataset_type == "molecule":
            self.atom_type_dim = getattr(cfg, "atom_type_dim", 5)
            self.charge_dim = getattr(cfg, "charge_dim", 1)
            self.total_input_dim = self.point_dim + self.atom_type_dim + self.charge_dim
        else:
            self.atom_type_dim = 0
            self.charge_dim = 0
            self.total_input_dim = self.point_dim

        self.coord_embedding = nn.Linear(self.total_input_dim, self.embedding_dim, bias=False)

        self.time_embedder = SinusoidalTimestep(self.embedding_dim)
        if self.dataset_type != "molecule":
            self.n_t_embed = nn.Sequential(
                nn.Linear(1, self.embedding_dim),
                nn.SiLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim),
            )

        self.blocks = nn.ModuleList([
            AdaLNBlock(self.embedding_dim, self.nheads, self.dim_feedforward)
            for _ in range(self.num_layers)
        ])
        self.backbone_output_dim = self.embedding_dim

        self.score_head = AdaLNFinalLayer(self.embedding_dim, self.total_input_dim)

        hidden_dim = getattr(cfg, "hidden_dim", 256)
        if self.dataset_type == "molecule":
            transformer_dim = getattr(cfg, "transformer_dim", 128)
            self.transformer_dim = transformer_dim
            self.n0_transformer_proj = nn.Linear(self.backbone_output_dim, transformer_dim)
            self.n0_transformer_layer = nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=getattr(cfg, "transformer_nhead", 4),
                dim_feedforward=self.dim_feedforward,
                batch_first=True,
                dropout=0.0,
            )
            self.n0_transformer_encoder = nn.TransformerEncoder(
                self.n0_transformer_layer,
                num_layers=getattr(cfg, "transformer_num_layers", 8),
            )
            max_n0 = getattr(cfg, "max_n0", 35)
            self.max_n0 = max_n0
            self.expected_n0_head = nn.Linear(transformer_dim, max_n0)
            self.nearest_atom_head = nn.Linear(transformer_dim, 1)
        else:
            self.expected_n0_head = nn.Sequential(
                nn.Linear(self.backbone_output_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.nearest_atom_head = None

        if self.dataset_type == "molecule":
            trans3_input_dim = self.backbone_output_dim + self.total_input_dim + 1
            self.add_transformer_proj = nn.Linear(trans3_input_dim, self.transformer_dim)
            self.add_transformer_layer = nn.TransformerEncoderLayer(
                d_model=self.transformer_dim,
                nhead=getattr(cfg, "transformer_nhead", 4),
                dim_feedforward=self.dim_feedforward,
                batch_first=True,
                dropout=0.0,
            )
            self.add_transformer_encoder = nn.TransformerEncoder(
                self.add_transformer_layer,
                num_layers=getattr(cfg, "transformer_num_layers", 8),
            )
            self.add_weight_head = nn.Linear(self.transformer_dim, 1)
            feature_output_dim = self.point_dim + self.atom_type_dim + self.charge_dim
            self.add_feature_mean_head = nn.Linear(self.transformer_dim, feature_output_dim)
            self.add_feature_std_head = nn.Linear(self.transformer_dim, feature_output_dim)
        else:
            self.insert_head = nn.Sequential(
                nn.Linear(self.backbone_output_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.point_dim * 2),
            )

    def forward(self, x, t, mask=None):
        """Same signature and returns as nonfused JumpNet (7-tuple)."""
        B, L, _ = x.shape
        if mask is None:
            mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)

        x_original = x.clone()

        h = self.coord_embedding(x)
        c = self.time_embedder(t)
        if self.dataset_type != "molecule":
            n_t = (~mask).sum(dim=1, keepdim=True).float()
            c = c + self.n_t_embed(n_t).unsqueeze(1)

        for block in self.blocks:
            h = block(h, c, key_padding_mask=mask)
        backbone_output = h

        score = self.score_head(backbone_output, c)

        if self.dataset_type == "molecule":
            n0_input = self.n0_transformer_proj(backbone_output)
            trans2_output = self.n0_transformer_encoder(n0_input, src_key_padding_mask=mask)
            pooled_trans2 = trans2_output.masked_fill(mask.unsqueeze(-1), 0.0).sum(dim=1) / (~mask).sum(dim=1, keepdim=True).clamp(min=1)
            n0_logits = self.expected_n0_head(pooled_trans2)

            nearest_atom_logits = self.nearest_atom_head(trans2_output).squeeze(-1)
            nearest_atom_logits = nearest_atom_logits.masked_fill(mask, float('-inf'))

            nearest_atom_probs = F.softmax(nearest_atom_logits, dim=-1)
            nearest_atom_indices = nearest_atom_probs.argmax(dim=-1)
            coords_original = x_original[:, :, :self.point_dim]
            B_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, L)
            nearest_coords = coords_original[B_idx, nearest_atom_indices]
            distances = torch.norm(coords_original - nearest_coords, dim=-1, keepdim=True)
            distances = distances * (~mask.unsqueeze(-1)).float()

            trans3_input = torch.cat([backbone_output, x_original, distances], dim=-1)
            trans3_output = self.add_transformer_encoder(
                self.add_transformer_proj(trans3_input), src_key_padding_mask=mask)

            add_weights = self.add_weight_head(trans3_output).squeeze(-1)
            add_weights = add_weights.masked_fill(mask, float('-inf'))
            add_feature_mean = self.add_feature_mean_head(trans3_output)
            add_feature_log_std = self.add_feature_std_head(trans3_output)

            n0_probs = F.softmax(n0_logits, dim=-1)
            n0_indices = torch.arange(self.max_n0, device=n0_logits.device, dtype=torch.float32)
            expected_n0 = (n0_probs * n0_indices.unsqueeze(0)).sum(dim=-1)
            predicted_log_n0 = torch.log(expected_n0 + 1e-6)

            return (score, predicted_log_n0, n0_logits, nearest_atom_logits, add_weights,
                    add_feature_mean, add_feature_log_std)
        else:
            pooled = backbone_output.masked_fill(mask.unsqueeze(-1), 0.0).sum(dim=1) / (~mask).sum(dim=1, keepdim=True).clamp(min=1)
            predicted_log_n0 = self.expected_n0_head(pooled).squeeze(-1)
            insert_params = self.insert_head(pooled)
            insert_mu, insert_log_std = insert_params.chunk(2, dim=-1)
            return score, predicted_log_n0, None, None, insert_mu, insert_log_std, None


class JumpNet(nn.Module):
    """Plain Transformer with optional input fusion.

    The default ``fused`` layout projects concatenated embeddings to embedding_dim.
    Explicit ``jump_net_arch: nonfused`` retains the original checkpoint keys/shapes.
    """

    def __init__(self, cfg):
        super().__init__()

        self.embedding_dim = cfg.embedding_dim
        self.point_dim = cfg.point_dim
        self.nheads = cfg.nheads
        self.dim_feedforward = cfg.dim_feedforward
        self.num_layers = cfg.num_layers

        self.use_egnn = getattr(cfg, "use_egnn", False)

        self.fuse_inputs = resolve_jump_net_arch(cfg) == "fused"

        self.backbone_output_dim = self.embedding_dim

        self.dataset_type = getattr(cfg, "dataset_type", "trip")
        self.point_dim = cfg.point_dim
        if self.dataset_type == "molecule":
            self.atom_type_dim = getattr(cfg, "atom_type_dim", 5)
            self.charge_dim = getattr(cfg, "charge_dim", 1)
            self.total_input_dim = self.point_dim + self.atom_type_dim + self.charge_dim
        else:
            self.atom_type_dim = 0
            self.charge_dim = 0
            self.total_input_dim = self.point_dim

        if self.use_egnn and self.dataset_type == "molecule":
            raise NotImplementedError("EGNN backbone not yet implemented. Set use_egnn=False for now.")
        else:
            if self.dataset_type == "molecule":
                self.coord_embedding = nn.Linear(self.point_dim, self.embedding_dim, bias=False)
                self.atom_type_embedding = nn.Linear(self.atom_type_dim, self.embedding_dim, bias=False)
                self.charge_embedding = nn.Linear(self.charge_dim, self.embedding_dim, bias=False)
            else:
                self.coord_embedding = nn.Linear(self.total_input_dim, self.embedding_dim, bias=False)

            if bool(getattr(cfg, "time_embedder_legacy", False)):
                self.time_embedder = SinusoidalTimestepLegacy(self.embedding_dim)
            else:
                self.time_embedder = SinusoidalTimestep(self.embedding_dim)

            if self.dataset_type != "molecule":
                self.n_t_embed = nn.Sequential(
                    nn.Linear(1, self.embedding_dim),
                    nn.SiLU(),
                    nn.Linear(self.embedding_dim, self.embedding_dim)
                )

            if self.dataset_type == "molecule":
                encoder_input_dim = self.embedding_dim * 4
            else:
                encoder_input_dim = self.embedding_dim * 3

            if self.fuse_inputs:
                self.f1 = nn.Sequential(
                    nn.Linear(encoder_input_dim, self.embedding_dim),
                    nn.SiLU(),
                )
                encoder_input_dim = self.embedding_dim

            self.input_norm = nn.LayerNorm(encoder_input_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=encoder_input_dim,
                nhead=self.nheads,
                dim_feedforward=self.dim_feedforward,
                batch_first=True,
                dropout=0.0
            )
            if not self.fuse_inputs:
                self.encoder_layer = encoder_layer
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

            self.backbone_output_dim = encoder_input_dim

        self.score_head = nn.Linear(self.backbone_output_dim, self.total_input_dim)

        transformer_dim = getattr(cfg, "transformer_dim", 128)
        self.transformer_dim = transformer_dim
        if self.dataset_type == "molecule":
            self.n0_transformer_proj = nn.Linear(self.backbone_output_dim, transformer_dim)

            self.n0_transformer_layer = nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=getattr(cfg, "transformer_nhead", 4),
                dim_feedforward=self.dim_feedforward,
                batch_first=True,
                dropout=0.0
            )
            self.n0_transformer_encoder = nn.TransformerEncoder(
                self.n0_transformer_layer,
                num_layers=getattr(cfg, "transformer_num_layers", 8)
            )
            max_n0 = getattr(cfg, "max_n0", 35)
            self.max_n0 = max_n0
            self.expected_n0_head = nn.Linear(transformer_dim, max_n0)
        else:
            hidden_dim = getattr(cfg, "hidden_dim", 256)
            self.expected_n0_head = nn.Sequential(
                nn.Linear(self.backbone_output_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        if self.dataset_type == "molecule":
            self.nearest_atom_head = nn.Linear(transformer_dim, 1)
        else:
            self.nearest_atom_head = None

        if self.dataset_type == "molecule":
            trans3_input_dim = self.backbone_output_dim + self.total_input_dim + 1
            self.add_transformer_proj = nn.Linear(trans3_input_dim, transformer_dim)

            self.add_transformer_layer = nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=getattr(cfg, "transformer_nhead", 4),
                dim_feedforward=self.dim_feedforward,
                batch_first=True,
                dropout=0.0
            )
            self.add_transformer_encoder = nn.TransformerEncoder(
                self.add_transformer_layer,
                num_layers=getattr(cfg, "transformer_num_layers", 8)
            )

            self.add_weight_head = nn.Linear(transformer_dim, 1)

            feature_output_dim = self.point_dim + self.atom_type_dim + self.charge_dim
            self.add_feature_mean_head = nn.Linear(transformer_dim, feature_output_dim)
            self.add_feature_std_head = nn.Linear(transformer_dim, feature_output_dim)
        else:
            hidden_dim = getattr(cfg, "hidden_dim", 256)
            self.insert_head = nn.Sequential(
                nn.Linear(self.backbone_output_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.point_dim * 2)
            )

    def forward(self, x, t, mask=None):
        B, L, _ = x.shape
        if mask is None:
            mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)

        x_original = x.clone()

        if self.use_egnn and self.dataset_type == "molecule":
            raise NotImplementedError("EGNN backbone not yet implemented. Set use_egnn=False for now.")
        else:
            if self.dataset_type == "molecule":
                coords = x[:, :, :self.point_dim]
                atom_types = x[:, :, self.point_dim:self.point_dim + self.atom_type_dim]
                charges = x[:, :, self.point_dim + self.atom_type_dim:]

                coord_emb = self.coord_embedding(coords)
                atom_type_emb = self.atom_type_embedding(atom_types)
                charge_emb = self.charge_embedding(charges)
                t_emb = self.time_embedder(t)

                h = torch.cat([coord_emb, atom_type_emb, charge_emb, t_emb], dim=-1)
            else:
                x_emb = self.coord_embedding(x)
                t_emb = self.time_embedder(t)

                n_t = (~mask).sum(dim=1, keepdim=True).float()
                n_t_emb = self.n_t_embed(n_t)
                n_t_emb = n_t_emb.unsqueeze(1).expand(-1, L, -1)

                h = torch.cat([x_emb, t_emb, n_t_emb], dim=-1)

            if self.fuse_inputs:
                h = self.f1(h)
            h = self.input_norm(h)
            backbone_output = self.encoder(h, src_key_padding_mask=mask)

        score = self.score_head(backbone_output)

        if self.dataset_type == "molecule":
            n0_input = self.n0_transformer_proj(backbone_output)
            trans2_output = self.n0_transformer_encoder(n0_input, src_key_padding_mask=mask)
            pooled_trans2 = trans2_output.masked_fill(mask.unsqueeze(-1), 0.0).sum(dim=1) / (~mask).sum(dim=1, keepdim=True).clamp(min=1)
            n0_logits = self.expected_n0_head(pooled_trans2)
        else:
            pooled = backbone_output.masked_fill(mask.unsqueeze(-1), 0.0).sum(dim=1) / (~mask).sum(dim=1, keepdim=True).clamp(min=1)
            predicted_log_n0 = self.expected_n0_head(pooled).squeeze(-1)
            n0_logits = None

        if self.dataset_type == "molecule":
            nearest_atom_logits = self.nearest_atom_head(trans2_output).squeeze(-1)
            nearest_atom_logits = nearest_atom_logits.masked_fill(mask, float('-inf'))
        else:
            nearest_atom_logits = None

        if self.dataset_type == "molecule":
            nearest_atom_probs = F.softmax(nearest_atom_logits, dim=-1)
            nearest_atom_indices = nearest_atom_probs.argmax(dim=-1)

            coords_original = x_original[:, :, :self.point_dim]

            B_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, L)
            nearest_coords = coords_original[B_idx, nearest_atom_indices]
            coord_diff = coords_original - nearest_coords
            distances = torch.norm(coord_diff, dim=-1, keepdim=True)
            distances = distances * (~mask.unsqueeze(-1)).float()

            trans3_input = torch.cat([backbone_output, x_original, distances], dim=-1)
            trans3_proj_input = self.add_transformer_proj(trans3_input)
            trans3_output = self.add_transformer_encoder(trans3_proj_input, src_key_padding_mask=mask)

            add_weights = self.add_weight_head(trans3_output).squeeze(-1)
            add_weights = add_weights.masked_fill(mask, float('-inf'))

            add_feature_mean = self.add_feature_mean_head(trans3_output)
            add_feature_log_std = self.add_feature_std_head(trans3_output)

            if n0_logits is not None:
                n0_probs = F.softmax(n0_logits, dim=-1)
                n0_indices = torch.arange(self.max_n0, device=n0_logits.device, dtype=torch.float32)
                expected_n0 = (n0_probs * n0_indices.unsqueeze(0)).sum(dim=-1)
                predicted_log_n0 = torch.log(expected_n0 + 1e-6)
            else:
                predicted_log_n0 = None

            return (score, predicted_log_n0, n0_logits, nearest_atom_logits, add_weights,
                    add_feature_mean, add_feature_log_std)
        else:
            pooled = backbone_output.masked_fill(mask.unsqueeze(-1), 0.0).sum(dim=1) / (~mask).sum(dim=1, keepdim=True).clamp(min=1)
            insert_params = self.insert_head(pooled)
            insert_mu, insert_log_std = insert_params.chunk(2, dim=-1)
            return score, predicted_log_n0, None, None, insert_mu, insert_log_std, None
