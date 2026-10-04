import torch
import torch.nn as nn
from typing import Optional

from .egnn_core import (
    EGNN, coord2diff, unsorted_segment_sum,
    remove_mean_with_mask, SinusoidsEmbeddingNew,
)

class EGNNAtomTypeBackbone(nn.Module):


    def __init__(
        self,
        num_atom_types: int = 5,
        hidden_nf: int = 64,
        n_layers: int = 4,
        attention: bool = False,
        tanh: bool = False,
        norm_constant: int = 0,
        inv_sublayers: int = 2,
        sin_embedding: bool = False,
        normalization_factor: int = 100,
        aggregation_method: str = 'sum',
        condition_time: bool = False,
        CoM0: bool = True,
        return_coords: bool = False,
        return_hidden: bool = False,

        return_pos_pred: bool = False,

        coord_clip: Optional[float] = 50.0,
        feat_clip: Optional[float] = 10.0,
        center_coords: bool = True,
        coord_norm_scale: float = 1.0,
        eps: float = 1e-8,

        embedding_out_mode: int = 1,

    ):
        super().__init__()

        self.num_atom_types = num_atom_types
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.condition_time = condition_time
        self.CoM0 = CoM0
        self.return_coords = return_coords
        self.return_hidden = return_hidden
        self.return_pos_pred = bool(return_pos_pred)
        self.coord_clip = coord_clip
        self.feat_clip = feat_clip
        self.center_coords = bool(center_coords)
        self.coord_norm_scale = float(coord_norm_scale)
        self.eps = float(eps)
        self.embedding_out_mode = int(embedding_out_mode)
        if self.embedding_out_mode not in (0, 1):
            raise ValueError(
                "embedding_out_mode must be 0 (legacy projection) or 1 "
                f"(skip unused projection), got {embedding_out_mode!r}"
            )


        in_node_nf = num_atom_types
        if condition_time:
            in_node_nf += 1


        context_node_nf = 0


        self.egnn = EGNN(
            in_node_nf=in_node_nf + context_node_nf,
            in_edge_nf=1,
            hidden_nf=hidden_nf,
            act_fn=nn.SiLU(),
            n_layers=n_layers,
            attention=attention,
            norm_diff=True,
            out_node_nf=hidden_nf,
            tanh=tanh,
            coords_range=15,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            use_embedding_out=(self.embedding_out_mode == 0),
        )


        self._edges_dict = {}

    def get_adj_matrix(self, n_nodes: int, batch_size: int, device: torch.device):

        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]


        rows, cols = [], []
        for batch_idx in range(batch_size):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    rows.append(i + batch_idx * n_nodes)
                    cols.append(j + batch_idx * n_nodes)

        edges = [
            torch.LongTensor(rows).to(device),
            torch.LongTensor(cols).to(device)
        ]

        if n_nodes not in self._edges_dict:
            self._edges_dict[n_nodes] = {}
        self._edges_dict[n_nodes][batch_size] = edges

        return edges

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        feat_gate: Optional[torch.Tensor] = None,
        input_embedding_add: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        device = x.device
        B, N, total_dim = x.shape


        expected_dim = 3 + self.num_atom_types
        assert total_dim == expected_dim, \
            f"Expected x.shape[2]={expected_dim} (3 coords + {self.num_atom_types} atom_types), got {total_dim}"


        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        coords_raw = x[:, :, :3]
        atom_types = x[:, :, 3:3+self.num_atom_types]

        mask = torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
        mask = (mask.to(device=device) > 0).to(dtype=x.dtype)
        node_mask = mask.unsqueeze(2)

        coords = coords_raw
        if self.center_coords:
            denom = node_mask.sum(dim=1, keepdim=True).clamp(min=self.eps)
            mean = (coords * node_mask).sum(dim=1, keepdim=True) / denom
            coords = coords - mean

        coord_scale = max(abs(self.coord_norm_scale), self.eps)
        if coord_scale != 1.0:
            coords = coords / coord_scale

        if self.coord_clip is not None:
            coords = coords.clamp(min=-float(self.coord_clip), max=float(self.coord_clip))
        if self.feat_clip is not None:
            atom_types = atom_types.clamp(min=-float(self.feat_clip), max=float(self.feat_clip))


        h = atom_types


        if self.condition_time:
            if t is None:
                raise ValueError("condition_time=True but t is None")
            assert t.shape == (B,), f"Expected t.shape=(B,), got {t.shape}"
            h_time = t.view(B, 1, 1).expand(B, N, 1)
            h = torch.cat([h, h_time], dim=2)

        edge_mask = (mask.unsqueeze(1) * mask.unsqueeze(2))

        eye_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        edge_mask = edge_mask * (~eye_mask).float()


        h_flat = h.view(B * N, -1)
        coords_flat = coords.view(B * N, 3)
        node_mask_flat = node_mask.view(B * N, 1)
        edge_mask_flat = edge_mask.view(B * N * N, 1)


        edges = self.get_adj_matrix(N, B, device)
        edges = [e.to(device) for e in edges]


        h_flat = h_flat * node_mask_flat
        if feat_gate is not None:
            gate = feat_gate
            if gate.dim() == 2:
                gate = gate.unsqueeze(-1)
            if gate.shape[:2] != (B, N):
                raise ValueError(
                    f"feat_gate must be [B,N] or [B,N,1] with B={B}, N={N}; "
                    f"got {tuple(feat_gate.shape)}"
                )
            gate = torch.nan_to_num(gate, nan=1.0, posinf=1.0, neginf=1.0)
            h_flat = h_flat * gate.reshape(B * N, 1).to(dtype=h_flat.dtype)
        input_embedding_add_flat = None
        if input_embedding_add is not None:
            expected_add_shape = (B, N, self.hidden_nf)
            if input_embedding_add.shape != expected_add_shape:
                raise ValueError(
                    "input_embedding_add must have shape "
                    f"{expected_add_shape}, got {tuple(input_embedding_add.shape)}"
                )
            input_embedding_add = torch.nan_to_num(
                input_embedding_add.to(device=device, dtype=h_flat.dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            input_embedding_add_flat = (
                input_embedding_add * node_mask
            ).reshape(B * N, self.hidden_nf)
        coords_flat = coords_flat * node_mask_flat


        h_out, coords_out, h_hidden = self.egnn(
            h_flat, coords_flat, edges,
            node_mask=node_mask_flat,
            edge_mask=edge_mask_flat,
            return_last_layer=True,
            input_embedding_add=input_embedding_add_flat,
        )

        h_out = torch.nan_to_num(h_out, nan=0.0, posinf=0.0, neginf=0.0)
        h_hidden = torch.nan_to_num(h_hidden, nan=0.0, posinf=0.0, neginf=0.0)


        vel = (coords_out - coords_flat).reshape(B, N, 3)
        finite_graph = torch.isfinite(vel).flatten(start_dim=1).all(dim=1)
        vel = torch.where(
            finite_graph.view(B, 1, 1),
            vel,
            torch.zeros_like(vel),
        )
        vel = vel * node_mask


        if self.CoM0:

            nm = node_mask
            denom = nm.sum(dim=1, keepdim=True).clamp(min=1.0)
            mean = (vel * nm).sum(dim=1, keepdim=True) / denom
            vel = vel - mean * nm


        h_out = h_out.view(B, N, self.hidden_nf)
        h_hidden = h_hidden.view(B, N, self.hidden_nf)
        coords_updated = coords_raw + vel * coord_scale

        pos_pred = vel * coord_scale

        h_out = torch.nan_to_num(h_out, nan=0.0, posinf=0.0, neginf=0.0)
        h_hidden = torch.nan_to_num(h_hidden, nan=0.0, posinf=0.0, neginf=0.0)
        coords_updated = torch.nan_to_num(coords_updated, nan=0.0, posinf=0.0, neginf=0.0)
        pos_pred = torch.nan_to_num(pos_pred, nan=0.0, posinf=0.0, neginf=0.0)


        h_out = h_out * node_mask
        h_hidden = h_hidden * node_mask
        coords_updated = coords_updated * node_mask


        returns = []


        returns.append(h_out)

        if self.return_coords:
            returns.append(coords_updated)

        if self.return_hidden:
            returns.append(h_hidden)

        if self.return_pos_pred:
            returns.append(pos_pred)

        if len(returns) == 1:
            return returns[0]
        else:
            return tuple(returns)
