"""Convert padded point-set batches to variable-length lists."""

import torch


def tensor_to_set_list(data: torch.Tensor):
    """Remove either +inf or -inf padding from a [B, K, D] tensor."""
    return [data[i, ~torch.isinf(data[i, :, 0])] for i in range(data.size(0))]
