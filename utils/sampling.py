import torch

def sample_n_from_val_n(
    val_n: torch.Tensor,
    num_samples: int,
    device=None,
):
    if device is None:
        device = val_n.device

    val_n = val_n.to(device).long()

    if val_n.numel() == 0:
        raise ValueError(f"val_n must not be empty. Got val_n with shape {val_n.shape}")

    idx = torch.randint(
        low=0,
        high=val_n.numel(),
        size=(num_samples,),
        device=device,
    )
    n_samples = val_n[idx].clamp(min=1)
    return n_samples
