import torch.nn as nn
import torch

class BaseMethod(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

    def prepare_train_epoch(self, train_data_raw):
        return {"x0": train_data_raw.to(self.device)}

    def prepare_val_data(self, val_data_raw):
        return {"x0": val_data_raw.to(self.device)}

    def prepare_batch(self, batch_raw: torch.Tensor, *, split: str = "train") -> dict:

        _ = split

        return {"x0": batch_raw.to(self.device, non_blocking=True)}

    def training_step(self, batch):
        raise NotImplementedError

    def eval_step(self, batch, generator=None):
        raise NotImplementedError

    def sample_sets(self, num_samples, device,val_n):
        raise NotImplementedError
