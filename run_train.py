import argparse
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from methods import ExistenceMethod, JumpMethod, MaskMethod, NoGroupMethod, PSDMethod
from trainer.trainer import Trainer
from utils.set_seed import set_seed


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("Config must be a YAML mapping")
    return SimpleNamespace(**values)


def load_qm9_split(path, cfg):
    with np.load(path, allow_pickle=False) as data:
        positions = torch.from_numpy(data["positions"]).float()
        charges = torch.from_numpy(data["charges"]).long()
        counts = torch.from_numpy(data["num_atoms"]).long()
    mask = charges != 0
    if bool(getattr(cfg, "qm9_preprocess_com0", False)):
        mean = (positions * mask.unsqueeze(-1)).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp(min=1).unsqueeze(-1)
        positions = (positions - mean) * mask.unsqueeze(-1)
    if bool(getattr(cfg, "qm9_preprocess_pos_scale", False)):
        positions = positions / float(cfg.qm9_pos_scale)
    lut = torch.full((10,), -1, dtype=torch.long)
    lut[6], lut[1], lut[7], lut[8], lut[9] = 0, 1, 2, 3, 4
    if ((charges < 0) | (charges > 9) | ((charges != 0) & (lut[charges.clamp(0, 9)] < 0))).any():
        raise ValueError("QM9 charges must be padded 0 or atomic numbers 1, 6, 7, 8, 9")
    types = F.one_hot(lut[charges].clamp(min=0), 5).float() * mask.unsqueeze(-1)
    if int(cfg.point_dim) != 8:
        raise ValueError("Molecule EFDM expects point_dim: 8 (xyz plus five atom types)")
    result = torch.cat((positions, types), -1)
    result[~mask] = float("inf")
    return result, counts


def load_training_data(cfg):
    dataset = str(cfg.dataset_type)
    if dataset == "molecule":
        root = Path(getattr(cfg, "qm9_data_dir", "data/qm9"))
        train_path, val_path = root / "train.npz", root / "valid.npz"
        if bool(getattr(cfg, "qm9_preprocess_pos_scale", False)):
            with np.load(train_path, allow_pickle=False) as data:
                positions = torch.from_numpy(data["positions"]).float()
                mask = torch.from_numpy(data["charges"]).long() != 0
            if bool(getattr(cfg, "qm9_preprocess_com0", False)):
                mean = (positions * mask.unsqueeze(-1)).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp(min=1).unsqueeze(-1)
                positions = (positions - mean) * mask.unsqueeze(-1)
            values = positions[mask]
            cfg.qm9_pos_scale = float(torch.sqrt((values.double() ** 2).mean() + 1e-12).item()) if values.numel() else 1.0
        train, train_n = load_qm9_split(train_path, cfg)
        val, val_n = load_qm9_split(val_path, cfg)
        return train, train_n, val, val_n
    if dataset not in ("synthetic", "trip", "earthquake_psd"):
        raise ValueError(f"Unsupported dataset_type: {dataset}")
    path = getattr(cfg, "data_path", None) or getattr(cfg, f"{dataset}_data_path", None)
    if path is None:
        if dataset == "synthetic":
            path = "data/bent/two_gaussian_train_val_data.pt"
        else:
            raise ValueError(f"Provide data_path for the paper's {dataset} split")
    train, train_n, val, val_n = torch.load(path, map_location="cpu", weights_only=False)
    return train, train_n, val, val_n


METHODS = {
    "existence": ExistenceMethod,
    "mask": MaskMethod,
    "nogroup": NoGroupMethod,
    "jump": JumpMethod,
    "psd": PSDMethod,
}


def build_method(method_name, cfg, device):
    if method_name == "existence_soft_count":
        cfg.existence_net_arch = "soft_count"
        method_name = "existence"
    elif method_name == "jump_adaptive":
        cfg.lambda_adaptive = True
        method_name = "jump"
    cfg.method_name = method_name
    return METHODS[method_name](cfg, device)


def main():
    parser = argparse.ArgumentParser(description="Train a point-set diffusion method")
    parser.add_argument("--method_name", choices=[*METHODS, "existence_soft_count", "jump_adaptive"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir")
    parser.add_argument("--resume_path")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--grad_clip", type=float)
    parser.add_argument("--dataset", choices=["synthetic", "trip", "earthquake_psd", "molecule"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    method_name = args.method_name or getattr(cfg, "method_name", "existence")
    for key, value in (("num_epochs", args.epochs), ("seed", args.seed), ("grad_clip", args.grad_clip), ("dataset_type", args.dataset)):
        if value is not None:
            setattr(cfg, key, value)
    set_seed(int(cfg.seed))
    train, train_n, val, val_n = load_training_data(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = args.run_dir or (str(Path(args.resume_path).parent) if args.resume_path else None)
    cfg.ckpt_dir = run_dir or str(Path("checkpoints") / datetime.now().strftime("%Y%m%d_%H%M%S"))
    method = build_method(method_name, cfg, device)
    if hasattr(method, "fit_data_statistics"):
        method.fit_data_statistics(train)
    if hasattr(method, "fit_n_distribution"):
        method.fit_n_distribution(train_n)
    if cfg.method_name == "nogroup":
        if bool(getattr(cfg, "use_dataloader", False)):
            raise ValueError("IDM requires use_dataloader: false so batches contain points")
        cfg.ema_units_per_epoch = int(train_n.sum().item())
    trainer = Trainer(method, cfg, train, val, device)
    if args.resume_path:
        trainer.resume_from_checkpoint(args.resume_path)
    trainer.train(cfg.num_epochs)


if __name__ == "__main__":
    main()
