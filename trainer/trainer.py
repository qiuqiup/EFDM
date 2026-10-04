import math
import random
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.ema import EMA


class Trainer:
    def __init__(self, method, cfg, train_data_raw, val_data_raw, device):
        self.method = method
        self.cfg = cfg
        self.device = device
        self.batch_size = int(cfg.batch_size)
        self.ckpt_dir = Path(cfg.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.use_dataloader = bool(getattr(cfg, "use_dataloader", False))
        if self.use_dataloader:
            self.train_data_raw = train_data_raw.cpu()
            self.val_data_raw = val_data_raw.cpu() if val_data_raw is not None else None
        else:
            self.train_data_raw = train_data_raw.to(device)
            self.val_data_raw = val_data_raw.to(device) if val_data_raw is not None else None

        self.optimizer = torch.optim.Adam(
            method.parameters(), lr=float(cfg.lr),
            weight_decay=float(getattr(cfg, "weight_decay", 0.0)),
        )
        self.base_lr = float(cfg.lr)
        self.warmup_samples = float(getattr(cfg, "lr_warmup_kimg", 0.0) or 0.0) * 1000.0
        self.processed_samples = 0
        self.grad_clip = getattr(cfg, "grad_clip", None)
        half_life_epochs = float(getattr(cfg, "ema_halflife_epochs", 0.0) or 0.0)
        self.ema_halflife_samples = (
            half_life_epochs * int(getattr(cfg, "ema_units_per_epoch", len(train_data_raw)))
            if half_life_epochs > 0
            else float(getattr(cfg, "ema_halflife_kimg", 0.0) or 0.0) * 1000.0
        )
        self.ema_rampup_ratio = getattr(cfg, "ema_rampup_ratio", None)
        decay = float(getattr(cfg, "ema_decay", 0.0) or 0.0)
        if self.ema_halflife_samples > 0:
            decay = 0.5 ** (self.batch_size / self.ema_halflife_samples)
        self.ema = EMA(method, decay=decay) if 0 < decay < 1 else None
        self.start_epoch = 0
        self.best_val = float("inf")

    def _loader(self, data, train):
        return DataLoader(
            TensorDataset(data), batch_size=self.batch_size, shuffle=train,
            num_workers=int(getattr(self.cfg, "dataloader_num_workers", 0)),
            pin_memory=bool(getattr(self.cfg, "dataloader_pin_memory", False)),
        )

    def _train_batches(self):
        if self.use_dataloader:
            for (raw,) in self._loader(self.train_data_raw, True):
                yield self.method.prepare_batch(raw, split="train")
        else:
            epoch_data = self.method.prepare_train_epoch(self.train_data_raw)
            n = next(iter(epoch_data.values())).shape[0]
            for idx in torch.randperm(n, device=self.device).split(self.batch_size):
                yield {key: value[idx] for key, value in epoch_data.items()}

    def _val_batches(self):
        if self.val_data_raw is None:
            return
        if self.use_dataloader:
            for (raw,) in self._loader(self.val_data_raw, False):
                yield self.method.prepare_batch(raw, split="val")
        else:
            data = self.method.prepare_val_data(self.val_data_raw)
            n = next(iter(data.values())).shape[0]
            for start in range(0, n, self.batch_size):
                sl = slice(start, start + self.batch_size)
                yield {key: value[sl] for key, value in data.items()}

    def _step(self, batch):
        loss, _ = self.method.training_step(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.warmup_samples > 0:
            scale = min(self.processed_samples / self.warmup_samples, 1.0)
            for group in self.optimizer.param_groups:
                group["lr"] = self.base_lr * scale
        norm = nn.utils.clip_grad_norm_(
            self.method.parameters(),
            float(self.grad_clip) if self.grad_clip is not None else float("inf"),
        )
        batch_size = next(iter(batch.values())).shape[0]
        if torch.isfinite(norm):
            self.optimizer.step()
            if self.ema is not None:
                if self.ema_halflife_samples > 0:
                    halflife = self.ema_halflife_samples
                    if self.ema_rampup_ratio is not None:
                        halflife = min(halflife, self.processed_samples * float(self.ema_rampup_ratio))
                    decay = 0.5 ** (batch_size / max(halflife, 1e-8))
                    self.ema.update(self.method, decay=decay)
                else:
                    self.ema.update(self.method)
        else:
            print("Skipping update with non-finite gradient norm")
        self.processed_samples += batch_size
        return float(loss.item()), batch_size

    def _evaluate(self):
        self.method.eval()
        total, count = 0.0, 0
        devices = [self.device.index or torch.cuda.current_device()] if self.device.type == "cuda" else []
        ema_context = self.ema.apply_to(self.method) if self.ema is not None else nullcontext()
        with torch.random.fork_rng(devices=devices), torch.no_grad(), ema_context:
            torch.manual_seed(int(getattr(self.cfg, "validation_seed", 123)))
            for batch in self._val_batches():
                loss, _ = self.method.eval_step(batch)
                size = next(iter(batch.values())).shape[0]
                total += float(loss.item()) * size
                count += size
        return total / count if count else None

    def _save_weights(self, name):
        path = self.ckpt_dir / name
        torch.save(self.method.state_dict(), path)
        if self.ema is not None:
            torch.save(self.ema.state_dict(), path.with_name(path.stem + "_ema.pth"))

    def _save_last(self, epoch):
        checkpoint = {
            "format_version": 2,
            "epoch": epoch,
            "processed_samples": self.processed_samples,
            "method_state": self.method.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val": self.best_val,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        if self.ema is not None:
            checkpoint["ema_state"] = self.ema.state_dict()
        torch.save(checkpoint, self.ckpt_dir / "last.pth")

    def resume_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "method_state" in checkpoint:
            self.method.load_state_dict(checkpoint["method_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.start_epoch = int(checkpoint["epoch"])
            self.processed_samples = int(checkpoint.get("processed_samples", 0))
            self.best_val = float(checkpoint.get("best_val", float("inf")))
            if self.ema is not None and "ema_state" in checkpoint:
                self.ema.load_state_dict(checkpoint["ema_state"])
            rng_state = checkpoint.get("rng_state")
            if rng_state is not None:
                random.setstate(rng_state["python"])
                np.random.set_state(rng_state["numpy"])
                torch.set_rng_state(rng_state["torch"])
                if torch.cuda.is_available() and rng_state["cuda"] is not None:
                    torch.cuda.set_rng_state_all(rng_state["cuda"])
        else:
            self.method.load_state_dict(checkpoint)
            if self.ema is not None:
                sidecar = Path(checkpoint_path).with_name(Path(checkpoint_path).stem + "_ema.pth")
                if sidecar.exists():
                    self.ema.load_state_dict(torch.load(sidecar, map_location="cpu", weights_only=False))
        print(f"Loaded {checkpoint_path}; next epoch is {self.start_epoch + 1}")

    def train(self, num_epochs):
        num_epochs = int(num_epochs)
        eval_interval = int(getattr(self.cfg, "eval_interval", 1))
        ckpt_interval = int(getattr(self.cfg, "ckpt_interval_epochs", 0) or 0)
        last_interval = int(getattr(self.cfg, "save_last_every_epochs", 1) or 0)
        use_validation = bool(getattr(self.cfg, "use_validation", True))
        if self.start_epoch == 0:
            self._save_weights("model_epoch_0.pth")
        for epoch in range(self.start_epoch + 1, num_epochs + 1):
            self.method.train()
            total, count = 0.0, 0
            for batch in self._train_batches():
                loss, size = self._step(batch)
                total += loss * size
                count += size
            train_loss = total / max(count, 1)
            if not math.isfinite(train_loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            is_last = epoch == num_epochs
            if use_validation and (epoch % eval_interval == 0 or is_last):
                val_loss = self._evaluate()
                print(f"epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}")
                if val_loss is not None and val_loss < self.best_val:
                    self.best_val = val_loss
                    self._save_weights("best_model.pth")
            else:
                print(f"epoch {epoch}: train={train_loss:.6f}")
            if last_interval and (epoch % last_interval == 0 or is_last):
                self._save_last(epoch)
            if ckpt_interval and (epoch % ckpt_interval == 0 or is_last):
                self._save_weights(f"model_epoch_{epoch}.pth")
        return self.best_val
