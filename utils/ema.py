from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn

class EMA:

    def __init__(self, model: nn.Module, *, decay: float):
        if not (0.0 < float(decay) < 1.0):
            raise ValueError(f"EMA decay must be in (0,1), got {decay}")
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self._init_from(model)

    @torch.no_grad()
    def _init_from(self, model: nn.Module) -> None:
        self.shadow = {}
        for name, p in model.named_parameters():
            if p is None:
                continue
            if not p.requires_grad:
                continue

            self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module, *, decay: float | None = None) -> None:

        d = self.decay if decay is None else float(decay)
        if not (0.0 <= d < 1.0):
            raise ValueError(f"EMA update decay must be in [0,1), got {d}")
        for name, p in model.named_parameters():
            if p is None or (not p.requires_grad):
                continue
            if name not in self.shadow:

                self.shadow[name] = p.detach().clone()
                continue
            s = self.shadow[name]

            if (s.device != p.device) or (s.dtype != p.dtype):
                s = s.to(device=p.device, dtype=p.dtype)
                self.shadow[name] = s
            s.mul_(d).add_(p.detach(), alpha=(1.0 - d))

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.shadow.items()}

    @torch.no_grad()
    def load_state_dict(self, sd: Dict[str, torch.Tensor], *, strict: bool = False) -> None:
        if not isinstance(sd, dict):
            raise TypeError(f"EMA.load_state_dict expected dict, got {type(sd)}")
        if strict:
            missing = set(self.shadow.keys()) - set(sd.keys())
            extra = set(sd.keys()) - set(self.shadow.keys())
            if missing or extra:
                raise KeyError(f"EMA strict load mismatch: missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")

        skipped = []
        for k, v in sd.items():
            if k in self.shadow:

                tgt = self.shadow[k]
                if torch.is_tensor(v) and tgt.shape != v.shape:

                    skipped.append(k)
                    continue
                self.shadow[k] = v.detach().to(device=tgt.device, dtype=tgt.dtype).clone()
        if skipped:
            print(f"[EMA] load_state_dict: skipped {len(skipped)} shape-mismatched key(s): "
                  f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """
        Copy EMA params into `model` in-place.
        """
        for name, p in model.named_parameters():
            if p is None or (not p.requires_grad):
                continue
            v = self.shadow.get(name, None)
            if v is None:
                continue
            p.data.copy_(v.to(device=p.device, dtype=p.dtype))

    @contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[None]:
        """
        Temporarily swap model params to EMA values, then restore.
        """
        backup: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p is None or (not p.requires_grad):
                continue
            backup[name] = p.detach().clone()
        try:
            self.copy_to(model)
            yield
        finally:
            for name, p in model.named_parameters():
                if p is None or (not p.requires_grad):
                    continue
                v = backup.get(name, None)
                if v is None:
                    continue
                p.data.copy_(v.to(device=p.device, dtype=p.dtype))
