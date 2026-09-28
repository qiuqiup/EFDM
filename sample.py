import argparse
from pathlib import Path

import numpy as np
import torch

from run_train import METHODS, build_method, load_config, load_training_data
from utils.set_seed import set_seed


def load_weights(method, checkpoint_path, weights):
    path = Path(checkpoint_path)
    direct_ema = path.stem.endswith("_ema")
    if direct_ema and weights == "raw":
        raise ValueError("A *_ema.pth checkpoint contains EMA weights only")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if direct_ema:
        raw_path = path.with_name(path.stem[:-4] + path.suffix)
        if raw_path.exists():
            raw = torch.load(raw_path, map_location="cpu", weights_only=False)
            method.load_state_dict(raw.get("method_state", raw))
        elif len(method.state_dict()) != len(checkpoint):
            raise ValueError(f"{path} needs its matching raw checkpoint to load model buffers")
        else:
            method.load_state_dict(checkpoint)
        ema_state = checkpoint
    else:
        method.load_state_dict(checkpoint.get("method_state", checkpoint))
        ema_state = checkpoint.get("ema_state") if "method_state" in checkpoint else None
        sidecar = path.with_name(path.stem + "_ema.pth")
        if ema_state is None and sidecar.exists():
            ema_state = torch.load(sidecar, map_location="cpu", weights_only=False)

    if weights == "ema" and ema_state is None:
        raise ValueError("EMA weights were requested but the checkpoint has none")
    if weights != "raw" and ema_state is not None:
        params = dict(method.named_parameters())
        unknown = set(ema_state) - set(params)
        if unknown:
            raise ValueError(f"EMA checkpoint has unknown parameters: {sorted(unknown)[:3]}")
        with torch.no_grad():
            for key, value in ema_state.items():
                params[key].copy_(value.to(device=params[key].device, dtype=params[key].dtype))
        print("Using EMA weights")
    return method


def pad_and_concat(batches, value):
    width = max(batch.shape[1] for batch in batches)
    padded = []
    for batch in batches:
        if batch.shape[1] == width:
            padded.append(batch)
        else:
            shape = (batch.shape[0], width, *batch.shape[2:])
            out = batch.new_full(shape, value)
            out[:, :batch.shape[1]] = batch
            padded.append(out)
    return torch.cat(padded)


def main():
    parser = argparse.ArgumentParser(description="Generate unconditional point-set samples")
    parser.add_argument("--method_name", choices=[*METHODS, "existence_soft_count", "jump_adaptive"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--pc-mode", choices=["none", "510p485c", "1000p485c"], default="none")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weights", choices=["auto", "raw", "ema"], default="auto")
    parser.add_argument("--vp-no-noise-final-step", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if args.num_samples < 1 or args.batch_size < 1:
        parser.error("num-samples and batch-size must be positive")
    set_seed(args.seed)
    cfg = load_config(args.config)
    if args.vp_no_noise_final_step is not None:
        cfg.vp_sde_no_noise_final_step = args.vp_no_noise_final_step
    method_name = args.method_name or getattr(cfg, "method_name", "existence")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = build_method(method_name, cfg, device)
    method = load_weights(method, args.checkpoint, args.weights)
    if hasattr(method, "fit_n_distribution"):
        _, train_n, _, _ = load_training_data(cfg)
        method.fit_n_distribution(train_n)
        method.n_distribution._rng = np.random.default_rng(args.seed)
    if cfg.method_name == "psd" and args.pc_mode != "none":
        parser.error("PSD sampling supports only --pc-mode none")
    method.eval()
    set_seed(args.seed)

    samples, probabilities, keeps, stats = [], [], [], []
    for start in range(0, args.num_samples, args.batch_size):
        count = min(args.batch_size, args.num_samples - start)
        if cfg.method_name == "psd":
            result = method.sample_sets(count, device=device)
        else:
            result = method.sample_sets(count, device=device, pc_mode=args.pc_mode)
        if cfg.method_name == "existence":
            x, p, keep = result
            probabilities.append(p.cpu())
            keeps.append(keep.cpu())
        else:
            x = result
        samples.append(x.cpu())
        stats.append(getattr(method, "last_sampling_stats", None))

    pad_value = float("inf") if cfg.method_name == "psd" else float("-inf")
    output_data = {
        "method_name": method_name,
        "samples": pad_and_concat(samples, pad_value),
        "sampling_stats": stats,
    }
    if probabilities:
        output_data["existence_prob"] = pad_and_concat(probabilities, 0.0)
        output_data["keep"] = pad_and_concat(keeps, False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_data, output)
    print(f"Saved {args.num_samples} samples to {output}")


if __name__ == "__main__":
    main()
