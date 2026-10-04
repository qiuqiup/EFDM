import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent


def load_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def split_sets(data):
    if not isinstance(data, torch.Tensor) or data.ndim != 3:
        raise ValueError("Expected a padded tensor with shape [sets, slots, coordinates]")
    valid = torch.isfinite(data).all(dim=-1)
    padding = torch.isinf(data).all(dim=-1)
    if not bool((valid | padding).all()):
        raise ValueError("Rows must be finite points or all-infinite padding")
    return [data[i, valid[i]] for i in range(len(data))]


def normalization_context(dataset, train, stats_path):
    if dataset == "bent":
        points = train[torch.isfinite(train).all(-1)].double()
        low, high = points.amin(0), points.amax(0)
        margin = (high - low) * 0.05
        bounds = torch.stack((low - margin, high + margin), dim=-1).numpy()
        mean = np.zeros(train.shape[-1], dtype=np.float64)
        std = np.ones(train.shape[-1], dtype=np.float64)
    else:
        if stats_path is None:
            raise ValueError(f"--stats is required for {dataset}")
        stats = load_tensor(stats_path)
        key = "space_bound_lonlat" if dataset == "earthquake" else "space_bound"
        bounds = np.asarray(stats[key], dtype=np.float64).reshape(train.shape[-1], 2)
        mean = np.asarray(stats["mean"], dtype=np.float64).reshape(-1)
        std = np.asarray(stats["std"], dtype=np.float64).reshape(-1)
    if not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Invalid coordinate bounds")
    return bounds, mean, std


def normalized_sets(sets, bounds, mean, std):
    low, high = bounds[:, 0], bounds[:, 1]
    output = []
    for item in sets:
        points = item.detach().cpu().numpy().astype(np.float64)
        raw = points * std + mean
        mapped = 2.0 * (raw - low) / (high - low) - 1.0
        output.append(np.clip(mapped, -1.0, 1.0))
    return output


def joint_sets(dataset, raw_sets, normalized):
    if dataset == "trip":
        return [item.detach().cpu().numpy().astype(np.float64) for item in raw_sets]
    return normalized


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, torch.Tensor):
        return json_value(value.item() if value.numel() == 1 else value.tolist())
    return value


def evaluate_spatial(args, samples):
    from metrics.evaluators import eval_feature_w1_binned, w1_cardinality
    from metrics.jmmd import median_sw_sigma_equal
    from metrics.jmmd_accelerated import evaluate_joint_conditional
    from metrics.wd_mmd import calibrate_train, distance_grids, evaluate_grids

    train_path = args.train_val or (
        ROOT / "data/bent/two_gaussian_train_val_data.pt" if args.dataset == "bent" else None
    )
    test_path = args.test or (
        ROOT / "data/bent/two_gaussian_test_data.pt" if args.dataset == "bent" else None
    )
    if train_path is None or test_path is None:
        raise ValueError("--train-val and --test are required for external datasets")
    train = load_tensor(train_path)[0]
    test = load_tensor(test_path)[0]
    generated_sets = split_sets(samples)
    train_sets = split_sets(train)
    real_sets = split_sets(test)
    if any(len(item) == 0 for item in generated_sets + train_sets + real_sets):
        raise ValueError("The reported full-set metrics require nonempty sets")
    if samples.shape[-1] != 2 or train.shape[-1] != 2 or test.shape[-1] != 2:
        raise ValueError("Spatial evaluation expects two coordinate channels")

    result = {
        "dataset": args.dataset,
        "n_generated": len(generated_sets),
        "n_test": len(real_sets),
        "card_w1": float(w1_cardinality(samples, test)),
    }
    print("Computing feature W1 metrics", flush=True)
    bins = [0, max(samples.shape[1], test.shape[1]) + 1]
    for feature in ("mean", "var", "skew", "kurt", "nn1"):
        result[f"{feature}_w1"] = float(
            eval_feature_w1_binned(generated_sets, real_sets, feature, bins)["overall_w1"]
        )

    bounds, mean, std = normalization_context(args.dataset, train, args.stats)
    gen_normalized = normalized_sets(generated_sets, bounds, mean, std)
    real_normalized = normalized_sets(real_sets, bounds, mean, std)
    train_normalized = normalized_sets(train_sets, bounds, mean, std)
    gen_joint = joint_sets(args.dataset, generated_sets, gen_normalized)
    real_joint = joint_sets(args.dataset, real_sets, real_normalized)
    train_joint = joint_sets(args.dataset, train_sets, train_normalized)

    print("Calibrating and computing JMMD and ICMMD", flush=True)
    sigma = median_sw_sigma_equal(train_joint, seed=0)
    jmmd, icmmd = evaluate_joint_conditional(
        gen_joint, real_joint, [sigma * 0.5, sigma, sigma * 2.0],
        seed=0, device=args.device,
    )
    result["sigma_train_equal"] = sigma
    for index, suffix in enumerate(("_bw05", "", "_bw20")):
        result[f"jmmd2{suffix}"] = jmmd[index]
        result[f"icmmd2{suffix}"] = (
            icmmd[index]["icmmd2"] if icmmd[index]["n_eligible"] else None
        )
        result[f"icmmd2_covered{suffix}"] = icmmd[index]["icmmd2_covered"]
    result["icmmd_coverage"] = icmmd[1]["coverage"]
    result["icmmd_eligible_mass"] = icmmd[1]["eligible_mass"]
    result["icmmd_n_eligible"] = icmmd[1]["n_eligible"]

    print("Calibrating train-distance scale for full-set sliced-W2 MMD", flush=True)
    train_m, _ = calibrate_train(
        train_normalized, device=args.device, batch_size=args.pair_batch_size
    )
    print("Computing full GG, TT, GT distance grids", flush=True)
    grids = distance_grids(
        gen_normalized, real_normalized, device=args.device,
        seed=0, batch_size=args.pair_batch_size,
    )
    result.update(evaluate_grids(grids, train_m))
    result["mmd_sigma_train"] = train_m
    result["coordinate_bounds"] = bounds.tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate the metrics reported in the manuscript")
    parser.add_argument("--dataset", choices=("bent", "trip", "earthquake", "qm9"), required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--train-val", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pair-batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.pair_batch_size < 1:
        parser.error("--pair-batch-size must be positive")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = load_tensor(args.samples)
    if not isinstance(payload, dict) or "samples" not in payload:
        raise ValueError("Expected the output dictionary from sample.py")
    if args.dataset == "qm9":
        from metrics.qm9 import evaluate_qm9
        result = evaluate_qm9(payload)
    else:
        result = evaluate_spatial(args, payload["samples"])
    output = args.output or args.samples.with_suffix(".metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_value(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
