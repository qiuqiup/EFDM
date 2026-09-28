#!/usr/bin/env python3
"""Generate the bent, fixed-width two-Gaussian point-set dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


MAX_POINTS = 110
SIGMA = 0.2
KAPPA = 1.0
SPLITS = {
    "train": (500, 20260908),
    "validation": (200, 20260909),
    "test": (2000, 20260910),
}


def generate_split(num_sets: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.Generator(np.random.PCG64(seed))
    coordinates = np.full((num_sets, MAX_POINTS, 2), np.inf, dtype=np.float32)
    counts = np.empty(num_sets, dtype=np.int64)

    for index in range(num_sets):
        a, b = rng.random(2)
        count = int(np.rint(30.0 + 40.0 * (a + b)))
        distance = 0.5 + 1.5 * a
        right = rng.random(count) < 0.5
        points = rng.normal(0.0, SIGMA, size=(count, 2))
        points[:, 0] += np.where(right, distance / 2.0, -distance / 2.0)
        coordinates[index, :count] = points[rng.permutation(count)]
        counts[index] = count

    data = torch.from_numpy(coordinates)
    n = torch.from_numpy(counts)
    valid = torch.arange(MAX_POINTS)[None, :] < n[:, None]
    result = data.clone()
    points = data[valid].double()
    points[:, 1] += KAPPA * points[:, 0].square()
    result[valid] = points.float()
    return result, n


def generate_all() -> dict[str, tuple[torch.Tensor, ...]]:
    train, train_n = generate_split(*SPLITS["train"])
    val, val_n = generate_split(*SPLITS["validation"])
    test, test_n = generate_split(*SPLITS["test"])
    return {
        "two_gaussian_train_val_data.pt": (train, train_n, val, val_n),
        "two_gaussian_test_data.pt": (test, test_n),
    }


def verify(root: Path, expected: dict[str, tuple[torch.Tensor, ...]]) -> None:
    for name, tensors in expected.items():
        stored = torch.load(root / name, map_location="cpu", weights_only=True)
        if len(stored) != len(tensors) or any(
            not torch.equal(actual, wanted)
            for actual, wanted in zip(stored, tensors)
        ):
            raise ValueError(f"Regenerated tensors differ: {root / name}")
        print(f"Verified {root / name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("bent_generated"))
    parser.add_argument("--verify-dir", type=Path,
                        help="Regenerate in memory and compare with files in this directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    generated = generate_all()
    if args.verify_dir is not None:
        verify(args.verify_dir, generated)
        return

    output_dir = args.output_dir.expanduser()
    existing = [name for name in generated if (output_dir / name).exists()]
    if existing and not args.overwrite:
        parser.error("Output files already exist: " + ", ".join(existing))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, tensors in generated.items():
        torch.save(tensors, output_dir / name)
        print(f"Saved {output_dir / name}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
