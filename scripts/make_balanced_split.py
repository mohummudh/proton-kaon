#!/usr/bin/env python3
"""
scripts/make_balanced_split.py

Build a species-balanced train/val split over the concatenated [p, k, m] tensor
that run_training.py builds in all-species mode (`data.proton: all`).

The training set holds --n-train images split as evenly as possible across the
three species; every remaining image goes to validation. Indices are written in
the concatenated frame (p occupies [0, n_p), k [n_p, n_p+n_k), m the rest), so
run_training.py / run_inference.py can consume the file directly.

Usage:
    python scripts/make_balanced_split.py \
        --data /Volumes/easystore/proton-kaon/images/pkm_48x48_raw_10-179wires.pt \
        --splits-dir /Volumes/easystore/proton-kaon/training \
        --n-train 9419 --tag bal9419 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import torch

SPECIES = ("p", "k", "m")


def species_counts(n_train, n_species):
    """Split n_train as evenly as possible; the remainder goes to the last
    species so the totals always add up exactly."""
    base = n_train // n_species
    counts = [base] * n_species
    for i in range(n_train - base * n_species):
        counts[-1 - i] += 1
    return counts


def build_split(sizes, n_train, seed):
    rng = np.random.default_rng(seed)
    per_species = species_counts(n_train, len(sizes))

    train_parts, val_parts, offset = [], [], 0
    for size, n_take in zip(sizes, per_species):
        if n_take > size:
            raise ValueError(f"requested {n_take} training images from a species with only {size}")
        order = rng.permutation(size) + offset
        train_parts.append(order[:n_take])
        val_parts.append(order[n_take:])
        offset += size

    return per_species, np.concatenate(train_parts), np.concatenate(val_parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="all-species image tensor with keys p/k/m")
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--n-train", type=int, required=True, help="total training images")
    parser.add_argument("--tag", required=True, help="data.tag; split is written as split_all_<tag>.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="overwrite an existing split file")
    args = parser.parse_args()

    data = torch.load(args.data, map_location="cpu", weights_only=True)
    sizes = [len(data[key]) for key in SPECIES]

    per_species, train_idx, val_idx = build_split(sizes, args.n_train, args.seed)

    out_path = Path(args.splits_dir) / f"split_all_{args.tag}.npz"
    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} already exists; pass --force to overwrite")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, train_idx=np.sort(train_idx), val_idx=np.sort(val_idx))

    total = sum(sizes)
    print(f"{args.data}\n  available: " + "  ".join(f"{k}={n}" for k, n in zip(SPECIES, sizes)) + f"  total={total}")
    print("  train:     " + "  ".join(f"{k}={n}" for k, n in zip(SPECIES, per_species)) + f"  total={len(train_idx)}")
    print("  val:       " + "  ".join(
        f"{k}={n - t}" for k, n, t in zip(SPECIES, sizes, per_species)) + f"  total={len(val_idx)}")
    print(f"Wrote {out_path} (seed={args.seed})")


if __name__ == "__main__":
    raise SystemExit(main())
