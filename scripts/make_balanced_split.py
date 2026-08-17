#!/usr/bin/env python3
"""
scripts/make_balanced_split.py

Build a species-balanced train/val split over the concatenated [p, k, m] tensor
that run_training.py builds in all-species mode (`data.proton: all`).

Indices are written in the concatenated frame (p occupies [0, n_p), k
[n_p, n_p+n_k), m the rest), so run_training.py / run_inference.py can consume
the file directly.

TWO MODES

  --n-train N            Balanced training set of N images total, split as evenly
                         as possible across the three species; *every* remaining
                         image goes to validation. Validation is therefore both
                         larger than training and unbalanced, because the species
                         have different totals.

  --pool-per-species N   Take N images from each species and split that pool at
    --train-frac F       --train-frac. Images beyond N are held out of both sets.
                         Train and val are then each exactly balanced, which is
                         what makes a sweep over F interpretable: the species
                         mixture is identical at every split ratio, so a change in
                         the result is a change in training statistics and nothing
                         else. N is bounded by the smallest species (8227 kaons).

Both modes draw one permutation per species from --seed, and the pool is taken
from the front of it. Training sets across different --train-frac values are
therefore NESTED: the 90% training set contains the 80% one. That is deliberate —
it means a sweep varies how much data the model sees without also varying which
data, so two runs differ by sample size alone.

Usage:
    # single balanced training set, everything else to validation
    python scripts/make_balanced_split.py \
        --data /Volumes/easystore/proton-kaon/images/pkm_48x48_raw_10-179wires.pt \
        --splits-dir /Volumes/easystore/proton-kaon/training \
        --n-train 9419 --tag bal9419 --seed 42

    # one rung of the split sweep: 8227 per species, 90/10
    python scripts/make_balanced_split.py \
        --data /Volumes/easystore/proton-kaon/images/pkm_48x48_raw_10-179wires.pt \
        --splits-dir /Volumes/easystore/proton-kaon/training \
        --pool-per-species 8227 --train-frac 0.9 --tag pool8227_tr90 --seed 42
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


def build_pooled_split(sizes, pool, train_frac, seed):
    """Take `pool` images per species, split each at `train_frac`, discard the rest.

    One permutation per species, seeded identically regardless of train_frac, and
    the pool and the training set are both taken from its front. Two consequences
    worth stating because a sweep depends on them:

      * the pool is the same set of events at every train_frac, so the sweep never
        compares a model to one trained on different images;
      * training sets nest, so going from 50/50 to 90/10 only ever adds events.

    Returns (per_species_train, per_species_val, train_idx, val_idx), with the
    index arrays in the concatenated [p, k, m] frame.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"--train-frac must be strictly between 0 and 1; got {train_frac}")
    for key, size in zip(SPECIES, sizes):
        if pool > size:
            raise ValueError(
                f"--pool-per-species {pool} exceeds the {size} images available for "
                f"species {key!r}; the smallest species sets the ceiling")

    rng = np.random.default_rng(seed)
    n_tr = int(round(pool * train_frac))
    if n_tr in (0, pool):
        raise ValueError(f"--train-frac {train_frac} leaves one side of the split empty "
                         f"at --pool-per-species {pool}")

    train_parts, val_parts, offset = [], [], 0
    for size in sizes:
        order = rng.permutation(size) + offset
        train_parts.append(order[:n_tr])
        val_parts.append(order[n_tr:pool])
        offset += size

    n_species = len(sizes)
    return ([n_tr] * n_species, [pool - n_tr] * n_species,
            np.concatenate(train_parts), np.concatenate(val_parts))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="all-species image tensor with keys p/k/m")
    parser.add_argument("--splits-dir", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--n-train", type=int,
                     help="total training images; all remaining images go to validation")
    mode.add_argument("--pool-per-species", type=int,
                     help="images taken from each species; the pool is split at "
                          "--train-frac and anything beyond it is held out of both sets")
    parser.add_argument("--train-frac", type=float,
                        help="fraction of the pool used for training "
                             "(--pool-per-species mode only), e.g. 0.9 for a 90/10 split")
    parser.add_argument("--tag", required=True, help="data.tag; split is written as split_all_<tag>.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="overwrite an existing split file")
    args = parser.parse_args()

    if args.pool_per_species is not None and args.train_frac is None:
        parser.error("--pool-per-species requires --train-frac")
    if args.n_train is not None and args.train_frac is not None:
        parser.error("--train-frac applies to --pool-per-species mode, not --n-train")

    data = torch.load(args.data, map_location="cpu", weights_only=True)
    sizes = [len(data[key]) for key in SPECIES]

    if args.pool_per_species is not None:
        per_train, per_val, train_idx, val_idx = build_pooled_split(
            sizes, args.pool_per_species, args.train_frac, args.seed)
    else:
        per_train, train_idx, val_idx = build_split(sizes, args.n_train, args.seed)
        per_val = [n - t for n, t in zip(sizes, per_train)]

    out_path = Path(args.splits_dir) / f"split_all_{args.tag}.npz"
    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} already exists; pass --force to overwrite")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, train_idx=np.sort(train_idx), val_idx=np.sort(val_idx))

    def row(label, counts, total):
        return f"  {label:<11}" + "  ".join(
            f"{k}={n}" for k, n in zip(SPECIES, counts)) + f"  total={total}"

    total = sum(sizes)
    print(f"{args.data}\n" + row("available:", sizes, total))
    print(row("train:", per_train, len(train_idx)))
    print(row("val:", per_val, len(val_idx)))
    held_out = total - len(train_idx) - len(val_idx)
    if held_out:
        print(row("held out:", [s - t - v for s, t, v in zip(sizes, per_train, per_val)],
                  held_out))
    frac = len(train_idx) / (len(train_idx) + len(val_idx))
    print(f"  split:     {frac:.1%} train / {1 - frac:.1%} val")
    print(f"Wrote {out_path} (seed={args.seed})")


if __name__ == "__main__":
    raise SystemExit(main())
