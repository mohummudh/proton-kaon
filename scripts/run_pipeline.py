#!/usr/bin/env python3
"""
scripts/run_pipeline.py

Run the full post-training pipeline for one model config in one go:

    1. run_inference.py           — latents/recon for protons (train/val), kaons, muons
    2. compute_features.py        — physics features + histograms + UMAP
    3. analyse_latents.py         — correlation, traversal, logistic, nonlinear, feature AUC
    4. extra/plot_umap_all.py     — publication UMAP + latent scatter plots

All outputs land in per-model folders keyed by the model name (which includes
a _speciesall tag for all-species models), so different models never collide:

    {inference_dir}/{model_name}/          train.npz, val.npz, kaon.npz, muon.npz, reducer.pkl
    figs/{model_name}/features/            feature histograms
    figs/{model_name}/umap/                UMAP plots
    figs/{model_name}/latents-features/    analysis figures
    figs/{model_name}/latents-features-muon/  muon-only analysis

Usage:
    python scripts/run_pipeline.py --config configs/run_0092_all_species_....yaml
    python scripts/run_pipeline.py --config ... --skip-inference --skip-features
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_model_name(cfg: dict) -> str:
    species_tag = "_speciesall" if cfg["data"].get("proton") == "all" else ""
    return (
        f"model_{cfg['model']['type']}"
        f"_latent{cfg['model']['latent']}"
        f"_ch{'_'.join(str(c) for c in cfg['model']['channels'])}"
        f"_beta{cfg['train']['beta']}"
        f"_lr{cfg['optimizer']['lr']}"
        f"_epoch{cfg['train']['epochs']}"
        f"_act{cfg['model']['activation']}"
        f"_kern{cfg['model']['kernel']}"
        f"_stride{cfg['model']['stride']}"
        f"_pad{cfg['model']['padding']}"
        f"_hw{'x'.join(str(d) for d in cfg['model']['input_hw'])}"
        f"_tx{cfg['data'].get('transform', 'none')}{species_tag}"
    )


def run_step(label: str, cmd: list) -> None:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print("$", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\n✗ {label} failed with exit code {result.returncode} — stopping.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Full post-training pipeline for one model.")
    parser.add_argument("--config", required=True, help="Path to model YAML config")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip run_inference.py (latents already computed)")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip compute_features.py (features.pkl already up to date)")
    parser.add_argument("--analyses", nargs="+",
                        choices=["correlation", "traversal", "logistic", "nonlinear", "feature_auc"],
                        default=None,
                        help="Restrict analyse_latents.py to these analyses (default: all)")
    parser.add_argument("--muon-image-path", default=None,
                        help="Override muon image file for inference (default: run_inference.py's default)")
    parser.add_argument("--umap-dims", nargs=2, type=int, default=None, metavar=("ZA", "ZB"),
                        help="Latent dims for plot_umap_all z-scatters (default: 4 7)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = build_model_name(cfg)
    model_path = Path(cfg["output"]["dir"]) / (model_name + ".pt")
    if not model_path.exists():
        sys.exit(f"Model weights not found: {model_path}\nTrain the model first.")

    print(f"Model:  {model_name}")
    print(f"Config: {args.config}")

    py = [sys.executable]

    if not args.skip_inference:
        cmd = py + ["scripts/run_inference.py", "--config", args.config, "--include-muons"]
        if args.muon_image_path:
            cmd += ["--muon-image-path", args.muon_image_path]
        run_step("Step 1/4: inference", cmd)
    else:
        print("\nStep 1/4: inference — skipped")

    if not args.skip_features:
        run_step("Step 2/4: features",
                 py + ["scripts/compute_features.py", "--config", args.config, "--include-muons"])
    else:
        print("\nStep 2/4: features — skipped")

    cmd = py + ["scripts/analyse_latents.py", "--config", args.config, "--include-muons"]
    if args.analyses:
        cmd += ["--analyses"] + args.analyses
    run_step("Step 3/4: latent analysis", cmd)

    cmd = py + ["scripts/extra/plot_umap_all.py", "--config", args.config]
    if args.umap_dims:
        cmd += ["--dims"] + [str(d) for d in args.umap_dims]
    run_step("Step 4/4: UMAP plots", cmd)

    inference_dir = Path(cfg["output"]["inference_dir"]) / model_name
    print(f"\n{'=' * 70}\nPipeline complete. Outputs:")
    print(f"  latents:   {inference_dir}")
    print(f"  figures:   {PROJECT_ROOT / 'figs' / model_name}")


if __name__ == "__main__":
    main()
