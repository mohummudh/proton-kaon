# Frozen-encoder simulation truth audit

This bounded pilot converts truth-isolated PILArNet particles using the existing
proton-calibrated response, then checks which downstream labels are recoverable
from the frozen eight-dimensional paper VAE (run 0093). It includes photons,
electrons, muons, pions and protons; this dataset contains no kaons.

The numerical findings and limitations are in [RESULTS.md](RESULTS.md).
Large input, image and latent caches stay on the external drive under
`/Volumes/easystore/proton-kaon/pilarnet_lariat/latent_truth/pilot_pixels_v2`.
The same ordered 2,500-particle sample is reused from `pilot_v1`; only the
predictor comparison and visualization have changed.

Run these commands from the repository root with the project environment:

```sh
.venv/bin/python experiments/pilarnet_lariat/latent_truth/prepare.py
.venv/bin/python experiments/pilarnet_lariat/latent_truth/encode.py
.venv/bin/python experiments/pilarnet_lariat/latent_truth/evaluate.py
.venv/bin/python experiments/pilarnet_lariat/latent_truth/counterfactual.py
.venv/bin/python experiments/pilarnet_lariat/latent_truth/tsne.py
.venv/bin/python experiments/pilarnet_lariat/latent_truth/validate.py
.venv/bin/python experiments/pilarnet_lariat/latent_truth/build_report.py --fragment /absolute/writable/thread/path/latent-truth.html
```

`prepare.py` refuses to overwrite an existing pilot. Use a different `--output`
path for a new sample and pass that path to subsequent commands (`--source` for
the report builder). All scripts default to `pilot_pixels_v2`.
`encode.py --reuse-encodings /path/to/pilot_v1`
reuses frozen encodings only after checking identical ordered images/metadata
and unchanged checkpoints/configs. No checkpoint is trained, overwritten or downloaded.
The scripts require the existing profile response, angle bank, verified HDF5
shard, Bethe–Bloch table, real reference caches and control model checkpoints.
`MPLCONFIGDIR` may be set to a writable cache directory.

All particles from a source event share a fixed train/development/test partition.
Scale fits and input PCA use training events; regularization uses development
events. Test-set uncertainty resamples whole events. Readouts include species,
dominant semantic category, energy/momentum, dE/dx and Bethe–Bloch profile
diagnostics, 3D morphology, retained charge, placement nuisances and original
coordinates. The original coordinates are deliberately removed by conversion.

Controls include three VAE seeds, an AE, an untrained network, the earlier
proton-only VAE, **pixel PCA8**, raw pixels and shuffled labels. There are no
engineered image descriptors or truth-derived predictor baselines. Physical
truth quantities are evaluation targets, not extra inputs to the heads.
Counterfactuals regenerate the same particle at ±20% waveform gain and +2° xz
placement; their baselines must reproduce the saved images and latents.

The paired t-SNE views use the same settings on VAE8 and pixel PCA8. Pixel PCA
fits only training-event log1p pixels and retains eight components. t-SNE fits
all sampled particles for visualization only; numerical tests use the full
eight-dimensional representations and unchanged event partitions. The two
t-SNE maps have independent coordinates and do not measure calibrated species
separation or simulation-to-real transfer.

Truth isolation and a proton-only response do not validate full-event
segmentation, kaon decontamination or simulation-to-real task transfer. Incoming
kinematics have an explicit metadata consistency filter; retained-energy truth
is a geometric proxy, not exact waveform ancestry. See the report for support,
selection effects and unsupported tasks.
