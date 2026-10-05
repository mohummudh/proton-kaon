# Frozen-encoder simulation truth audit

This bounded pilot converts truth-isolated PILArNet particles using the existing
proton-calibrated response, then checks which downstream labels are recoverable
from the frozen eight-dimensional paper VAE (run 0093). It includes photons,
electrons, muons, pions and protons; this dataset contains no kaons.

The numerical findings and limitations are in [RESULTS.md](RESULTS.md).
Large input, image and latent caches stay on the external drive under
`/Volumes/easystore/proton-kaon/pilarnet_lariat/latent_truth/pilot_v1`.

Run these commands from the repository root with the project environment:

```sh
.venv/bin/python experiments/pilarnet_lariat/latent_truth/prepare.py
.venv/bin/python experiments/pilarnet_lariat/latent_truth/encode.py
.venv/bin/python experiments/pilarnet_lariat/latent_truth/evaluate.py
.venv/bin/python experiments/pilarnet_lariat/latent_truth/counterfactual.py
.venv/bin/python experiments/pilarnet_lariat/latent_truth/validate.py
.venv/bin/python experiments/pilarnet_lariat/latent_truth/build_report.py --fragment /absolute/writable/thread/path/latent-truth.html
```

`prepare.py` refuses to overwrite an existing pilot. Use a different `--output`
path for a new sample and pass that path to subsequent commands (`--source` for
the report builder). No checkpoint is trained, overwritten or downloaded.
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
proton-only VAE, image PCA, image summaries, raw pixels and shuffled labels.
Counterfactuals regenerate the same particle at ±20% waveform gain and +2° xz
placement; their baselines must reproduce the saved images and latents.

Truth isolation and a proton-only response do not validate full-event
segmentation, kaon decontamination or simulation-to-real task transfer. Incoming
kinematics have an explicit metadata consistency filter; retained-energy truth
is a geometric proxy, not exact waveform ancestry. See the report for support,
selection effects and unsupported tasks.
