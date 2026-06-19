# Proton–Kaon VAE: A Build-It-From-Scratch Methodology Manual

> **What this document is.** A complete, self-contained engineering and scientific
> manual for the `proton-kaon` project. It explains *what* every stage does, *why*
> it was designed that way, *how* the data flows, the *exact* parameters and magic
> numbers (and where they come from), the data structures at every boundary, the
> output artifacts, and the known pitfalls. It is written so that a competent
> physicist-programmer could rebuild the entire method from a blank directory with
> only this document and the raw ROOT files.
>
> **Scope.** This manual documents the code as it actually exists in the repository
> (verified against source, not from memory). Where the code, the `README.md`, and
> the on-disk artifacts disagree, those discrepancies are called out explicitly —
> they matter for reproducibility and for the paper.
>
> **Audience.** Someone who knows Python, PyTorch, and basic particle-physics
> calorimetry, but knows nothing about this repository.

---

## Table of Contents

1. [Scientific premise and the central question](#1-scientific-premise)
2. [The detector and the raw data](#2-detector-and-raw-data)
3. [End-to-end architecture (the pipeline DAG)](#3-architecture)
4. [Repository layout — every file](#4-repository-layout)
5. [Environment, dependencies, storage model](#5-environment)
6. [Stage 1 — ROOT ingestion (`open_root`, `Event`)](#6-stage1-ingestion)
7. [Stage 2 — Clustering (track finding)](#7-stage2-clustering)
8. [Stage 3 — Quality & fiducial cuts](#8-stage3-cuts)
9. [Stage 4 — Cross-plane cluster matching](#9-stage4-matching)
10. [Stage 5 — Beamline + reco merge + Bethe–Bloch χ² PID](#10-stage5-chi2)
11. [Stage 6 — Image preprocessing](#11-stage6-images)
12. [Stage 7 — The β-VAE model](#12-stage7-model)
13. [Stage 8 — The loss function](#13-stage8-loss)
14. [Stage 9 — Training](#14-stage9-training)
15. [Stage 10 — Hyperparameter sweeps](#15-stage10-sweeps)
16. [Stage 11 — Inference (latents & reconstructions)](#16-stage11-inference)
17. [Stage 12 — Physics feature engineering (25 features)](#17-stage12-features)
18. [Stage 13 — Latent-space analysis (5 analyses)](#18-stage13-analysis)
19. [Auxiliary particle samples: muons & CSDA-kaons](#19-aux-samples)
20. [Physics validation (GMM / MVN / CSDA-range)](#20-physics-validation)
21. [The data-alignment contract & integrity pitfalls](#21-alignment)
22. [Output artifact catalog & naming grammar](#22-artifacts)
23. [End-to-end runbook (exact commands)](#23-runbook)
24. [Appendix A — Feature formula reference](#appendix-a)
25. [Appendix B — Config schema](#appendix-b)
26. [Appendix C — Glossary](#appendix-c)

---

<a name="1-scientific-premise"></a>
## 1. Scientific premise and the central question

Protons and kaons traversing liquid argon leave distinct ionisation signatures.
The dominant discriminator is the **Bragg peak**: as a charged particle slows and
stops, its specific energy loss `dE/dx` rises sharply near the track end. The shape
and position of that rise depend on the particle's mass (via the **Bethe–Bloch**
relation), so at a fixed beam momentum a proton (≈938 MeV/c²) and a kaon
(≈494 MeV/c²) produce measurably different end-of-track profiles. Kaons can also
**decay in flight** (e.g. K→μν, K→ππ), producing kinks and secondary tracks that
protons never show.

The traditional PID method is a **χ² fit** of the measured `dE/dx`-vs-residual-range
curve against the expected Bethe–Bloch curve for each hypothesis. This project keeps
that χ² as a baseline, but asks a different, **unsupervised** question:

> If we train a generative model (a convolutional **β-VAE**) to *reconstruct*
> calorimetry images **using only protons and no particle labels**, does its latent
> space spontaneously organise around physically meaningful quantities — and do
> kaons (which the model has never seen) fall into a separable region of that space?

This is an **anomaly/novelty-detection framing**: the VAE learns "what a proton
looks like", and kaons are evaluated as out-of-distribution. The learned latents are
then *probed* against handcrafted physics features to interpret what the network
encoded, and against external truth (beamline mass, CSDA range) to validate that the
separation is physical rather than an artifact.

**Two design decisions that look like bugs but are intentional** (confirmed with the
project owner):

- **Features come from the full track; the VAE sees only the last 50 wires.** The
  feature code computes quantities over the *entire* cluster, while the network is
  fed only the track endpoint (the Bragg region). If a latent still predicts a
  global feature like total charge, that is a *stronger* scientific result: it means
  the endpoint alone is informative about the whole track.
- **Track direction is unambiguous.** LArIAT is a test-beam experiment: every
  particle enters from the upstream face and travels in a known direction along the
  beamline. Wire-row order therefore always runs entry → stop. Bragg-position
  features (`max_ADC_position`, `end_vs_start_ratio`, `monotonic_rise_fraction`, …)
  are physically well-defined with no left/right ambiguity.

---

<a name="2-detector-and-raw-data"></a>
## 2. The detector and the raw data

**LArIAT** (Liquid Argon In A Testbeam) is a liquid-argon time projection chamber
(LArTPC). Ionisation electrons drift to two sensing **wire planes**:

- **Induction plane** — wires see a bipolar induced signal as charge drifts past.
- **Collection plane** — wires collect the charge; the cleaner calorimetric signal.

Each plane has **240 wires** (`WIRES_PER_PLANE = 240`, so `CHANNELS_PER_EVENT = 480`).
A single event is a 2-D image per plane: **wire number (space, along the beam) ×
time tick (drift time)**. ADC counts are the pixel intensities.

### 2.1 Input files (all on an external drive, *not* in git)

The pipeline reads from `/Volumes/easystore/…`. Nothing under that path is version
controlled; only code and a handful of paper figures are. The canonical inputs:

| File | Contents | Tree name | ADC layout |
|---|---|---|---|
| `…/raw/p_1track_protons_600_1600.root` | "picky" protons, momentum 600–1600, reco-required 1 track | `ana/raw` | channel-mapped |
| `…/raw/rawExtracted_350_650.root` | "picky" kaon candidates, momentum 350–650 | `ana/raw;352` | flat |
| `…/raw/Muons_50_300/RAW_muons.root` | dedicated muon sample (momentum 50–300) | `anatree/raw` | flat |
| `…/protons.txt`, `…/kaons.txt` | Bethe–Bloch tables (residual-range, expected dE/dx) | — | 2-column text |
| `…/proton-deuteron/csv/picky+match.csv` | beamline metadata per event: `p`, `m`, `beamline_mass` | — | CSV |
| `…/bruno/root/primary_trk_dedx_rr.csv` | LArIAT reco: `trkrr`, `trkdedx` (`;`-joined arrays) | — | CSV |
| `…/proton-kaon/docs/kaon_df_plane_1_thr_DAQ.csv` | confirmed-kaon event list + log-likelihoods | — | CSV |
| `…/proton-kaon/docs/proton_df_plane_1_thr_DAQ.csv` | proton log-likelihoods | — | CSV |

> The trailing `;352` on the kaon tree is a **ROOT cycle number**. ROOT keeps
> multiple versions of an object under the same key; `ana/raw;352` selects cycle 352.
> If you regenerate the kaon file you will almost certainly need a different cycle —
> do not hard-code 352 blindly.

### 2.2 The two raw ADC layouts (why `Event` has two loaders)

The proton and kaon ROOT files were produced by different LArSoft jobs and store the
raw ADC differently. `src/event.py` detects which by inspecting branch names:

- **Channel-mapped** (proton file): branches `raw_rawadc` (flat ADC payload) and
  `raw_channel` (per-channel id). The payload length must be divisible by the number
  of channels; it is reshaped to `(n_channels, n_ticks)`. Channels with id `< 240`
  are induction; `id − 240` indexes collection. Missing wires stay zero. This is the
  sparse, channel-addressed format.
- **Flat** (kaon and muon files): branch `rawadc1`, a dense `480 × n_ticks` block.
  First 240 rows = induction, next 240 = collection. No channel map.

Both loaders end with `self.collection` and `self.induction` as dense
`(240, n_ticks)` `float` matrices, **rows = wires, columns = time ticks**. Everything
downstream assumes this orientation.

> **Pedestal subtraction (muons).** `RAW_muons.root` is pedestal-subtracted, so its
> ADC values are centred near 0 and can be negative. The muon extraction script
> clips `image_intensity` to `[0, ∞)` *before* image-making, otherwise the later
> `log1p` transform produces NaNs.

---

<a name="3-architecture"></a>
## 3. End-to-end architecture (the pipeline DAG)

```
                ┌─────────────────────────────────────────────────────────┐
  ROOT files →  │ dataset.py                                               │
  (p, k)        │  open_root → extract_clusters → cluster_cuts → matching  │
                │  → merge beamline(p,m,mass) → merge reco(trkrr,trkdedx)  │
                │  → Bethe-Bloch χ² (kaon/proton hypothesis)               │
                └───────────────┬─────────────────────────────────────────┘
                                │  col.pkl, ind.pkl   (matched cluster DataFrames)
              ┌─────────────────┴──────────────┬───────────────────────────┐
              ▼                                 ▼                           ▼
   ┌───────────────────────┐      ┌──────────────────────────┐   ┌──────────────────┐
   │ image_making.py        │      │ compute_features.py       │   │ (χ² + log-L are  │
   │ cut→pad→downsample→     │      │ 25 features per cluster   │   │  carried in the  │
   │ stack→[transform]       │      │ (collection plane only)   │   │  feature table)  │
   │ → pk_*.pt {"p","k"}     │      │ → features.pkl            │   └──────────────────┘
   └───────────┬────────────┘      └─────────────┬─────────────┘
               │ 48×48 2-ch tensors               │
               ▼                                  │
   ┌───────────────────────┐                      │
   │ run_training.py        │                      │
   │ β-VAE on PROTONS ONLY  │   (run_sweep.py drives many of these locally/remote)
   │ → model_*.pt + logs    │                      │
   └───────────┬────────────┘                      │
               │ weights                            │
               ▼                                    │
   ┌───────────────────────┐                       │
   │ run_inference.py       │   encode μ for        │
   │ train/val/kaon (+muon, │   each subset         │
   │ +csda) → *.npz         │                       │
   └───────────┬────────────┘                       │
               │ latents/recon/RE                    │
               ▼                                     ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ analyse_latents.py  (correlation, traversal, logistic,        │
   │                      nonlinear, feature_auc)                   │
   │ + notebooks: separation(GMM), mvn_classification, csda-length │
   │ → figs/<model>/latents-features/*.png                          │
   └──────────────────────────────────────────────────────────────┘
```

Auxiliary branches (muons, CSDA-kaons) feed the same `inference → analysis`
machinery and are described in §19.

### 3.1 Compute environments

The repo targets three places:

- **Laptop / Apple Silicon** — device auto-selects `cuda → mps → cpu`. Image-making
  and analysis run here against the external drive.
- **`cdt`** — a GPU box reached over SSH; `run_sweep.py` (default remote mode)
  rsyncs data + splits up, runs training per config, pulls models back, optionally
  deletes the remote copy. Multi-GPU concurrency via `CUDA_VISIBLE_DEVICES`.
- **CSF** (Manchester SLURM cluster) — `jobs/sweep.sh` is an `sbatch` script
  (`gpuA40GB`) that runs `run_sweep.py --local` with `configs/csf.yaml` path
  overrides.

---

<a name="4-repository-layout"></a>
## 4. Repository layout — every file

```
src/                         importable library (`src` namespace package)
├── open_root.py             ROOT → DataFrame of (file_path, event_index, run, subrun, event)
├── event.py                 Event class: load raw ADC, build plane matrices, 4 clustering algos
├── clustering.py            extract_clusters(): bulk per-event cluster extraction → DataFrame
├── cuts.py                  cluster_cuts / image_cuts / reco_track_cuts
├── matching.py              greedy 1-to-1 induction↔collection pairing
├── chi2.py                  parse reco arrays, Bethe-Bloch χ² scoring, particle_hypothesis
├── bethe_bloch.py           bb_file(): load (resrange, dedx) table
├── transforms.py            apply_transform(): 12 ADC preprocessing transforms
├── images.py                cut_start / pad_image / pad_image_batch_gpu / downsample_image
├── event_display.py         is_valid_cluster(), plot_event_displays() (debug viz)
├── features/
│   ├── calorimetry.py       19 energy/Bragg-profile features
│   ├── topology.py          6 geometric/morphological features
│   └── plot.py              hist() and plot_umap() helpers
├── models/configVAE.py      configurable convolutional VAE  ← the live model
├── losses/vae.py            weighted MSE + KL, vae_loss()
├── train/
│   ├── train.py             training loop with early stopping  ← the live loop
│   ├── logger.py            save_run_log() → JSON
│   ├── plot.py              plot_training() loss curves
│   └── vae.py               ⚠ DEAD legacy standalone script (imports src.models.vae, which no longer exists)
└── inference/
    ├── inference.py         inference(): batched encode → (latents μ, recon, RE)
    └── plot.py              empty

scripts/                     entry points (run with `uv run python scripts/<x>.py`)
├── dataset.py               Stage 1–5: ROOT → col.pkl/ind.pkl  (hard-coded paths, no argparse)
├── image_making.py          Stage 6: clusters → pk_*.pt  (also `--muon` mode)
├── run_training.py          Stage 9: YAML config + CLI overrides → model_*.pt
├── run_sweep.py             Stage 10: grid sweep, remote (SSH/rsync) or `--local`
├── run_inference.py         Stage 11: model → train/val/kaon(+muon/+csda) .npz
├── compute_features.py      Stage 12: clusters → features.pkl + histograms + UMAP
├── analyse_latents.py       Stage 13: 5 latent analyses → figs
├── extract_csv_kaons.py     CSDA-kaon extraction from ROOT by event list (no cuts)
└── extra/                   auxiliary tools (see §19, §20)
    ├── image_making_muons.py      OLD muon proxy: ≥180-wire tracks from the proton ROOT
    ├── image_making_muons_art.py  NEW muon sample: from dedicated RAW_muons.root
    ├── csda_kaon_cleaning.py      RF-based 1-cluster-per-event selection for CSDA-kaons
    ├── csda_kaon_labeler.py       Bokeh GUI for manual cluster labeling
    ├── inspect_solidity.py        visual diagnostic for the solidity feature
    ├── inspect_n_local_maxima.py  visual diagnostic for n_local_maxima
    ├── plot_kaon_vertex_distribution.py  derives the cluster_cuts fiducial windows
    ├── plot_latents.py / plot_umap_all.py  latent/UMAP visualisations of all species
    ├── umap_explorer.py           interactive Bokeh UMAP↔image explorer
    ├── save_kaon_displays.py      save event displays (scan or CSV mode)
    ├── muons.sh / csda-kaons.sh   orchestration command sequences

configs/                     YAML configs (see §Appendix B)
├── default.yaml             base config (⚠ stale: input_hw 256, latent 4, beta 10)
├── run_0066_*.yaml          THE PAPER MODEL (latent8, relu, beta0.5, 48×48, log1p)
├── run_0006_*.yaml, run_0091_*.yaml   other sweep winners kept as named configs
├── sweep.yaml               192-run architecture grid
├── sweep_transforms.yaml    6-run transform grid
├── remote.yaml(.example)    SSH/rsync profile for `cdt` (git-ignored)
└── csf.yaml                 path overrides for the SLURM cluster

jobs/sweep.sh                SLURM batch script
notebooks/                   exploratory mirrors + physics validation (git-ignored)
figs/                        outputs (mostly git-ignored; a few paper figures kept)
logs/                        run logs + training-history JSON (git-ignored)
docs/                        this manual
```

> **Dead code to ignore.** `src/train/vae.py` is an old standalone trainer that does
> `from src.models.vae import VAE` and `from src.losses.vae import vae_loss`. The
> model module was later renamed to `src/models/configVAE.py`, so this import now
> fails — the file is non-functional legacy. The `*.egg-info/SOURCES.txt` likewise
> still lists `src/models/vae.py`; the egg-info is stale. The live model is
> `src/models/configVAE.py`; the live loop is `src/train/train.py`.

---

<a name="5-environment"></a>
## 5. Environment, dependencies, storage model

- **Python ≥ 3.11.9**, managed with **`uv`** (there is a `uv.lock`). The project is
  an editable install (`pip install -e .` / `uv sync`); `src` becomes importable as
  a namespace package (no `__init__.py` needed in subpackages because scripts also
  prepend the project root to `sys.path`).
- **Key dependencies** (`pyproject.toml`): `torch`, `numpy`, `pandas`,
  `scikit-learn`, `scikit-image`, `scipy`, `matplotlib`, `seaborn`, `umap-learn`,
  `uproot`, `awkward`, `bokeh` (for the interactive labelers), `nflows` (declared but
  **unused** — normalising-flow PID was prototyped then dropped), `pyyaml`, `tqdm`.
- **Device policy.** Almost every script chooses
  `cuda → mps → cpu`. Inference uses a small batch size (8) and calls
  `torch.mps.empty_cache()` between batches to survive on Apple GPUs.
- **Storage.** All datasets, models, latents and features live under
  `/Volumes/easystore/proton-kaon/` (external drive). Git tracks only source, a few
  paper figures, and configs. Re-pathing for a new machine means editing the
  hard-coded constants in `dataset.py`, `image_making.py`, `compute_features.py`,
  `run_inference.py` defaults, and the `output.*` blocks of the YAML configs (or
  supplying an overrides YAML like `csf.yaml`).

Directory convention on the data drive:

```
/Volumes/easystore/proton-kaon/
├── raw/         input ROOT files
├── clusters/    col.pkl, ind.pkl, muon_*.pkl, csv_kaon_*.pkl
├── images/      pk_*.pt, muon_*.pt, csv_kaon_*.pt
├── models/      model_*.pt, *_curves.png, sweep_configs/
├── training/    split_<protonkey>.npz  (train/val index split)
├── inference/   <model_name>/{train,val,kaon,muon,csda_kaon}.npz, reducer.pkl
├── features/    features.pkl
└── docs/        external CSVs (log-likelihoods, selected-kaon lists)
```

---

<a name="6-stage1-ingestion"></a>
## 6. Stage 1 — ROOT ingestion (`open_root`, `Event`)

### 6.1 `src/open_root.py`

`open_root(file_path, tree_name=None)` returns a DataFrame with one row per event:
`file_path, event_index, run, subrun, event`. If `tree_name` is omitted it auto-finds
the first `TTree` that contains `{run, subrun, event}`. `_select_tree` is reused by
the cluster extractor so the tree is opened once.

`(run, subrun, event)` is the **universal event key** used to join everything
(beamline metadata, reco arrays, log-likelihoods) throughout the pipeline.

### 6.2 `src/event.py` — the `Event` class

```python
Event(filepath=None, tree=None, index=0, threshold=1, plot=True)
```

- Requires an already-open `tree`; `index` selects the event row.
- `load()` dispatches to `_load_channel_mapped_event` or `_load_flat_event` based on
  branch names (§2.2) and fills `self.collection`, `self.induction`.
- Clustering algorithms (all operate on one plane matrix):
  - **`connectedregions(matrix, threshold)`** — the production algorithm. Binary
    mask `matrix > threshold`, `skimage.measure.label` (8-connectivity by default),
    `regionprops(..., intensity_image=matrix)`. Returns `(labeled, regions)`.
  - `longestcluster` — keep the single largest-area region.
  - `max_adc_ratio` — keep the region with the largest max/min ADC ratio.
  - `search_from_max_adc` — BFS flood-fill from the global ADC maximum, with an
    auto-threshold = `max_adc / 6` if none supplied.
  - `direction`, `master` — stubs (not implemented).
- Plotting helpers (`plot`, `visualiseclusters`, `plotconnectedregions`) are for
  notebooks; `plt` y-axes are inverted so time runs upward.

Only `connectedregions` is used in production extraction. The others exist for
exploratory comparison of clustering strategies.

---

<a name="7-stage2-clustering"></a>
## 7. Stage 2 — Clustering (track finding)

### 7.1 `src/clustering.py :: extract_clusters`

```python
extract_clusters(events_df, particle_type, threshold=15, max_events=None, tree_name=None)
```

For each event it builds an `Event`, runs `connectedregions` on **both** planes, and
emits one record per connected region. Crucially:

- **Collection threshold = `threshold` (15). Induction threshold = `threshold // 2`
  (7).** The induction signal is weaker/bipolar, so it gets a lower bar. (The debug
  display in `event_display.py` uses 15/7 to match.)
- **No height/width filtering happens here** — every connected region above
  threshold is kept. Filtering is a later, separate stage (this is why muons can be
  recovered later: the raw extraction never discarded them).
- Exceptions per event are caught and skipped (robustness over completeness).

Each cluster record (a future DataFrame row):

| Field | Meaning |
|---|---|
| `event_idx` | running index in the events DataFrame |
| `run, subrun, event` | event key |
| `file_path, event_index` | provenance back to ROOT |
| `particle_type` | the label passed in (`proton`/`kaon`/`muon`) — provenance, *not* truth |
| `plane` | `collection` or `induction` |
| `cluster_idx` | index of this region within its plane for this event |
| `bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col` | region bounding box |
| `width` | `bbox_max_col − bbox_min_col` = **time-tick span** |
| `height` | `bbox_max_row − bbox_min_row` = **wire span** (≈ track length) |
| `image_intensity` | 2-D ADC crop of the bbox, shape `(height, width)` = `(wires, ticks)` |
| `column_maxes` | `image_intensity.max(axis=1)` — see below |

### 7.2 The single most important derived array: `column_maxes`

```python
column_maxes = region.image_intensity.max(axis=1)   # length == height == n_wires
```

Despite the name, this is the **per-wire maximum ADC**: for each wire (row) it is the
peak ADC across all time ticks. Because wires run along the beam direction, this is
the **longitudinal charge profile of the track** — a `dE/dx`-like trace from entry
(index 0) to stop (last index). **Every "profile" feature in §17 is computed from
`column_maxes`, and the Bragg peak lives near its high-index end.**

Keep the orientation straight:

```
image_intensity:  rows = wires (space, beam direction),  cols = time ticks (drift)
height  = number of wires the track spans  ≈ track length
width   = number of time ticks             ≈ track transverse extent in drift
column_maxes[i] = peak ADC on wire i = charge deposited as the track crosses wire i
```

### 7.3 Real counts (from `logs/dataset.log`, 2026-05-04)

```
proton events: 16,036   → 366,533 raw clusters
kaon   events: 20,035   → 1,725,928 raw clusters
```

(The kaon file has far more clusters per event — noise + secondaries — which is why
cuts and matching are aggressive.)

---

<a name="8-stage3-cuts"></a>
## 8. Stage 3 — Quality & fiducial cuts (`src/cuts.py`)

Three cut functions, used at different stages.

### 8.1 `cluster_cuts(clusters_df, lower=1, upper=179)` — used in `dataset.py` with `lower=10`

Applied right after extraction. Filters:

1. **Length window:** `lower < height < upper` → with `lower=10, upper=179`,
   keeps `10 < height < 179` wires. The **upper bound of 179 is a hard ceiling**:
   anything ≥180 wires (long, punch-through, muon-like) is discarded here. This is
   precisely why muons must be recovered from a separate extraction (§19).
2. **Non-degenerate profile:** `len(set(column_maxes)) > 1` — drop flat/constant
   columns (dead or saturated).
3. **Fiducial / beam-spot window** on the *vertex*:
   `vertex_x = bbox_min_col + argmax(image_intensity[0])` (the time tick of the ADC
   peak in the **first wire** of the cluster — i.e. where the track enters). Plus a
   wire-entry window on `bbox_min_row`:

   | plane | `bbox_min_row` window | `vertex_x` window |
   |---|---|---|
   | collection | `12 < min_row < 37` | `789 < vertex_x < 1927` |
   | induction  | `11 < min_row < 35` | `786 < vertex_x < 1794` |

   These windows isolate tracks that **enter through the beam spot** and reject
   cosmic/edge activity. **Where the numbers come from:** they were read off the
   2-D histograms produced by `scripts/extra/plot_kaon_vertex_distribution.py`,
   which extracts *all* clusters with no cuts and plots the `(min_row, vertex_x)`
   distribution to reveal the beam acceptance region. If you regenerate the data or
   change the detector, **re-derive these windows from that script** — they are not
   physical constants.

Result row count (dataset.log): protons 21,708; kaons 24,382 after cuts.

> **`event_display.py :: is_valid_cluster` is an approximation, not the truth.** It
> replays the cuts for the debug display but uses `bbox_max_col` (`maxc`) for the
> time window instead of the true `vertex_x`. Use it only for visual sanity-checking,
> never as the authoritative filter.

### 8.2 `image_cuts(col, ind, lower=1, upper=179, width=1500)` — used in `image_making.py` & `compute_features.py`

Applied to the **matched** `col`/`ind` tables before turning them into images or
features. It removes any row failing `lower < height < upper` **and** `width < width`,
but does so on the **union of failing indices across both planes**:

```python
removed = removed_in_col ∪ removed_in_ind
col_cut = col.drop(removed);  ind_cut = ind.drop(removed)
```

This guarantees `col_cut` and `ind_cut` stay **row-aligned and equal length** — a
hard requirement, because the two planes are later stacked positionally into a
2-channel image. (See §21 for why this positional contract is the project's biggest
validity risk.)

For the proton/kaon image set the call is `image_cuts(col, ind, lower=10)`. For muons
it is `image_cuts(..., lower=175, upper=10_000_000, width=473)` — long tracks,
narrow in drift.

### 8.3 `reco_track_cuts(trk)`

Cleans the external reco CSV before the χ²: parses the `;`-joined `trkrr`
(residual range) and `trkdedx` arrays via `chi2.parse_array`, drops `dedx > 100`
spikes (`filter_arrays`), requires non-empty and equal-length `rr`/`dedx`.

---

<a name="9-stage4-matching"></a>
## 9. Stage 4 — Cross-plane cluster matching (`src/matching.py`)

A real track appears as one cluster in *each* plane; we must pair them. `matching()`
calls `pair_clusters()` then re-joins full cluster data.

`pair_clusters(df, height_weight=1, row_weight=1, col_weight=1)`:

1. Split into induction and collection sub-tables (by `plane` substring).
2. **Cartesian product within each `(run, subrun, event)`** — every induction
   cluster paired with every collection cluster in that event.
3. **Match score** (lower = better), a weighted sum of squared bbox deltas:

   ```
   score =  col_weight · ( (Δmin_col)² + (Δmax_col)² )     # time-tick agreement
          + row_weight · ( (Δmin_row)² + (Δmax_row)² )     # wire agreement
          + height_weight · (Δheight)²                     # length agreement
   ```
   where Δ = collection − induction for each bbox coordinate. All weights default 1.
4. **Greedy 1-to-1 assignment** per event: sort pairs by score ascending; walk the
   list, accept a pair only if neither its induction nor collection cluster is
   already used. This yields at most `min(n_ind, n_col)` matches per event.

`matching()` returns `(collection_df, induction_df)`: the surviving pairs joined back
to the full per-plane cluster records (so each carries its own `image_intensity`,
`column_maxes`, bbox, etc.). The two returned frames are **row-aligned**: row *i* of
`collection_df` is the same physical track as row *i* of `induction_df`.

Matched counts (dataset.log): **10,467 proton** and **8,383 kaon** tracks.

---

<a name="10-stage5-chi2"></a>
## 10. Stage 5 — Beamline + reco merge + Bethe–Bloch χ² PID (`scripts/dataset.py`, `src/chi2.py`, `src/bethe_bloch.py`)

`dataset.py` is a **linear, hard-coded, no-argparse "run-once" script**. After
matching it:

1. **Concatenates** kaons then protons: `col = concat([k_col, p_col])`,
   `ind = concat([k_ind, p_ind])`. **Order matters** — kaon rows precede proton rows
   in `col.pkl`/`ind.pkl`. (Downstream filtering by `particle_type` recovers each
   species in its original relative order.)
2. **Left-merges beamline metadata** from `picky+match.csv` on the event key:
   `p` (momentum flag/value), `m` (mass), `beamline_mass` (the spectrometer-measured
   mass — the closest thing to truth, see §20).
3. **Inner-merges reco arrays** `trkrr`, `trkdedx` from the reco CSV (after
   `reco_track_cuts`). *Inner* means tracks without reconstructed `dE/dx` are
   dropped — this trims the candidate list to **18,752 total** (col.pkl row count).
4. **Bethe–Bloch χ²** via `bb_file()` + `do_chi_squared()`.

### 10.1 The χ² (`src/chi2.py :: chi2_track_alek`)

For a track with residual-range/`dE/dx` samples `(rᵢ, dᵢ)` and an expected curve
`bb[·]` indexed by residual range:

```
for each hit i:
    ro   = round( round(rᵢ / 0.05) · 0.05 , 2)     # snap residual range to a 0.05 grid
    x_l  = index of ro in the expected-curve table
    χ²  += (dᵢ − bb[x_l])² / bb[x_l]                # Pearson-style, normalised by expectation
return χ² / N                                       # reduced χ² (per hit)
```

`build_res_index()` precomputes the `ro → index` lookup (rounded to 2 dp) so the
inner loop is O(1). The denominator is the **expected** `dE/dx` (no ε clamp), and the
normalisation is by the **number of hits** — this exactly reproduces a collaborator's
("Alek's") reference implementation; the function name encodes that compatibility.

`do_chi_squared()` computes `chi_squared_kaon` and `chi_squared_proton` for every
track and sets:

```
particle_hypothesis = 0 (kaon)  if  chi_squared_kaon < chi_squared_proton
                    = 1 (proton) otherwise
```

`bb_file(PATH)` loads a 2-column table (residual range, expected `dE/dx`) and returns
`(bb, res)` with `res = flip(resrange)`.

### 10.2 Output of Stage 1–5

```
/Volumes/easystore/proton-kaon/clusters/col.pkl   # collection-plane matched tracks
/Volumes/easystore/proton-kaon/clusters/ind.pkl   # induction-plane, row-aligned to col.pkl
```

Each row carries: event key, provenance, `particle_type` (proton/kaon by source
file), bbox + `height`/`width`, `image_intensity`, `column_maxes`, beamline `p/m/
beamline_mass`, reco `trkrr/trkdedx`, `chi_squared_kaon/proton`, `particle_hypothesis`.
**These two pickles are the hinge of the project**: both image-making and
feature-engineering start from them.

---

<a name="11-stage6-images"></a>
## 11. Stage 6 — Image preprocessing (`scripts/image_making.py`, `src/images.py`)

Goal: turn each matched track into a fixed-size **(2, H, W)** tensor — channel 0 =
collection, channel 1 = induction — focused on the **track endpoint** (Bragg region).

### 11.1 The four image operations (`src/images.py`)

- **`cut_start(image, target=50)`** → `image[-target:, :]`. Keeps the **last 50
  wires** — the stopping/Bragg end of the track. (Despite the name "cut_start", it
  keeps the *end*.) This is the asymmetry described in §1: the VAE sees only the
  endpoint.
- **`pad_image_batch_gpu(images, target_wh=(1502, 51), device, batch_size=32,
  cut_rows=None)`** — GPU-batched canvas placement:
  - if `cut_rows` set, apply `cut_start` first;
  - for each image, find `v = argmax(row 0)` — the peak time-tick on the first wire
    of the (cut) image — and place the image on a `target_h × target_w` zero canvas
    so that peak lands at the horizontal centre (`x0 = clamp(target_w//2 − v)`),
    vertically at the top (`y0 = 0`).
  - This **horizontally aligns every track by its entry-into-window position in drift
    time**, removing absolute drift position as a nuisance variable.
- **`downsample_image`** — scipy `zoom` bilinear (used by older code paths).
- **`pad_image`** — single-image CPU version of the canvas placement.

### 11.2 The production pipeline (`scripts/image_making.py`, default proton/kaon mode)

```
col.pkl, ind.pkl
  → image_cuts(lower=10)                       # 10<height<179, width<1500, union-aligned
  → split by particle_type into p_c,p_i,k_c,k_i (collection/induction lists)
  → pad_image_batch_gpu(cut_rows=50)           # last 50 wires, peak-centered on a (1502×51) canvas
  → F.interpolate(size=(256,256), bilinear)    # ⚠ current default is 256×256 (see note)
  → stack([collection, induction]) → (N,2,H,W)
  → save RAW                pk_256x256_raw_10-179wires.pt   {"p":…, "k":…}
  → save log1p (compat)     pk_256x256_log1p_10-179wires.pt {"p":…, "k":…}
```

The tensors are saved as a dict with **keys `"p"` (protons) and `"k"` (kaons)**, each
shape `(N, 2, H, W)`. Row order within `"p"` is the proton rows of `col.pkl` after
`image_cuts`; within `"k"`, the kaon rows. **This order is the alignment key** for
matching latents to features later (§21).

> **Raw vs transform.** Image-making saves **raw ADC** (`_raw_`) as the primary file
> and a `log1p` copy for backward compatibility. The intended design is: *save raw,
> apply the transform at training time* (`run_training.py` calls
> `apply_transform`). This lets one image file feed a whole transform sweep.

> ⚠ **256 vs 48 — the version drift.** The *current* `image_making.py` writes
> **256×256** (`pk_256x256_*`). But the **paper model trains on 48×48**
> (`pk_48x48_raw_10-179wires.pt`), which was produced by an **older** version of this
> same script. `run_training.py` will `F.interpolate` a 256 file down to the config's
> `input_hw` if they differ, but to reproduce the paper exactly you want the 48×48
> file. Treat the image resolution as part of the experiment identity, not an
> incidental detail.

### 11.3 Muon mode (`image_making.py --muon`)

Reads `muon_col.pkl`/`muon_ind.pkl`, applies `image_cuts(lower=175, upper=10_000_000,
width=473)`, pads with `cut_rows=50`, interpolates to **48×48**, saves
`muon_48x48_raw.pt` / `muon_48x48_log1p.pt` with key **`"m"`**. (Counts: 18,034
matched muon pairs → 8,964 after the width<473 image cut → an `(8964, 2, 48, 48)`
tensor; from `logs/image_making.log`.)

---

<a name="12-stage7-model"></a>
## 12. Stage 7 — The β-VAE model (`src/models/configVAE.py`)

A fully configurable convolutional VAE.

```python
VAE(input_hw=(48,48), latent=4, channels=(32,64,128,256),
    kernel=5, stride=2, padding=2, activation="softplus", p_enc=0.0)
```

### 12.1 Encoder

For each channel width `c` in `channels`, a block:
`Conv2d(in, c, kernel, stride, padding) → BatchNorm2d(c) → activation → Dropout2d(p_enc)`.
Input channels = 2 (collection, induction).

With `stride=2`, each block halves H and W, so after `len(channels)` blocks the
feature map is `input_hw / 2^len(channels)`. **`scale = 2 ** len(channels)`.** For the
paper model (48×48, 4 channels): `scale=16`, encoded map `3×3`, flattened size
`256·3·3 = 2304`.

Two linear heads on the flattened features produce the latent parameters:
`fc_mu : Linear(flat, latent)` and `fc_logvar : Linear(flat, latent)`.

### 12.2 Reparameterisation & decoder

- `reparameterise(μ, logσ²) = μ + ε·exp(0.5·logσ²)`, `ε ~ N(0, I)`.
- `fc_dec : Linear(latent, flat)`, reshaped back to `(bottleneck_ch, h_enc, w_enc)`.
- Decoder mirrors the encoder with `ConvTranspose2d(..., output_padding=1)` blocks
  (each doubles H, W) and `BatchNorm + activation`, ending with a final
  `ConvTranspose2d → 2 channels` followed by the activation. Because inputs are
  non-negative (log/transformed ADC), a non-negative output activation (softplus/relu)
  is appropriate.
- `forward(x) → (recon, mu, logvar, z)`.

### 12.3 Activation choices

`ACTIVATIONS = {softplus, relu, gelu, silu, leaky_relu}`. The paper model uses
`relu`; `default.yaml` uses `softplus`.

> **Architectural constraint (enforced by the sweep validator).** `input_hw` must be
> divisible by `2^len(channels)`, and the encoded map must be ≥ 1×1. So 48×48 works
> with up to 4 channel blocks (48/16 = 3); a 5th block (scale 32) would fail.

---

<a name="13-stage8-loss"></a>
## 13. Stage 8 — The loss function (`src/losses/vae.py`)

```
vae_loss(recon, x, mu, logvar, beta) = weighted_mse(recon, x) + beta · KL(mu, logvar)
```

- **Weighted MSE** — signal pixels matter more than background:

  ```
  mask = (target > 0.01)
  per_pixel = (recon − target)²
  weighted  = per_pixel · (1 + (weight − 1)·mask)     # weight = 10.0 (hard-coded)
  loss = weighted.sum_over_pixels.mean_over_batch
  ```
  So a signal pixel's error counts **10×** a background pixel's. This stops the VAE
  from trivially reconstructing the (mostly empty) canvas and ignoring the track.
  The `weight=10` lives in `weighted_mse_loss`'s default and is *not* exposed in the
  config — change it in code if you need to.

- **KL divergence** — standard closed form, mean over batch:
  `KL = 0.5 · Σ_dim (exp(logσ²) + μ² − 1 − logσ²)`.

- **β** scales the KL term (the "β-VAE" knob). The paper model uses **β = 0.5**
  (reconstruction-favouring, looser latent regularisation); `default.yaml` uses 10.

`recon_loss_mse` (plain summed MSE) also exists but is unused by `vae_loss`.

---

<a name="14-stage9-training"></a>
## 14. Stage 9 — Training (`scripts/run_training.py`, `src/train/train.py`)

### 14.1 What gets trained on

```python
p = data[cfg["data"]["proton"]]          # default key "p" → PROTONS ONLY
p = apply_transform(p, cfg["data"]["transform"])
```

**The VAE trains exclusively on protons.** Kaons (`"k"`) are never shown during
training — they are held out for the inference/anomaly step. This is the conceptual
heart of the method (§1): learn the proton manifold, then see where kaons land.

### 14.2 Config + CLI overrides

`run_training.py --config X.yaml` loads the YAML, then any of these CLI flags
override the corresponding field: `--latent --beta --lr --epochs --batch_size
--proton --channels --kernel --activation --transform`. The transform is applied to
the raw tensor; if the tensor's H,W ≠ `model.input_hw`, it is bilinearly resized.

### 14.3 Train/val split (deterministic & reused)

```
split_path = splits_dir / f"split_{cfg['data']['proton']}.npz"   # e.g. split_p.npz
```
If it exists, load `train_idx`/`val_idx`; else create via
`train_test_split(test_size=val_split, random_state=random_seed)` and **save it**.
The same `split_p.npz` is later consumed by inference and analysis to recover which
proton rows are train vs val — so **never regenerate it after training** or the
latent↔feature alignment breaks (§21).

### 14.4 The loop (`src/train/train.py`)

Standard VAE loop with:
- Adam (`lr`, `weight_decay` from config), `DataLoader(shuffle=True)` for train.
- Per-epoch tracking of total / reconstruction / KL loss for both splits.
- **Early stopping:** `patience=20`, `min_delta=1e-4` on val loss; keeps a deep copy
  of the best `state_dict` and restores it at the end.
- Returns the best model + six loss-history lists.

### 14.5 Outputs & naming

The model name is a deterministic encoding of the config (this exact string is the
join key for every downstream stage):

```
model_vae_latent8_ch32_64_128_256_beta0.5_lr0.001_epoch200_actrelu_kern5_stride2_pad2_hw48x48_txlog1p
```

Saved:
- `{output.dir}/{name}.pt` — `state_dict`
- `{output.dir}/{name}_curves.png` — train/val total/recon/KL curves
- `logs/{name}.json` — timestamp, device, full config, dataset sizes, per-epoch
  history (`save_run_log`).

---

<a name="15-stage10-sweeps"></a>
## 15. Stage 10 — Hyperparameter sweeps (`scripts/run_sweep.py`)

A Cartesian-grid runner over a `base` config plus a `grid`.

- **Grid expansion:** `iter_grid` takes `itertools.product` over grid axes; each combo
  is applied to a deep-copied base via dotted keys (`set_nested("model.latent", 8)`),
  and recorded under `cfg["sweep"] = {index, parameters}`.
- **Validation** (`validate_training_config`): activation ∈ valid set, transform ∈
  valid set, `input_hw` divisible by `2^len(channels)`, non-empty encoded map. Invalid
  combos are skipped (not failed).
- **`configs/sweep.yaml`** — the architecture grid:
  `latent ∈ {2,4,8,16,32,48,64,128} × channels ∈ {3 configs} × beta ∈ {0.5,1,5,10}
  × activation ∈ {softplus,relu}` = **192 runs**. The paper model is **run index 66**
  of this grid (hence `run_0066`).
- **`configs/sweep_transforms.yaml`** — fixes architecture, varies only
  `data.transform ∈ {none, log1p, sqrt, mpv_linear, mpv_tanh, hill_mpv}` = 6 runs.

### 15.1 Two execution modes

- **Remote (default).** Reads `configs/remote.yaml` (host, project dir, python cmd,
  data source, splits source). It: ssh-checks the project dir, mkdirs remote
  folders, rsyncs the image tensor and `split_*.npz` up (skips if present unless
  `--force-data`), then per config writes a generated YAML, rsyncs it up, runs
  `run_training.py` remotely (optionally pinned to a GPU via `CUDA_VISIBLE_DEVICES`),
  rsyncs back `{name}.pt`, `_curves.png`, `{name}.json`, and (optionally) deletes the
  remote `.pt`. With `--gpu-devices auto` it queries `nvidia-smi` for GPUs with
  ≥ `--min-free-gpu-gb` free and runs them concurrently via a thread pool.
- **Local (`--local --overrides csf.yaml`).** Writes each generated config to
  `{output.dir}/sweep_configs/run_XXXX_*.yaml` and runs `run_training.py` in-process
  sequentially. This is what `jobs/sweep.sh` does under SLURM.

Useful flags: `--start-index`, `--limit`, `--resume` (skip existing models),
`--keep-going` (continue past failures), `--dry-run`.

---

<a name="16-stage11-inference"></a>
## 16. Stage 11 — Inference (`scripts/run_inference.py`, `src/inference/inference.py`)

`inference(model, data)`:
- encodes in batches of 8 on `mps/cpu`;
- **uses the mean `μ` (not a sampled `z`) as the latent vector** — deterministic,
  reproducible embeddings;
- returns `(latent_vectors (N, latent), recon_all (N,2,H,W), RE_per_sample (N,))`
  where RE = per-sample mean squared reconstruction error over all pixels/channels.

`run_inference.py`:
1. Rebuilds the model name from the config, loads `{name}.pt`.
2. Loads the image tensor, applies the config transform to `"p"` and `"k"`.
3. Loads `split_p.npz`, makes `train_subset`/`val_subset` from protons.
4. Runs inference on **train**, **val**, and the **full kaon set**.
5. Writes `{inference_dir}/{name}/{train,val,kaon}.npz`, each with arrays
   `latents`, `recon`, `re`.
6. Optional `--include-muons` (loads `--muon-image-path`, key `"m"`, → `muon.npz`)
   and `--csda-kaon-path` (key `"k"`, → `csda_kaon.npz`).

> ⚠ **Muon path trap.** The `--muon-image-path` default is
> `muon_48x48_raw_180+wires.pt` — the **OLD** proxy sample. The **NEW** pipeline
> writes `muon_48x48_raw.pt` / `muon_48x48_log1p.pt`. If you run the new muon
> extraction but forget to override the path, you will silently infer on the old
> 457-track proxy while your features describe the new sample. **Always pass
> `--muon-image-path` explicitly** (as `muons.sh` does). See §21.

---

<a name="17-stage12-features"></a>
## 17. Stage 12 — Physics feature engineering (`scripts/compute_features.py`, `src/features/`)

### 17.1 What it computes and on what

- Loads `col.pkl`, `ind.pkl`, applies `image_cuts(lower=10)` (must mirror
  image-making so the row order matches the image tensor).
- **Iterates `col.iterrows()` — i.e. features use the COLLECTION plane only**, even
  though the VAE consumes both planes. Each row provides `image_intensity` (full
  bbox, not the cut-50 endpoint) and `column_maxes`.
- Computes **25 features** = 9 image-based + 16 profile-based (see Appendix A for
  formulas), plus carries `run/subrun/event/particle_type/height/chi_squared_*`.
- Any feature that raises is set to `NaN` (robust).

> The repo `README.md` says "22 features"; the code computes **25**. Trust the code.
> Note also: `image_intensity` here is the **full track** crop, so `total_adc`,
> `bragg_rise_slope`, etc. describe the whole track — deliberately mismatched with
> the VAE's endpoint-only view (§1).

### 17.2 External truth/baseline columns merged in

- **Log-likelihoods** from `kaon_df_plane_1_thr_DAQ.csv` + `proton_df_plane_1_thr_DAQ.csv`
  (concatenated, deduplicated on event key), left-merged → `log_likelihood_kaon`,
  `log_likelihood_proton`.
- **Cleaning:** `log_likelihood_* < −5000 → NaN`; `chi_squared_* > 10 → NaN` (clip
  pathological values so plots/regressions aren't dominated by outliers).

Saved to `{features_dir}/features.pkl`. Then:
- **Histograms** for every feature + `height` + χ² + log-L, overlaying
  proton/kaon/muon/csda_kaon densities (`src/features/plot.py :: hist`).
- **UMAP**: loads train/val/kaon latents, builds (or reuses a cached `reducer.pkl`)
  a `umap.UMAP(n_neighbors=30, min_dist=0.1)` fit on all latents, transforms each
  population, and saves per-feature scatter plots colored by feature value
  (`plot_umap`, 3 or 4 panels). The proton features are aligned to train/val latents
  by indexing `feat_df[particle_type=='proton'].iloc[train_idx / val_idx]`.

### 17.3 The two optional species

`--include-muons` reads `muon_col.pkl`/`muon_ind.pkl` (`image_cuts(lower=175,
upper=10M, width=473)`) and appends rows with `particle_type='muon'`.
`--csda-kaons` reads `csv_kaon_col_clean.pkl` and appends `particle_type='csda_kaon'`.

---

<a name="18-stage13-analysis"></a>
## 18. Stage 13 — Latent-space analysis (`scripts/analyse_latents.py`)

The flagship analysis script. Output dir: `figs/<model_name>/latents-features/`.
CLI: `--config`, `--analyses {correlation,traversal,logistic,nonlinear,feature_auc}`
(default all), `--features`, `--include-muons`, `--csda-kaons`.

### 18.1 The active feature subset

```python
CALO = ["mean_adc", "total_adc"]
TOPO = ["solidity"]
```
The full lists (21 calo + 5 topo) are present but **commented out**. As shipped, the
analysis focuses on **three** interpretable features — the cleanest calorimetry
summaries plus the one topology feature that captures kaon kinks. To analyse more,
uncomment.

### 18.2 The alignment reconstruction (the crux, repeated in several analyses)

Latents come back as three separate arrays; features come back as one big DataFrame.
To line them up, the code rebuilds a full-length latent matrix indexed by DataFrame
position:

```python
all_proton = features[features.particle_type=="proton"]
all_kaon   = features[features.particle_type=="kaon"]
latent_z = zeros((len(features), n_dims))
latent_z[all_proton.index[train_idx]] = train_latents
latent_z[all_proton.index[val_idx]]   = val_latents
latent_z[all_kaon.index]              = kaon_latents
```

This **assumes** the *i*-th proton row of `features.pkl` is the same physical track as
the *i*-th proton image in the tensor (and the same `train_idx`/`val_idx` split).
That assumption is the **positional alignment contract** of §21 — there is no
`(run,subrun,event,cluster_idx)` join here, only row order.

### 18.3 The five analyses

1. **`correlation`** — Spearman ρ between each active feature and each latent dim
   (heatmap, p/k combined and per-particle); feature-to-feature correlation; and a
   **variance decomposition** (per-dim, per-category mean univariate linear R²,
   stacked bar). Outputs: `disentanglement_heatmap.png`,
   `feature_correlation.png`, `variance_decomposition.png` (+ `_proton`/`_kaon`).

2. **`traversal`** — load the model, take an anchor `μ = train_latents[10]` and
   per-dim `σ = std(train_latents)`. For each latent dim, sweep that coordinate from
   −2σ to +2σ in 9 steps (others fixed at the anchor), decode, and tile the
   collection-channel reconstructions into a grid. Reveals what each dimension
   "means" generatively. Output: `latent_traversal.png`.

3. **`logistic`** — supervised *probe* of separability (the latents themselves are
   unsupervised; this only measures them). Binary **proton (val) vs kaon**:
   - `X = [val_latents ; kaon_latents]`, `y = [0…, 1…]`.
   - Latent **subsets**: each single dim; all pairs (if `n_dims ≤ 8`); all dims.
   - 5-fold stratified CV; **Logistic Regression** (`StandardScaler` + balanced LR)
     and an **MLP** (32,16) trained per-fold with balanced sample weights.
   - Reports AUC + accuracy per subset, event-level agreement (both correct / LR
     only / MLP only / both wrong), and **hard cases** (events both classifiers get
     wrong, split into "hard kaons that look like protons" and vice-versa).
   - Plots: `linear_probe.png`, `hard_cases.png`, and raw-image panels of the hard
     kaons/protons (mapping subset positions back to the image tensor via
     `kaon_orig_indices` and `val_idx`).
   - Runs **twice**: all kaons, then **"picky" kaons** (`p==1` in `picky+match.csv`),
     suffix `_picky`.
   - If `--csda-kaons`: extra 60/40 binary probes csda-kaon vs proton / kaon / muon
     (`csda_kaon_logistic_probes.png`).

4. **`nonlinear`** — how much feature information is in the latents, and whether it's
   linear. For each active feature: predict it from **all** latents with **Ridge**
   (linear) vs an **MLP** (16,16); 5-fold CV R²; the **gap** (MLP − Ridge) measures
   non-linearly encoded information. Also per-particle R² (proton/kaon/[muon]),
   error metrics (MAE/MSE/RMSE/MAPE), **permutation importance** per latent dim, and
   **mutual information** per dim. Outputs: `nonlinear_r2.png`, `error_metrics.png`,
   `permutation_importance_{category}[_proton/_kaon].png`,
   `mutual_information_{category}.png`. If muon latents exist they are folded into a
   combined p/k/muon regression.

5. **`feature_auc`** — per *class*, can the latent space recover a feature's
   high/low split? For each feature and each species (proton, kaon, [muon],
   [csda_kaon]) independently: binarise the feature at its median, train LR on that
   species' latents (5-fold CV), report AUC. Output: grouped bar chart
   `feature_auc.png`. This answers "is feature F encoded *within* protons, *within*
   kaons, …?" rather than "does F separate the species?".

### 18.4 Muon-only section

With `--include-muons`, after the main analyses the script runs a **separate**
muon-only block (`figs/<model>/latents-features-muon/`): muon correlation heatmap,
muon traversal (anchored at the muon latent mean), binary muon-vs-proton /
muon-vs-kaon probes, and a combined p/k/muon non-linear regression. If muon feature
count ≠ latent count it **truncates to the minimum** (a silent alignment band-aid —
see §21).

---

<a name="19-aux-samples"></a>
## 19. Auxiliary particle samples: muons & CSDA-kaons

### 19.1 Muons — two generations (do not confuse them)

| | **OLD** (`image_making_muons.py`) | **NEW** (`image_making_muons_art.py`) |
|---|---|---|
| Source | the **proton** ROOT file | dedicated **`RAW_muons.root`** (`anatree/raw`) |
| Definition | clusters with `height ≥ 180` & `width < 1500` (punch-through proxies) | genuine muon events |
| Cuts | `cluster_cuts(lower=179, upper=10000)` then `matching` | clip ADC≥0, `cluster_cuts(lower=175)`, `matching` |
| Cluster pkl | writes `muon_col.pkl`/`muon_ind.pkl` | **overwrites** the same `muon_col.pkl`/`muon_ind.pkl` |
| Images | writes `muon_48x48_raw_180+wires.pt` directly | **does not** make images — `image_making.py --muon` does, → `muon_48x48_raw.pt` |
| Count | ~457 matched tracks | 18,034 matched → 8,964 after image cut |

The NEW pipeline is the intended one (real muons). The danger is the **shared
filenames** (§21): the NEW run overwrites the cluster pickles but the images land
under a *different* name than the OLD run, while `run_inference.py`'s default still
points at the OLD image file.

### 19.2 CSDA-kaons (high-purity kaon reference)

"CSDA-kaons" are kaons confirmed by an external selection (CSDA range / track length
consistency — see §20), listed in `kaon_df_plane_1_thr_DAQ.csv` (432 events). They
serve as a **clean kaon reference** (vs the broader, contaminated "kaon candidate"
sample from `rawExtracted`).

Pipeline:
1. **`extract_csv_kaons.py`** — for the 432 events, extract **all** clusters from
   ROOT with **no quality/length cuts** (maximise recovery), save unmatched per-plane
   pickles (for the manual labeler), then `matching` + `width<1500` →
   `csv_kaon_col.pkl`/`csv_kaon_ind.pkl` and a first-pass image tensor
   `csv_kaon_48x48_raw.pt`. (Real run: 26,148 clusters → 8,056 matched → 8,016 kept;
   multiple clusters per event.)
2. **Cluster cleaning — pick one track per event.** Two interchangeable tools:
   - **`csda_kaon_cleaning.py`** (automatic) — compute the 25 features for every
     csda-kaon cluster, train a 3-class **RandomForest** (proton/muon/kaon) on the
     reference `features.pkl`, score each cluster's `P(kaon)`, keep the highest per
     event (fallback: lowest `match_score`). Writes `csv_kaon_*_clean.pkl` and
     `csv_kaon_48x48_raw_clean.pt`, plus a diagnostic feature grid.
   - **`csda_kaon_labeler.py`** (manual, Bokeh) — a GUI showing all clusters per
     event per plane; you click the true kaon track in each plane; saves the same
     `*_clean` artifacts. Auto-initialises to the tallest cluster.
3. Downstream: `run_inference.py --csda-kaon-path …_clean.pt` → `csda_kaon.npz`;
   `compute_features.py --csda-kaons`; `analyse_latents.py --csda-kaons`.

### 19.3 Feature diagnostics

`inspect_solidity.py` and `inspect_n_local_maxima.py` visualise exactly how those two
features behave on real proton/kaon/muon tracks (solidity = signal vs convex-hull
fill, drawn with the hull outline; n_local_maxima = raw vs `uniform_filter1d(size=15)`
smoothed profile with detected peaks). Use these to sanity-check that a feature
measures what you think.

---

<a name="20-physics-validation"></a>
## 20. Physics validation (the notebooks not mirrored in scripts)

These live only in `notebooks/` (git-ignored). They are the scientific payoff and the
ground-truth cross-checks.

- **`separation.ipynb`** — fits a 2-component **Gaussian Mixture Model** (full
  covariance, 20 restarts) to the proton+kaon latents and measures how well the
  *unsupervised* clusters line up with the proton/kaon labels. This is the
  "does the latent space separate species without supervision?" result, complementing
  the supervised logistic probe.
- **`mvn_classification.ipynb`** — fits a **multivariate Gaussian per known species**
  (proton, muon, csda_kaon) in latent space, then classifies the broad kaon-candidate
  sample by maximum likelihood. A generative, reference-driven PID built on the clean
  species samples.
- **`csda-length.ipynb`** — validates the kaon selection against **beamline mass**.
  Loads `selected_kaon_csda.csv` and `selected_kaon_model.csv`, merges the
  spectrometer `beamline_mass`, and histograms it; a clean peak near the **kaon mass
  (~494 MeV/c²)** confirms the selection is physically real, not a latent artifact.
  (`beamline_mass` is the closest thing to truth in the whole project: it is measured
  by the upstream beamline spectrometer, independent of the TPC image.)

> **Reproducibility note for the paper:** the *flow-threshold* + *beamline-mass*
> selection referenced in some discussions is **not** committed as a script — the
> repo's separation logic is the GMM notebook, and `nflows` (in `pyproject.toml`) is
> unused. If the paper claims a normalising-flow selection, that code must be added.

---

<a name="21-alignment"></a>
## 21. The data-alignment contract & integrity pitfalls

This section is the most important one for anyone reproducing results. The pipeline
joins latents to features **by row position, not by event key**, in several places.
Get the order wrong and every correlation/AUC silently becomes meaningless.

### 21.1 The positional alignment contract

The chain that must hold, link by link:

```
col.pkl  ──image_cuts(lower=10)──▶  ordered rows
   │                                    │
   ├─ split by particle_type 'proton' ──┴──▶ image tensor "p"  (image_making.py)
   │                                    └──▶ feature rows proton (compute_features.py)
   └─ split by particle_type 'kaon'  ──────▶ image tensor "k"  &  feature rows kaon

split_p.npz (train_idx/val_idx)  indexes the PROTON order identically in
   training, inference, compute_features UMAP, and analyse_latents.
```

For latents to match features, **`col.pkl` must be byte-for-byte the same file, with
the same row order, when image-making, feature-computation, and the saved
`split_p.npz` were produced.** There is **no `(run,subrun,event,cluster_idx)` key
stored alongside the image tensors or the `.npz` latents** — only order. If you
re-run `dataset.py` (which concatenates kaons-then-protons and inner-merges reco,
both order-affecting) after training, you must regenerate images, features, splits,
and re-train/re-infer together. This is the single biggest validity threat in the
project.

**Mitigation if you rebuild from scratch:** persist `(run, subrun, event,
cluster_idx)` next to every image tensor and every `.npz`, and join on that key in
`compute_features.py`/`analyse_latents.py` instead of `.iloc[train_idx]`. The data is
all there in `col.pkl`; it simply isn't carried forward.

### 21.2 The muon filename trap

- OLD muon images: `muon_48x48_raw_180+wires.pt` (457 proxy tracks from the proton
  file).
- NEW muon images: `muon_48x48_raw.pt` (8,964 real muons), but the NEW extractor
  **overwrites** `muon_col.pkl`/`muon_ind.pkl`.
- `run_inference.py --muon-image-path` **defaults to the OLD file.**

So a careless run computes muon *features* from the NEW `muon_col.pkl` but muon
*latents* from the OLD image tensor. The analysis then "fixes" the length mismatch by
**truncating to the shorter array** — producing a confidently wrong, misaligned muon
analysis. **Always pass `--muon-image-path .../muon_48x48_log1p.pt` (or `_raw.pt`)
explicitly**, exactly as `scripts/extra/muons.sh` does.

### 21.3 `mean_adc` is a confounded feature

`mean_adc = np.mean(image_intensity)` averages over the **entire bounding box,
including all the zero background pixels**. It therefore conflates true charge density
with fill fraction and track geometry. `median_adc = np.median(image[image>0])` (signal
pixels only) is the clean charge-density probe. Be careful interpreting `mean_adc`
correlations — and note `mean_adc` is one of the three active analysis features.

### 21.4 Stale defaults

- `configs/default.yaml` is **internally inconsistent**: `input_hw: [256,256]` but
  `path: pk_48x48_raw_…`, `latent: 4`, `beta: 10`, `epochs: 500`. It is *not* the
  paper model. Use a named `run_XXXX.yaml` (the paper model is `run_0066`).
- `image_making.py` default now outputs **256×256**; the 48×48 training file was made
  by an older revision. Resolution is part of experiment identity.
- `src/train/vae.py` and `*.egg-info/SOURCES.txt` reference a `src/models/vae.py`
  that no longer exists. Dead.
- `README.md` says 22 features (it's 25) and shows figure names like
  `bragg_peak_position.png`/`max_ADC_postion.png` that don't all match the code's
  `max_ADC_position` — figure paths in the README drifted from the feature names.

### 21.5 Other gotchas

- The kaon tree cycle `ana/raw;352` is file-specific.
- `dataset.py` has **no argparse**: editing paths means editing source.
- The `weight=10` in `weighted_mse_loss` is hard-coded, not in the config.
- Inference uses **μ** (the mean), not a sampled `z` — correct for reproducibility,
  but remember it when comparing to the training stochasticity.
- `compute_features.py` features come from the **collection plane only**; the VAE used
  both planes. Intentional, but worth stating in any writeup.

---

<a name="22-artifacts"></a>
## 22. Output artifact catalog & naming grammar

### 22.1 The model-name grammar (the universal join key)

Built identically by `run_training.py`, `run_inference.py`, `compute_features.py`,
`analyse_latents.py`, `run_sweep.py`, and the notebooks:

```
model_{type}_latent{L}_ch{c0_c1_…}_beta{B}_lr{LR}_epoch{E}
     _act{ACT}_kern{K}_stride{S}_pad{P}_hw{H}x{W}_tx{TRANSFORM}
```

Example (paper): `model_vae_latent8_ch32_64_128_256_beta0.5_lr0.001_epoch200_actrelu_kern5_stride2_pad2_hw48x48_txlog1p`.

### 22.2 Artifacts by stage

| Stage | Path | Contents |
|---|---|---|
| 1–5 | `clusters/col.pkl`, `clusters/ind.pkl` | matched tracks + beamline + reco + χ² |
| 6 | `images/pk_{HW}_{raw|log1p}_10-179wires.pt` | `{"p":…, "k":…}` tensors `(N,2,H,W)` |
| 6 (muon) | `images/muon_48x48_{raw|log1p}.pt` | `{"m":…}` |
| 6 (csda) | `images/csv_kaon_48x48_raw[_clean].pt` | `{"k":…}` |
| 9 | `models/{name}.pt`, `{name}_curves.png` | weights + loss curves |
| 9 | `logs/{name}.json` | full config + per-epoch history |
| 9 | `training/split_{protonkey}.npz` | `train_idx`, `val_idx` |
| 11 | `inference/{name}/{train,val,kaon,muon,csda_kaon}.npz` | `latents`, `recon`, `re` |
| 11/12 | `inference/{name}/reducer.pkl` | cached fitted UMAP |
| 12 | `features/features.pkl` | 25 features + meta + χ² + log-L |
| 12 | `figs/{name}/features/*.png`, `figs/{name}/umap/*.png` | histograms, UMAPs |
| 13 | `figs/{name}/latents-features/*.png` | the 5 analyses |
| 13 (muon) | `figs/{name}/latents-features-muon/*.png` | muon-only analyses |

`.npz` arrays: `latents` `(N, latent)`, `recon` `(N, 2, H, W)`, `re` `(N,)`.

---

<a name="23-runbook"></a>
## 23. End-to-end runbook (exact commands)

Assumes `uv` env is synced and the external drive is mounted. Use the **paper config**
`configs/run_0066_…txlog1p.yaml` (referred to below as `$CFG`).

### 23.1 Core proton/kaon pipeline

```bash
# Stage 1–5: ROOT → matched cluster DataFrames (edit paths in the script first)
uv run python scripts/dataset.py
#   → clusters/col.pkl, clusters/ind.pkl   (≈10.5k proton, ≈8.4k kaon, 18,752 total)

# Stage 6: images  (NOTE: current default writes 256×256; the paper uses 48×48 —
#          use the script revision / config that produces pk_48x48_raw_10-179wires.pt)
uv run python scripts/image_making.py
#   → images/pk_{HW}_{raw,log1p}_10-179wires.pt

# Stage 9: train the β-VAE on PROTONS ONLY
uv run python scripts/run_training.py --config $CFG
#   → models/{name}.pt, {name}_curves.png, logs/{name}.json, training/split_p.npz
#   (or drive a whole grid: uv run python scripts/run_sweep.py --sweep configs/sweep.yaml [--local --overrides configs/csf.yaml])

# Stage 11: latents/reconstructions for train/val/kaon
uv run python scripts/run_inference.py --config $CFG
#   → inference/{name}/{train,val,kaon}.npz

# Stage 12: physics features + histograms + UMAP
uv run python scripts/compute_features.py --config $CFG
#   → features/features.pkl, figs/{name}/{features,umap}/*.png

# Stage 13: latent analyses
uv run python scripts/analyse_latents.py --config $CFG
#   → figs/{name}/latents-features/*.png
```

### 23.2 Add muons (`scripts/extra/muons.sh`)

```bash
uv run python scripts/extra/image_making_muons_art.py        # RAW_muons.root → muon_col/ind.pkl
uv run python scripts/image_making.py --muon                 # → images/muon_48x48_{raw,log1p}.pt
uv run python scripts/run_inference.py --config $CFG --include-muons \
    --muon-image-path /Volumes/easystore/proton-kaon/images/muon_48x48_log1p.pt   # ← explicit path!
uv run python scripts/compute_features.py --config $CFG --include-muons
uv run python scripts/analyse_latents.py  --config $CFG --include-muons
uv run python scripts/extra/plot_umap_all.py --config $CFG
```

### 23.3 Add CSDA-kaons (`scripts/extra/csda-kaons.sh`)

```bash
uv run python scripts/extract_csv_kaons.py                                   # ROOT → csv_kaon_*.pkl + raw images
uv run python scripts/extra/csda_kaon_cleaning.py --config $CFG              # RF → *_clean.pkl + *_raw_clean.pt
#   (or: bokeh serve scripts/extra/csda_kaon_labeler.py --args --config $CFG  # manual labeling)
uv run python scripts/run_inference.py    --config $CFG \
    --csda-kaon-path /Volumes/easystore/proton-kaon/images/csv_kaon_48x48_raw_clean.pt
uv run python scripts/compute_features.py --config $CFG --csda-kaons --include-muons
uv run python scripts/analyse_latents.py  --config $CFG --include-muons --csda-kaons \
    --analyses feature_auc logistic
uv run python scripts/extra/plot_umap_all.py --config $CFG
```

### 23.4 Physics validation (notebooks)

Run `notebooks/separation.ipynb` (GMM), `notebooks/mvn_classification.ipynb`
(per-species MVN PID), and `notebooks/csda-length.ipynb` (beamline-mass cross-check)
against the produced `.npz`/CSV artifacts.

### 23.5 Minimal from-scratch order

```
dataset.py → image_making.py → run_training.py → run_inference.py
          → compute_features.py → analyse_latents.py
```
Everything else (muons, CSDA-kaons, sweeps, interactive explorers, validation
notebooks) hangs off this spine.

---

<a name="appendix-a"></a>
## Appendix A — Feature formula reference

`img` = full-bbox `image_intensity` (collection plane). `cm` = `column_maxes`
(per-wire peak ADC, the longitudinal profile, length = `height`). Index 0 = track
entry, last index = stopping end.

### Image-based (9) — `src/features/calorimetry.py` + `topology.py`

| Feature | Formula / definition | Physics intuition |
|---|---|---|
| `total_adc` | `log(Σ img)` | total deposited charge (log) |
| `mean_adc` | `mean(img)` over **all** bbox pixels (incl. zeros) | ⚠ confounded with fill (§21.3) |
| `median_adc` | `median(img[img>0])` | robust typical `dE/dx`; clean density probe |
| `max_adc` | `max(img)` | Bragg-spike pixel height |
| `std_adc` | `std(img[img>0])` | spread of signal pixel intensities |
| `adc_entropy` | normalised Shannon entropy of 50-bin signal histogram | diversity of deposition levels |
| `n_pixels` | `Σ(img>0)` | track volume proxy |
| `solidity` | largest region's `area / convex_hull_area` | <1 ⇒ kinks/bends (kaon decay) |
| `fill_fraction` | `Σ(img>0) / (H·W)` | size-independent occupancy |

### Profile-based (16) — from `cm`

| Feature | Formula / definition | Physics intuition |
|---|---|---|
| `bragg_peak_height` | `max(cm)` | peak `dE/dx` |
| `max_ADC_position` | `argmax(cm)/len(cm)` | normalised peak position (→1 for stopping protons) |
| `bragg_peak_ratio` | `max(cm)/mean(cm)` | Bragg prominence over track average |
| `bragg_peak_to_median` | `max(cm)/median(cm)` | robust prominence |
| `end_vs_start_ratio` | `mean(last 10%) / mean(first 10%)` | magnitude of the Bragg rise |
| `last_quartile_mean` | `mean(cm[last 25%])` | elevated near Bragg for protons |
| `first_quartile_mean` | `mean(cm[first 25%])` | entry-region baseline |
| `bragg_rise_slope` | slope of linear fit `cm vs wire` | steeper for rising protons |
| `peak_integral_fraction` | `Σ(last 15%) / Σ(cm>0)` | charge concentration at the end |
| `bragg_peak_width` | FWHM via `peak_widths(rel_height=0.5)` | narrow proton spike vs flat kaon |
| `profile_cv` | `std/mean` of `cm>0` | unevenness of energy loss |
| `monotonic_rise_fraction` | fraction of `diff(smooth₃(cm))>0` | consistency of rise toward Bragg |
| `relative_peak_energy` | `Σ(±10% window around peak) / Σ(cm>0)` | sharpness of the spike |
| `profile_skewness` | `skew(cm>0)` | right-skew from Bragg tail |
| `profile_kurtosis` | `log(kurtosis(cm>0))` | flat/concentrated profile shape |
| `n_local_maxima` | peaks in `uniform_filter1d(cm, size=15)` | multiple bumps ⇒ kaon secondaries/decay |

---

<a name="appendix-b"></a>
## Appendix B — Config schema

```yaml
data:
  path:        <image .pt file>            # dict tensor with keys proton/kaon
  proton:      p                           # KEY selecting the training tensor (proton-only)
  kaon:        k                           # KEY for the held-out kaon tensor
  transform:   log1p                       # one of src/transforms.py VALID_TRANSFORMS
  val_split:   0.1                         # fraction held out for validation
  random_seed: 42                          # split reproducibility
  # features_path: <override for analyse_latents>   (optional)

model:
  type:        vae
  latent:      8                           # latent dimensionality
  input_hw:    [48, 48]                    # must be divisible by 2**len(channels)
  channels:    [32, 64, 128, 256]          # encoder block widths (decoder mirrors)
  kernel:      5
  stride:      2
  padding:     2
  dropout:     0                           # Dropout2d rate in encoder (p_enc)
  activation:  relu                        # softplus|relu|gelu|silu|leaky_relu

optimizer:
  lr:           0.001
  weight_decay: 0.0001

train:
  epochs:     200
  batch_size: 32
  beta:       0.5                          # KL weight (β-VAE)
  patience:   20                           # early-stopping patience (optional, default 20)
  min_delta:  0.0001                       # early-stopping tolerance (optional)

output:
  dir:           <models dir>
  splits_dir:    <splits dir>              # split_<protonkey>.npz lives here
  inference_dir: <inference dir>

# sweep: {index, parameters}   ← stamped automatically by run_sweep.py
```

**Transforms** (`src/transforms.py`): `none, log1p, sqrt, cbrt, minmax,
log1p_minmax, clamp99_minmax, tanh100, tanh500, mpv_linear, mpv_tanh, hill_mpv`.
The `mpv_*` / `hill_mpv` transforms normalise each plane by its most-probable ADC
value (collection ≈ 483, induction ≈ 247) so the two channels share a scale despite
their natural ~2× difference. `log1p` is the paper choice.

---

<a name="appendix-c"></a>
## Appendix C — Glossary

- **LArIAT** — Liquid Argon In A Testbeam; the LArTPC test-beam experiment supplying
  the data.
- **Collection / induction plane** — the two 240-wire sensing planes; channels 0–2.
- **ADC** — digitised charge per (wire, time-tick) pixel.
- **Cluster / region** — a connected above-threshold blob = a candidate track in one
  plane.
- **`column_maxes`** — per-wire maximum ADC; the longitudinal `dE/dx`-like profile
  (the most-used derived quantity).
- **Bragg peak** — the sharp `dE/dx` rise at a stopping track's end.
- **Residual range (`trkrr`)** — distance from a hit to the track end; the x-axis of
  Bethe–Bloch.
- **χ² / `particle_hypothesis`** — Bethe–Bloch goodness-of-fit PID baseline; 0 = kaon,
  1 = proton.
- **`beamline_mass`** — spectrometer-measured mass upstream of the TPC; the project's
  closest proxy for truth.
- **β-VAE** — variational autoencoder with a tunable KL weight β.
- **Latent (μ)** — the encoder mean vector used as the deterministic embedding.
- **RE** — per-sample reconstruction error (mean squared error over pixels).
- **CSDA-kaon** — high-purity kaon selected by CSDA-range/length, one clean track per
  event.
- **Picky** — the curated "picky" beam selection (`p==1`) used for the highest-quality
  subset.
- **CSF / cdt** — the SLURM cluster / the SSH GPU box used for training sweeps.

---

*End of manual. This document reflects the repository state as read from source on
2026-06-19; verify `dataset.py`/`image_making.py` paths and the active model config
before reproducing, and treat §21 (alignment) as mandatory reading.*
