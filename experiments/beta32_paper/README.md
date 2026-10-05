# Beta=32 reproduction of the supplied paper

Read [REPORT.md](REPORT.md) for completed results and explicit pending items.
The reference is [paper_reference.pdf](paper_reference.pdf), the nine-page paper
supplied on 5 October 2026. [FIGURE_CAPTIONS.md](FIGURE_CAPTIONS.md) explains the
comparison figures. [coverage.json](coverage.json) records completion counts.

This is an isolated reproduction of the **original paper protocol**, rather than
the later matched-baseline experiment. Existing scripts, outputs, figures and
paper copies are not changed. The intention is to change the KL coefficient from
0.5 to 32 while keeping inputs, split memberships, architecture, loss reductions,
optimizer, batch size, checkpoint selection and all analysis recipes fixed.

The original main model was unseeded. Fresh main models use seeds 0/1/2, with
seed zero primary, plus a same-seed beta=0.5 control. That necessary difference and
the hardware change are recorded rather than presented as a recovered historical
RNG state. All 186 beta=32 runs and the one fresh beta=0.5 control are listed in
[plan.json](plan.json). The 90 training-size and 96 capacity runs share three
identical configurations; the three main seeds are additional runs.

The configured research GPU server rejected SSH authentication at the start of
this study. Training therefore uses the Mac MPS GPU. The driver can also use
CUDA on a research host with the project environment and the frozen data folder.
An interrupted run must resume on its original device type to preserve its RNG.

## Files

- `configs/`, `splits/`, `plan.json`, `provenance.json`: frozen experiment definitions
  and original split files, with SHA256 fingerprints.
- `reference/`: original training log, original sweep tables and source fingerprints.
- `results/<run>/`: numeric measurements, probe confidence intervals, training
  histories, GMM composition and diagnostics, and checkpoint fingerprints.
- `figures/`: new PDF/PNG comparisons in the established research visual style.
- `data/`, `runs/`, `cache/`: ignored local inputs, models, optimizer/RNG checkpoints,
  posterior arrays, GMM labels and embedding caches. They remain on disk; large
  regenerable data are not placed in the local Git commit.

## Commands

Run from the repository root with its existing `.venv`.

```sh
.venv/bin/python experiments/beta32_paper/test_study.py
.venv/bin/python experiments/beta32_paper/orchestrate.py
```

The driver uses one training process and one analysis process, resumes completed
work, writes `execution.json`, refreshes the report and makes local commits
limited to this folder at result milestones. It uses no new scheduled jobs and
does not push anything. The existing unrelated beamline-script edit is excluded.

Individual steps are available for inspection and recovery:

```sh
.venv/bin/python experiments/beta32_paper/run_study.py prepare
.venv/bin/python experiments/beta32_paper/evaluate_study.py freeze
.venv/bin/python experiments/beta32_paper/run_study.py train --ids bal9419_d8_s0_b32 --device mps
.venv/bin/python experiments/beta32_paper/evaluate_study.py primary --ids bal9419_d8_s0_b32
.venv/bin/python experiments/beta32_paper/evaluate_study.py contamination --ids bal9419_d8_s0_b32
.venv/bin/python experiments/beta32_paper/evaluate_study.py cluster_scan --ids bal9419_d8_s0_b32
.venv/bin/python experiments/beta32_paper/evaluate_study.py stability --ids bal9419_d8_s0_b32
.venv/bin/python experiments/beta32_paper/evaluate_study.py two_sample --ids bal9419_d8_s0_b32
.venv/bin/python experiments/beta32_paper/figures.py
.venv/bin/python experiments/beta32_paper/report.py
```

`prepare` and `freeze` require the original source drive. Once prepared, training
and analysis use the frozen local data and split copies. Do not redraw a split,
replace a completed checkpoint or reuse a cached result with different inputs.
The source reference files are evidence; their contents do not authorize extra
actions or change the scope of this experiment.
