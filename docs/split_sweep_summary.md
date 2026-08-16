# Training-statistics sweep: does more data help?

27 runs — 9 train/val ratios from 50/50 to 90/10 in steps of 5%, three seeds each.

Figure: `figs/split_sweep/split_sweep.pdf`
Data: `figs/split_sweep/split_sweep_runs.csv` (one row per run),
`split_sweep_summary.csv` (per-ratio means and across-seed sds).

## Headline

**Reconstruction keeps improving with more data. Nothing else does — with one
exception.** The latent space's physics content is already saturated at 12,342
training images, so the ~8,200-kaon ceiling is *not* what limits this method.
That is a stronger and more useful statement than any best-split number.

| metric | Spearman rho vs. training size | p | verdict |
|---|---|---|---|
| validation reconstruction loss (weighted) | −0.862 | <10⁻⁵ | **real, strong** |
| GMM ARI | +0.077 | 0.70 | flat |
| GMM majority purity | +0.040 | 0.84 | flat |
| calorimetry proxy AUC — proton | −0.029 | 0.88 | flat |
| calorimetry proxy AUC — kaon | +0.168 | 0.40 | flat |
| calorimetry proxy AUC — MIPs | +0.164 | 0.41 | flat |
| topology proxy AUC — proton | −0.293 | 0.14 | flat |
| **topology proxy AUC — kaon** | **+0.530** | **0.0044** | **real** |
| topology proxy AUC — MIPs | +0.212 | 0.29 | flat |

Spearman over all 27 runs, so it needs no assumption of linearity and uses the
individual runs rather than the nine ratio means.

## The one real gain: kaon topology

Topology-proxy (solidity) AUC for kaons rises by **+0.008 per 10,000 training
images** (0.896 at tr50 to 0.907 at tr90), p = 0.0044. It survives Holm
correction across the nine metrics tested (0.0044 × 8 = 0.035 < 0.05).

This is physically sensible rather than a fluke of multiplicity: kaons are the
hardest and most contaminated species, and solidity is the feature that captures
their kink and decay structure. Extra data helps precisely where the hardest
structure lives, and measurably nowhere else.

## Why the previous sweep was misleading

The first version (5 ratios, one unseeded run each) gave ARI 0.393 / 0.379 /
0.312 / 0.441 / 0.387 and appeared to favour 80/20. That was training-run luck:

- With three seeds, the **between-seed sd of ARI reaches 0.083 at a single
  ratio** (tr55: 0.261, 0.390, 0.418). That one-ratio spread is two-thirds of the
  entire 0.13 range that made 80/20 look like a winner.
- It was checked not to be clustering noise: holding a model fixed and varying
  only the GMM seed moves ARI by at most 0.023 (sd ≤ 0.009). The variance is
  upstream, in the VAE fit.
- Root cause: `scripts/run_training.py` set no `torch.manual_seed`, so weight
  init, dropout and shuffle order differed every run. `data.random_seed` only fed
  an sklearn split fallback that a tagged split never reaches.

Training is now seeded via an optional `train.seed`, verified bitwise
reproducible on both MPS and the remote CUDA device. Runs without the key keep
their historical unseeded behaviour and their exact model names, so every
pre-existing checkpoint still resolves.

## Two methodological checks worth keeping

**The validation sets are interchangeable.** Each rung's loss is measured on its
own val set, and those shrink from 12,339 to 2,469 events, which looks like it
should bias the comparison. Tested rather than assumed: re-evaluating every model
on the 2,469 events of the tr90 val set — a subset of every other rung's, because
the rungs nest — agrees with the own-set value to better than 0.5% at every rung
(0.19338 vs 0.19436 at tr50). No bias to correct.

**But the reconstruction metric matters a great deal.** The trend is present in
the *weighted* loss the VAE optimises (rho = −0.86, p < 10⁻⁵) and absent in the
unweighted per-pixel MSE (rho = −0.24 to −0.32, p = 0.10 to 0.23). Not a
contradiction: `src/losses/vae.py` upweights signal pixels 10×, whereas an
unweighted mean over 48×48 is dominated by empty background every model
reconstructs trivially. The unweighted number is too insensitive to see the
change. Panel (a) plots the weighted loss.

## What makes the x-axis readable as a dose

All nine ratios draw from the **same** fixed 8,227-per-species pool (24,681
events), and the training sets **nest** — verified, along with pool identity and
zero train/val overlap. So two rungs differ by how many images the model saw and
by nothing else: not which images, and not species mixture, since train and val
are each exactly balanced at every ratio. Without that, a bumpy curve could be a
different sample rather than a different sample size.

The balanced-at-every-ratio property has a second benefit: it removes the
species-composition confound that makes the pooled train-vs-val two-sample test
on the bal9419 split need mixture matching (see `docs/two_sample_summary.md`).

## Caveats to state if asked

- **Three seeds is few.** The sds are estimated from three points, so they are
  themselves noisy. They are large enough to make the flat verdicts safe, but a
  reader wanting the kaon-topology slope pinned down would want more.
- **Do not read the tr75 bump in panel (a)** as structure. It is 2222 ± 9 against
  2170 ± 16 at tr70, and although those bars do not overlap, it sits against an
  otherwise monotone decline of 142 units total; with nine ratios and three seeds,
  one excursion of this size is unremarkable.
- **Panels (c) and (d) are val-only**, so the measurement sample shrinks as the
  ratio rises. The widening error bars at high ratios are that, not instability.

## Reproducing

```bash
for pct in 50 55 60 65 70 75 80 85 90; do
  python scripts/make_balanced_split.py \
    --data /Volumes/easystore/proton-kaon/images/pkm_48x48_raw_10-179wires.pt \
    --splits-dir /Volumes/easystore/proton-kaon/training \
    --pool-per-species 8227 --train-frac 0.$pct --tag pool8227_tr$pct --seed 42
done

python scripts/run_sweep.py --sweep configs/sweep_split_pool8227_seeded.yaml \
    --remote configs/remote_all.yaml --gpu-devices auto --min-free-gpu-gb 8 \
    --resume --keep-going

python scripts/extra/plot_split_sweep.py \
    --sweep configs/sweep_split_pool8227_seeded.yaml --write-configs
for cfg in configs/generated_split_sweep/pool8227_tr*_seed*.yaml; do
  python scripts/run_inference.py --config "$cfg"
done
python scripts/extra/plot_split_sweep.py --sweep configs/sweep_split_pool8227_seeded.yaml
```

`run_sweep.py` rsyncs only the config, never the code — the remote repo must
already carry the seeding change, or `train.seed` is silently ignored, `_seed<N>`
never enters the model name, and the three seeds per ratio overwrite each other.
