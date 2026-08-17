# Latent-capacity sweep: how many dimensions does the physics need?

96 runs — latent dim 4 to 128 in steps of 4, three seeds each, data split fixed at
50/50 (12,342 train / 12,339 val, exactly species-balanced on both sides).

Figure: `figs/latent_sweep/latent_sweep.pdf`
Data: `figs/latent_sweep/latent_sweep_runs.csv`, `latent_sweep_summary.csv`

## Headline

Three different answers on three axes, which is what makes this worth reporting:

| axis | behaviour | Spearman vs latent dim |
|---|---|---|
| reconstruction | improves strongly, saturates by latent ~48 | rho = −0.93, p ~ 10⁻⁴¹ |
| unsupervised species structure (GMM) | **no capacity effect at all** | rho = −0.13, p = 0.20 |
| linear decodability of physics proxies | real, monotone, but small | rho = +0.58 to +0.94, all p < 10⁻⁹ |

**Contrast with the split sweep.** There the physics content was flat in the amount
of *data*. Here it is *not* flat in the amount of *capacity* — it improves. So the
representation is data-saturated but mildly capacity-limited, and those are
genuinely different statements about the same latent space.

## Nothing collapses, and that is a finding about the loss

Every latent dimension stays active at every latent dim tested: the conventional
`Var(mu) > 0.01` count sits exactly on the diagonal from 4 to 128, and even at
latent 128 no dimension falls near the threshold.

The reason is the loss balance, not the latent size. The reconstruction term is a
weighted MSE **summed** over 2×48×48 pixels while the KL is a mean over dimensions,
so at beta = 0.5 the KL is **0.5% of the objective at latent 4, rising only to 6.5%
at latent 128**. There is essentially no pressure to compress. This model is a
lightly regularised autoencoder rather than a tightly-constrained beta-VAE, and
posterior collapse never arises.

That invalidates the standard active-units diagnostic here, so the sweep uses the
**participation ratio** (sum var)²/sum(var²) instead, which is continuous and
threshold-free. It shows the effect the binary count cannot: effective dimensions
grow sub-linearly, with PR/latent falling from 0.95 at latent 4 to 0.55 at latent
128. Surplus capacity is *diluted across* dimensions, not switched off.

If a reviewer expects to see posterior collapse and does not, this is the answer,
and it is worth stating in the paper rather than leaving as an apparent anomaly.

## Reconstruction saturates around latent 48

2880 (latent 4) → 2307 (8) → 1493 (32) → 1374 (48) → 1312 (128). 95% of the total
gain is reached by **latent 48**; beyond that the curve is flat to within the
between-seed spread.

## The physics proxies improve — modestly

All six proxy AUCs rise monotonically and significantly, and the trend is robust:
restricting to latent >= 32 leaves every one of them at p < 10⁻⁵. So this is a real
capacity effect and not an artefact of latent 4 being too small.

But the effect sizes are small. Going from the paper's latent 8 to latent 128:

| proxy | latent 4 | latent 8 | latent 128 | 8 → 128 |
|---|---|---|---|---|
| calorimetry, proton | 0.916 | 0.951 | 0.972 | +0.021 |
| calorimetry, kaon | 0.792 | 0.853 | 0.885 | +0.033 |
| calorimetry, MIPs | 0.722 | 0.827 | 0.858 | +0.031 |
| topology, proton | 0.768 | 0.812 | 0.856 | +0.044 |
| topology, kaon | 0.915 | 0.896 | 0.901 | +0.005 |
| topology, MIPs | 0.770 | 0.771 | 0.791 | +0.019 |

Two to four AUC points for sixteen times the latent dimension. Statistically
unambiguous, practically marginal.

### The kaon topology curve is U-shaped

Topology AUC for kaons is *highest at latent 4* (0.915), falls to 0.841 by latent
40, then recovers to ~0.905 by latent 120. Its Spearman rho accordingly strengthens
from +0.58 over the full range to +0.91 when restricted to latent >= 32 — the early
decline and the later rise are fighting each other.

A plausible reading is that at latent 4 there is only room for the few most
dominant factors, and kaon solidity is one of them, so it is trivially linearly
accessible; as capacity grows the information spreads across dimensions and becomes
less linearly accessible before there is enough room to represent it cleanly again.
That is speculation offered to be tested, not a conclusion — the sweep establishes
the shape, not the mechanism.

## What this means for the paper's choice of latent 8

Latent 8 is **not** optimal on reconstruction (latent 48 is 40% better) and not
optimal on proxy decodability (latent 128 is 2–4 AUC points better). It is
defensible on parsimony and interpretability, and on the fact that the claim the
paper actually makes — that species structure emerges without labels — does not
improve with capacity at all. That is the argument to make; "8 was enough" is not
supported as a statement about reconstruction.

If a reviewer asks why not more dimensions, the honest answer is that more
dimensions buy better reconstruction and slightly better linear readout, but no
better unsupervised class structure, and cost interpretability.

## Caveats

- **The probe grows with the x-axis.** Proxy AUC is a logistic regression on the
  latents, so its feature count is the latent dim. Cross-validation means surplus
  dimensions carrying no information cannot inflate it, and at the 50/50 split each
  species has 4113 validation events, a comfortable 32:1 ratio even at latent 128.
  A *fall* at the top end would be weaker evidence than a rise, since added probe
  variance can only depress the AUC; no such fall is observed.
- **The clustering also grows.** A full-covariance k=3 mixture fits ~24,800
  covariance parameters at latent 128 against 25,418 events. It converges with
  sklearn's regularisation, but panel (d)'s occasional low outliers (latent 40, 48,
  104, 128) sit alongside large between-seed bars and are most likely bad fits
  rather than structure. `n_init` is held at 20 throughout, matching
  cluster_latents.py, so the numbers stay comparable to the paper model's.
- **Three seeds** per point. Enough to make the flat verdict on clustering safe and
  the monotone proxy trends unambiguous, but the sds are themselves estimated from
  three values.

## Reproducing

```bash
python scripts/run_sweep.py --sweep configs/sweep_latent_pool8227_tr50.yaml \
    --remote configs/remote_all.yaml --gpu-devices auto --min-free-gpu-gb 8 \
    --resume --keep-going

python scripts/extra/plot_latent_sweep.py --write-configs
for cfg in configs/generated_latent_sweep/*.yaml; do
  python scripts/run_inference.py --config "$cfg"
done
python scripts/extra/plot_latent_sweep.py
```

Run the inference loop **serially**. Two concurrent passes over the same configs
corrupt the `.npz` outputs, and the corruption is invisible to an existence check
because `np.load` on an `.npz` is lazy — it only verifies the CRC when an array is
actually read. Validate with a forced full read:

```bash
python -c "
import glob, numpy as np, os
for d in glob.glob('/Volumes/easystore/proton-kaon/inference/*/'):
    for n in ('train.npz','val.npz','kaon.npz','muon.npz','species_split.npz'):
        try:
            z = np.load(os.path.join(d, n))
            for k in z.files: _ = z[k][...].sum()
        except Exception as e: print('BAD', d, n, type(e).__name__)
"
```
