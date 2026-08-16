# Train vs. validation latents: does the split hold up?

Short version for the 12 Aug meeting. Two tests, one figure, four comparisons.
**Every comparison comes back indistinguishable.** The one that does not is
confounded by construction and is explained below.

Figure: `figs/<model>/two-sample/paper/paper_two_sample.pdf`
Model: `run_0093`, latent dim 8, 9419 balanced training images (3139 p / 3140 k /
3140 MIPs), 18238 validation.

---

## Why this test exists, and what a positive result would mean

Train and val here are a **random split of one sample**. So under the null they
genuinely *are* the same distribution — there is no covariate shift to find. A
significant result would therefore not mean "the validation set is different"; it
would mean the *encoder* treats images it was trained on differently from images
it was not. That is memorisation. This is a check on the model, not on the data.

Two consequences worth stating in the paper:

1. **Significance is nearly free at this n.** With thousands of events per group,
   these tests can flag shifts far too small to matter physically. So every
   number below is paired with a *noise floor*: what the same statistic returns
   when the train/val labels are deliberately shuffled. The question is never
   "is p < 0.05" but "is the observed value outside the grey band".
2. **A null result is the desired result.** Which means the test has to be shown
   to have teeth — see the negative control at the bottom.

## The two tests

| | test | what it gives us | why this one |
|---|---|---|---|
| (a) | **C2ST** — classifier two-sample test | AUC of a small MLP trained to tell train latents from val latents, scored on held-out folds. 0.5 = nothing to separate. | This is the one you asked for: "train a binary classifier to distinguish train vs. val; inability to distinguish confirms same distribution". It is also the most adversarial test available — the classifier is free to find *any* structure that separates the two groups — and its answer is on a scale every reader already knows. |
| (b) | **Energy distance** (Székely–Rizzo), permutation-calibrated | The formal joint test. Zero if and only if the two distributions are identical, in any number of dimensions, and its estimator is unbiased under the null. | This is the p-value to quote. It has the clean definition of good and bad that you wanted, and unlike the per-dimension KS tests it sees changes in the *correlation structure* between latent dimensions, not just per-axis shifts. |

**Earth mover / Wasserstein is dropped**, as agreed. In 8 dimensions the
empirical estimator converges at n^(−1/8), so what you compute is dominated by
finite-sample bias rather than by any real difference: two samples drawn from the
*identical* distribution give a comfortably non-zero distance. It has no
interpretable scale, which is exactly what energy distance does have. It is still
computed and sits in `summary.csv` so a reviewer question can be answered without
a re-run, but it is out of the figure and out of the text.

**The KS tests stay, in the supplement.** Per-dimension KS + Holm correction is
the right tool for naming *which* latent dimension is responsible if a joint test
ever fires. As a headline it invites a multiplicity argument the joint tests
don't need. Plots: `ecdf_diff_<comparison>.pdf`, `marginal_effects.pdf`.

## Results

Final run (`two-sample/paper/`): 1999 permutations for the marginal and energy
nulls, 5 subsample draws each for energy and C2ST, 199 permutation nulls per C2ST
draw. AUC is the mean over the draws; the ± is the spread across them. The null sd
is empirical, from the permutation null, not the analytic Hanley–McNeil formula —
the analytic one assumes away the refit instability that moves a cross-validated
AUC most, and comes out ~1.5× too narrow here.

| comparison | n train / val | C2ST AUC (MLP) | vs. chance | energy p | KS dims flagged | verdict |
|---|---|---|---|---|---|---|
| All species (matched) | 9419 / 15259 | 0.5026 ± 0.0029 | +0.6 sd | 0.64 | 0 of 8 | indistinguishable |
| Proton | 3139 / 7327 | 0.4911 ± 0.0031 | −1.1 sd | 1.00 | 0 of 8 | indistinguishable |
| Kaon | 3140 / 5087 | 0.5053 ± 0.0046 | +0.7 sd | 0.42 | 0 of 8 | indistinguishable |
| MIPs | 3140 / 5824 | 0.4956 ± 0.0050 | −0.5 sd | 0.67 | 0 of 8 | indistinguishable |
| *All species (unmatched)* | *9419 / 18238* | *0.5230 ± 0.0018* | *+5.1 sd* | *0.001* | *2 of 8* | **confounded — see below** |

A logistic-regression classifier was run alongside the MLP and agrees throughout
(0.5050, 0.4928, 0.5019, 0.5042 on the four matched rows; 0.5311 on the unmatched
one). So the null result is not an artefact of one model class being too weak to
find the difference.

Largest per-dimension displacement anywhere in the matched comparisons: **0.03 σ**
(all species) to 0.06 σ (kaon), against a same-distribution noise floor of the
same size. Nothing that could matter physically.

These numbers reproduce an earlier 999-permutation pass (`two-sample/default/`)
on every qualitative point; the p-values move around, as p-values with no effect
behind them do, but no verdict changes and no AUC moves by more than 0.006.

### The unmatched row is a composition artefact, not memorisation

This is the one number that looks bad, and it is worth being able to explain it
on the spot. The balanced split makes the **training** set exact thirds
(3139/3140/3140) and dumps *everything* left over into validation — and the three
species have different totals (10466 p, 8227 k, 8964 MIPs). So the two pools have
different species mixtures:

- train: 33.3% p / 33.3% k / 33.3% MIPs
- val: 40.2% p / 27.9% k / 31.9% MIPs

Species are separable in latent space **by construction** — that is the whole
result of the paper. So pooling train and val without matching the mixture
compares 33% kaons against 28% kaons, and the test correctly reports that those
differ. It has nothing to do with train/val membership. `All species (matched)`
resamples the validation pool to the training mixture first; that is the row to
quote, and it is null.

**The split sweep fixes this by construction.** The new `pool8227_tr*` splits take
8227 events from each species and split *that* pool, so train and val are each
exactly balanced at every ratio and the matched/unmatched distinction disappears.

### The test has teeth

Two checks that the null result is real rather than the pipeline failing quietly:

- **Negative control** (`two-sample/control/`): train/val labels reshuffled, so
  the null is true by construction. Comes back clean, as it must.
- **The unmatched row fires.** A 6.9-percentage-point shift in species mixture
  produces AUC 0.523 at p = 0.001, +5.1 null sd, and 2 of 8 KS dimensions
  Holm-significant. So the machinery does detect a real difference of a size we
  can name — it simply does not detect one between train and val.

## Suggested wording for the paper

> Train and validation sets are a random split of a single sample, so a detected
> difference would indicate encoder memorisation rather than covariate shift. A
> classifier trained to distinguish train from validation latents achieves
> AUC = 0.503 (permutation null 0.500 ± 0.005), and the permutation-calibrated
> energy distance does not reject equality of the two distributions (p = 0.64);
> the same holds for each species individually (AUC 0.491–0.505, energy
> p = 0.42–1.00). The largest per-dimension displacement is 0.03 σ, within the
> same-distribution noise floor. Species composition was matched between the two
> pools before pooling, since the balanced training split leaves the validation
> remainder unbalanced and species are separable in the latent space by
> construction.

## Reproducing

```bash
python scripts/latent_two_sample.py \
    --config configs/run_0093_all_species_bal9419_*.yaml \
    --tests ks energy c2st --label paper \
    --n-perm 1999 --energy-repeats 5 --c2st-perm 199 --c2st-repeats 5
```

Re-render the figures from a finished run without recomputing anything:

```bash
python scripts/plot_two_sample.py --results figs/<model>/two-sample/paper/results.json
```

Full auto-generated detail, including the per-dimension KS table and the
interpretation notes, is in `report.md` next to each `results.json`.
