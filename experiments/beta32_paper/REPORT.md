# Paper reproduction with beta = 32

**Status: in progress; pending entries are not results.**

4/187 training runs complete; 4/187 representation summaries complete.

The reference is the nine-page `paper_reference.pdf` copied from the supplied `neurips.pdf`. This study concerns that paper, including Appendices A-C. It does not substitute the later event/run-disjoint baseline protocol.

## First completed main-model findings

- KL is 17.35% of the selected validation objective; beta=32 does not yield 50/50 after retraining.
- k=37 overall tag agreement changes from 81.78% to 79.09%.
- Kaon-group tag purity changes from 75.30% to 69.21%; the primary run does not retain the paper's above-75%-for-all-tags claim.
- The unsupervised kaon-window mass shift remains positive: +98.63 MeV versus +104.69 MeV historically.
- These are seed-zero main-model findings. The report below distinguishes the fresh paired control, additional seeds and still-pending appendix results.

## What was held fixed

Original 27,657 two-plane 48 x 48 images; log1p; four encoder blocks [32, 64, 128, 256]; latent dimension eight for the main model; weighted squared error summed over pixels with 10x weight above 0.01; summed Gaussian KL; Adam lr=0.001, weight_decay=0.0001; batch size 32; 200-epoch cap; stochastic validation; patience 20; min_delta=0.0001; restore the best accepted checkpoint. The original 9,419/18,238 split is copied byte-for-byte: train 3,139/3,140/3,140 and validation 7,327/5,087/5,824 for proton/kaon/MIPs. Labels remain absent from the VAE loss and unsupervised GMM fitting.

The only intended objective change is beta=0.5 to beta=32. The original main checkpoint was unseeded, so identical original initialization cannot be reproduced. New main runs use explicit seeds 0/1/2, with seed zero primary. A fresh beta=0.5 seed-zero control uses the same initialization and shuffle seed as the primary beta=32 run. Historical comparisons and paired comparisons are distinguished. Hardware is the Mac MPS GPU; bitwise equality to historical GPU runs is not claimed.

The isolated trainer was checked against the original trainer on an uneven-batch, two-epoch fixture: losses and selected model weights matched exactly. Input, split and original source hashes are recorded in `provenance.json`, `plan.json`, and `reference/sources.json`. Checkpoints retain optimizer and CPU/device/shuffle RNG states for epoch-level restart.

## Loss balance and reconstruction

The selected checkpoint is epoch 31. Its logged stochastic validation reconstruction is 2325.89, unweighted KL 15.26, beta x KL 488.30, and total loss 2814.19. KL contributes **17.35%**, not an assumed 50%.

| Quantity | Historical beta=0.5 | beta=32 seed 0 | Change |
|---|---:|---:|---:|
| Validation reconstruction (stochastic) | 2160.31 | 2325.89 | 165.57 |
| Validation KL | 59.71 | 15.26 | -44.45 |
| KL share | 0.0136 | 0.1735 | 0.1599 |
| Active mean coordinates (Var > 0.01) | 8 | 8 | 0 |
| Participation ratio | 7.744 | 7.837 | 0.094 |
| Total variance of posterior means | 66.55 | 8.33 | -58.22 |

Total objectives have different beta weights and should not be interpreted as a direct reconstruction ranking. Mean-decoded validation reconstruction and KL are also retained in `scan_metrics.json`; they are separate from the original sampled objective.

## Figures 1-3: species structure and physical probes

The detector example images and their interpretation are unchanged input data. New t-SNE projections use the original raw posterior means, perplexity 30, PCA initialization, automatic learning rate, 1,000 iterations and seed zero. Projections are fitted separately; positions and orientations are not comparable between models, and visual spacing is not a clustering metric.

The original six AUC values reproduce the paper to its displayed precision. New probes use the same within-species validation median split, five stratified folds, standardization inside each fold, and logistic reader. Intervals resample frozen out-of-fold predictions 2,000 times; they omit VAE training uncertainty.

| Proxy / tag | Historical AUC | beta=32 AUC | Change | beta=32 95% interval |
|---|---:|---:|---:|---|
| mean_adc / proton | 0.952 | 0.937 | -0.015 | 0.932-0.942 |
| mean_adc / kaon | 0.857 | 0.842 | -0.015 | 0.831-0.852 |
| mean_adc / muon | 0.823 | 0.825 | +0.002 | 0.815-0.836 |
| solidity / proton | 0.804 | 0.808 | +0.004 | 0.798-0.817 |
| solidity / kaon | 0.908 | 0.884 | -0.024 | 0.874-0.893 |
| solidity / muon | 0.788 | 0.776 | -0.012 | 0.764-0.787 |

4 of six linear proxy AUCs decrease in the primary historical comparison. This measures access to the named observed proxies within each tag, not true-species accuracy.

## Figure 4 and Appendix B: unsupervised clustering and mass corroboration

Full-covariance GMMs use raw, unstandardized posterior means and the original pooled training/validation rows, 20 starts, default covariance regularization, and seed zero. Cluster majority tags are read back after fitting. Their purities are descriptive agreement with imperfect beamline tags; they are not independently held-out naming accuracy.

| Quantity at k=37 | Historical beta=0.5 | beta=32 seed 0 | Change |
|---|---:|---:|---:|
| Overall tag agreement | 0.818 | 0.791 | -0.027 |
| Clusters at least 80% pure | 22 | 22 | 0 |
| Sample fraction in those clusters | 0.636 | 0.573 | -0.063 |
| Proton-group tag purity | 0.866 | 0.826 | -0.040 |
| Kaon-group tag purity | 0.753 | 0.692 | -0.061 |
| MIP-group tag purity | 0.819 | 0.854 | 0.035 |
| At least 90%-kaon clusters | 5 | 1 | -4 |
| Kaon candidates in those clusters | 2981 | 894 | -2087 |
| Combined kaon-tag purity | 0.937 | 0.968 | 0.031 |
| Kaon-window candidates in proton-majority clusters | 1304 | 1349 | 45 |
| Fraction of kaon window flagged | 0.159 | 0.164 | 0.005 |
| Median mass in proton-majority group [MeV] | 566.54 | 561.74 | -4.80 |
| Median mass in kaon-majority group [MeV] | 461.85 | 463.11 | 1.26 |
| Median mass shift [MeV] | 104.69 | 98.63 | -6.06 |

The baseline k=37 composition, five clean kaon clusters, 2,981 clean-cluster kaon candidates, and +104.69 MeV mass shift reproduce the paper. The beta=32 mass split is positive in the expected direction.

| Tag | Historical mass-versus-proton-fraction rho | beta=32 rho | beta=32 p | Qualifying clusters |
|---|---:|---:|---:|---:|
| proton | 0.457 | 0.512 | 0.005373 | 28 |
| kaon | 0.644 | 0.780 | 1.644e-06 | 27 |
| muon | -0.179 | -0.750 | 0.0003353 | 18 |

Each correlation uses clusters containing at least 50 candidates of the tag being tested. Under this explicit rule the historical proton correlation is rho=0.457, p=0.0217, so the paper's statement of no proton correlation is not reproduced by this check. The historical kaon rho=0.644 does reproduce the published 0.64. These p-values are descriptive and uncorrected for multiple comparisons.

### Table 1 and full k=3-50 scan

| k | Historical purity | beta=32 purity | beta=32 best kaon purity | beta=32 fraction in >=85% clusters | beta=32 smallest cluster |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.734 | 0.646 | 0.526 | 0.246 | 6805 |
| 8 | 0.729 | 0.698 | 0.615 | 0.269 | 1987 |
| 12 | 0.737 | 0.767 | 0.850 | 0.364 | 1086 |
| 15 | 0.803 | 0.766 | 0.807 | 0.297 | 1174 |
| 37 | 0.818 | 0.791 | 0.968 | 0.443 | 181 |
| 50 | 0.821 | 0.816 | 1.000 | 0.508 | 169 |

11/48 points in the full beta=32 cluster-count scan have completed.

### k=36-41 stability, five GMM seeds

| k | Mean mass shift [MeV] | Seed SD [MeV] | Mean pairwise ARI |
|---:|---:|---:|---:|
| 36 | 93.33 | 7.35 | 0.602 |
| 37 | 96.90 | 1.71 | 0.579 |
| 38 | 93.84 | 6.74 | 0.576 |
| 39 | 96.02 | 10.91 | 0.594 |
| 40 | 104.09 | 7.04 | 0.601 |
| 41 | 97.49 | 4.53 | 0.551 |

## Appendix A.1: training/validation distributions

Mixture-matched pooled C2ST MLP AUC = 0.5046 +/- 0.0027; energy permutation p = 0.6310; Holm-significant coordinates = 0/8. The historical paper reports 0.5026 +/- 0.0029, p=0.64, and 0/8. The original protocol uses 1,999 marginal/energy permutations, five energy draws, five classifier draws, 199 classifier null permutations per draw and five folds.

## Appendix A.2, Figures 5-6: training size and latent capacity

The full original grids are configured: 30 nested training sizes x three seeds, and latent dimensions 4,8,...,128 x three seeds at the fixed 12,342-image split. The three latent-eight/tr50 runs are shared by the grids. Together with three main seeds and the paired weak-beta control this is 187 unique runs.

The reconstruction panels use original sampled validation loss at the selected epoch. The six proxy AUCs use each split's validation rows. GMM k=3, active coordinates, 95%-variance coordinate count and participation ratio follow the original source. An extra common-validation weighted mean-decoding check is explicitly separate from the original metric.

Completed training-size summaries: 0/90. Completed capacity-only summaries: 0/93 (plus the three shared tr50/latent-eight runs).

Until each grid is complete, neither the original few-thousand-image plateau nor stability over capacity nor the original 48-dimensional reconstruction saturation is established at beta=32. Stronger regularization could change all three.

## Appendix C, Figures 7-8: beamline context and anchored comparison

Figure 7 is independent of the VAE. Its measured beamline spectrum, 4,009-event Poisson fit, 488.3 +/- 1.2 MeV kaon peak, 36.2 MeV width, chi-squared/dof=0.91 and fitted 61.2%/25.1%/13.8% kaon/proton/light window fractions remain unchanged. They precede the analysis selection and remain contextual, rather than a measured composition of the selected 8,227 kaon candidates. No refit is needed to change beta.

At beta=32 the anchored comparison assigns 1,886 kaon-window candidates to the proton density, 5,540 to the kaon density and 801 to the MIP density. The proton-minus-kaon median mass shift is 88.83 MeV (paper: 1,692 proton assignments and +103.4 MeV).

The tag-conditioned injection-recovery slope is 0.469. Calibrated proton estimates are 14.61% for picky and 19.38% for non-picky (paper: 16.9% and 17.2%). These remain model-dependent density estimates, not truth labels.

The picky/non-picky calibrated gap is 4.77 percentage points. The original near-equality across the quality flag is not preserved in this primary density-model comparison.

## Paired control and interpretation

| Proxy / tag | Fresh beta=0.5 seed 0 | beta=32 seed 0 | Paired change |
|---|---:|---:|---:|
| mean_adc / proton | 0.954 | 0.937 | -0.017 |
| mean_adc / kaon | 0.869 | 0.842 | -0.026 |
| mean_adc / muon | 0.818 | 0.825 | +0.007 |
| solidity / proton | 0.814 | 0.808 | -0.006 |
| solidity / kaon | 0.901 | 0.884 | -0.017 |
| solidity / muon | 0.783 | 0.776 | -0.007 |

5/6 AUCs decrease in the same-seed comparison. The MIP calorimetry reader improves; the other five decrease. This distinguishes the beta intervention from the historical model's unknown initialization, while remaining one training-seed comparison.

| Quantity | Fresh beta=0.5 seed 0 | beta=32 seed 0 | Paired change |
|---|---:|---:|---:|
| Validation reconstruction | 2129.99 | 2325.89 | 195.89 |
| Validation KL | 59.41 | 15.26 | -44.15 |
| k=37 majority tag agreement | 0.824 | 0.791 | -0.033 |
| k=37 kaon-group tag purity | 0.754 | 0.692 | -0.062 |
| k=37 kaon candidates in >=90%-kaon clusters | 1668 | 894 | -774 |
| k=37 median mass shift [MeV] | 98.03 | 98.63 | 0.60 |

The same-seed mass shift is essentially retained, despite lower clustering agreement and lower kaon-group tag purity. The small historical mass-shift difference should not therefore be attributed entirely to beta.

The observational limitations of the paper persist: beamline tags are imperfect; mass defines the kaon window; pooled GMM descriptions reuse tags for naming and scoring; t-SNE is visualization; crop and selection can shape the populations. Statements of novelty and detector physics are contextual and cannot change numerically with beta. Stronger regularization must be assessed by the measured KL share and the completed proxy/clustering results, not by treating beta=32 as a guaranteed 50/50 objective.

## Reproduce and resume

From the repository root:

```sh
.venv/bin/python experiments/beta32_paper/test_study.py
.venv/bin/python experiments/beta32_paper/orchestrate.py
```

The driver resumes saved runs, evaluates completed checkpoints, updates this report, and creates local commits limited to this folder at the main-result and final milestones. Large input tensors, checkpoints, posterior arrays and intermediate caches stay local and are ignored by Git; configurations, hashes, numerical result tables, reference paper, report and final figures are committed. No existing paper figures or paper copies are replaced.
