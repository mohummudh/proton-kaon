# PILArNet simulation truth in the frozen LArIAT latent space

This is a bounded diagnostic using **2,500 particles** (photon: 500, pion: 500, muon: 500, proton: 500, electron: 500), selected from the first 1,200 events of one checksum-verified shard. No kaons are available. The frozen paper VAE is run 0093, with eight latent dimensions; all predictions use its posterior mean. The response was calibrated on protons and remains exploratory for other species.

## Main findings

- Species and coarse track/shower information are recoverable. The nonlinear species probe reaches 72.3% balanced accuracy (event-bootstrap 95% interval 68.9%–75.3%); the linear probe reaches 61.1%. Five-species chance is 20%. Simple image summaries reach 78.2% with the same linear classifier.
- Proton incoming energy is recoverable, but incomplete: log-energy R² is 0.615 with a linear probe and 0.701 with trees. The nonlinear mean absolute error is 18.2 MeV on 103 held-out protons. Image summaries reach 0.834 with a linear probe.
- Fine proton endpoint calorimetry is poorly recoverable from the latent vectors. Endpoint dE/dx R² is 0.008 (linear) and 0.016 (trees), versus 0.488/0.591 from image summaries. The proton Bragg ratio gives 0.068/0.104 in the latent, versus 0.599 from linear image summaries. This is evidence of a useful image-level signal that these latent readouts fail to retain, rather than proof that no conceivable readout can recover it.
- Full muon incoming energy is weakly recoverable: linear R² 0.010; nonlinear 0.300. Only 29.8% of its full deposited energy is geometrically retained on average, compared with 98.7% for protons. Endpoint-only images cannot supply an unrestricted calorimetric energy measurement.
- Electron/photon deposited energy and broad transverse spread are recoverable. Pions are the hardest class: nonlinear recall is 37%, with 33% assigned to muons. Electron incoming-energy truth is accepted for 56.2% of the sample, photon truth for 32.0%; their incoming-energy results are conditional on this metadata-quality subset.
- Original vertices/directions are not recovered by linear probes. They are deliberately removed by independent particle placement. Same-interaction pairing reaches only AUROC 0.591, versus 1.000 using original vertices. Isolated endpoint vectors are insufficient for reconstructing the original event without retaining its geometry.
- The MC and real proton latents are strongly distinguishable (unmatched AUROC 0.944). This is not yet a validated simulation-to-real transfer test; differences in incoming/TPC energy, selection and detector response remain.
- Bethe–Bloch truth diagnostic: median proton dE/dx/model ratio 0.988, median absolute log-profile deviation 0.023. The density-ratio readout gives latent tree R² 0.019, versus 0.578 from image summaries. All three control VAE seeds and the AE also poorly recover proton endpoint dE/dx; this study does not isolate which architecture, bottleneck or loss choice causes it.

## Protocol and leakage controls

Source: `train/generic_v2_77600_v2.h5`; published SHA-256 `3d48f9a6fed80ebe479d85ad3c88604430b480d8c08ee43a896b88c66dedbf00`.
Frozen checkpoint SHA-256: `51fa1194fad6432e50c44647253758e195bf434306976de87ef6570cf2539e33`.
Frozen response SHA-256: `37713c1c59af6aceb625f1c098f82ff5468ac362180abcbf07519ea0d76bcaa3`.

One response, recombination rule and fit-only LArIAT direction bank are used for every species. No per-species image tuning, simulated model training or checkpoint modification is performed. Candidates are randomized and capped at 500 accepted particles per species; 349 attempted candidates fail the conversion/quality cuts. Selection requires 20–10,000 voxels, at least five occupied collection rows, and a dominant connected component carrying at least half the positive signal in each plane. These cuts bias the accepted shower/secondary sample. The same source event always occupies one 60/20/20 probe partition. Scaler/PCA fits use probe-training events only; regularization is chosen on development events. Test labels are used for reporting and explicitly labelled matched-population diagnostics only. CIs use 150 event-cluster bootstrap draws; this is an exploratory pilot with many correlated targets.

| species | dev | test | train |
| --- | --- | --- | --- |
| proton | 120 | 103 | 277 |
| pion | 113 | 112 | 275 |
| muon | 99 | 107 | 294 |
| electron | 110 | 106 | 284 |
| photon | 93 | 113 | 294 |

Incoming KE = sqrt(p²+m²)−m with the recorded momentum convention GeV/c→MeV/c. First-fragment kinematics are used only if fragment momentum spread ≤5%, KE>0 and deposited energy≤1.15 KE. This is a consistency filter, not independent validation of generator metadata. dx numerically follows the validated cm convention in the conversion; the dataset-card mm description is not used. Retained energy is a mean two-plane voxel-centre geometric proxy; it is not exact waveform ancestry. Track endpoint targets require >90% unique deposition times; shower sum(dx) is collective path, not shower length. 3D linearity and width come from unweighted voxel covariance. Bethe–Bloch scores use the existing proton-deuteron proton table without shifting or changing energy labels.

## Held-out classification

| task | representation | probe | balanced_accuracy | low | high | n_test |
| --- | --- | --- | --- | --- | --- | --- |
| species | paper_vae | logistic | 0.611 | 0.569 | 0.640 | 541 |
| species | paper_vae | permuted_train | 0.202 | — | — | 541 |
| species | paper_vae | trees | 0.723 | 0.689 | 0.753 | 541 |
| species | proton_vae | logistic | 0.478 | 0.444 | 0.514 | 541 |
| species | vae_s0 | logistic | 0.598 | 0.560 | 0.628 | 541 |
| species | vae_s1 | logistic | 0.601 | 0.557 | 0.642 | 541 |
| species | vae_s2 | logistic | 0.598 | 0.558 | 0.631 | 541 |
| species | ae_s0 | logistic | 0.624 | 0.580 | 0.654 | 541 |
| species | random_s0 | logistic | 0.293 | 0.255 | 0.328 | 541 |
| species | input_pca8 | logistic | 0.521 | 0.484 | 0.558 | 541 |
| species | image_summaries | logistic | 0.782 | 0.754 | 0.813 | 541 |
| species | pixels | logistic | 0.600 | 0.554 | 0.638 | 541 |
| species | retained_energy_oracle | logistic | 0.420 | 0.391 | 0.452 | 541 |
| semantic | paper_vae | logistic | 0.757 | 0.695 | 0.811 | 541 |
| semantic | paper_vae | trees | 0.826 | 0.766 | 0.871 | 541 |
| semantic | proton_vae | logistic | 0.617 | 0.541 | 0.677 | 541 |
| semantic | vae_s0 | logistic | 0.747 | 0.691 | 0.791 | 541 |
| semantic | vae_s1 | logistic | 0.781 | 0.730 | 0.829 | 541 |
| semantic | vae_s2 | logistic | 0.752 | 0.683 | 0.800 | 541 |
| semantic | ae_s0 | logistic | 0.799 | 0.745 | 0.841 | 541 |
| semantic | random_s0 | logistic | 0.543 | 0.480 | 0.601 | 541 |
| semantic | input_pca8 | logistic | 0.732 | 0.673 | 0.782 | 541 |
| semantic | image_summaries | logistic | 0.880 | 0.827 | 0.926 | 541 |
| semantic | pixels | logistic | 0.696 | 0.625 | 0.767 | 541 |
| semantic | retained_energy_oracle | logistic | 0.675 | 0.616 | 0.725 | 541 |

Semantic classification is the dominant **truth** label of an isolated particle (shower/track/Michel/delta), not pixel segmentation. Species may be inferred partly from their generated energy distributions. The five-class energy-balanced diagnostic has insufficient common retained-energy support; the energy-only oracle is reported to expose that shortcut.

## Within-species truth readouts

R² uses log1p(target) for energy, momentum, dE/dx, sizes and ratios; 3D linearity, fractions, angles and positions use their original scale. Physical-unit R²/MAE and support are in the CSV. R²≤0 means no advantage over the held-out target mean. Linear and nonlinear probes answer different accessibility questions. Input PCA has eight dimensions; image summaries have 32, and raw pixels have 4,608. A stronger summary baseline does not isolate latent dimension from architecture/objective effects.

### Proton

| target | probe | n_test | r2 | r2_low | r2_high | physical_mae |
| --- | --- | --- | --- | --- | --- | --- |
| incoming_ke_mev | ridge | 103 | 0.615 | 0.524 | 0.705 | 22.430 |
| incoming_ke_mev | trees | 103 | 0.701 | 0.587 | 0.814 | 18.174 |
| momentum_mev | ridge | 103 | 0.609 | 0.516 | 0.703 | 44.509 |
| momentum_mev | trees | 103 | 0.692 | 0.578 | 0.810 | 35.666 |
| deposited_mev | ridge | 103 | 0.555 | 0.434 | 0.682 | 19.867 |
| deposited_mev | trees | 103 | 0.661 | 0.529 | 0.780 | 15.103 |
| geom_retained_energy_mev | ridge | 103 | 0.566 | 0.445 | 0.711 | 16.678 |
| geom_retained_energy_mev | trees | 103 | 0.692 | 0.569 | 0.826 | 11.838 |
| geom_retained_electrons | ridge | 103 | 0.716 | 0.641 | 0.792 | 328847.272 |
| geom_retained_electrons | trees | 103 | 0.857 | 0.791 | 0.911 | 217834.407 |
| path_cm | ridge | 103 | 0.668 | 0.568 | 0.771 | 3.225 |
| path_cm | trees | 103 | 0.785 | 0.632 | 0.902 | 2.357 |
| mean_dedx_mev_cm | ridge | 103 | 0.467 | 0.332 | 0.572 | 1.529 |
| mean_dedx_mev_cm | trees | 103 | 0.539 | 0.405 | 0.657 | 1.272 |
| endpoint_dedx_mev_cm | ridge | 103 | 0.008 | -0.039 | 0.030 | 2.808 |
| endpoint_dedx_mev_cm | trees | 103 | 0.016 | -0.086 | 0.123 | 2.576 |
| bragg_ratio | ridge | 86 | 0.068 | -0.015 | 0.146 | 0.326 |
| bragg_ratio | trees | 86 | 0.104 | -0.022 | 0.270 | 0.297 |
| linearity_3d | ridge | 103 | 0.383 | 0.269 | 0.479 | 0.003 |
| linearity_3d | trees | 103 | 0.647 | 0.488 | 0.764 | 0.002 |
| transverse_rms_cm | ridge | 103 | 0.160 | 0.064 | 0.316 | 0.031 |
| transverse_rms_cm | trees | 103 | 0.232 | 0.144 | 0.350 | 0.030 |
| proton_bb_density_ratio | ridge | 103 | -0.001 | -0.048 | 0.021 | 0.055 |
| proton_bb_density_ratio | trees | 103 | 0.019 | -0.128 | 0.114 | 0.053 |
| proton_bb_log_distance | ridge | 103 | 0.001 | -0.045 | 0.025 | 0.063 |
| proton_bb_log_distance | trees | 103 | 0.013 | -0.142 | 0.127 | 0.062 |
| proton_bb_best_rr_offset_cm | ridge | 103 | 0.000 | -0.051 | 0.031 | 0.939 |
| proton_bb_best_rr_offset_cm | trees | 103 | 0.020 | -0.076 | 0.120 | 0.931 |
| extent_3d_cm | ridge | 103 | 0.663 | 0.560 | 0.768 | 3.239 |
| extent_3d_cm | trees | 103 | 0.784 | 0.629 | 0.903 | 2.361 |
| chord_over_path | ridge | 103 | -0.006 | -0.245 | -0.002 | 0.016 |
| chord_over_path | trees | 103 | -0.004 | -0.489 | 0.027 | 0.017 |
| visible_charge_fraction | ridge | 103 | -0.015 | -0.305 | 0.052 | 0.010 |
| visible_charge_fraction | trees | 103 | -0.030 | -1.088 | 0.109 | 0.010 |
| geometric_energy_fraction | ridge | 103 | 0.031 | -0.076 | 0.171 | 0.026 |
| geometric_energy_fraction | trees | 103 | 0.088 | -0.023 | 0.265 | 0.025 |
| assigned_xz_deg | ridge | 103 | 0.219 | 0.049 | 0.322 | 1.776 |
| assigned_xz_deg | trees | 103 | 0.340 | 0.225 | 0.415 | 1.585 |
| assigned_yz_deg | ridge | 103 | -0.022 | -0.112 | 0.070 | 3.907 |
| assigned_yz_deg | trees | 103 | 0.060 | -0.052 | 0.166 | 3.723 |
| vertex_x_cm | ridge | 103 | 0.001 | -0.077 | 0.025 | 26.960 |
| vertex_x_cm | trees | 103 | -0.029 | -0.117 | 0.028 | 27.250 |
| vertex_y_cm | ridge | 103 | -0.014 | -0.167 | 0.034 | 54.943 |
| vertex_y_cm | trees | 103 | -0.027 | -0.145 | 0.039 | 55.120 |
| vertex_z_cm | ridge | 103 | -0.092 | -0.237 | 0.003 | 51.636 |
| vertex_z_cm | trees | 103 | -0.066 | -0.249 | 0.054 | 51.173 |
| source_direction_x | ridge | 103 | -0.106 | -0.227 | -0.020 | 0.510 |
| source_direction_x | trees | 103 | 0.008 | -0.093 | 0.078 | 0.489 |
| source_direction_y | ridge | 103 | -0.024 | -0.115 | 0.028 | 0.527 |
| source_direction_y | trees | 103 | -0.077 | -0.177 | 0.005 | 0.534 |
| source_direction_z | ridge | 103 | -0.024 | -0.120 | 0.027 | 0.497 |
| source_direction_z | trees | 103 | -0.037 | -0.147 | 0.042 | 0.497 |

### Pion

| target | probe | n_test | r2 | r2_low | r2_high | physical_mae |
| --- | --- | --- | --- | --- | --- | --- |
| incoming_ke_mev | ridge | 112 | 0.290 | 0.074 | 0.437 | 87.402 |
| incoming_ke_mev | trees | 112 | 0.465 | 0.316 | 0.573 | 75.756 |
| momentum_mev | ridge | 112 | 0.272 | 0.061 | 0.421 | 96.114 |
| momentum_mev | trees | 112 | 0.435 | 0.281 | 0.548 | 84.023 |
| deposited_mev | ridge | 112 | 0.470 | 0.306 | 0.561 | 35.799 |
| deposited_mev | trees | 112 | 0.686 | 0.606 | 0.758 | 33.553 |
| geom_retained_energy_mev | ridge | 112 | 0.314 | 0.215 | 0.379 | 14.551 |
| geom_retained_energy_mev | trees | 112 | 0.719 | 0.626 | 0.780 | 10.875 |
| geom_retained_electrons | ridge | 112 | 0.489 | 0.273 | 0.614 | 303797.256 |
| geom_retained_electrons | trees | 112 | 0.802 | 0.724 | 0.848 | 225682.484 |
| path_cm | ridge | 112 | 0.591 | 0.443 | 0.684 | 15.440 |
| path_cm | trees | 112 | 0.713 | 0.637 | 0.785 | 14.656 |
| mean_dedx_mev_cm | ridge | 112 | 0.256 | 0.064 | 0.383 | 0.844 |
| mean_dedx_mev_cm | trees | 112 | 0.483 | 0.298 | 0.581 | 0.692 |
| endpoint_dedx_mev_cm | ridge | 112 | 0.086 | -0.086 | 0.221 | 3.027 |
| endpoint_dedx_mev_cm | trees | 112 | 0.398 | 0.314 | 0.471 | 2.491 |
| bragg_ratio | ridge | 108 | 0.048 | -0.149 | 0.182 | 0.699 |
| bragg_ratio | trees | 108 | 0.341 | 0.232 | 0.410 | 0.594 |
| linearity_3d | ridge | 112 | 0.092 | -0.133 | 0.248 | 0.003 |
| linearity_3d | trees | 112 | 0.287 | 0.194 | 0.585 | 0.002 |
| transverse_rms_cm | ridge | 112 | 0.420 | 0.221 | 0.600 | 0.195 |
| transverse_rms_cm | trees | 112 | 0.468 | 0.258 | 0.651 | 0.178 |
| extent_3d_cm | ridge | 112 | 0.588 | 0.442 | 0.680 | 14.958 |
| extent_3d_cm | trees | 112 | 0.715 | 0.637 | 0.784 | 14.117 |
| chord_over_path | ridge | 112 | -0.040 | -0.108 | 0.001 | 0.023 |
| chord_over_path | trees | 112 | -0.054 | -0.129 | 0.014 | 0.023 |
| visible_charge_fraction | ridge | 112 | -0.014 | -0.094 | 0.073 | 0.049 |
| visible_charge_fraction | trees | 112 | -0.024 | -0.105 | 0.091 | 0.049 |
| geometric_energy_fraction | ridge | 112 | 0.328 | 0.168 | 0.500 | 0.169 |
| geometric_energy_fraction | trees | 112 | 0.378 | 0.268 | 0.492 | 0.154 |
| assigned_xz_deg | ridge | 112 | 0.117 | -0.013 | 0.220 | 1.720 |
| assigned_xz_deg | trees | 112 | 0.121 | -0.082 | 0.242 | 1.684 |
| assigned_yz_deg | ridge | 112 | -0.051 | -0.167 | 0.019 | 3.448 |
| assigned_yz_deg | trees | 112 | -0.088 | -0.224 | 0.016 | 3.420 |
| vertex_x_cm | ridge | 112 | -0.011 | -0.096 | 0.031 | 25.180 |
| vertex_x_cm | trees | 112 | -0.019 | -0.121 | 0.047 | 25.120 |
| vertex_y_cm | ridge | 112 | -0.022 | -0.139 | 0.023 | 45.486 |
| vertex_y_cm | trees | 112 | -0.055 | -0.200 | 0.014 | 46.748 |
| vertex_z_cm | ridge | 112 | 0.001 | -0.081 | 0.026 | 48.167 |
| vertex_z_cm | trees | 112 | -0.004 | -0.119 | 0.047 | 48.048 |
| source_direction_x | ridge | 112 | -0.042 | -0.140 | 0.003 | 0.454 |
| source_direction_x | trees | 112 | -0.072 | -0.199 | 0.029 | 0.460 |
| source_direction_y | ridge | 112 | -0.057 | -0.217 | 0.019 | 0.596 |
| source_direction_y | trees | 112 | -0.091 | -0.237 | -0.011 | 0.601 |
| source_direction_z | ridge | 112 | -0.028 | -0.127 | 0.040 | 0.455 |
| source_direction_z | trees | 112 | -0.046 | -0.199 | 0.069 | 0.457 |

### Muon

| target | probe | n_test | r2 | r2_low | r2_high | physical_mae |
| --- | --- | --- | --- | --- | --- | --- |
| incoming_ke_mev | ridge | 107 | 0.010 | -0.206 | 0.194 | 524.287 |
| incoming_ke_mev | trees | 107 | 0.300 | 0.178 | 0.385 | 431.208 |
| momentum_mev | ridge | 107 | -0.004 | -0.218 | 0.190 | 520.271 |
| momentum_mev | trees | 107 | 0.296 | 0.166 | 0.387 | 436.656 |
| deposited_mev | ridge | 107 | 0.190 | 0.006 | 0.355 | 113.174 |
| deposited_mev | trees | 107 | 0.192 | 0.043 | 0.305 | 114.417 |
| geom_retained_energy_mev | ridge | 107 | 0.120 | -0.085 | 0.245 | 6.073 |
| geom_retained_energy_mev | trees | 107 | 0.326 | 0.210 | 0.425 | 5.531 |
| geom_retained_electrons | ridge | 107 | 0.244 | -0.466 | 0.589 | 137295.427 |
| geom_retained_electrons | trees | 107 | 0.461 | 0.241 | 0.622 | 126945.649 |
| path_cm | ridge | 107 | 0.231 | 0.025 | 0.397 | 57.889 |
| path_cm | trees | 107 | 0.283 | 0.138 | 0.392 | 56.324 |
| mean_dedx_mev_cm | ridge | 107 | 0.179 | -0.019 | 0.303 | 0.309 |
| mean_dedx_mev_cm | trees | 107 | 0.376 | 0.246 | 0.469 | 0.255 |
| endpoint_dedx_mev_cm | ridge | 107 | -0.054 | -0.372 | 0.198 | 1.727 |
| endpoint_dedx_mev_cm | trees | 107 | 0.244 | 0.100 | 0.369 | 1.574 |
| bragg_ratio | ridge | 107 | -0.007 | -0.172 | 0.126 | 0.497 |
| bragg_ratio | trees | 107 | 0.170 | 0.022 | 0.296 | 0.452 |
| linearity_3d | ridge | 107 | 0.180 | -0.104 | 0.434 | 0.001 |
| linearity_3d | trees | 107 | 0.451 | 0.242 | 0.606 | 0.001 |
| transverse_rms_cm | ridge | 107 | 0.055 | -0.111 | 0.143 | 0.371 |
| transverse_rms_cm | trees | 107 | 0.074 | -0.132 | 0.169 | 0.368 |
| extent_3d_cm | ridge | 107 | 0.230 | 0.025 | 0.396 | 55.496 |
| extent_3d_cm | trees | 107 | 0.285 | 0.142 | 0.393 | 53.865 |
| chord_over_path | ridge | 107 | -0.014 | -0.088 | 0.034 | 0.015 |
| chord_over_path | trees | 107 | -0.003 | -0.174 | 0.099 | 0.015 |
| visible_charge_fraction | ridge | 107 | 0.030 | -0.034 | 0.074 | 0.211 |
| visible_charge_fraction | trees | 107 | -0.030 | -0.147 | 0.057 | 0.215 |
| geometric_energy_fraction | ridge | 107 | 0.231 | 0.065 | 0.373 | 0.179 |
| geometric_energy_fraction | trees | 107 | 0.308 | 0.193 | 0.400 | 0.169 |
| assigned_xz_deg | ridge | 107 | 0.200 | -0.014 | 0.320 | 1.763 |
| assigned_xz_deg | trees | 107 | 0.184 | 0.018 | 0.291 | 1.741 |
| assigned_yz_deg | ridge | 107 | 0.042 | -0.060 | 0.100 | 3.670 |
| assigned_yz_deg | trees | 107 | 0.211 | 0.076 | 0.293 | 3.363 |
| vertex_x_cm | ridge | 107 | -0.063 | -0.167 | -0.000 | 34.114 |
| vertex_x_cm | trees | 107 | -0.038 | -0.164 | 0.045 | 33.131 |
| vertex_y_cm | ridge | 107 | -0.096 | -0.246 | -0.008 | 73.068 |
| vertex_y_cm | trees | 107 | -0.141 | -0.304 | -0.015 | 74.688 |
| vertex_z_cm | ridge | 107 | 0.033 | -0.035 | 0.090 | 56.848 |
| vertex_z_cm | trees | 107 | -0.045 | -0.174 | 0.036 | 59.616 |
| source_direction_x | ridge | 107 | -0.029 | -0.107 | 0.004 | 0.481 |
| source_direction_x | trees | 107 | -0.038 | -0.142 | 0.026 | 0.482 |
| source_direction_y | ridge | 107 | -0.006 | -0.094 | 0.033 | 0.487 |
| source_direction_y | trees | 107 | -0.039 | -0.148 | 0.027 | 0.488 |
| source_direction_z | ridge | 107 | -0.085 | -0.221 | -0.023 | 0.535 |
| source_direction_z | trees | 107 | -0.061 | -0.217 | 0.027 | 0.522 |

### Electron

| target | probe | n_test | r2 | r2_low | r2_high | physical_mae |
| --- | --- | --- | --- | --- | --- | --- |
| incoming_ke_mev | ridge | 57 | 0.561 | 0.311 | 0.626 | 30.507 |
| incoming_ke_mev | trees | 57 | 0.599 | 0.521 | 0.769 | 30.145 |
| momentum_mev | ridge | 57 | 0.561 | 0.309 | 0.625 | 30.504 |
| momentum_mev | trees | 57 | 0.596 | 0.518 | 0.770 | 30.128 |
| deposited_mev | ridge | 106 | 0.898 | 0.860 | 0.931 | 261.590 |
| deposited_mev | trees | 106 | 0.932 | 0.908 | 0.957 | 249.847 |
| geom_retained_energy_mev | ridge | 106 | 0.910 | 0.871 | 0.938 | 39.253 |
| geom_retained_energy_mev | trees | 106 | 0.967 | 0.957 | 0.976 | 35.800 |
| geom_retained_electrons | ridge | 106 | 0.905 | 0.864 | 0.934 | 1188974.630 |
| geom_retained_electrons | trees | 106 | 0.967 | 0.956 | 0.975 | 1051661.843 |
| mean_dedx_mev_cm | ridge | 106 | 0.011 | -0.053 | 0.047 | 0.072 |
| mean_dedx_mev_cm | trees | 106 | 0.058 | -0.082 | 0.153 | 0.072 |
| linearity_3d | ridge | 106 | -0.036 | -0.206 | 0.029 | 0.031 |
| linearity_3d | trees | 106 | 0.201 | 0.028 | 0.302 | 0.026 |
| transverse_rms_cm | ridge | 106 | 0.775 | 0.714 | 0.832 | 0.743 |
| transverse_rms_cm | trees | 106 | 0.805 | 0.747 | 0.865 | 0.682 |
| extent_3d_cm | ridge | 106 | 0.698 | 0.625 | 0.773 | 22.553 |
| extent_3d_cm | trees | 106 | 0.792 | 0.721 | 0.856 | 19.012 |
| n_fragments | ridge | 106 | 0.812 | 0.737 | 0.866 | 4.167 |
| n_fragments | trees | 106 | 0.807 | 0.721 | 0.868 | 4.361 |
| shower_fraction | ridge | 106 | 0.653 | 0.485 | 0.746 | 0.209 |
| shower_fraction | trees | 106 | 0.718 | 0.540 | 0.825 | 0.156 |
| visible_charge_fraction | ridge | 106 | 0.045 | -0.072 | 0.162 | 0.075 |
| visible_charge_fraction | trees | 106 | 0.045 | -0.096 | 0.199 | 0.073 |
| geometric_energy_fraction | ridge | 106 | 0.694 | 0.585 | 0.768 | 0.144 |
| geometric_energy_fraction | trees | 106 | 0.698 | 0.592 | 0.787 | 0.142 |
| assigned_xz_deg | ridge | 106 | 0.013 | -0.092 | 0.079 | 1.996 |
| assigned_xz_deg | trees | 106 | 0.016 | -0.136 | 0.117 | 1.977 |
| assigned_yz_deg | ridge | 106 | -0.046 | -0.148 | 0.006 | 3.403 |
| assigned_yz_deg | trees | 106 | -0.107 | -0.243 | -0.024 | 3.473 |
| vertex_x_cm | ridge | 106 | -0.040 | -0.158 | -0.005 | 30.683 |
| vertex_x_cm | trees | 106 | -0.108 | -0.252 | -0.033 | 31.177 |
| vertex_y_cm | ridge | 106 | -0.017 | -0.091 | 0.029 | 56.443 |
| vertex_y_cm | trees | 106 | -0.008 | -0.113 | 0.085 | 56.663 |
| vertex_z_cm | ridge | 106 | -0.023 | -0.114 | 0.008 | 49.630 |
| vertex_z_cm | trees | 106 | -0.077 | -0.175 | -0.009 | 51.396 |
| source_direction_x | ridge | 106 | -0.097 | -0.220 | -0.019 | 0.522 |
| source_direction_x | trees | 106 | -0.063 | -0.203 | 0.008 | 0.520 |
| source_direction_y | ridge | 106 | -0.053 | -0.153 | 0.009 | 0.505 |
| source_direction_y | trees | 106 | -0.155 | -0.271 | -0.069 | 0.527 |
| source_direction_z | ridge | 106 | 0.031 | -0.032 | 0.059 | 0.496 |
| source_direction_z | trees | 106 | 0.027 | -0.063 | 0.082 | 0.493 |

### Photon

| target | probe | n_test | r2 | r2_low | r2_high | physical_mae |
| --- | --- | --- | --- | --- | --- | --- |
| incoming_ke_mev | ridge | 33 | 0.255 | 0.007 | 0.396 | 20.714 |
| incoming_ke_mev | trees | 33 | 0.328 | -0.023 | 0.558 | 20.863 |
| momentum_mev | ridge | 33 | 0.255 | 0.007 | 0.396 | 20.714 |
| momentum_mev | trees | 33 | 0.328 | -0.023 | 0.558 | 20.863 |
| deposited_mev | ridge | 113 | 0.647 | 0.579 | 0.712 | 17.563 |
| deposited_mev | trees | 113 | 0.724 | 0.669 | 0.774 | 16.443 |
| geom_retained_energy_mev | ridge | 113 | 0.743 | 0.665 | 0.807 | 9.867 |
| geom_retained_energy_mev | trees | 113 | 0.797 | 0.751 | 0.838 | 10.378 |
| geom_retained_electrons | ridge | 113 | 0.743 | 0.667 | 0.806 | 290667.686 |
| geom_retained_electrons | trees | 113 | 0.796 | 0.748 | 0.838 | 306885.623 |
| mean_dedx_mev_cm | ridge | 113 | -0.013 | -0.057 | 0.019 | 0.079 |
| mean_dedx_mev_cm | trees | 113 | 0.007 | -0.083 | 0.080 | 0.078 |
| linearity_3d | ridge | 113 | 0.079 | -0.088 | 0.126 | 0.046 |
| linearity_3d | trees | 113 | 0.134 | -0.129 | 0.208 | 0.043 |
| transverse_rms_cm | ridge | 113 | 0.376 | 0.260 | 0.505 | 0.701 |
| transverse_rms_cm | trees | 113 | 0.329 | 0.218 | 0.440 | 0.737 |
| extent_3d_cm | ridge | 113 | 0.248 | 0.131 | 0.338 | 22.230 |
| extent_3d_cm | trees | 113 | 0.250 | 0.077 | 0.386 | 22.038 |
| n_fragments | ridge | 113 | 0.302 | 0.176 | 0.383 | 0.811 |
| n_fragments | trees | 113 | 0.284 | 0.150 | 0.387 | 0.816 |
| visible_charge_fraction | ridge | 113 | 0.033 | -0.028 | 0.076 | 0.049 |
| visible_charge_fraction | trees | 113 | 0.073 | -0.053 | 0.129 | 0.048 |
| geometric_energy_fraction | ridge | 113 | 0.061 | -0.013 | 0.116 | 0.119 |
| geometric_energy_fraction | trees | 113 | 0.042 | -0.069 | 0.138 | 0.120 |
| assigned_xz_deg | ridge | 113 | 0.072 | -0.069 | 0.151 | 1.797 |
| assigned_xz_deg | trees | 113 | -0.036 | -0.165 | 0.050 | 1.843 |
| assigned_yz_deg | ridge | 113 | -0.090 | -0.239 | -0.007 | 3.302 |
| assigned_yz_deg | trees | 113 | -0.103 | -0.281 | -0.003 | 3.312 |
| vertex_x_cm | ridge | 113 | -0.063 | -0.173 | -0.006 | 28.414 |
| vertex_x_cm | trees | 113 | -0.082 | -0.227 | 0.002 | 28.600 |
| vertex_y_cm | ridge | 113 | -0.014 | -0.133 | 0.013 | 53.377 |
| vertex_y_cm | trees | 113 | -0.056 | -0.194 | 0.002 | 54.038 |
| vertex_z_cm | ridge | 113 | -0.024 | -0.094 | 0.009 | 45.526 |
| vertex_z_cm | trees | 113 | -0.036 | -0.123 | 0.018 | 45.683 |
| source_direction_x | ridge | 113 | -0.079 | -0.172 | -0.032 | 0.543 |
| source_direction_x | trees | 113 | -0.119 | -0.233 | -0.038 | 0.545 |
| source_direction_y | ridge | 113 | -0.026 | -0.133 | 0.002 | 0.487 |
| source_direction_y | trees | 113 | -0.044 | -0.163 | 0.018 | 0.487 |
| source_direction_z | ridge | 113 | -0.008 | -0.100 | 0.037 | 0.475 |
| source_direction_z | trees | 113 | -0.044 | -0.161 | 0.027 | 0.482 |

## Reconstruction and response sensitivity

The decoder is evaluated at the posterior mean. Reconstruction ADC is expm1(max(decoded log image,0)). Ratios below are descriptive image quantities, not calibrated charge or energy closure. The normalized wire-profile error compares row marginals on the same resized image grid; it is not a physical dE/dx measurement.

| species | plane | n_test | median_decoded_input_adc_sum_ratio | median_decoded_input_adc_peak_ratio | median_normalized_wire_profile_l1_error |
| --- | --- | --- | --- | --- | --- |
| proton | collection | 103 | 0.649 | 0.526 | 0.240 |
| proton | induction | 103 | 0.984 | 0.608 | 0.390 |
| pion | collection | 112 | 0.903 | 0.669 | 0.258 |
| pion | induction | 112 | 1.244 | 0.813 | 0.379 |
| muon | collection | 107 | 0.938 | 0.709 | 0.238 |
| muon | induction | 107 | 1.150 | 0.874 | 0.323 |
| electron | collection | 106 | 0.732 | 0.402 | 0.456 |
| electron | induction | 106 | 1.235 | 0.446 | 0.504 |
| photon | collection | 113 | 0.948 | 0.484 | 0.438 |
| photon | induction | 113 | 1.247 | 0.526 | 0.519 |

| species | variant | n | median_distance | median_relative_distance | pid_change_fraction | median_energy_probe_fractional_shift |
| --- | --- | --- | --- | --- | --- | --- |
| electron | angle_plus_2deg | 25 | 0.689 | 0.190 | 0.000 | 0.080 |
| electron | baseline | 25 | 0.000 | 0.000 | 0.000 | 0.000 |
| electron | gain_0.8 | 25 | 0.330 | 0.091 | 0.080 | -0.049 |
| electron | gain_1.2 | 25 | 0.288 | 0.079 | 0.000 | 0.014 |
| muon | angle_plus_2deg | 25 | 1.505 | 0.507 | 0.000 | -0.031 |
| muon | baseline | 25 | 0.000 | 0.000 | 0.000 | 0.000 |
| muon | gain_0.8 | 25 | 0.303 | 0.102 | 0.000 | 0.079 |
| muon | gain_1.2 | 25 | 0.256 | 0.086 | 0.000 | -0.071 |
| photon | angle_plus_2deg | 25 | 0.611 | 0.189 | 0.040 | 0.001 |
| photon | baseline | 25 | 0.000 | 0.000 | 0.000 | 0.000 |
| photon | gain_0.8 | 25 | 0.262 | 0.081 | 0.120 | -0.011 |
| photon | gain_1.2 | 25 | 0.221 | 0.068 | 0.040 | 0.012 |
| pion | angle_plus_2deg | 25 | 1.228 | 0.330 | 0.120 | -0.016 |
| pion | baseline | 25 | 0.000 | 0.000 | 0.000 | 0.000 |
| pion | gain_0.8 | 25 | 0.326 | 0.088 | 0.080 | 0.070 |
| pion | gain_1.2 | 25 | 0.260 | 0.070 | 0.040 | -0.057 |
| proton | angle_plus_2deg | 25 | 0.842 | 0.265 | 0.000 | 0.001 |
| proton | baseline | 25 | 0.000 | 0.000 | 0.000 | 0.000 |
| proton | gain_0.8 | 25 | 0.282 | 0.089 | 0.000 | 0.048 |
| proton | gain_1.2 | 25 | 0.232 | 0.073 | 0.000 | -0.032 |

Counterfactuals use 25 held-out particles per species. ±20% waveform gain is applied before threshold and recropping; a +2° xz rotation preserves the source deposits but can change volume/crop support. Distances are standardized on simulation probe-training latents and divided by the median distance between random distinct particles of the same species. These are sensitivity diagnostics, not proof of invariance. Regenerating every baseline reproduced its saved image and latent vector.

## Domain and original-event geometry

| species | auc_real_vs_simulation | low | high | n_real_test | n_simulation_test | note |
| --- | --- | --- | --- | --- | --- | --- |
| proton | 0.944 | 0.928 | 0.959 | 1500 | 103 | Unmatched distributions; LArIAT MIP reference is not pure muon truth. No cross-domain task-transfer claim. |
| muon | 0.780 | 0.735 | 0.838 | 1154 | 107 | Unmatched distributions; LArIAT MIP reference is not pure muon truth. No cross-domain task-transfer claim. |

| representation | n_test_pairs | n_test_events | auc | low | high | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| paper_vae | 589 | 160 | 0.591 | 0.548 | 0.647 | 0.569 |
| input_pca8 | 589 | 160 | 0.562 | 0.511 | 0.611 | 0.548 |
| image_summaries | 589 | 160 | 0.702 | 0.643 | 0.750 | 0.622 |
| original_vertex_oracle | 589 | 160 | 1.000 | 1.000 | 1.000 | 1.000 |

A second domain check equalizes real/MC counts within fixed 20 MeV incoming-KE bands in each event partition. Real labels use upstream beamline momentum; simulated labels refer to the generated particle. TPC energy, upstream material, reconstruction and selections remain unmatched.

| auc | low | high | n_train | n_dev | n_test | energy_bin_width_mev | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.907 | 0.778 | 0.991 | 116 | 46 | 30 | 20 | Equal real/MC counts within 20 MeV incoming-KE bands in each event partition; upstream real beam energy differs from TPC energy. Selection, shape and response remain unmatched. |

A real-trained closed-set proton/kaon/MIP head is also applied descriptively:

| species | kaon | muon | proton |
| --- | --- | --- | --- |
| electron | 0.432 | 0.484 | 0.084 |
| muon | 0.246 | 0.620 | 0.134 |
| photon | 0.638 | 0.248 | 0.114 |
| pion | 0.258 | 0.612 | 0.130 |
| proton | 0.078 | 0.526 | 0.396 |

These allocations are **not** validated species probabilities or beam-composition estimates. For example, assigning a simulated photon to the kaon region does not establish a real kaon contamination rate. The real MIP category is not pure simulated-muon truth.

## Tasks this pilot cannot validate

- Kaon recognition and proton/kaon decontamination: no kaon simulation truth in this release.
- Full-scene instance/pixel segmentation and vertex finding: particles were extracted using truth; original shared geometry is removed.
- Parent/daughter, decay ancestry and interaction-process classification: no adequate ancestry/process truth in these arrays.
- Unrestricted total energy reconstruction or calibrated LArIAT task performance: endpoint cropping and unresolved domain differences.

## Figure captions and reproducibility

`latent_species.pdf`: (a) real-training PCA of standardized eight-dimensional latents, with a random real reference subset and all converted particles; this 2D view is illustrative. (b) held-out row-normalized five-class confusion matrix for the nonlinear latent probe. Numerical probes use all eight dimensions.

`truth_preservation.pdf`: identical linear probes of the eight-dimensional latent and 32-dimensional image summaries. Entries are within-species held-out R²; asterisks denote log1p targets. Colour spans −0.2 to 1, while annotations retain exact values including any outside that range. Dashes indicate unsupported/unapplied targets. Confidence intervals and all controls are in the CSV.

Full caches and truth results: `/Volumes/easystore/proton-kaon/pilarnet_lariat/latent_truth/pilot_v1`. Figures and copied CSVs: `/Users/user/code/research/proton-kaon/output/pilarnet_lariat/latent_truth/pilot_v1`.

Run `prepare.py`, `encode.py`, `evaluate.py`, `counterfactual.py`, then `build_report.py`. Use another output path to change an existing pilot. All checkpoints remain frozen and large data stay on the external drive.
