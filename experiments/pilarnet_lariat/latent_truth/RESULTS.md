# VAE versus pixel PCA: simulation truth and t-SNE

Updated comparison on the same 2,500 converted particles (photon: 500, pion: 500, muon: 500, proton: 500, electron: 500). No engineered image descriptors or truth-derived predictors are used. Simulation truth supplies evaluation labels only. The main baseline is **eight principal components of the actual pixels**, matching the frozen VAE’s eight latent dimensions.

## Pixel baseline

Flatten both 48×48 planes into 4,608 values after the same single log1p(ADC) transform used by the VAE. PCA centres each pixel and fits **training events only**, without per-pixel variance scaling or labels. Eight components retain 47.5% of training pixel variance. Development/test images are projected with that fixed basis. This is an in-domain MC-fitted PCA baseline; the VAE remains frozen from real LArIAT training. Their representation-training data therefore differ, so this does not isolate architecture under identical training conditions. Prediction heads standardize each 8D representation on training events, select regularization on development events, and report test performance. VAE and pixel PCA use identical linear and nonlinear head settings.

| species | dev | test | train |
| --- | --- | --- | --- |
| proton | 120 | 103 | 277 |
| pion | 113 | 112 | 275 |
| muon | 99 | 107 | 294 |
| electron | 110 | 106 | 284 |
| photon | 93 | 113 | 294 |

## t-SNE

Settings: perplexity 30, seed 9105, PCA initialization, automatic learning rate, 1500 iterations. Both 8D representations are standardized using their training events before separate t-SNE fits. All 2,500 particles participate in this descriptive visualization. PID labels are added only afterwards for colour/shape. The coordinates of the two plots are independent; global distances and cluster sizes cannot be compared across maps. **No t-SNE coordinates enter the prediction heads.**

| representation | kl_divergence | trustworthiness_15_neighbors | iterations | seconds |
| --- | --- | --- | --- | --- |
| VAE (8D) | 1.000 | 0.986 | 1499 | 4.308 |
| Pixel PCA (8D) | 0.772 | 0.993 | 1499 | 3.857 |

## Held-out species and semantic classification

| task | representation | probe | balanced_accuracy | low | high | n_test |
| --- | --- | --- | --- | --- | --- | --- |
| species | paper_vae | logistic | 0.611 | 0.569 | 0.640 | 541 |
| species | paper_vae | permuted_train | 0.202 | — | — | 541 |
| species | paper_vae | trees | 0.723 | 0.689 | 0.753 | 541 |
| species | input_pca8 | logistic | 0.521 | 0.484 | 0.558 | 541 |
| species | input_pca8 | permuted_train | 0.189 | — | — | 541 |
| species | input_pca8 | trees | 0.633 | 0.600 | 0.667 | 541 |
| semantic | paper_vae | logistic | 0.757 | 0.695 | 0.811 | 541 |
| semantic | paper_vae | trees | 0.826 | 0.766 | 0.871 | 541 |
| semantic | input_pca8 | logistic | 0.732 | 0.673 | 0.782 | 541 |
| semantic | input_pca8 | trees | 0.791 | 0.734 | 0.845 | 541 |

Five-species chance is 20%. Semantic classification predicts the dominant truth category of an isolated particle, not voxel segmentation. The generated species have different energy distributions; the five-class energy-balanced diagnostic lacks common support, so this remains a conditional simulation pilot.

## Within-species physical readouts

Energy, momentum, dE/dx, lengths and ratios use log1p targets; positions, angles, fractions and 3D linearity use native targets. R² is reported on the fitted target scale. Physical-unit mean absolute errors and R² are in the CSV. Confidence intervals resample source events (150 draws). The endpoint dE/dx and Bragg conclusions must be judged against pixel PCA, rather than the earlier descriptor baseline. Failure of both 8D representations does not demonstrate a VAE-specific loss of information.

### Proton

| target | representation | probe | n_test | r2 | r2_low | r2_high | physical_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| incoming_ke_mev | paper_vae | ridge | 103 | 0.615 | 0.524 | 0.705 | 22.430 |
| incoming_ke_mev | paper_vae | trees | 103 | 0.701 | 0.587 | 0.814 | 18.174 |
| incoming_ke_mev | input_pca8 | ridge | 103 | 0.649 | 0.539 | 0.754 | 18.390 |
| incoming_ke_mev | input_pca8 | trees | 103 | 0.566 | 0.403 | 0.723 | 20.232 |
| momentum_mev | paper_vae | ridge | 103 | 0.609 | 0.516 | 0.703 | 44.509 |
| momentum_mev | paper_vae | trees | 103 | 0.692 | 0.578 | 0.810 | 35.666 |
| momentum_mev | input_pca8 | ridge | 103 | 0.642 | 0.531 | 0.751 | 36.958 |
| momentum_mev | input_pca8 | trees | 103 | 0.557 | 0.393 | 0.720 | 40.218 |
| deposited_mev | paper_vae | ridge | 103 | 0.555 | 0.434 | 0.682 | 19.867 |
| deposited_mev | paper_vae | trees | 103 | 0.661 | 0.529 | 0.780 | 15.103 |
| deposited_mev | input_pca8 | ridge | 103 | 0.679 | 0.569 | 0.795 | 15.520 |
| deposited_mev | input_pca8 | trees | 103 | 0.575 | 0.426 | 0.720 | 17.588 |
| geom_retained_energy_mev | paper_vae | ridge | 103 | 0.566 | 0.445 | 0.711 | 16.678 |
| geom_retained_energy_mev | paper_vae | trees | 103 | 0.692 | 0.569 | 0.826 | 11.838 |
| geom_retained_energy_mev | input_pca8 | ridge | 103 | 0.676 | 0.567 | 0.815 | 12.627 |
| geom_retained_energy_mev | input_pca8 | trees | 103 | 0.619 | 0.502 | 0.751 | 14.427 |
| geom_retained_electrons | paper_vae | ridge | 103 | 0.716 | 0.641 | 0.792 | 328847.272 |
| geom_retained_electrons | paper_vae | trees | 103 | 0.857 | 0.791 | 0.911 | 217834.407 |
| geom_retained_electrons | input_pca8 | ridge | 103 | 0.819 | 0.755 | 0.885 | 235634.856 |
| geom_retained_electrons | input_pca8 | trees | 103 | 0.783 | 0.678 | 0.881 | 245555.946 |
| path_cm | paper_vae | ridge | 103 | 0.668 | 0.568 | 0.771 | 3.225 |
| path_cm | paper_vae | trees | 103 | 0.785 | 0.632 | 0.902 | 2.357 |
| path_cm | input_pca8 | ridge | 103 | 0.764 | 0.641 | 0.869 | 2.395 |
| path_cm | input_pca8 | trees | 103 | 0.666 | 0.458 | 0.848 | 2.629 |
| mean_dedx_mev_cm | paper_vae | ridge | 103 | 0.467 | 0.332 | 0.572 | 1.529 |
| mean_dedx_mev_cm | paper_vae | trees | 103 | 0.539 | 0.405 | 0.657 | 1.272 |
| mean_dedx_mev_cm | input_pca8 | ridge | 103 | 0.464 | 0.304 | 0.607 | 1.422 |
| mean_dedx_mev_cm | input_pca8 | trees | 103 | 0.386 | 0.175 | 0.561 | 1.505 |
| endpoint_dedx_mev_cm | paper_vae | ridge | 103 | 0.008 | -0.039 | 0.030 | 2.808 |
| endpoint_dedx_mev_cm | paper_vae | trees | 103 | 0.016 | -0.086 | 0.123 | 2.576 |
| endpoint_dedx_mev_cm | input_pca8 | ridge | 103 | 0.015 | -0.042 | 0.031 | 2.772 |
| endpoint_dedx_mev_cm | input_pca8 | trees | 103 | -0.073 | -0.213 | -0.026 | 2.785 |
| bragg_ratio | paper_vae | ridge | 86 | 0.068 | -0.015 | 0.146 | 0.326 |
| bragg_ratio | paper_vae | trees | 86 | 0.104 | -0.022 | 0.270 | 0.297 |
| bragg_ratio | input_pca8 | ridge | 86 | 0.059 | -0.036 | 0.136 | 0.339 |
| bragg_ratio | input_pca8 | trees | 86 | 0.049 | -0.070 | 0.226 | 0.307 |
| linearity_3d | paper_vae | ridge | 103 | 0.383 | 0.269 | 0.479 | 0.003 |
| linearity_3d | paper_vae | trees | 103 | 0.647 | 0.488 | 0.764 | 0.002 |
| linearity_3d | input_pca8 | ridge | 103 | 0.443 | 0.313 | 0.544 | 0.002 |
| linearity_3d | input_pca8 | trees | 103 | 0.607 | 0.413 | 0.752 | 0.002 |
| transverse_rms_cm | paper_vae | ridge | 103 | 0.160 | 0.064 | 0.316 | 0.031 |
| transverse_rms_cm | paper_vae | trees | 103 | 0.232 | 0.144 | 0.350 | 0.030 |
| transverse_rms_cm | input_pca8 | ridge | 103 | 0.168 | 0.072 | 0.295 | 0.031 |
| transverse_rms_cm | input_pca8 | trees | 103 | 0.060 | -0.192 | 0.154 | 0.034 |
| proton_bb_density_ratio | paper_vae | ridge | 103 | -0.001 | -0.048 | 0.021 | 0.055 |
| proton_bb_density_ratio | paper_vae | trees | 103 | 0.019 | -0.128 | 0.114 | 0.053 |
| proton_bb_density_ratio | input_pca8 | ridge | 103 | 0.006 | -0.050 | 0.031 | 0.054 |
| proton_bb_density_ratio | input_pca8 | trees | 103 | -0.070 | -0.218 | -0.023 | 0.056 |
| proton_bb_log_distance | paper_vae | ridge | 103 | 0.001 | -0.045 | 0.025 | 0.063 |
| proton_bb_log_distance | paper_vae | trees | 103 | 0.013 | -0.142 | 0.127 | 0.062 |
| proton_bb_log_distance | input_pca8 | ridge | 103 | 0.007 | -0.045 | 0.033 | 0.062 |
| proton_bb_log_distance | input_pca8 | trees | 103 | -0.089 | -0.269 | -0.022 | 0.066 |
| proton_bb_best_rr_offset_cm | paper_vae | ridge | 103 | 0.000 | -0.051 | 0.031 | 0.939 |
| proton_bb_best_rr_offset_cm | paper_vae | trees | 103 | 0.020 | -0.076 | 0.120 | 0.931 |
| proton_bb_best_rr_offset_cm | input_pca8 | ridge | 103 | 0.009 | -0.037 | 0.031 | 0.933 |
| proton_bb_best_rr_offset_cm | input_pca8 | trees | 103 | -0.057 | -0.154 | 0.003 | 0.954 |
| extent_3d_cm | paper_vae | ridge | 103 | 0.663 | 0.560 | 0.768 | 3.239 |
| extent_3d_cm | paper_vae | trees | 103 | 0.784 | 0.629 | 0.903 | 2.361 |
| extent_3d_cm | input_pca8 | ridge | 103 | 0.762 | 0.637 | 0.867 | 2.394 |
| extent_3d_cm | input_pca8 | trees | 103 | 0.662 | 0.453 | 0.848 | 2.619 |
| chord_over_path | paper_vae | ridge | 103 | -0.006 | -0.245 | -0.002 | 0.016 |
| chord_over_path | paper_vae | trees | 103 | -0.004 | -0.489 | 0.027 | 0.017 |
| chord_over_path | input_pca8 | ridge | 103 | -0.007 | -0.279 | 0.004 | 0.017 |
| chord_over_path | input_pca8 | trees | 103 | -0.042 | -0.774 | 0.012 | 0.018 |
| visible_charge_fraction | paper_vae | ridge | 103 | -0.015 | -0.305 | 0.052 | 0.010 |
| visible_charge_fraction | paper_vae | trees | 103 | -0.030 | -1.088 | 0.109 | 0.010 |
| visible_charge_fraction | input_pca8 | ridge | 103 | -0.008 | -0.218 | 0.093 | 0.010 |
| visible_charge_fraction | input_pca8 | trees | 103 | -0.023 | -1.247 | 0.217 | 0.010 |
| geometric_energy_fraction | paper_vae | ridge | 103 | 0.031 | -0.076 | 0.171 | 0.026 |
| geometric_energy_fraction | paper_vae | trees | 103 | 0.088 | -0.023 | 0.265 | 0.025 |
| geometric_energy_fraction | input_pca8 | ridge | 103 | -0.013 | -0.098 | 0.070 | 0.029 |
| geometric_energy_fraction | input_pca8 | trees | 103 | 0.015 | -0.062 | 0.113 | 0.026 |
| assigned_xz_deg | paper_vae | ridge | 103 | 0.219 | 0.049 | 0.322 | 1.776 |
| assigned_xz_deg | paper_vae | trees | 103 | 0.340 | 0.225 | 0.415 | 1.585 |
| assigned_xz_deg | input_pca8 | ridge | 103 | 0.303 | 0.111 | 0.448 | 1.651 |
| assigned_xz_deg | input_pca8 | trees | 103 | 0.354 | 0.214 | 0.472 | 1.567 |
| assigned_yz_deg | paper_vae | ridge | 103 | -0.022 | -0.112 | 0.070 | 3.907 |
| assigned_yz_deg | paper_vae | trees | 103 | 0.060 | -0.052 | 0.166 | 3.723 |
| assigned_yz_deg | input_pca8 | ridge | 103 | -0.009 | -0.088 | 0.061 | 3.856 |
| assigned_yz_deg | input_pca8 | trees | 103 | -0.040 | -0.142 | 0.038 | 3.937 |
| vertex_x_cm | paper_vae | ridge | 103 | 0.001 | -0.077 | 0.025 | 26.960 |
| vertex_x_cm | paper_vae | trees | 103 | -0.029 | -0.117 | 0.028 | 27.250 |
| vertex_x_cm | input_pca8 | ridge | 103 | -0.004 | -0.071 | 0.030 | 26.435 |
| vertex_x_cm | input_pca8 | trees | 103 | -0.064 | -0.144 | -0.014 | 27.698 |
| vertex_y_cm | paper_vae | ridge | 103 | -0.014 | -0.167 | 0.034 | 54.943 |
| vertex_y_cm | paper_vae | trees | 103 | -0.027 | -0.145 | 0.039 | 55.120 |
| vertex_y_cm | input_pca8 | ridge | 103 | -0.012 | -0.141 | 0.010 | 54.853 |
| vertex_y_cm | input_pca8 | trees | 103 | -0.017 | -0.185 | 0.054 | 54.645 |
| vertex_z_cm | paper_vae | ridge | 103 | -0.092 | -0.237 | 0.003 | 51.636 |
| vertex_z_cm | paper_vae | trees | 103 | -0.066 | -0.249 | 0.054 | 51.173 |
| vertex_z_cm | input_pca8 | ridge | 103 | -0.071 | -0.205 | 0.013 | 51.137 |
| vertex_z_cm | input_pca8 | trees | 103 | -0.095 | -0.273 | 0.056 | 51.721 |
| source_direction_x | paper_vae | ridge | 103 | -0.106 | -0.227 | -0.020 | 0.510 |
| source_direction_x | paper_vae | trees | 103 | 0.008 | -0.093 | 0.078 | 0.489 |
| source_direction_x | input_pca8 | ridge | 103 | -0.060 | -0.168 | 0.005 | 0.497 |
| source_direction_x | input_pca8 | trees | 103 | -0.072 | -0.204 | 0.013 | 0.496 |
| source_direction_y | paper_vae | ridge | 103 | -0.024 | -0.115 | 0.028 | 0.527 |
| source_direction_y | paper_vae | trees | 103 | -0.077 | -0.177 | 0.005 | 0.534 |
| source_direction_y | input_pca8 | ridge | 103 | -0.020 | -0.083 | 0.019 | 0.522 |
| source_direction_y | input_pca8 | trees | 103 | -0.014 | -0.115 | 0.073 | 0.523 |
| source_direction_z | paper_vae | ridge | 103 | -0.024 | -0.120 | 0.027 | 0.497 |
| source_direction_z | paper_vae | trees | 103 | -0.037 | -0.147 | 0.042 | 0.497 |
| source_direction_z | input_pca8 | ridge | 103 | -0.041 | -0.109 | 0.002 | 0.498 |
| source_direction_z | input_pca8 | trees | 103 | -0.038 | -0.148 | 0.034 | 0.498 |

### Pion

| target | representation | probe | n_test | r2 | r2_low | r2_high | physical_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| incoming_ke_mev | paper_vae | ridge | 112 | 0.290 | 0.074 | 0.437 | 87.402 |
| incoming_ke_mev | paper_vae | trees | 112 | 0.465 | 0.316 | 0.573 | 75.756 |
| incoming_ke_mev | input_pca8 | ridge | 112 | 0.171 | -0.013 | 0.283 | 91.737 |
| incoming_ke_mev | input_pca8 | trees | 112 | 0.378 | 0.223 | 0.505 | 80.467 |
| momentum_mev | paper_vae | ridge | 112 | 0.272 | 0.061 | 0.421 | 96.114 |
| momentum_mev | paper_vae | trees | 112 | 0.435 | 0.281 | 0.548 | 84.023 |
| momentum_mev | input_pca8 | ridge | 112 | 0.151 | -0.031 | 0.265 | 101.596 |
| momentum_mev | input_pca8 | trees | 112 | 0.347 | 0.189 | 0.478 | 88.940 |
| deposited_mev | paper_vae | ridge | 112 | 0.470 | 0.306 | 0.561 | 35.799 |
| deposited_mev | paper_vae | trees | 112 | 0.686 | 0.606 | 0.758 | 33.553 |
| deposited_mev | input_pca8 | ridge | 112 | 0.631 | 0.543 | 0.702 | 33.729 |
| deposited_mev | input_pca8 | trees | 112 | 0.648 | 0.544 | 0.722 | 35.137 |
| geom_retained_energy_mev | paper_vae | ridge | 112 | 0.314 | 0.215 | 0.379 | 14.551 |
| geom_retained_energy_mev | paper_vae | trees | 112 | 0.719 | 0.626 | 0.780 | 10.875 |
| geom_retained_energy_mev | input_pca8 | ridge | 112 | 0.599 | 0.502 | 0.656 | 13.000 |
| geom_retained_energy_mev | input_pca8 | trees | 112 | 0.671 | 0.563 | 0.740 | 12.409 |
| geom_retained_electrons | paper_vae | ridge | 112 | 0.489 | 0.273 | 0.614 | 303797.256 |
| geom_retained_electrons | paper_vae | trees | 112 | 0.802 | 0.724 | 0.848 | 225682.484 |
| geom_retained_electrons | input_pca8 | ridge | 112 | 0.713 | 0.637 | 0.749 | 266771.161 |
| geom_retained_electrons | input_pca8 | trees | 112 | 0.782 | 0.697 | 0.827 | 253834.426 |
| path_cm | paper_vae | ridge | 112 | 0.591 | 0.443 | 0.684 | 15.440 |
| path_cm | paper_vae | trees | 112 | 0.713 | 0.637 | 0.785 | 14.656 |
| path_cm | input_pca8 | ridge | 112 | 0.672 | 0.578 | 0.745 | 14.817 |
| path_cm | input_pca8 | trees | 112 | 0.693 | 0.608 | 0.757 | 15.219 |
| mean_dedx_mev_cm | paper_vae | ridge | 112 | 0.256 | 0.064 | 0.383 | 0.844 |
| mean_dedx_mev_cm | paper_vae | trees | 112 | 0.483 | 0.298 | 0.581 | 0.692 |
| mean_dedx_mev_cm | input_pca8 | ridge | 112 | 0.135 | 0.012 | 0.208 | 0.910 |
| mean_dedx_mev_cm | input_pca8 | trees | 112 | 0.369 | 0.211 | 0.482 | 0.775 |
| endpoint_dedx_mev_cm | paper_vae | ridge | 112 | 0.086 | -0.086 | 0.221 | 3.027 |
| endpoint_dedx_mev_cm | paper_vae | trees | 112 | 0.398 | 0.314 | 0.471 | 2.491 |
| endpoint_dedx_mev_cm | input_pca8 | ridge | 112 | 0.106 | -0.056 | 0.238 | 2.925 |
| endpoint_dedx_mev_cm | input_pca8 | trees | 112 | 0.273 | 0.152 | 0.376 | 2.753 |
| bragg_ratio | paper_vae | ridge | 108 | 0.048 | -0.149 | 0.182 | 0.699 |
| bragg_ratio | paper_vae | trees | 108 | 0.341 | 0.232 | 0.410 | 0.594 |
| bragg_ratio | input_pca8 | ridge | 108 | 0.082 | -0.120 | 0.250 | 0.661 |
| bragg_ratio | input_pca8 | trees | 108 | 0.235 | 0.113 | 0.328 | 0.638 |
| linearity_3d | paper_vae | ridge | 112 | 0.092 | -0.133 | 0.248 | 0.003 |
| linearity_3d | paper_vae | trees | 112 | 0.287 | 0.194 | 0.585 | 0.002 |
| linearity_3d | input_pca8 | ridge | 112 | 0.186 | -0.031 | 0.435 | 0.003 |
| linearity_3d | input_pca8 | trees | 112 | 0.257 | 0.057 | 0.559 | 0.002 |
| transverse_rms_cm | paper_vae | ridge | 112 | 0.420 | 0.221 | 0.600 | 0.195 |
| transverse_rms_cm | paper_vae | trees | 112 | 0.468 | 0.258 | 0.651 | 0.178 |
| transverse_rms_cm | input_pca8 | ridge | 112 | 0.411 | 0.196 | 0.580 | 0.193 |
| transverse_rms_cm | input_pca8 | trees | 112 | 0.396 | 0.185 | 0.534 | 0.194 |
| extent_3d_cm | paper_vae | ridge | 112 | 0.588 | 0.442 | 0.680 | 14.958 |
| extent_3d_cm | paper_vae | trees | 112 | 0.715 | 0.637 | 0.784 | 14.117 |
| extent_3d_cm | input_pca8 | ridge | 112 | 0.672 | 0.579 | 0.746 | 14.293 |
| extent_3d_cm | input_pca8 | trees | 112 | 0.702 | 0.618 | 0.767 | 14.463 |
| chord_over_path | paper_vae | ridge | 112 | -0.040 | -0.108 | 0.001 | 0.023 |
| chord_over_path | paper_vae | trees | 112 | -0.054 | -0.129 | 0.014 | 0.023 |
| chord_over_path | input_pca8 | ridge | 112 | -0.033 | -0.106 | 0.013 | 0.023 |
| chord_over_path | input_pca8 | trees | 112 | -0.070 | -0.169 | -0.003 | 0.024 |
| visible_charge_fraction | paper_vae | ridge | 112 | -0.014 | -0.094 | 0.073 | 0.049 |
| visible_charge_fraction | paper_vae | trees | 112 | -0.024 | -0.105 | 0.091 | 0.049 |
| visible_charge_fraction | input_pca8 | ridge | 112 | -0.006 | -0.064 | 0.040 | 0.050 |
| visible_charge_fraction | input_pca8 | trees | 112 | -0.080 | -0.162 | -0.008 | 0.051 |
| geometric_energy_fraction | paper_vae | ridge | 112 | 0.328 | 0.168 | 0.500 | 0.169 |
| geometric_energy_fraction | paper_vae | trees | 112 | 0.378 | 0.268 | 0.492 | 0.154 |
| geometric_energy_fraction | input_pca8 | ridge | 112 | 0.326 | 0.228 | 0.431 | 0.174 |
| geometric_energy_fraction | input_pca8 | trees | 112 | 0.336 | 0.208 | 0.433 | 0.158 |
| assigned_xz_deg | paper_vae | ridge | 112 | 0.117 | -0.013 | 0.220 | 1.720 |
| assigned_xz_deg | paper_vae | trees | 112 | 0.121 | -0.082 | 0.242 | 1.684 |
| assigned_xz_deg | input_pca8 | ridge | 112 | 0.170 | 0.030 | 0.272 | 1.674 |
| assigned_xz_deg | input_pca8 | trees | 112 | 0.235 | 0.015 | 0.366 | 1.593 |
| assigned_yz_deg | paper_vae | ridge | 112 | -0.051 | -0.167 | 0.019 | 3.448 |
| assigned_yz_deg | paper_vae | trees | 112 | -0.088 | -0.224 | 0.016 | 3.420 |
| assigned_yz_deg | input_pca8 | ridge | 112 | -0.048 | -0.143 | 0.022 | 3.393 |
| assigned_yz_deg | input_pca8 | trees | 112 | -0.143 | -0.356 | -0.012 | 3.625 |
| vertex_x_cm | paper_vae | ridge | 112 | -0.011 | -0.096 | 0.031 | 25.180 |
| vertex_x_cm | paper_vae | trees | 112 | -0.019 | -0.121 | 0.047 | 25.120 |
| vertex_x_cm | input_pca8 | ridge | 112 | -0.003 | -0.091 | 0.031 | 24.991 |
| vertex_x_cm | input_pca8 | trees | 112 | -0.005 | -0.102 | 0.050 | 24.744 |
| vertex_y_cm | paper_vae | ridge | 112 | -0.022 | -0.139 | 0.023 | 45.486 |
| vertex_y_cm | paper_vae | trees | 112 | -0.055 | -0.200 | 0.014 | 46.748 |
| vertex_y_cm | input_pca8 | ridge | 112 | -0.038 | -0.174 | 0.004 | 45.644 |
| vertex_y_cm | input_pca8 | trees | 112 | -0.054 | -0.207 | 0.020 | 45.346 |
| vertex_z_cm | paper_vae | ridge | 112 | 0.001 | -0.081 | 0.026 | 48.167 |
| vertex_z_cm | paper_vae | trees | 112 | -0.004 | -0.119 | 0.047 | 48.048 |
| vertex_z_cm | input_pca8 | ridge | 112 | 0.010 | -0.067 | 0.031 | 48.086 |
| vertex_z_cm | input_pca8 | trees | 112 | -0.029 | -0.171 | 0.077 | 48.329 |
| source_direction_x | paper_vae | ridge | 112 | -0.042 | -0.140 | 0.003 | 0.454 |
| source_direction_x | paper_vae | trees | 112 | -0.072 | -0.199 | 0.029 | 0.460 |
| source_direction_x | input_pca8 | ridge | 112 | -0.026 | -0.132 | 0.051 | 0.450 |
| source_direction_x | input_pca8 | trees | 112 | -0.085 | -0.197 | 0.031 | 0.461 |
| source_direction_y | paper_vae | ridge | 112 | -0.057 | -0.217 | 0.019 | 0.596 |
| source_direction_y | paper_vae | trees | 112 | -0.091 | -0.237 | -0.011 | 0.601 |
| source_direction_y | input_pca8 | ridge | 112 | -0.045 | -0.189 | 0.021 | 0.589 |
| source_direction_y | input_pca8 | trees | 112 | -0.039 | -0.189 | 0.058 | 0.588 |
| source_direction_z | paper_vae | ridge | 112 | -0.028 | -0.127 | 0.040 | 0.455 |
| source_direction_z | paper_vae | trees | 112 | -0.046 | -0.199 | 0.069 | 0.457 |
| source_direction_z | input_pca8 | ridge | 112 | -0.032 | -0.122 | 0.024 | 0.456 |
| source_direction_z | input_pca8 | trees | 112 | -0.097 | -0.285 | 0.067 | 0.468 |

### Muon

| target | representation | probe | n_test | r2 | r2_low | r2_high | physical_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| incoming_ke_mev | paper_vae | ridge | 107 | 0.010 | -0.206 | 0.194 | 524.287 |
| incoming_ke_mev | paper_vae | trees | 107 | 0.300 | 0.178 | 0.385 | 431.208 |
| incoming_ke_mev | input_pca8 | ridge | 107 | 0.143 | 0.004 | 0.248 | 479.981 |
| incoming_ke_mev | input_pca8 | trees | 107 | 0.252 | 0.110 | 0.355 | 445.588 |
| momentum_mev | paper_vae | ridge | 107 | -0.004 | -0.218 | 0.190 | 520.271 |
| momentum_mev | paper_vae | trees | 107 | 0.296 | 0.166 | 0.387 | 436.656 |
| momentum_mev | input_pca8 | ridge | 107 | 0.133 | -0.012 | 0.241 | 483.064 |
| momentum_mev | input_pca8 | trees | 107 | 0.235 | 0.095 | 0.337 | 451.799 |
| deposited_mev | paper_vae | ridge | 107 | 0.190 | 0.006 | 0.355 | 113.174 |
| deposited_mev | paper_vae | trees | 107 | 0.192 | 0.043 | 0.305 | 114.417 |
| deposited_mev | input_pca8 | ridge | 107 | 0.254 | 0.068 | 0.423 | 110.638 |
| deposited_mev | input_pca8 | trees | 107 | 0.274 | 0.091 | 0.441 | 112.042 |
| geom_retained_energy_mev | paper_vae | ridge | 107 | 0.120 | -0.085 | 0.245 | 6.073 |
| geom_retained_energy_mev | paper_vae | trees | 107 | 0.326 | 0.210 | 0.425 | 5.531 |
| geom_retained_energy_mev | input_pca8 | ridge | 107 | 0.270 | 0.024 | 0.422 | 5.612 |
| geom_retained_energy_mev | input_pca8 | trees | 107 | 0.354 | 0.190 | 0.451 | 5.184 |
| geom_retained_electrons | paper_vae | ridge | 107 | 0.244 | -0.466 | 0.589 | 137295.427 |
| geom_retained_electrons | paper_vae | trees | 107 | 0.461 | 0.241 | 0.622 | 126945.649 |
| geom_retained_electrons | input_pca8 | ridge | 107 | 0.460 | 0.128 | 0.665 | 127159.849 |
| geom_retained_electrons | input_pca8 | trees | 107 | 0.548 | 0.323 | 0.720 | 115571.429 |
| path_cm | paper_vae | ridge | 107 | 0.231 | 0.025 | 0.397 | 57.889 |
| path_cm | paper_vae | trees | 107 | 0.283 | 0.138 | 0.392 | 56.324 |
| path_cm | input_pca8 | ridge | 107 | 0.325 | 0.104 | 0.512 | 56.297 |
| path_cm | input_pca8 | trees | 107 | 0.356 | 0.150 | 0.529 | 56.426 |
| mean_dedx_mev_cm | paper_vae | ridge | 107 | 0.179 | -0.019 | 0.303 | 0.309 |
| mean_dedx_mev_cm | paper_vae | trees | 107 | 0.376 | 0.246 | 0.469 | 0.255 |
| mean_dedx_mev_cm | input_pca8 | ridge | 107 | 0.306 | 0.099 | 0.401 | 0.293 |
| mean_dedx_mev_cm | input_pca8 | trees | 107 | 0.427 | 0.215 | 0.563 | 0.262 |
| endpoint_dedx_mev_cm | paper_vae | ridge | 107 | -0.054 | -0.372 | 0.198 | 1.727 |
| endpoint_dedx_mev_cm | paper_vae | trees | 107 | 0.244 | 0.100 | 0.369 | 1.574 |
| endpoint_dedx_mev_cm | input_pca8 | ridge | 107 | 0.139 | -0.041 | 0.246 | 1.661 |
| endpoint_dedx_mev_cm | input_pca8 | trees | 107 | 0.225 | 0.104 | 0.328 | 1.577 |
| bragg_ratio | paper_vae | ridge | 107 | -0.007 | -0.172 | 0.126 | 0.497 |
| bragg_ratio | paper_vae | trees | 107 | 0.170 | 0.022 | 0.296 | 0.452 |
| bragg_ratio | input_pca8 | ridge | 107 | 0.086 | -0.068 | 0.173 | 0.480 |
| bragg_ratio | input_pca8 | trees | 107 | 0.170 | 0.047 | 0.259 | 0.460 |
| linearity_3d | paper_vae | ridge | 107 | 0.180 | -0.104 | 0.434 | 0.001 |
| linearity_3d | paper_vae | trees | 107 | 0.451 | 0.242 | 0.606 | 0.001 |
| linearity_3d | input_pca8 | ridge | 107 | 0.330 | 0.004 | 0.540 | 0.001 |
| linearity_3d | input_pca8 | trees | 107 | 0.466 | 0.168 | 0.773 | 0.001 |
| transverse_rms_cm | paper_vae | ridge | 107 | 0.055 | -0.111 | 0.143 | 0.371 |
| transverse_rms_cm | paper_vae | trees | 107 | 0.074 | -0.132 | 0.169 | 0.368 |
| transverse_rms_cm | input_pca8 | ridge | 107 | 0.112 | -0.071 | 0.233 | 0.360 |
| transverse_rms_cm | input_pca8 | trees | 107 | 0.062 | -0.122 | 0.183 | 0.361 |
| extent_3d_cm | paper_vae | ridge | 107 | 0.230 | 0.025 | 0.396 | 55.496 |
| extent_3d_cm | paper_vae | trees | 107 | 0.285 | 0.142 | 0.393 | 53.865 |
| extent_3d_cm | input_pca8 | ridge | 107 | 0.323 | 0.103 | 0.509 | 53.923 |
| extent_3d_cm | input_pca8 | trees | 107 | 0.352 | 0.147 | 0.525 | 53.921 |
| chord_over_path | paper_vae | ridge | 107 | -0.014 | -0.088 | 0.034 | 0.015 |
| chord_over_path | paper_vae | trees | 107 | -0.003 | -0.174 | 0.099 | 0.015 |
| chord_over_path | input_pca8 | ridge | 107 | -0.038 | -0.159 | 0.050 | 0.015 |
| chord_over_path | input_pca8 | trees | 107 | -0.109 | -0.299 | 0.040 | 0.016 |
| visible_charge_fraction | paper_vae | ridge | 107 | 0.030 | -0.034 | 0.074 | 0.211 |
| visible_charge_fraction | paper_vae | trees | 107 | -0.030 | -0.147 | 0.057 | 0.215 |
| visible_charge_fraction | input_pca8 | ridge | 107 | 0.042 | -0.054 | 0.103 | 0.208 |
| visible_charge_fraction | input_pca8 | trees | 107 | -0.037 | -0.165 | 0.102 | 0.211 |
| geometric_energy_fraction | paper_vae | ridge | 107 | 0.231 | 0.065 | 0.373 | 0.179 |
| geometric_energy_fraction | paper_vae | trees | 107 | 0.308 | 0.193 | 0.400 | 0.169 |
| geometric_energy_fraction | input_pca8 | ridge | 107 | 0.302 | 0.129 | 0.469 | 0.169 |
| geometric_energy_fraction | input_pca8 | trees | 107 | 0.330 | 0.183 | 0.483 | 0.168 |
| assigned_xz_deg | paper_vae | ridge | 107 | 0.200 | -0.014 | 0.320 | 1.763 |
| assigned_xz_deg | paper_vae | trees | 107 | 0.184 | 0.018 | 0.291 | 1.741 |
| assigned_xz_deg | input_pca8 | ridge | 107 | 0.203 | 0.052 | 0.324 | 1.763 |
| assigned_xz_deg | input_pca8 | trees | 107 | 0.162 | -0.040 | 0.304 | 1.774 |
| assigned_yz_deg | paper_vae | ridge | 107 | 0.042 | -0.060 | 0.100 | 3.670 |
| assigned_yz_deg | paper_vae | trees | 107 | 0.211 | 0.076 | 0.293 | 3.363 |
| assigned_yz_deg | input_pca8 | ridge | 107 | -0.011 | -0.072 | 0.020 | 3.721 |
| assigned_yz_deg | input_pca8 | trees | 107 | 0.008 | -0.090 | 0.094 | 3.706 |
| vertex_x_cm | paper_vae | ridge | 107 | -0.063 | -0.167 | -0.000 | 34.114 |
| vertex_x_cm | paper_vae | trees | 107 | -0.038 | -0.164 | 0.045 | 33.131 |
| vertex_x_cm | input_pca8 | ridge | 107 | -0.030 | -0.126 | 0.032 | 33.464 |
| vertex_x_cm | input_pca8 | trees | 107 | -0.027 | -0.182 | 0.049 | 33.163 |
| vertex_y_cm | paper_vae | ridge | 107 | -0.096 | -0.246 | -0.008 | 73.068 |
| vertex_y_cm | paper_vae | trees | 107 | -0.141 | -0.304 | -0.015 | 74.688 |
| vertex_y_cm | input_pca8 | ridge | 107 | -0.094 | -0.260 | 0.011 | 72.432 |
| vertex_y_cm | input_pca8 | trees | 107 | -0.111 | -0.315 | 0.033 | 73.509 |
| vertex_z_cm | paper_vae | ridge | 107 | 0.033 | -0.035 | 0.090 | 56.848 |
| vertex_z_cm | paper_vae | trees | 107 | -0.045 | -0.174 | 0.036 | 59.616 |
| vertex_z_cm | input_pca8 | ridge | 107 | -0.004 | -0.125 | 0.083 | 57.956 |
| vertex_z_cm | input_pca8 | trees | 107 | -0.096 | -0.259 | 0.015 | 60.699 |
| source_direction_x | paper_vae | ridge | 107 | -0.029 | -0.107 | 0.004 | 0.481 |
| source_direction_x | paper_vae | trees | 107 | -0.038 | -0.142 | 0.026 | 0.482 |
| source_direction_x | input_pca8 | ridge | 107 | -0.018 | -0.118 | 0.028 | 0.480 |
| source_direction_x | input_pca8 | trees | 107 | -0.125 | -0.247 | -0.031 | 0.507 |
| source_direction_y | paper_vae | ridge | 107 | -0.006 | -0.094 | 0.033 | 0.487 |
| source_direction_y | paper_vae | trees | 107 | -0.039 | -0.148 | 0.027 | 0.488 |
| source_direction_y | input_pca8 | ridge | 107 | 0.004 | -0.079 | 0.034 | 0.486 |
| source_direction_y | input_pca8 | trees | 107 | 0.003 | -0.102 | 0.053 | 0.474 |
| source_direction_z | paper_vae | ridge | 107 | -0.085 | -0.221 | -0.023 | 0.535 |
| source_direction_z | paper_vae | trees | 107 | -0.061 | -0.217 | 0.027 | 0.522 |
| source_direction_z | input_pca8 | ridge | 107 | -0.050 | -0.184 | 0.005 | 0.529 |
| source_direction_z | input_pca8 | trees | 107 | -0.056 | -0.180 | 0.033 | 0.521 |

### Electron

| target | representation | probe | n_test | r2 | r2_low | r2_high | physical_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| incoming_ke_mev | paper_vae | ridge | 57 | 0.561 | 0.311 | 0.626 | 30.507 |
| incoming_ke_mev | paper_vae | trees | 57 | 0.599 | 0.521 | 0.769 | 30.145 |
| incoming_ke_mev | input_pca8 | ridge | 57 | 0.733 | 0.537 | 0.848 | 42.846 |
| incoming_ke_mev | input_pca8 | trees | 57 | 0.585 | 0.478 | 0.706 | 30.388 |
| momentum_mev | paper_vae | ridge | 57 | 0.561 | 0.309 | 0.625 | 30.504 |
| momentum_mev | paper_vae | trees | 57 | 0.596 | 0.518 | 0.770 | 30.128 |
| momentum_mev | input_pca8 | ridge | 57 | 0.740 | 0.541 | 0.853 | 40.985 |
| momentum_mev | input_pca8 | trees | 57 | 0.582 | 0.474 | 0.706 | 30.377 |
| deposited_mev | paper_vae | ridge | 106 | 0.898 | 0.860 | 0.931 | 261.590 |
| deposited_mev | paper_vae | trees | 106 | 0.932 | 0.908 | 0.957 | 249.847 |
| deposited_mev | input_pca8 | ridge | 106 | 0.842 | 0.765 | 0.900 | 402.901 |
| deposited_mev | input_pca8 | trees | 106 | 0.934 | 0.904 | 0.959 | 246.250 |
| geom_retained_energy_mev | paper_vae | ridge | 106 | 0.910 | 0.871 | 0.938 | 39.253 |
| geom_retained_energy_mev | paper_vae | trees | 106 | 0.967 | 0.957 | 0.976 | 35.800 |
| geom_retained_energy_mev | input_pca8 | ridge | 106 | 0.871 | 0.817 | 0.906 | 71.269 |
| geom_retained_energy_mev | input_pca8 | trees | 106 | 0.961 | 0.943 | 0.976 | 33.004 |
| geom_retained_electrons | paper_vae | ridge | 106 | 0.905 | 0.864 | 0.934 | 1188974.630 |
| geom_retained_electrons | paper_vae | trees | 106 | 0.967 | 0.956 | 0.975 | 1051661.843 |
| geom_retained_electrons | input_pca8 | ridge | 106 | 0.866 | 0.812 | 0.899 | 2151476.521 |
| geom_retained_electrons | input_pca8 | trees | 106 | 0.960 | 0.942 | 0.975 | 978426.621 |
| mean_dedx_mev_cm | paper_vae | ridge | 106 | 0.011 | -0.053 | 0.047 | 0.072 |
| mean_dedx_mev_cm | paper_vae | trees | 106 | 0.058 | -0.082 | 0.153 | 0.072 |
| mean_dedx_mev_cm | input_pca8 | ridge | 106 | 0.004 | -0.062 | 0.029 | 0.072 |
| mean_dedx_mev_cm | input_pca8 | trees | 106 | -0.041 | -0.292 | 0.116 | 0.077 |
| linearity_3d | paper_vae | ridge | 106 | -0.036 | -0.206 | 0.029 | 0.031 |
| linearity_3d | paper_vae | trees | 106 | 0.201 | 0.028 | 0.302 | 0.026 |
| linearity_3d | input_pca8 | ridge | 106 | -0.026 | -0.187 | 0.036 | 0.031 |
| linearity_3d | input_pca8 | trees | 106 | 0.158 | -0.050 | 0.273 | 0.027 |
| transverse_rms_cm | paper_vae | ridge | 106 | 0.775 | 0.714 | 0.832 | 0.743 |
| transverse_rms_cm | paper_vae | trees | 106 | 0.805 | 0.747 | 0.865 | 0.682 |
| transverse_rms_cm | input_pca8 | ridge | 106 | 0.720 | 0.636 | 0.806 | 0.848 |
| transverse_rms_cm | input_pca8 | trees | 106 | 0.822 | 0.767 | 0.872 | 0.659 |
| extent_3d_cm | paper_vae | ridge | 106 | 0.698 | 0.625 | 0.773 | 22.553 |
| extent_3d_cm | paper_vae | trees | 106 | 0.792 | 0.721 | 0.856 | 19.012 |
| extent_3d_cm | input_pca8 | ridge | 106 | 0.658 | 0.576 | 0.751 | 24.229 |
| extent_3d_cm | input_pca8 | trees | 106 | 0.793 | 0.721 | 0.866 | 18.168 |
| n_fragments | paper_vae | ridge | 106 | 0.812 | 0.737 | 0.866 | 4.167 |
| n_fragments | paper_vae | trees | 106 | 0.807 | 0.721 | 0.868 | 4.361 |
| n_fragments | input_pca8 | ridge | 106 | 0.774 | 0.677 | 0.865 | 4.670 |
| n_fragments | input_pca8 | trees | 106 | 0.804 | 0.731 | 0.867 | 4.480 |
| shower_fraction | paper_vae | ridge | 106 | 0.653 | 0.485 | 0.746 | 0.209 |
| shower_fraction | paper_vae | trees | 106 | 0.718 | 0.540 | 0.825 | 0.156 |
| shower_fraction | input_pca8 | ridge | 106 | 0.586 | 0.427 | 0.699 | 0.234 |
| shower_fraction | input_pca8 | trees | 106 | 0.745 | 0.571 | 0.870 | 0.135 |
| visible_charge_fraction | paper_vae | ridge | 106 | 0.045 | -0.072 | 0.162 | 0.075 |
| visible_charge_fraction | paper_vae | trees | 106 | 0.045 | -0.096 | 0.199 | 0.073 |
| visible_charge_fraction | input_pca8 | ridge | 106 | 0.033 | -0.064 | 0.152 | 0.075 |
| visible_charge_fraction | input_pca8 | trees | 106 | -0.032 | -0.152 | 0.075 | 0.077 |
| geometric_energy_fraction | paper_vae | ridge | 106 | 0.694 | 0.585 | 0.768 | 0.144 |
| geometric_energy_fraction | paper_vae | trees | 106 | 0.698 | 0.592 | 0.787 | 0.142 |
| geometric_energy_fraction | input_pca8 | ridge | 106 | 0.629 | 0.524 | 0.722 | 0.157 |
| geometric_energy_fraction | input_pca8 | trees | 106 | 0.707 | 0.608 | 0.788 | 0.139 |
| assigned_xz_deg | paper_vae | ridge | 106 | 0.013 | -0.092 | 0.079 | 1.996 |
| assigned_xz_deg | paper_vae | trees | 106 | 0.016 | -0.136 | 0.117 | 1.977 |
| assigned_xz_deg | input_pca8 | ridge | 106 | 0.041 | -0.094 | 0.128 | 1.959 |
| assigned_xz_deg | input_pca8 | trees | 106 | 0.029 | -0.100 | 0.112 | 1.950 |
| assigned_yz_deg | paper_vae | ridge | 106 | -0.046 | -0.148 | 0.006 | 3.403 |
| assigned_yz_deg | paper_vae | trees | 106 | -0.107 | -0.243 | -0.024 | 3.473 |
| assigned_yz_deg | input_pca8 | ridge | 106 | -0.061 | -0.175 | -0.000 | 3.399 |
| assigned_yz_deg | input_pca8 | trees | 106 | -0.125 | -0.262 | -0.047 | 3.496 |
| vertex_x_cm | paper_vae | ridge | 106 | -0.040 | -0.158 | -0.005 | 30.683 |
| vertex_x_cm | paper_vae | trees | 106 | -0.108 | -0.252 | -0.033 | 31.177 |
| vertex_x_cm | input_pca8 | ridge | 106 | -0.031 | -0.158 | 0.010 | 30.735 |
| vertex_x_cm | input_pca8 | trees | 106 | -0.073 | -0.207 | -0.005 | 31.058 |
| vertex_y_cm | paper_vae | ridge | 106 | -0.017 | -0.091 | 0.029 | 56.443 |
| vertex_y_cm | paper_vae | trees | 106 | -0.008 | -0.113 | 0.085 | 56.663 |
| vertex_y_cm | input_pca8 | ridge | 106 | -0.043 | -0.117 | -0.000 | 57.538 |
| vertex_y_cm | input_pca8 | trees | 106 | 0.029 | -0.070 | 0.104 | 55.863 |
| vertex_z_cm | paper_vae | ridge | 106 | -0.023 | -0.114 | 0.008 | 49.630 |
| vertex_z_cm | paper_vae | trees | 106 | -0.077 | -0.175 | -0.009 | 51.396 |
| vertex_z_cm | input_pca8 | ridge | 106 | -0.026 | -0.121 | 0.013 | 50.004 |
| vertex_z_cm | input_pca8 | trees | 106 | -0.057 | -0.174 | 0.022 | 50.771 |
| source_direction_x | paper_vae | ridge | 106 | -0.097 | -0.220 | -0.019 | 0.522 |
| source_direction_x | paper_vae | trees | 106 | -0.063 | -0.203 | 0.008 | 0.520 |
| source_direction_x | input_pca8 | ridge | 106 | -0.050 | -0.199 | 0.018 | 0.512 |
| source_direction_x | input_pca8 | trees | 106 | -0.018 | -0.181 | 0.062 | 0.502 |
| source_direction_y | paper_vae | ridge | 106 | -0.053 | -0.153 | 0.009 | 0.505 |
| source_direction_y | paper_vae | trees | 106 | -0.155 | -0.271 | -0.069 | 0.527 |
| source_direction_y | input_pca8 | ridge | 106 | -0.083 | -0.198 | -0.020 | 0.510 |
| source_direction_y | input_pca8 | trees | 106 | -0.113 | -0.258 | -0.036 | 0.515 |
| source_direction_z | paper_vae | ridge | 106 | 0.031 | -0.032 | 0.059 | 0.496 |
| source_direction_z | paper_vae | trees | 106 | 0.027 | -0.063 | 0.082 | 0.493 |
| source_direction_z | input_pca8 | ridge | 106 | 0.015 | -0.058 | 0.050 | 0.502 |
| source_direction_z | input_pca8 | trees | 106 | -0.019 | -0.111 | 0.037 | 0.505 |

### Photon

| target | representation | probe | n_test | r2 | r2_low | r2_high | physical_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| incoming_ke_mev | paper_vae | ridge | 33 | 0.255 | 0.007 | 0.396 | 20.714 |
| incoming_ke_mev | paper_vae | trees | 33 | 0.328 | -0.023 | 0.558 | 20.863 |
| incoming_ke_mev | input_pca8 | ridge | 33 | 0.381 | 0.006 | 0.624 | 20.649 |
| incoming_ke_mev | input_pca8 | trees | 33 | 0.521 | 0.245 | 0.659 | 18.641 |
| momentum_mev | paper_vae | ridge | 33 | 0.255 | 0.007 | 0.396 | 20.714 |
| momentum_mev | paper_vae | trees | 33 | 0.328 | -0.023 | 0.558 | 20.863 |
| momentum_mev | input_pca8 | ridge | 33 | 0.381 | 0.006 | 0.624 | 20.649 |
| momentum_mev | input_pca8 | trees | 33 | 0.521 | 0.245 | 0.659 | 18.641 |
| deposited_mev | paper_vae | ridge | 113 | 0.647 | 0.579 | 0.712 | 17.563 |
| deposited_mev | paper_vae | trees | 113 | 0.724 | 0.669 | 0.774 | 16.443 |
| deposited_mev | input_pca8 | ridge | 113 | 0.743 | 0.677 | 0.791 | 16.586 |
| deposited_mev | input_pca8 | trees | 113 | 0.745 | 0.677 | 0.812 | 15.106 |
| geom_retained_energy_mev | paper_vae | ridge | 113 | 0.743 | 0.665 | 0.807 | 9.867 |
| geom_retained_energy_mev | paper_vae | trees | 113 | 0.797 | 0.751 | 0.838 | 10.378 |
| geom_retained_energy_mev | input_pca8 | ridge | 113 | 0.862 | 0.833 | 0.889 | 7.585 |
| geom_retained_energy_mev | input_pca8 | trees | 113 | 0.884 | 0.857 | 0.909 | 7.273 |
| geom_retained_electrons | paper_vae | ridge | 113 | 0.743 | 0.667 | 0.806 | 290667.686 |
| geom_retained_electrons | paper_vae | trees | 113 | 0.796 | 0.748 | 0.838 | 306885.623 |
| geom_retained_electrons | input_pca8 | ridge | 113 | 0.860 | 0.832 | 0.886 | 226066.499 |
| geom_retained_electrons | input_pca8 | trees | 113 | 0.884 | 0.857 | 0.910 | 214633.912 |
| mean_dedx_mev_cm | paper_vae | ridge | 113 | -0.013 | -0.057 | 0.019 | 0.079 |
| mean_dedx_mev_cm | paper_vae | trees | 113 | 0.007 | -0.083 | 0.080 | 0.078 |
| mean_dedx_mev_cm | input_pca8 | ridge | 113 | -0.013 | -0.064 | 0.022 | 0.082 |
| mean_dedx_mev_cm | input_pca8 | trees | 113 | 0.000 | -0.065 | 0.067 | 0.079 |
| linearity_3d | paper_vae | ridge | 113 | 0.079 | -0.088 | 0.126 | 0.046 |
| linearity_3d | paper_vae | trees | 113 | 0.134 | -0.129 | 0.208 | 0.043 |
| linearity_3d | input_pca8 | ridge | 113 | 0.086 | -0.029 | 0.139 | 0.045 |
| linearity_3d | input_pca8 | trees | 113 | 0.062 | -0.203 | 0.156 | 0.047 |
| transverse_rms_cm | paper_vae | ridge | 113 | 0.376 | 0.260 | 0.505 | 0.701 |
| transverse_rms_cm | paper_vae | trees | 113 | 0.329 | 0.218 | 0.440 | 0.737 |
| transverse_rms_cm | input_pca8 | ridge | 113 | 0.301 | 0.175 | 0.437 | 0.740 |
| transverse_rms_cm | input_pca8 | trees | 113 | 0.247 | 0.128 | 0.365 | 0.775 |
| extent_3d_cm | paper_vae | ridge | 113 | 0.248 | 0.131 | 0.338 | 22.230 |
| extent_3d_cm | paper_vae | trees | 113 | 0.250 | 0.077 | 0.386 | 22.038 |
| extent_3d_cm | input_pca8 | ridge | 113 | 0.258 | 0.097 | 0.375 | 21.141 |
| extent_3d_cm | input_pca8 | trees | 113 | 0.235 | 0.057 | 0.373 | 22.327 |
| n_fragments | paper_vae | ridge | 113 | 0.302 | 0.176 | 0.383 | 0.811 |
| n_fragments | paper_vae | trees | 113 | 0.284 | 0.150 | 0.387 | 0.816 |
| n_fragments | input_pca8 | ridge | 113 | 0.291 | 0.128 | 0.395 | 0.821 |
| n_fragments | input_pca8 | trees | 113 | 0.256 | 0.090 | 0.383 | 0.826 |
| visible_charge_fraction | paper_vae | ridge | 113 | 0.033 | -0.028 | 0.076 | 0.049 |
| visible_charge_fraction | paper_vae | trees | 113 | 0.073 | -0.053 | 0.129 | 0.048 |
| visible_charge_fraction | input_pca8 | ridge | 113 | 0.016 | -0.054 | 0.056 | 0.049 |
| visible_charge_fraction | input_pca8 | trees | 113 | 0.047 | -0.076 | 0.113 | 0.048 |
| geometric_energy_fraction | paper_vae | ridge | 113 | 0.061 | -0.013 | 0.116 | 0.119 |
| geometric_energy_fraction | paper_vae | trees | 113 | 0.042 | -0.069 | 0.138 | 0.120 |
| geometric_energy_fraction | input_pca8 | ridge | 113 | 0.032 | -0.060 | 0.096 | 0.120 |
| geometric_energy_fraction | input_pca8 | trees | 113 | -0.012 | -0.119 | 0.097 | 0.120 |
| assigned_xz_deg | paper_vae | ridge | 113 | 0.072 | -0.069 | 0.151 | 1.797 |
| assigned_xz_deg | paper_vae | trees | 113 | -0.036 | -0.165 | 0.050 | 1.843 |
| assigned_xz_deg | input_pca8 | ridge | 113 | 0.119 | -0.030 | 0.206 | 1.688 |
| assigned_xz_deg | input_pca8 | trees | 113 | 0.066 | -0.051 | 0.156 | 1.696 |
| assigned_yz_deg | paper_vae | ridge | 113 | -0.090 | -0.239 | -0.007 | 3.302 |
| assigned_yz_deg | paper_vae | trees | 113 | -0.103 | -0.281 | -0.003 | 3.312 |
| assigned_yz_deg | input_pca8 | ridge | 113 | -0.117 | -0.227 | -0.017 | 3.293 |
| assigned_yz_deg | input_pca8 | trees | 113 | -0.148 | -0.300 | -0.047 | 3.334 |
| vertex_x_cm | paper_vae | ridge | 113 | -0.063 | -0.173 | -0.006 | 28.414 |
| vertex_x_cm | paper_vae | trees | 113 | -0.082 | -0.227 | 0.002 | 28.600 |
| vertex_x_cm | input_pca8 | ridge | 113 | -0.046 | -0.157 | 0.030 | 28.077 |
| vertex_x_cm | input_pca8 | trees | 113 | -0.097 | -0.237 | 0.002 | 28.486 |
| vertex_y_cm | paper_vae | ridge | 113 | -0.014 | -0.133 | 0.013 | 53.377 |
| vertex_y_cm | paper_vae | trees | 113 | -0.056 | -0.194 | 0.002 | 54.038 |
| vertex_y_cm | input_pca8 | ridge | 113 | -0.052 | -0.167 | -0.015 | 54.105 |
| vertex_y_cm | input_pca8 | trees | 113 | -0.041 | -0.169 | 0.013 | 53.753 |
| vertex_z_cm | paper_vae | ridge | 113 | -0.024 | -0.094 | 0.009 | 45.526 |
| vertex_z_cm | paper_vae | trees | 113 | -0.036 | -0.123 | 0.018 | 45.683 |
| vertex_z_cm | input_pca8 | ridge | 113 | -0.054 | -0.121 | -0.012 | 46.345 |
| vertex_z_cm | input_pca8 | trees | 113 | -0.077 | -0.173 | -0.015 | 46.732 |
| source_direction_x | paper_vae | ridge | 113 | -0.079 | -0.172 | -0.032 | 0.543 |
| source_direction_x | paper_vae | trees | 113 | -0.119 | -0.233 | -0.038 | 0.545 |
| source_direction_x | input_pca8 | ridge | 113 | -0.068 | -0.160 | -0.018 | 0.542 |
| source_direction_x | input_pca8 | trees | 113 | -0.077 | -0.174 | -0.010 | 0.546 |
| source_direction_y | paper_vae | ridge | 113 | -0.026 | -0.133 | 0.002 | 0.487 |
| source_direction_y | paper_vae | trees | 113 | -0.044 | -0.163 | 0.018 | 0.487 |
| source_direction_y | input_pca8 | ridge | 113 | -0.034 | -0.141 | 0.009 | 0.490 |
| source_direction_y | input_pca8 | trees | 113 | -0.063 | -0.212 | 0.004 | 0.492 |
| source_direction_z | paper_vae | ridge | 113 | -0.008 | -0.100 | 0.037 | 0.475 |
| source_direction_z | paper_vae | trees | 113 | -0.044 | -0.161 | 0.027 | 0.482 |
| source_direction_z | input_pca8 | ridge | 113 | -0.071 | -0.193 | -0.001 | 0.486 |
| source_direction_z | input_pca8 | trees | 113 | -0.101 | -0.236 | -0.005 | 0.490 |

## Controls and limits

Additional controls use raw pixels, the proton-only VAE, three VAE seeds, an AE, an untrained encoder and shuffled labels. These are pixel/learned representations. Original-coordinate truth and derived physical profiles remain targets, not additional model inputs. The interaction-pair head concatenates the two representations directly.

| representation | n_test_pairs | n_test_events | auc | low | high | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| paper_vae | 589 | 160 | 0.718 | 0.665 | 0.767 | 0.658 |
| input_pca8 | 589 | 160 | 0.684 | 0.634 | 0.740 | 0.635 |

This sample uses truth-isolated particles and one exploratory proton-calibrated readout response. Incoming kinematics are accepted only when fragment momenta agree within 5% and energy closes; the retained-energy label is a voxel-centre geometric proxy, not exact waveform ancestry. Original shared vertices/directions were removed during placement. Kaon truth, full-event segmentation/ancestry and calibrated real-data task transfer are not tested. The previous domain and response-sensitivity results remain diagnostics of the same fixed sample.

## Provenance

Source: `train/generic_v2_77600_v2.h5`, published SHA-256 `3d48f9a6fed80ebe479d85ad3c88604430b480d8c08ee43a896b88c66dedbf00`.
Checkpoint SHA-256: `51fa1194fad6432e50c44647253758e195bf434306976de87ef6570cf2539e33`.
Response SHA-256: `37713c1c59af6aceb625f1c098f82ff5468ac362180abcbf07519ea0d76bcaa3`.
Ordered manifest SHA-256: `96945da969311c91f925360405efb56d3fe6a573e0425ad2e9f04d87f343904d`.

Data and numerical results: `/Volumes/easystore/proton-kaon/pilarnet_lariat/latent_truth/pilot_pixels_v2`. Plots and copied CSVs: `/Users/user/code/research/proton-kaon/output/pilarnet_lariat/latent_truth/pilot_pixels_v2`.

`tsne_pixel_baseline.png`: identical t-SNE settings on VAE8 and pixel PCA8; all selected simulation particles. `truth_pixel_baseline.png`: identical linear probes on these two eight-dimensional representations; asterisks indicate log1p targets and dashes indicate unsupported targets. Numerical readouts use all eight dimensions.
