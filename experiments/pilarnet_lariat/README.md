# PILArNet-M → approximate LArIAT images

All experiment code is isolated in this folder. It imports the existing image
padding and figure style without changing the training pipeline. Dataset files,
calibration caches and converted images stay on the external drive under
`/Volumes/easystore/proton-kaon/pilarnet_lariat`.

## Full dataset and proton calibration

The full download includes every file at the pinned public repository revision:
167,504,316,198 bytes across 26 files. `download.py` resumes `.partial` files,
checks expected sizes, verifies the published LFS SHA256 values, and atomically
renames verified files. `full/download_status.json` records progress. Do not
treat a `.partial` file as a complete HDF5 dataset.

The calibration uses **incoming kinetic energy**, as requested. LArIAT has a
beamline momentum match for all 10,466 proton images. The relativistic relation
`K = sqrt(p² + m²) - m` gives kinetic energy, with proton mass 938.272 MeV/c².
The existing feature-making proton order supplies the tensor's metadata order;
equal row counts are checked. The PILArNet metadata has mass values near
938.272 and momentum values near 0.3–1.0. The explicit convention is mass in
MeV/c² and momentum in GeV/c; the card does not specify momentum units, so the
cache report also checks its consistency with deposited energy. This convention
must be audited rather than silently assuming both columns use the same scale.

Beamline kinetic energy is not automatically kinetic energy at the TPC face:
material upstream removes energy. No upstream-loss correction is available in
the current labels. The calibration records this limitation; it does not warp
track lengths or redistribute energy to hide it.

`calibrate.py prepare` saves raw reference images, energy labels and deterministic
partitions. Twenty percent of the distinct LArIAT runs are reserved; duplicate
images and repeated events remain in a single partition. There are 8,878 fit
images and 1,588 validation images. An effective joint xz/yz angle prior is
estimated from stereo tangents in **fit-only endpoint crops**. This includes
scattering and is not a recovered initial beam-direction distribution.

`calibrate.py fit` caches primary-like protons from the training shard and holds
out entire PILArNet events. It fits global collection and induction amplitude
factors plus the shaping peak. The objective compares ADC summaries, occupancy,
wire extent, time width and stopping-profile fractions within fixed incoming-KE
bins. Low acceptance incurs a penalty. The fit has a finite search budget and
is a pilot, not a guarantee of a converged or unique physical calibration.

The held-out report lists energy-bin support, remaining distribution distances,
acceptance and a feature-based domain-classifier AUC after matching energy-bin
populations. An AUC near 0.5 is chance; a high value reveals remaining differences.
Matching summary distributions does not demonstrate identical image distributions.

The initial 320-proton pilot reduced the fit distance from 2.118 to 1.222,
but the held-out distance was 1.556 and the domain-classifier AUC was **1.0**.
The images remain readily distinguishable. Only the 150–200 and 200–250 MeV
validation bins had the required sample support; no held-out simulated protons
covered 250–800 MeV. This response is not ready to claim proton agreement or
to produce the requested matched full dataset. More energy-balanced protons and
an audit of the beamline-to-TPC energy difference are needed before bulk use.
The full download can continue independently. The bulk command below is for
use after evaluating a satisfactory calibration; it does not enforce a pass
criterion automatically.

```sh
.venv/bin/python experiments/pilarnet_lariat/download.py \
  --output /Volumes/easystore/proton-kaon/pilarnet_lariat/full

.venv/bin/python experiments/pilarnet_lariat/calibrate.py prepare
.venv/bin/python experiments/pilarnet_lariat/calibrate.py fit \
  --wait-for-input --max-particles 320 --max-evaluations 48

.venv/bin/python experiments/pilarnet_lariat/apply_all.py --wait
```

The bulk converter scans all verified HDF5 files as they become available. It
freezes the response and fit-only angle bank, records their checksums, and applies
the same response to every non-LED particle group, including electrons and photons.
It never refits by PID. Particles with no surviving signal, invalid path lengths
or an incompatible crop are recorded with rejection reasons. For displaced or
very short groups, the direction estimate uses the interaction-to-visible-deposit
vector and is explicitly flagged as approximate; true generator starts/directions
are not available in this schema. The nearest-visible gap is preserved.

Converted output consists of resumable, compressed HDF5 parts rather than millions
of separate waveform files. Each image has incoming KE, PID, original event/group
IDs, acceptance and component fractions, direction-quality and energy-extrapolation
flags. Images are raw ADC `(2,48,48)`; apply `log1p` once for the current model.
LED is amorphous event-level deposition and is counted separately, not assigned
a fabricated beam-particle image. `converted/conversion_status.json` and per-file
`progress.json` record progress. A completed bulk conversion is an experimental
transfer dataset, not a claim that the proton agreement has passed validation.

Run checks with:

```sh
.venv/bin/python -m unittest discover -s experiments/pilarnet_lariat/tests
```

This prototype converts labelled 3D deposits into signed collection/induction
wire–time waveforms and the project's two-plane `(2, 48, 48)` model input.
It is a detector-response approximation for transfer experiments, not a validated
LArIAT simulation or evidence of model performance on another detector.

The first run uses six actual events from the public
[PILArNet-M test split](https://huggingface.co/datasets/DeepLearnPhysics/PILArNet-M),
pinned to revision `f32f36bd1c17d707d0a24f0c63ec16419475c20f`. It produced
24 particle pairs: 3 protons, 13 muons and 8 charged pions. Four candidates
were rejected because their interaction vertex was displaced from the track.
Processing stopped at 24 accepted particles. PILArNet-M has **no kaon class**.

## What the conversion does

1. Join fragments with the same particle group ID. Retain muons, charged pions
   and protons with at least 20 voxels. This isolates one truth particle;
   it does not collect its daughter particles or reproduce event reconstruction.
2. Convert 3-mm voxel coordinates to centimetres. Estimate the initial tangent
   from deposits within 3 cm of the interaction vertex, reject tracks whose
   closest deposit is more than 1 cm from that vertex, and rigidly rotate/translate
   the deposits to enter the LArIAT front face near `(x,y,z)=(22.5,0,0)` cm.
   Rotation preserves physical lengths, scattering and deposited energy.
3. Convert the winning particle's deposited energy and voxel path length to
   ionization electrons using Modified Box recombination. The supplied electron
   counts are available as an alternative mode, without applying recombination
   again; their original simulation conventions still need checking.
4. Integrate charge uniformly within each coarse voxel, clip to the active
   volume, attenuate for electron lifetime, map drift distance to arrival time,
   and apply approximate diffusion. Project onto the two stereo wire coordinates.
5. Apply a unipolar collection pulse and a bipolar induction pulse, scaled by
   hardware gain and configurable plane response factors. Save the signed
   `(2,240,3072)` waveforms, with collection first, induction second.
6. Threshold at 15 ADC in collection and 7 ADC in induction, select the largest
   positive connected component, and report fragmentation. Reuse the existing
   50-wire endpoint crop, time alignment, `(51,1502)` canvas, bilinear resizing
   to `(48,48)`, and `log1p`. The raw tensor is available for the training loader
   to transform once; do not apply `log1p` again to the transformed tensor.

The wire coordinate increases downstream in both views, so `origin='upper'`
shows a beam-aligned particle progressing from top to bottom. Its time-axis
slope follows the drift component of its 3D direction. The two images come from
one common physical rotation, rather than independent image rotations.

Long through-going particles are cropped at the detector boundary; that crop
is not a stopping endpoint. The converter does not apply the real sample's
fiducial, minimum-height or cross-plane matching cuts.

## Detector parameters and their status

The supplied detector paper is
[1911.10379v2.pdf](</Users/user/CDT/PhD Projects/LARIAT PAPERS/1911.10379v2.pdf>).
The public [LArIAT software](https://github.com/ArCS-FNAL/lariatsoft) provides
geometry, clock and response configuration. Defaults target **Run II**:
different runs have different wire pitches and calibrations.

| Parameter | Default | Basis / limitation |
| --- | --- | --- |
| Active dimensions | 47.3 cm main drift × 40 cm height × 90 cm length | Paper Sec. 5, approximate rectangular acceptance |
| Wire geometry | 240 wires per plane, 4 mm pitch, ±60° to beam | Paper Table 1; analytic stereo normals, not imported channel-map geometry |
| Wire coordinate | `u_col=0.5y+√3z/2`, `u_ind=−0.5y+√3z/2` | Analytic convention; wire-0 offset from [GDML](https://github.com/ArCS-FNAL/lariatsoft/blob/develop/Geo/gdml/lariat.gdml), sign/channel mapping needs data validation |
| Sampling / readout | 128 ns / 3072 ticks | Detector paper and [clock configuration](https://github.com/ArCS-FNAL/lariatsoft/blob/develop/Utilities/detectorclocks_lariat.fcl) |
| Trigger offset | 24.2 µs before trigger | Clock configuration; detailed simulation/calibration time offsets omitted |
| Drift field | 0.4865 kV/cm | Run II [detector configuration](https://github.com/ArCS-FNAL/lariatsoft/blob/develop/Utilities/detectorproperties_lariat.fcl); paper reports related measured/nominal values |
| Drift speed | 0.148 cm/µs | Approximation from paper Fig. 34 drift window; should be fitted for the data period |
| Lifetime | 1600 µs | MC default; actual lifetime changes with run/day |
| Recombination | Modified Box, α=0.93, β=0.212 | Parameters cited in paper; voxel-level quenching approximates step-level physics |
| Electronics | 3 µs shaping, 25 mV/fC, 0.5 mV/ADC | Paper/readout and [signal configuration](https://github.com/ArCS-FNAL/lariatsoft/blob/develop/Utilities/signalservices_lariat.fcl) |
| Plane response factors | Collection 0.50, induction 0.66 | Signal configuration; used with a toy pulse, not the original field-response calculation |
| Beam angular spread | Optional Gaussian 5° in xz, 3.3° in yz | [Beamlike MC prior](https://github.com/ArCS-FNAL/lariatsoft/blob/develop/JobConfigurations/prodsingle/mc_gen_prodsingle_beamlike.fcl), **not a measured beam distribution** |
| Diffusion / induction-lobe timing | Configurable priors | Need calibration; noise is disabled by default |

For an individual deposit, the approximate mapping is
`wire = round((u - u0)/pitch)` and
`tick = trigger_tick + (x/v_d + plane_gap + t_dep)/Δt`.
Plane-gap delays are approximate; deposition time is retained relative to the
particle's earliest deposit.

The hardware gain implies about **0.00801 ADC per electron at the pulse peak**,
before the response factors. This is not an integrated-area conversion.
The paper's Table 4 electron↔ADC calibration applies to processed integrated
hit area, with ADC×tick units; using it directly as a per-tick amplitude gain
would mix different quantities. Matching our ADC requires comparing the final
waveforms and preprocessing against real LArIAT pulses.

## Reproduce and inspect

The additional HDF5 reader can be installed with
`uv pip install --python .venv/bin/python -r experiments/pilarnet_lariat/requirements.txt`.
Run from the repository root:

```sh
.venv/bin/python experiments/pilarnet_lariat/fetch_sample.py \
  --output /Volumes/easystore/proton-kaon/pilarnet_lariat/input

.venv/bin/python experiments/pilarnet_lariat/convert.py \
  --input /Volumes/easystore/proton-kaon/pilarnet_lariat/input/event_*.npz \
  --output /Volumes/easystore/proton-kaon/pilarnet_lariat/prototype \
  --beam-jitter --max-particles 24

.venv/bin/python experiments/pilarnet_lariat/gallery.py \
  --input /Volumes/easystore/proton-kaon/pilarnet_lariat/prototype

.venv/bin/python -m unittest discover -s experiments/pilarnet_lariat/tests -p 'test_lariat_forward.py'
```

The bounded downloader fetched only 2 MiB for these six events, instead of the
whole 7-GB HDF5 file. Its default transfer cap is 100 MiB. Local HDF5 files can
also be supplied directly to the converter using `--event-index`.
Use `--xz-deg` / `--yz-deg` for a fixed entrance direction, and `--charge-mode
electrons` for supplied charge. The response parameters live in
`experiments/pilarnet_lariat/configs/run2.yaml`.

Outputs under `/Volumes/easystore/proton-kaon/pilarnet_lariat/prototype`:

- `manifest.json`: placement, accepted charge, drift/diffusion, pulse ranges,
  connected components and rejection reasons for every processed candidate.
- Per-particle `.npz`: placed coordinates, energies, charges, signed waveforms,
  padded images, raw 48×48 images and transformed images.
- `lariat_like_raw.pt`: species tensors `p`, `m`, `pi` and combined `all`.
- `lariat_like_log1p.pt`: transformed combined tensor `all`.

Example PNG/PDF figures are saved under `output/pilarnet_lariat`.
Tests check rotation without reflection/rescaling, the two stereo slopes,
recombination bounds, drift timing, pulse gain/area units, charge accounting,
fragment grouping and the model input shape.

## What needs matching before a physics evaluation

Use beamline/reconstructed entrance directions to fit the joint xz/yz angle
distribution and entry-position distribution. A narrow nominal +z beam or the
existing MC widths only give a starting point. Then match isolated real pulses:
ADC scale, bipolar shape, shaping time, noise, lifetime and threshold survival.
Finally apply the same clustering, plane matching, fiducial and crop selections
as the real dataset, and compare pixel distributions, slopes and stopping profiles
before passing the images through the frozen model.

PILArNet-M's 3-mm voxels cannot recover finer spatial structure or true
sub-voxel arrival times. At the default drift speed, one voxel spans about
16 time ticks; uniform integration adds an assumption, not new information.
The original simulated smearing/charge conventions must be audited before adding
LArIAT diffusion. The estimated tangent, coarse recombination, mean-drift
diffusion, simplified field response, absent electronics noise/saturation/channel
variation, and missing daughter groups all limit realism. No trained-model
inference or external performance claim has been made in this prototype.
