# PILArNet-M → approximate LArIAT images

All experiment code is isolated in this folder. It imports the existing image
padding and figure style without changing the training pipeline. Dataset files,
calibration caches and converted images stay on the external drive under
`/Volumes/easystore/proton-kaon/pilarnet_lariat`.

## Proton physics check with the existing Bethe–Bloch model

The current objective is **similar physical stopping and ADC profiles**, with
real fluctuations allowed. Incoming kinetic energy remains the primary label.
`proton_profiles.py` uses the existing model at
`/Volumes/easystore/proton-deuteron/protons.txt`, with the residual-range reversal
used by `src/bethe_bloch.py`. Its SHA256 is
`12bf19856e02374bec7788090584d55310401be7b25be7a59c4138b9efbb8d0d`.
Bin averages are computed by integrating the table, avoiding a point-value
comparison across the rapidly changing Bragg region. Original table provenance
has not been recovered; range energies are diagnostic stopping-energy estimates.

**A step-length unit correction supersedes the original pilot fit.** The pinned
[PILArNet-M card](https://huggingface.co/datasets/DeepLearnPhysics/PILArNet-M)
says `dx` is in mm. Numerical checks of 320 cached protons instead support cm:
the fit-particle median predicted/supplied electron ratio is 1.0012 when numeric
`dx` is treated as cm, versus 0.2535 with the card's mm convention. Sum(dx) divided
by the endpoint chord is 1.0063 under the cm interpretation. This is an inference
from local data, not confirmation by the publisher. The configurable conversion
is now `pilarnet_dx_cm_per_unit=1.0`. Old responses without that convention cannot
be resumed or applied to the full dataset. Original 24-particle previews predate
this correction and must be regenerated before quantitative use.

The reconstructed reference is
`/Volumes/easystore/proton-deuteron/protons/hist_bbox_100a_RecoBBox100A_20250815T193002.root`.
Its native collection-track calorimetry supplies dE/dx, residual range, pitch and
3D positions; matched hits supply integrated ADC areas and pulse widths. The
local analyser source fills per-hit `hit_dEds` using unmatched hit keys, so that
field is excluded. There are 941 unique single WC-matched tracks after excluding
ambiguous image events and conflicting reconstruction duplicates: 814 fit and
127 held-out events. A contained endpoint, Bragg rise and calorimetry coverage
identify 792 stopping candidates. An additional raw-cluster endpoint audit finds
458/941 reconstructed endpoints within three wires of the raw signal endpoint.
These are quality proxies, not truth-level stopping labels.

In residual-range bins supported over 0.5–25 cm, median absolute log deviations
from the model are 0.0212/0.0226 for fit/held-out PILArNet protons. LArIAT gives
0.3487/0.3234, with median dE/dx/model ratios 0.706/0.729. A positive residual-range
offset improves some real profiles and correlates with incomplete raw endpoints;
it is recorded as a diagnostic only. It never shifts tracks or changes energies.
Mean-loss versus reconstructed-loss conventions, calibration bias, incomplete
tracks and non-stopping contamination require further checking.

Incoming-energy pairs use ±5 MeV, the same partition, stopping proxies and at
least four common calorimetry bins. They select the nearest energy, without
optimizing dE/dx or ADC similarity. This yields 22 pairs (20 fit, **only two
held-out**) with median PILArNet/reconstructed range ratio 2.412. A separate
TPC-range-energy diagnostic additionally requires range agreement within 25%,
raw endpoint agreement within three wires and selects by physical dE/dx; it
yields 221 pairs (180 fit, 41 held-out), median range ratio 1.010. Incoming beam
KE is retained alongside each diagnostic estimate. These selected pairs do not
establish population agreement or a verified beamline-to-TPC energy correction.
The calorimetric energy integral is typically only about 64% of the range
estimate, another unresolved reconstruction/calibration discrepancy.

Fit-only hit-area conventions give 0.09331 collection and 0.03517 induction
ADC×ticks/electron, close to the detector paper's processed-hit calibrations.
This is circular with the existing calorimetry calibration and is not an
independent electronics-gain measurement. The toy pulse uses an explicit
**positive integrated-area** normalization, with a signed negative induction
lobe, rather than treating ADC×ticks as a peak gain. Fit-only pulse widths give
a 2.84 µs shaping peak. A pilot fits just two global raw-image amplitude
corrections (1.20 collection, 0.95 induction) on 32 TPC-range fit pairs; it does
not warp dE/dx, lengths, energies or pixels. Re-simulation includes thresholds
and crop selection after fitting.

Of 32 held-out TPC-range image pairs, 21 have nearby held-out real stopping
protons within ±5 MeV, 25% range, 3° xz and 5° yz. The median normalized absolute
row-maximum profile discrepancies are 0.364 collection / 0.557 induction, versus
0.348 / 0.533 for real-to-real variation: ratios **1.045 / 1.044**. This supports
similar ADC profiles in this small, physically selected diagnostic. Paired real
entry positions/directions are used as a geometry control; independent angle
sampling, energy coverage, completeness and incoming-KE agreement remain
unvalidated. It is not sufficient evidence to freeze a response for all species.

All caches, source audits, matches, responses and validation reports are under
`/Volumes/easystore/proton-kaon/pilarnet_lariat/profile_matching`. Reproduce after
preparing the reference and 320-proton cache:

```sh
.venv/bin/python experiments/pilarnet_lariat/proton_profiles.py prepare
.venv/bin/python experiments/pilarnet_lariat/proton_profiles.py judge
.venv/bin/python experiments/pilarnet_lariat/proton_profiles.py compare --max-pairs 32
.venv/bin/python experiments/pilarnet_lariat/proton_profiles.py fit-adc
.venv/bin/python experiments/pilarnet_lariat/proton_profiles.py compare --max-pairs 32 --fitted
.venv/bin/python experiments/pilarnet_lariat/proton_profiles.py validate-adc
MPLCONFIGDIR=/private/tmp/proton-profile-matplotlib .venv/bin/python \
  experiments/pilarnet_lariat/plot_proton_profiles.py --fitted
```

The two figures in `output/pilarnet_lariat/proton_profiles` show the dE/dx
comparison, held-out collection ADC profiles and representative held-out image
pairs. Bands span the 16th–84th percentiles; the incoming ADC panel has only two
events. The image pair is selected at the median physical-profile discrepancy,
without choosing the best-looking ADC match. Original legacy bulk conversion
expects `calibrate.py` outputs; it does not consume this exploratory profile fit.

## Full dataset and original calibration workflow

The full download includes every file at the pinned public repository revision:
167,504,316,198 bytes across 26 files. `download.py` resumes `.partial` files,
checks expected sizes, verifies the published LFS SHA256 values, and atomically
renames verified files. `full/download_status.json` records progress. Do not
treat a `.partial` file as a complete HDF5 dataset.
`--background` detaches the process so it survives an interrupted chat turn;
`full/download.log` and the status JSON record its progress and PID. A process
lock prevents concurrent downloaders writing the same output, and SHA256 checks
run sequentially to avoid competing full-file reads on the external HDD.

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

The original 320-proton summary-distribution pilot used the card's incorrect
inferred step-length convention, so its fitted response and distance metrics
are obsolete. The cache also has inadequate held-out support above 250 MeV.
Use the physical profile checks above to diagnose the response before refitting
this older workflow. More energy-balanced protons and an audit of the beamline
and TPC energy references are needed before bulk conversion. The full download
continues independently; the bulk command below is for a future satisfactory
calibration and does not enforce a scientific pass criterion automatically.

```sh
.venv/bin/python experiments/pilarnet_lariat/download.py \
  --output /Volumes/easystore/proton-kaon/pilarnet_lariat/full --background

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
   hardware gain and configurable plane response factors, or explicitly supplied
   positive integrated-area calibrations. Save the signed
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
