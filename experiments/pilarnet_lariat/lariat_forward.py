"""Approximate Run II LArIAT readout for labelled 3D energy deposits.

Coordinates: x = distance from the shield plane toward the cathode, y = up,
z = beam direction from the TPC front face, all in cm. Output axes are
(collection/induction, wire, time tick), matching this project's Event loader.
This is a configurable forward-model prototype, not LArSoft detector simulation.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from skimage.measure import label, regionprops


@dataclass(frozen=True)
class LArIATResponse:
    voxel_cm: float = 0.3
    # Numeric convention inferred from geometry and supplied electrons.
    # The pinned card says mm, but PILArNet-M stored dx behaves as cm.
    pilarnet_dx_cm_per_unit: float = 1.0
    wire_pitch_cm: float = 0.4
    wires: int = 240
    ticks: int = 3072
    sample_us: float = 0.128
    drift_cm_us: float = 0.148
    field_kv_cm: float = 0.4865
    density_g_cm3: float = 1.38
    main_drift_cm: float = 47.3
    height_cm: float = 40.0
    length_cm: float = 90.0
    # Analytic stereo geometry. Offset is derived from GDML wire-0 centre.
    wire_origin_cm: float = -8.4195036381
    trigger_tick: float = 24.2 / 0.128
    induction_gap_us: float = 2.3
    collection_gap_us: float = 4.3
    lifetime_us: float = 1600.0
    # Configurable priors, not measured Run II diffusion coefficients.
    transverse_diffusion_cm2_s: float = 13.0
    longitudinal_diffusion_cm2_s: float = 4.8
    shaping_peak_us: float = 3.0
    adc_peak_per_electron: float = 25.0 * 1.602176634e-4 / 0.5
    collection_response_scale: float = 0.50
    induction_response_scale: float = 0.66
    induction_lobe_delay_us: float = 2.0
    induction_negative_ratio: float = 0.85
    # Integrated positive-pulse calibration, rather than a per-tick ADC gain.
    collection_area_adc_ticks_per_electron: float | None = None
    induction_positive_area_adc_ticks_per_electron: float | None = None
    noise_rms_adc: float = 0.0
    subvoxel_samples: int = 3
    collection_threshold_adc: float = 15.0
    induction_threshold_adc: float = 7.0

    def __post_init__(self):
        positive = ('voxel_cm', 'pilarnet_dx_cm_per_unit', 'wire_pitch_cm', 'sample_us', 'drift_cm_us',
                    'field_kv_cm', 'density_g_cm3', 'main_drift_cm', 'height_cm',
                    'length_cm', 'lifetime_us', 'shaping_peak_us',
                    'adc_peak_per_electron', 'induction_lobe_delay_us')
        for name in positive:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        for name in ('wires', 'ticks', 'subvoxel_samples'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('collection_area_adc_ticks_per_electron',
                     'induction_positive_area_adc_ticks_per_electron'):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be positive and finite when provided')
        for name in ('transverse_diffusion_cm2_s', 'longitudinal_diffusion_cm2_s',
                     'noise_rms_adc', 'collection_threshold_adc',
                     'induction_threshold_adc', 'collection_response_scale',
                     'induction_response_scale', 'induction_negative_ratio'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative and finite')


def particle_groups(points, clusters, extras):
    """Map the official PILArNet-M variable-length arrays to particle groups.

    Group IDs join fragments of ONE particle. They do not recover a decay tree.
    The metadata vertex belongs to the interaction, not necessarily this particle.
    """
    points, clusters, extras = map(np.asarray, (points, clusters, extras))
    if points.ndim != 2 or points.shape[1] != 8:
        raise ValueError('point must have shape (N, 8)')
    if clusters.ndim != 2 or clusters.shape[1] != 6 or extras.shape != (len(clusters), 5):
        raise ValueError('cluster and cluster_extra must have shapes (M, 6) and (M, 5)')
    if not all(np.isfinite(a).all() for a in (points, clusters, extras)):
        raise ValueError('Input arrays contain nonfinite values')
    if not np.equal(clusters[:, :6], np.floor(clusters[:, :6])).all():
        raise ValueError('Cluster counts and labels must be integers')
    counts = clusters[:, 0].astype(int)
    if np.any(counts < 0) or counts.sum() != len(points):
        raise ValueError('Cluster point counts do not match the point array')
    edges = np.r_[0, np.cumsum(counts)]
    for group in np.unique(clusters[:, 2].astype(int)):
        if group < 0:
            continue
        indices = np.flatnonzero(clusters[:, 2] == group)
        pids = np.unique(clusters[indices, 5].astype(int))
        if len(pids) != 1:
            raise ValueError(f'Conflicting PID labels in group {group}')
        yield {
            'group_id': int(group), 'pid': int(pids[0]),
            'points': np.concatenate([points[edges[i]:edges[i+1]] for i in indices]),
            'vertex_voxels': extras[indices[0], 2:5],
            'momentum': float(extras[indices[0], 1]),
            'mass_mev': float(extras[indices[0], 0]),
        }


def rotation_between(source, target):
    """Proper rotation (no reflection or rescaling), including antiparallel axes."""
    source, target = (np.asarray(v, dtype=float) for v in (source, target))
    if source.shape != (3,) or target.shape != (3,):
        raise ValueError('Directions must be 3-vectors')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Directions must be finite')
    if np.linalg.norm(source) == 0 or np.linalg.norm(target) == 0:
        raise ValueError('Directions cannot be zero')
    a, b = source / np.linalg.norm(source), target / np.linalg.norm(target)
    cosine = np.clip(a @ b, -1, 1)
    if cosine > 1 - 1e-12:
        return np.eye(3)
    if cosine < -1 + 1e-12:
        axis = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    v = np.cross(a, b)
    cross = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + cross + cross @ cross / (1 + cosine)


def place_particle(points, vertex_voxels, response, target_direction=(0, 0, 1),
                   entry_cm=(22.5, 0, 0), source_direction=None, allow_displaced=False):
    """Place a primary particle at the beam entrance, preserving all distances.

    Without a true direction, estimate an initial tangent from deposits within
    3 cm of the interaction vertex. Reject displaced secondaries and tiny tracks.
    This geometric estimate is recorded; it is not generator truth.
    """
    xyz = np.asarray(points[:, :3], dtype=float) * response.voxel_cm
    vertex = np.asarray(vertex_voxels, dtype=float) * response.voxel_cm
    relative = xyz - vertex
    distances = np.linalg.norm(relative, axis=1)
    direction_method = 'provided' if source_direction is not None else 'local PCA at interaction vertex'
    if source_direction is None:
        if distances.min() > 1.0:
            if not allow_displaced:
                raise ValueError('Particle is displaced from the interaction vertex; true start needed')
            # For conversion-gap showers / displaced secondaries this retains
            # the vertex-to-first-visible gap. It is explicitly approximate,
            # not a recovered particle start or generator direction.
            near = relative[distances <= distances.min()+.6]
            source_direction = near.mean(axis=0)
            direction_method = 'interaction vertex to nearest visible deposits (approximate)'
        else:
            near = relative[distances <= 3.0]
            if len(near) < 5:
                if not allow_displaced:
                    raise ValueError('Too few deposits to estimate the initial track direction')
                source_direction = near.mean(axis=0)
                if np.linalg.norm(source_direction) == 0:
                    source_direction = np.array([0.,0.,1.])
                direction_method = 'small-deposit centroid direction (approximate)'
            else:
                _, _, vectors = np.linalg.svd(near - near.mean(axis=0), full_matrices=False)
                source_direction = vectors[0]
                if source_direction @ near.mean(axis=0) < 0:
                    source_direction = -source_direction
    rotation = rotation_between(source_direction, target_direction)
    placed = relative @ rotation.T + np.asarray(entry_cm, dtype=float)
    return placed, {'source_direction_estimate': np.asarray(source_direction).tolist(),
                    'target_direction': np.asarray(target_direction).tolist(),
                    'entry_cm': list(entry_cm), 'rotation': rotation.tolist(),
                    'direction_method': direction_method,
                    'nearest_deposit_to_vertex_cm': float(distances.min())}


def ionization_electrons(energy_mev, dx_cm, response):
    """Forward Modified Box recombination, using dE/dx = voxel energy/path length.

    Alpha=0.93 and beta=0.212 are the ArgoNeuT parameters cited by the LArIAT
    paper. Voxel aggregation makes this an approximation to step-level quenching.
    Do not apply this again to already quenched electron counts.
    """
    energy, dx = np.asarray(energy_mev, float), np.asarray(dx_cm, float)
    if energy.shape != dx.shape or not np.isfinite(energy).all() or not np.isfinite(dx).all():
        raise ValueError('Energy and path lengths must be aligned finite arrays')
    if np.any(energy < 0) or np.any(dx <= 0):
        raise ValueError('Energy must be nonnegative and path length positive')
    beta = 0.212 / (response.density_g_cm3 * response.field_kv_cm)
    argument = 0.93 + beta * energy / dx
    return np.clip(dx * np.log(argument) / (beta * 23.6e-6), 0, energy / 23.6e-6)


def wire_coordinates(xyz_cm, response):
    """Stereo wire normals, ordered collection then induction.

    Both wire numbers increase downstream for a +z beam. Wire angle +/-60 deg
    is measured relative to z; the measured coordinate is perpendicular to wire.
    The signs are the analytic convention corresponding to the GDML planes.
    """
    yz = np.asarray(xyz_cm)[:, 1:3]
    normals = np.array([[0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2]])
    return (yz @ normals.T - response.wire_origin_cm) / response.wire_pitch_cm


def response_kernels(response):
    """Toy semi-Gaussian electronics and bipolar induction field response.

    The default hardware gain normalizes the pulse peak. Optional reconstructed
    hit-area calibrations instead normalize the sum of positive samples in
    ADC×ticks/electron; they are never used as per-tick peak gains. The negative
    induction lobe remains signed and is excluded from that positive area.
    """
    time = np.arange(0, 12 * response.shaping_peak_us, response.sample_us)
    u = time / response.shaping_peak_us
    collection = u**3 * np.exp(3 * (1 - u))
    collection /= collection.max()
    delay = max(1, round(response.induction_lobe_delay_us / response.sample_us))
    induction = np.pad(collection, (0, delay))
    induction -= response.induction_negative_ratio * np.pad(collection, (delay, 0))
    induction /= induction.max()
    gain = response.adc_peak_per_electron
    kernels = [collection * gain * response.collection_response_scale,
               induction * gain * response.induction_response_scale]
    areas = (response.collection_area_adc_ticks_per_electron,
             response.induction_positive_area_adc_ticks_per_electron)
    for plane, area in enumerate(areas):
        if area is not None:
            basis = (collection, induction)[plane]
            kernels[plane] = basis * area / np.maximum(basis, 0).sum()
    return tuple(kernels)


def simulate_readout(xyz_cm, electrons, deposition_ns, response, seed=0, windowed=False):
    """Subvoxel integration, lifetime, diffusion, wire/time binning and response.

    Diffusion uses the charge-weighted mean drift time of this particle. No space
    charge, neighbouring-wire field response, channel gains or ADC saturation.
    """
    xyz, charge, times = map(lambda a: np.asarray(a, float),
                             (xyz_cm, electrons, deposition_ns))
    if xyz.ndim != 2 or xyz.shape[1] != 3 or charge.shape != (len(xyz),) or times.shape != charge.shape:
        raise ValueError('Expected xyz (N,3), electrons (N,), and deposition_ns (N,)')
    if len(xyz) == 0 or not all(np.isfinite(a).all() for a in (xyz, charge, times)) or np.any(charge < 0):
        raise ValueError('Readout inputs must be nonempty, finite, and charge nonnegative')
    n = response.subvoxel_samples
    offsets = (np.arange(n) + 0.5) / n - 0.5
    shifts = np.array(list(product(offsets, repeat=3))) * response.voxel_cm
    locations = (xyz[:, None, :] + shifts).reshape(-1, 3)
    weights = np.repeat(charge / len(shifts), len(shifts))
    time_ns = np.repeat(times, len(shifts))
    inside = ((locations[:, 0] >= 0) & (locations[:, 0] <= response.main_drift_cm)
              & (np.abs(locations[:, 1]) <= response.height_cm / 2)
              & (locations[:, 2] >= 0) & (locations[:, 2] <= response.length_cm))
    volume_charge = float(weights[inside].sum())
    locations, weights, time_ns = locations[inside], weights[inside], time_ns[inside]
    if weights.sum() <= 0:
        raise ValueError('No charge survives the detector-volume acceptance')
    drift = locations[:, 0] / response.drift_cm_us
    weights *= np.exp(-drift / response.lifetime_us)
    mean_drift = float(np.average(drift, weights=weights))
    sigma_wire = np.sqrt(2 * response.transverse_diffusion_cm2_s * mean_drift / 1e6) / response.wire_pitch_cm
    sigma_tick = np.sqrt(2 * response.longitudinal_diffusion_cm2_s * mean_drift / 1e6) / response.drift_cm_us / response.sample_us
    wires = wire_coordinates(locations, response)
    kernels = response_kernels(response)
    tick_coordinates = [response.trigger_tick + (drift + gap + time_ns / 1000) / response.sample_us
                        for gap in (response.collection_gap_us, response.induction_gap_us)]
    wire_lo, wire_hi, tick_lo, tick_hi = 0, response.wires, 0, response.ticks
    if windowed:
        # Include the exact support used by gaussian_filter (truncate=4) and
        # the causal pulse. Cutting surrounding zeros preserves the model crop.
        wire_margin = int(4 * sigma_wire + .5) + 1
        tick_margin = int(4 * sigma_tick + .5) + 1
        wire_lo = max(0, int(np.floor(wires.min() + .5)) - wire_margin)
        wire_hi = min(response.wires, int(np.floor(wires.max() + .5)) + wire_margin + 1)
        tick_lo = max(0, int(np.floor(min(t.min() for t in tick_coordinates))) - tick_margin)
        tick_hi = min(response.ticks, int(np.ceil(max(t.max() for t in tick_coordinates)))
                      + tick_margin + max(map(len, kernels)) + 1)
        if wire_hi <= wire_lo or tick_hi <= tick_lo:
            raise ValueError('No charge falls in the instrumented readout window')
    output, audit = [], []
    for plane, gap_us in enumerate((response.collection_gap_us, response.induction_gap_us)):
        ticks = tick_coordinates[plane]
        wi = np.floor(wires[:, plane] + 0.5).astype(int)
        ti = np.floor(ticks).astype(int)
        frac = ticks - ti
        histogram = np.zeros((wire_hi-wire_lo, tick_hi-tick_lo), dtype=float)
        accepted = 0.0
        for tick_offset, portion in ((0, 1-frac), (1, frac)):
            valid = (wi >= 0) & (wi < response.wires) & (ti+tick_offset >= 0) & (ti+tick_offset < response.ticks)
            np.add.at(histogram, (wi[valid]-wire_lo, ti[valid]+tick_offset-tick_lo), weights[valid]*portion[valid])
            accepted += float(np.sum(weights[valid]*portion[valid]))
        blurred = gaussian_filter(histogram, (sigma_wire, sigma_tick), mode='constant')
        shaped = fftconvolve(blurred, kernels[plane][None, :], mode='full')[:, :tick_hi-tick_lo]
        # Remove FFT roundoff before adding explicitly requested noise.
        shaped[np.abs(shaped) < 1e-9] = 0
        if response.noise_rms_adc:
            shaped += np.random.default_rng(seed + plane).normal(0, response.noise_rms_adc, shaped.shape)
        output.append(shaped.astype(np.float32))
        audit.append({'accepted_electrons': accepted,
                      'after_diffusion_electrons': float(blurred.sum()),
                      'peak_adc': float(shaped.max()), 'minimum_adc': float(shaped.min()),
                      'waveform_area_adc_ticks': float(shaped.sum())})
    return np.stack(output), {'input_electrons': float(charge.sum()),
                             'inside_volume_electrons': volume_charge,
                             'after_lifetime_electrons': float(weights.sum()),
                             'mean_drift_us': mean_drift, 'sigma_wires': float(sigma_wire),
                             'sigma_ticks': float(sigma_tick), 'planes': audit,
                             'window_origin_wire_tick': [wire_lo, tick_lo],
                             'window_shape': [wire_hi-wire_lo, tick_hi-tick_lo]}


def prepare_model_input(waveforms, response):
    """Reuse the project's positive-region crop, 50-wire pad and interpolation.

    The selected particle is already isolated using simulation truth. Pick the
    highest-charge connected component in each view and report fragmentation.
    Full event clustering/matching and the sample's fiducial cuts are not emulated.
    """
    import torch
    import torch.nn.functional as F
    from src.images import pad_image_batch_gpu

    images, audit = [], []
    for plane, threshold in enumerate((response.collection_threshold_adc, response.induction_threshold_adc)):
        waveform = waveforms[plane]
        regions = regionprops(label(waveform > threshold), intensity_image=waveform)
        if not regions:
            raise ValueError(f'No connected component above threshold in plane {plane}')
        region = max(regions, key=lambda r: r.image_intensity.sum())
        image = region.image_intensity.astype(np.float32)
        if image.shape[1] > 1502:
            raise ValueError('Selected component is wider than the model canvas')
        above_threshold_charge = waveform[waveform > threshold].sum()
        audit.append({'bbox_wire_tick': list(map(int, region.bbox)),
                      'components': len(regions),
                      'selected_signal_fraction': min(1.0, float(image.sum()/above_threshold_charge)),
                      'component_shape': list(image.shape),
                      'retained_wires': min(50, image.shape[0])})
        padded = pad_image_batch_gpu([image], device='cpu', cut_rows=50)[0]
        images.append(padded)
    padded = np.stack(images)
    model_raw = F.interpolate(torch.from_numpy(padded)[None], size=(48, 48),
                              mode='bilinear', align_corners=False)[0]
    return model_raw.numpy(), torch.log1p(model_raw).numpy(), padded, audit
