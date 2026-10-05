"""Physics/unit checks for the approximate 3D -> LArIAT converter."""

from dataclasses import replace
import unittest

import numpy as np

from experiments.pilarnet_lariat.lariat_forward import (LArIATResponse, ionization_electrons,
                                particle_groups, rotation_between, wire_coordinates,
                                simulate_readout, response_kernels, prepare_model_input)


class ForwardModelTests(unittest.TestCase):
    def test_rigid_rotation_and_antiparallel_case(self):
        for target in ([0, 0, 1], [0, 0, -1], [1, 2, 3]):
            matrix = rotation_between([0, 0, 1], target)
            np.testing.assert_allclose(matrix @ [0, 0, 1], np.array(target)/np.linalg.norm(target), atol=1e-12)
            np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(matrix), 1)

    def test_both_views_run_downstream_and_have_opposite_vertical_slopes(self):
        r = LArIATResponse()
        wires = wire_coordinates(np.array([[20, 0, 0], [20, 0, 10], [20, 1, 0]]), r)
        np.testing.assert_allclose(wires[1]-wires[0], np.sqrt(3)/2*10/r.wire_pitch_cm)
        np.testing.assert_allclose(wires[2]-wires[0], [0.5/r.wire_pitch_cm, -0.5/r.wire_pitch_cm])

    def test_recombination_bounds_and_non_linearity(self):
        r = LArIATResponse()
        energy = np.array([0, 0.6, 3.0])
        electrons = ionization_electrons(energy, np.array([0.3]*3), r)
        self.assertEqual(electrons[0], 0)
        self.assertTrue(np.all(electrons <= energy / 23.6e-6))
        self.assertGreater(electrons[1]/energy[1], electrons[2]/energy[2])
        with self.assertRaises(ValueError):
            ionization_electrons([1], [0], r)

    def test_area_units_lifetime_and_no_double_counting_across_subvoxels(self):
        r = replace(LArIATResponse(), subvoxel_samples=1,
                    transverse_diffusion_cm2_s=0, longitudinal_diffusion_cm2_s=0)
        waveform, audit = simulate_readout([[20, 0, 10]], [20000], [0], r)
        expected_charge = 20000*np.exp(-(20/r.drift_cm_us)/r.lifetime_us)
        self.assertAlmostEqual(audit['after_lifetime_electrons'], expected_charge)
        for plane, kernel in enumerate(response_kernels(r)):
            self.assertAlmostEqual(audit['planes'][plane]['accepted_electrons'], expected_charge)
            self.assertAlmostEqual(waveform[plane].sum(), expected_charge*kernel.sum(), delta=0.02)
        self.assertLess(waveform[1].min(), 0)

    def test_drift_time_mapping_and_peak_gain(self):
        r = replace(LArIATResponse(), subvoxel_samples=1, lifetime_us=1e12,
                    transverse_diffusion_cm2_s=0, longitudinal_diffusion_cm2_s=0)
        # Place deposit at an exact tick to separate sampling from gain response.
        x = (1000-r.trigger_tick-r.collection_gap_us/r.sample_us)*r.sample_us*r.drift_cm_us
        wave, _ = simulate_readout([[x, 0, 10]], [10000], [0], r)
        wire, tick = np.unravel_index(np.argmax(wave[0]), wave[0].shape)
        self.assertEqual(tick, 1000+int(np.argmax(response_kernels(r)[0])))
        self.assertAlmostEqual(wave[0, wire, tick], 10000*r.adc_peak_per_electron*r.collection_response_scale, delta=1e-4)

    def test_subvoxel_integration_and_front_boundary_acceptance(self):
        r = replace(LArIATResponse(), lifetime_us=1e12,
                    transverse_diffusion_cm2_s=0, longitudinal_diffusion_cm2_s=0)
        _, interior = simulate_readout([[20, 0, 10]], [27000], [0], r)
        self.assertAlmostEqual(interior['inside_volume_electrons'], 27000)
        # With three samples along z, two of three lie on/inside the front face.
        _, boundary = simulate_readout([[20, 0, 0]], [27000], [0], r)
        self.assertAlmostEqual(boundary['inside_volume_electrons'], 18000)

    def test_windowed_readout_preserves_waveforms_and_images(self):
        r = LArIATResponse()
        xyz = [[20, 0, 10], [20.2, .2, 10.5], [20.4, -.2, 11]]
        full, _ = simulate_readout(xyz, [60000]*3, [0]*3, r)
        window, audit = simulate_readout(xyz, [60000]*3, [0]*3, r, windowed=True)
        w, t = audit['window_origin_wire_tick']
        np.testing.assert_allclose(full[:, w:w+window.shape[1], t:t+window.shape[2]], window,
                                   rtol=1e-6, atol=1e-6)
        a, b = prepare_model_input(full, r)[0], prepare_model_input(window, r)[0]
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

    def test_fragment_union_and_conflicting_labels(self):
        points = np.arange(32).reshape(4, 8)
        clusters = np.array([[1, 0, -1, -1, 4, -1], [1, 1, 7, 0, 1, 4], [2, 2, 7, 0, 1, 4]])
        extras = np.zeros((3, 5))
        groups = list(particle_groups(points, clusters, extras))
        self.assertEqual(len(groups), 1)
        np.testing.assert_array_equal(groups[0]['points'], points[1:])
        clusters[2, 5] = 2
        with self.assertRaises(ValueError):
            list(particle_groups(points, clusters, extras))

    def test_training_tensor_and_signed_induction_mask(self):
        r = LArIATResponse()
        wave, _ = simulate_readout([[20, 0, 10]], [50000], [0], r)
        raw, transformed, padded, crop = prepare_model_input(wave, r)
        self.assertEqual(raw.shape, (2, 48, 48))
        self.assertEqual(padded.shape, (2, 51, 1502))
        self.assertTrue(np.isfinite(transformed).all())
        self.assertTrue(np.all(raw >= 0))
        np.testing.assert_allclose(transformed, np.log1p(raw), rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
