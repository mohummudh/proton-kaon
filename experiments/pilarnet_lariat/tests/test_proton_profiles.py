"""Check physical binning, energy references and integrated ADC calibration."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from experiments.pilarnet_lariat.lariat_forward import LArIATResponse, response_kernels
from experiments.pilarnet_lariat.proton_profiles import (
    binned_profile, fit_adc_profiles, match_protons, range_energy)


class ProtonProfileTests(unittest.TestCase):
    def test_path_weighted_energy_density_and_missing_bins(self):
        profile, counts = binned_profile([.1, .2, .8], [2, 10, 6], [.3, .1, .4])
        self.assertAlmostEqual(profile[0], 4.)
        self.assertAlmostEqual(profile[1], 6.)
        self.assertTrue(np.isnan(profile[2]))
        self.assertEqual(counts[:3].tolist(), [2, 1, 0])

    def test_range_energy_integrates_deposits_without_extrapolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            table = Path(temporary)/'range.txt'
            np.savetxt(table, [[0, 2], [1, 2], [2, 2]])
            energy = range_energy(np.array([0, .5, 2, 3]), table)
            np.testing.assert_allclose(energy[:3], [0, 1, 4])
            self.assertTrue(np.isnan(energy[3]))

    def test_matching_respects_energy_reference_and_partition(self):
        real = pd.DataFrame({'partition': ['fit', 'validation'],
            'incoming_ke_mev': [200, 201], 'range_energy_mev': [100, 101],
            'length_cm': [10, 10], 'stopping_proxy': [True, True]})
        fake = pd.DataFrame({'partition': ['validation'], 'incoming_ke_mev': [100],
            'length_cm': [10], 'stopping_proxy': [True]})
        a, b = np.ones((2, 13))*5, np.ones((1, 13))*5
        self.assertTrue(match_protons(real, a, fake, b, 'incoming_ke_mev').empty)
        result = match_protons(real, a, fake, b, 'range_energy_mev', True)
        self.assertEqual(result.real_row.tolist(), [1])

    def test_area_calibration_is_not_peak_gain_or_signed_net_area(self):
        for shaping in (1., 3., 6.):
            response = replace(LArIATResponse(), shaping_peak_us=shaping,
                collection_area_adc_ticks_per_electron=.0932,
                induction_positive_area_adc_ticks_per_electron=.0351)
            collection, induction = response_kernels(response)
            self.assertAlmostEqual(collection.sum(), .0932)
            self.assertAlmostEqual(np.maximum(induction, 0).sum(), .0351)
            self.assertLess(induction.min(), 0)
            self.assertLess(induction.sum(), .0351)

    def test_incoming_matches_exclude_nonstopping_particles(self):
        real = pd.DataFrame({'partition': ['fit'], 'incoming_ke_mev': [100],
            'length_cm': [10], 'stopping_proxy': [True]})
        fake = pd.DataFrame({'partition': ['fit', 'fit'], 'incoming_ke_mev': [100, 100],
            'length_cm': [2, 10], 'stopping_proxy': [False, True]})
        result = match_protons(real, np.ones((1, 13)), fake, np.ones((2, 13)), 'incoming_ke_mev')
        self.assertEqual(result.pilarnet_row.tolist(), [1])

    def test_amplitude_fit_does_not_use_heldout_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            pd.DataFrame({'partition': ['fit']*20+['validation']*2}).to_csv(
                output/'tpc_range_image_pairs.csv', index=False)
            simulated = np.ones((22, 2, 48, 48), dtype=np.float32)*80
            observed = simulated*1.5
            # Held-out events prefer a conflicting scale at the search boundary.
            observed[20:] = simulated[20:]*4
            np.savez(output/'tpc_range_image_pairs.npz', real=observed, pilarnet=simulated)
            response = replace(LArIATResponse(), collection_area_adc_ticks_per_electron=.0932,
                induction_positive_area_adc_ticks_per_electron=.0351)
            (output/'reco_response.yaml').write_text(yaml.safe_dump(asdict(response)))
            fit_adc_profiles(output)
            report = json.loads((output/'adc_fit_report.json').read_text())
            self.assertEqual(report['fit_pairs'], 20)
            np.testing.assert_allclose(report['effective_raw_projection_amplitude_multipliers'], [1.5, 1.5])
            fitted = yaml.safe_load((output/'profile_response.yaml').read_text())
            self.assertAlmostEqual(fitted['collection_area_adc_ticks_per_electron'], .0932*1.5)


if __name__ == '__main__':
    unittest.main()
