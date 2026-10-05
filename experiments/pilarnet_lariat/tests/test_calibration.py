import unittest
import numpy as np

from experiments.pilarnet_lariat.calibrate import kinetic_energy, conditional_distance


class EnergyCalibrationTests(unittest.TestCase):
    def test_massive_and_massless_energy_units(self):
        # 0.55158138 GeV/c is consistent with the ~149 MeV proton deposit
        # in the official example; treating it as MeV/c is not.
        self.assertAlmostEqual(float(kinetic_energy(.55158138*1000,938.27197)),150.12,delta=.1)
        np.testing.assert_allclose(kinetic_energy([0,100,200],[0,0,0]),[0,100,200])
        self.assertAlmostEqual(float(kinetic_energy(600,938.272)),175.440,delta=.01)
        with self.assertRaises(ValueError):kinetic_energy(-1,938.272)

    def test_energy_bins_do_not_compare_nonoverlapping_ranges(self):
        real=np.ones((30,16));fake=np.ones((30,16))
        distance,details=conditional_distance(real,np.full(30,125.),fake,np.full(30,225.))
        self.assertEqual(distance,1e6)
        self.assertFalse(any(d['supported'] for d in details))


if __name__=='__main__':unittest.main()
