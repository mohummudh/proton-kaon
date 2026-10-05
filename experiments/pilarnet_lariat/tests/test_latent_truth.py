"""Guard event isolation and the distinction between truth and image proxies."""

import unittest
import numpy as np

from experiments.pilarnet_lariat.latent_truth.prepare import geometry, partition, truth_features
from experiments.pilarnet_lariat.lariat_forward import LArIATResponse, rotation_between


class TruthPilotTests(unittest.TestCase):
    def test_particle_groups_from_same_event_share_partition(self):
        event = 'train/generic_v2_77600_v2.h5:event27'
        labels = [partition(event) for _ in range(100)]
        self.assertEqual(len(set(labels)), 1)
        self.assertEqual(partition(event), partition(event))
        self.assertEqual(set(partition(f'event{i}') for i in range(200)), {'train', 'dev', 'test'})

    def test_geometry_is_rigid_motion_invariant(self):
        points = np.zeros((20, 8))
        points[:, 0] = np.linspace(0, 10, 20)
        points[:, 1] = np.sin(points[:, 0])
        points[:, 2] = np.cos(points[:, 0])
        expected = geometry(points, .3)
        moved = points.copy()
        moved[:, :3] = points[:, :3]@rotation_between([1, 0, 0], [1, 2, 3]).T+[300, 200, 100]
        actual = geometry(moved, .3)
        for key in expected:
            self.assertAlmostEqual(expected[key], actual[key])

    def test_conflicting_fragment_momenta_do_not_become_incoming_truth(self):
        p = np.zeros((20, 8))
        p[:, 0] = np.arange(20)
        p[:, 3] = .5; p[:, 5] = np.arange(20); p[:, 7] = .1
        particle = {'group_id': 7, 'pid': 2, 'momentum': .4,
                    'mass_mev': 105.658, 'vertex_voxels': [0, 0, 0]}
        clusters = np.array([[10, 1, 7, 3, 1, 2], [10, 2, 7, 3, 1, 2]])
        extras = np.array([[105.658, .4, 0, 0, 0], [105.658, .2, 0, 0, 0]])
        f = truth_features(particle, p, clusters, extras, LArIATResponse())
        self.assertFalse(f['kinematics_reliable'])
        self.assertEqual(f['interaction_id'], 3)
        self.assertEqual(f['semantic_purity'], 1)
        self.assertEqual(f['deposited_mev'], 10)
        self.assertAlmostEqual(f['path_cm'], 2)


if __name__ == '__main__':
    unittest.main()
