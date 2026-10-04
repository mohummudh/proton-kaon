"""Small invariant checks for the scientific evaluation contract."""
import unittest
import numpy as np
import pandas as pd
import torch
from representation_baselines import OUT, endpoint_features, mask_batch, active_mask_batch
from evaluate_representation_baselines import mass_contrast, boot_weights


class BaselineContract(unittest.TestCase):
    def test_fixed_patch_fraction_and_reproducibility(self):
        x = torch.ones(7, 2, 48, 48)
        a = mask_batch(x, torch.Generator().manual_seed(2))
        b = mask_batch(x, torch.Generator().manual_seed(2))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.all(a.sum((1, 2, 3)) == 48 * 48 / 4))
        self.assertEqual(a.shape[1], 1)  # same mask broadcasts to both planes

    def test_active_mask_selects_signal_patches_only(self):
        x = torch.zeros(3, 2, 48, 48)
        x[0, 0, :6, :6] = 1
        x[1, 1, 6:12, 6:12] = 2
        x[1, 0, 12:18, 12:18] = 3
        x[2, :, ::6, ::6] = 1
        a = active_mask_batch(x, torch.Generator().manual_seed(42))
        b = active_mask_batch(x, torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(a.shape, (3, 1, 48, 48))
        active = torch.nn.functional.max_pool2d((x > 0).any(1, keepdim=True).float(), 6, 6)
        masked = torch.nn.functional.max_pool2d(a, 6, 6)
        self.assertFalse(bool(((masked > 0) & (active == 0)).any()))
        self.assertEqual(masked.flatten(1).sum(1).tolist(), [1., 1., 16.])

    def test_endpoint_features_finite_empty_and_signal(self):
        x = np.zeros((2, 2, 48, 48), dtype=float)
        x[1, :, 20:24, 5:42] = 3
        f = endpoint_features(x)
        self.assertEqual(f.shape, (2, 32))
        self.assertTrue(np.isfinite(f).all())
        np.testing.assert_allclose(f[1, :16], f[1, 16:])

    def test_bootstrap_never_splits_event_candidates(self):
        for w in boot_weights(np.array(['a', 'a', 'b', 'c', 'c']), n=20):
            self.assertEqual(w[0], w[1]); self.assertEqual(w[3], w[4])

    def test_profile_follows_wire_rows(self):
        x = np.zeros((1, 2, 48, 48))
        x[:, :, 40, 12] = 10
        f = endpoint_features(x)
        self.assertAlmostEqual(f[0, 10], 40 / 47, places=6)

    def test_mass_cannot_change_selection(self):
        n = 240
        df = pd.DataFrame({'partition': ['dev'] * 80 + ['test'] * 160,
                           'species': 'kaon', 'picky': 1,
                           'momentum': np.tile(np.linspace(400, 1200, 80), 3),
                           'beamline_mass': np.linspace(350, 650, n)})
        score = np.random.default_rng(4).normal(size=n)
        a = mass_contrast(score, df, 'test', .2, 1)
        df.beamline_mass = np.random.default_rng(5).permutation(df.beamline_mass)
        b = mass_contrast(score, df, 'test', .2, 1)
        np.testing.assert_array_equal(a['idx'], b['idx'])
        np.testing.assert_array_equal(a['selected'], b['selected'])
        self.assertAlmostEqual(a['n_flagged'] / a['n'], .2, delta=.025)

    def test_real_manifest_disjoint(self):
        df = pd.read_csv(OUT / 'manifest.csv')
        self.assertEqual(df.groupby('event_key').partition.nunique().max(), 1)
        self.assertEqual(df.groupby('image_sha256').partition.nunique().max(), 1)
        train = df[df.partition == 'train']; held = df[df.partition == 'run_test']
        self.assertFalse(set(train.run) & set(held.run))
        self.assertEqual(df[df.partition == 'excluded_conflicting_tag'].event_key.nunique(), 26)


if __name__ == '__main__': unittest.main()
