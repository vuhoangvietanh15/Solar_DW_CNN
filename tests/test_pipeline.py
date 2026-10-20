import tempfile
import unittest
from pathlib import Path
import numpy as np
from solar_dwcnn.data import mask_values, prepare, normalize


class DataTests(unittest.TestCase):
    def test_mask_never_exposes_targets_including_edges(self):
        x = np.random.default_rng(5).normal(size=(2, 512)).astype(np.float32)
        for phase in range(8):
            a, y = mask_values(x, [phase, phase])
            changed = x.copy()
            changed[y[..., 1].astype(bool)] += 12345
            b, _ = mask_values(changed, [phase, phase])
            np.testing.assert_array_equal(a, b)
            self.assertGreater(y[..., 1].sum(), 0)

    def test_heldout_data_cannot_change_normalization(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            days = ['2025_03_20', '2025_03_23', '2025_03_24']
            for i, day in enumerate(days):
                (root/day).mkdir()
                f = np.arange(16)*.01 + 610
                v = np.arange(16)*.001 + i
                np.savetxt(root/day/f'{day}_10_00_00_0_PSD.csv', np.c_[f, v], delimiter=',')
            meta = prepare(root, root/'a', [days[0]], [days[1]], [days[2]])
            p = next((root/days[2]).glob('*.csv'))
            a = np.loadtxt(p, delimiter=','); a[:, 1] += 100
            np.savetxt(p, a, delimiter=',')
            again = prepare(root, root/'b', [days[0]], [days[1]], [days[2]])
            self.assertEqual(meta['normalization'], again['normalization'])
            n = meta['normalization']
            x = np.arange(16).astype(np.float32)
            np.testing.assert_allclose(normalize(x, meta)*n['scale_db']+n['center_db'], x, atol=1e-5)
            with self.assertRaises(ValueError):
                prepare(root, root/'bad', [days[0]], [days[0]], [days[2]])
            a[:, 0] += .1
            np.savetxt(p, a, delimiter=',')
            with self.assertRaisesRegex(ValueError, 'frequency grid'):
                prepare(root, root/'wrong', [days[0]], [days[1]], [days[2]])


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from solar_dwcnn.runtime import configure
        configure(allow_cpu=True)

    def test_haar_reconstruction_energy_and_gradient(self):
        from solar_dwcnn.model import HaarDWT, HaarIDWT
        import tensorflow as tf
        x = tf.Variable(np.random.default_rng(1).normal(size=(2, 512, 3)).astype(np.float32))
        with tf.GradientTape() as tape:
            coeff = HaarDWT()(x)
            recon = HaarIDWT()(coeff)
            loss = tf.reduce_sum(recon**2)
        np.testing.assert_allclose(recon, x, atol=5e-7)
        np.testing.assert_allclose(tf.reduce_sum(coeff**2), tf.reduce_sum(x**2), rtol=1e-6)
        self.assertTrue(np.isfinite(tape.gradient(loss, x)).all())

    def test_save_reload_and_blind_inference(self):
        from solar_dwcnn.model import DW_CNN, masked_mse
        from solar_dwcnn.infer import blind_predict
        import tensorflow as tf
        net = DW_CNN(32, 4)
        x = np.random.default_rng(3).normal(size=(2, 32)).astype(np.float32)
        bx, by = mask_values(x, [0, 1])
        net.compile(optimizer='adam', loss=masked_mse)
        self.assertTrue(np.isfinite(net.train_on_batch(bx, by)))
        before = blind_predict(net, x, batch_size=1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'model.keras'
            net.save(path)
            loaded = tf.keras.models.load_model(path, compile=False)
            after = blind_predict(loaded, x, batch_size=2)
            np.testing.assert_allclose(before, after, atol=1e-6)
        for index in [0, 13, 31]:
            perturbed = x.copy(); perturbed[0, index] += 100
            out = blind_predict(net, perturbed)
            np.testing.assert_allclose(before[0, index], out[0, index], atol=1e-6)

    def test_loss_ignores_visible_bins(self):
        from solar_dwcnn.model import masked_mse
        y = np.zeros((1, 16, 2), dtype=np.float32); y[0, 3, 1] = 1
        pred = np.full((1, 16, 1), 100, dtype=np.float32); pred[0, 3, 0] = 2
        np.testing.assert_allclose(masked_mse(y, pred), [4.])


if __name__ == '__main__':
    unittest.main()
