import unittest

import dsgp4
import numpy as np
import sgp4.earth_gravity
import sgp4.io
import sgp4.propagation
import torch

# Real deep-space TLEs from the Vallado SGP4 verification set (SGP4-VER.TLE), chosen to
# cover every deep-space code path: no resonance, 24h synchronous resonance, 12h
# (Molniya) resonance on both sides of the e=0.7 polynomial switch, and the
# low-inclination (< 0.2 rad) Lyddane branch of the lunar-solar periodics.
DEEP_SPACE_TLES = [
    # irez=0, i=11.46 deg (Lyddane branch), e=0.145
    ("1 04632U 70093B   04031.91070959 -.00000084  00000-0  10000-3 0  9955",
     "2 04632  11.4628 273.1101 1450506 207.6000 143.9350  1.20231981 44145"),
    # irez=2, e=0.688
    ("1 08195U 75081A   06176.33215444  .00000099  00000-0  11873-3 0   813",
     "2 08195  64.1586 279.0717 6877146 264.7651  20.2257  2.00491383225656"),
    # irez=2, e=0.707
    ("1 09880U 77021A   06176.56157475  .00000421  00000-0  10000-3 0  9814",
     "2 09880  64.5968 349.3786 7069051 270.0229  16.3320  2.00813614112380"),
    # irez=1, i=9.50 deg
    ("1 09998U 74033F   05148.79417928 -.00000112  00000-0  00000+0 0  4480",
     "2 09998   9.4958 313.1750 0270971 327.5225  30.8097  1.16186785 45878"),
    # irez=1, i=11.44 deg
    ("1 14128U 83058A   06176.02844893 -.00000158  00000-0  10000-3 0  9627",
     "2 14128  11.4384  35.2134 0011562  26.4582 333.5652  0.98870114 46093"),
]

LEO_TLE = (
    "1 25544U 98067A   24087.49097222  .00016717  00000+0  10270-3 0  9993",
    "2 25544  51.6400  82.2420 0006290  58.9900  53.5550 15.50000000000000",
)

# times since epoch [min]: backward and forward, several resonance integration steps (720 min)
TSINCES = [-4320.0, -719.0, 0.0, 360.0, 1440.0, 10080.0]


def _python_sgp4(line1, line2, tsince):
    satrec = sgp4.io.twoline2rv(line1, line2, whichconst=sgp4.earth_gravity.wgs84)
    return np.array(sgp4.propagation.sgp4(satrec, float(tsince)))


def _batch_inputs():
    """Every deep-space TLE (plus one LEO) at every time, flattened into one batch."""
    lines, times = [], []
    for pair in DEEP_SPACE_TLES + [LEO_TLE]:
        for t in TSINCES:
            lines.append(pair)
            times.append(t)
    return lines, times


class BatchedDeepSpaceTestCase(unittest.TestCase):
    def setUp(self):
        self.previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        torch.set_default_dtype(self.previous_dtype)

    def test_initialize_returns_single_batch_object(self):
        tles = [dsgp4.tle.TLE(list(p)) for p in DEEP_SPACE_TLES + [LEO_TLE]]
        _, batch = dsgp4.initialize_tle(tles, gravity_constant_name="wgs-84")
        self.assertIsInstance(batch, dsgp4.tle.TLE)
        self.assertEqual(batch._method, ["d"] * len(DEEP_SPACE_TLES) + ["n"])
        self.assertEqual(batch._irez.tolist(), [0, 2, 2, 1, 1, 0])
        self.assertEqual(batch._isimp.tolist()[:len(DEEP_SPACE_TLES)], [1] * len(DEEP_SPACE_TLES))

    def test_batched_deep_space_matches_python_sgp4_and_unbatched(self):
        lines, times = _batch_inputs()
        tles = [dsgp4.tle.TLE(list(p)) for p in lines]
        _, batch = dsgp4.initialize_tle(tles, gravity_constant_name="wgs-84")
        out = dsgp4.sgp4_batched(batch, torch.tensor(times)).detach().numpy()

        ref = np.stack([_python_sgp4(l1, l2, t) for (l1, l2), t in zip(lines, times)])
        np.testing.assert_allclose(out[:, 0], ref[:, 0], rtol=0.0, atol=1e-6)   # km
        np.testing.assert_allclose(out[:, 1], ref[:, 1], rtol=0.0, atol=1e-9)   # km/s

        unbatched = []
        for (l1, l2), t in zip(lines, times):
            tle = dsgp4.tle.TLE([l1, l2])
            dsgp4.initialize_tle(tle, gravity_constant_name="wgs-84")
            unbatched.append(dsgp4.propagate(tle, torch.tensor([[t]])).detach().numpy().reshape(2, 3))
        unbatched = np.stack(unbatched)
        np.testing.assert_allclose(out[:, 0], unbatched[:, 0], rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(out[:, 1], unbatched[:, 1], rtol=0.0, atol=1e-9)

    def test_batched_deep_space_gradients_match_unbatched(self):
        idx = [1, 3]  # one 12h and one 24h resonant satellite
        tsince = torch.tensor([1440.0, 4000.0])
        tles = [dsgp4.tle.TLE(list(DEEP_SPACE_TLES[i])) for i in idx]
        elements, batch = dsgp4.initialize_tle(tles, gravity_constant_name="wgs-84", with_grad=True)
        state = dsgp4.sgp4_batched(batch, tsince)
        grads_b = torch.autograd.grad(state[:, 0, :].sum(), elements)

        for k, i in enumerate(idx):
            tle = dsgp4.tle.TLE(list(DEEP_SPACE_TLES[i]))
            el = dsgp4.initialize_tle(tle, gravity_constant_name="wgs-84", with_grad=True)
            s = dsgp4.propagate(tle, tsince[k].reshape(1, 1))
            g_u, = torch.autograd.grad(s[0, :].sum(), el)
            self.assertTrue(torch.all(torch.isfinite(grads_b[k])))
            np.testing.assert_allclose(grads_b[k].numpy(), g_u.numpy(), rtol=1e-6, atol=1e-6)


class BatchedDeepSpaceFloat32TestCase(unittest.TestCase):
    def setUp(self):
        self.previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)

    def tearDown(self):
        torch.set_default_dtype(self.previous_dtype)

    def test_float32_batch_matches_float32_unbatched(self):
        lines, times = _batch_inputs()
        tles = [dsgp4.tle.TLE(list(p)) for p in lines]
        _, batch = dsgp4.initialize_tle(tles, gravity_constant_name="wgs-84")
        out = dsgp4.sgp4_batched(batch, torch.tensor(times))
        self.assertEqual(out.dtype, torch.float32)
        for (l1, l2), t, o in zip(lines, times, out):
            tle = dsgp4.tle.TLE([l1, l2])
            dsgp4.initialize_tle(tle, gravity_constant_name="wgs-84")
            u = dsgp4.propagate(tle, torch.tensor([[t]])).reshape(2, 3)
            # float32 batched and unbatched round differently; both are float32 SGP4
            self.assertLess(torch.linalg.norm(o[0] - u[0]).item(), 5.0)


if __name__ == "__main__":
    unittest.main()
