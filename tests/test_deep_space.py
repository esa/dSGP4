import unittest

import dsgp4
import numpy as np
import sgp4.earth_gravity
import sgp4.io
import sgp4.propagation
import torch


DEEP_SPACE_TLES = [
    (
        "1 14128U 83058A   06176.02844893 -.00000158  00000-0  10000-3 0  9627",
        "2 14128  11.4384  35.2134 0011562  26.4582 333.5652  0.98870114 46093",
    ),
    (
        "1 09998U 74033F   05148.79417928 -.00000112  00000-0  00000+0 0  4480",
        "2 09998   9.4958 313.1750 0270971 327.5225  30.8097  1.16186785 45878",
    ),
]


LEO_TLE = (
    "1 25544U 98067A   24087.49097222  .00016717  00000+0  10270-3 0  9993",
    "2 25544  51.6400  82.2420 0006290  58.9900  53.5550 15.50000000000000",
)


def _reference_state(line1, line2, tsince):
    satrec = sgp4.io.twoline2rv(line1, line2, whichconst=sgp4.earth_gravity.wgs84)
    return np.array(sgp4.propagation.sgp4(satrec, float(tsince)))


class DeepSpaceParityTestCase(unittest.TestCase):
    def test_deep_space_scalar_matches_reference(self):
        times = torch.tensor([0.0, 60.0, 720.0, 1440.0])

        for line1, line2 in DEEP_SPACE_TLES:
            tle = dsgp4.tle.TLE([line1, line2])
            dsgp4.initialize_tle(tle, gravity_constant_name="wgs-84")

            self.assertEqual(tle._method, "d")

            out = dsgp4.propagate(tle, times, initialized=True).detach().numpy()
            ref = np.stack([_reference_state(line1, line2, t) for t in times.numpy()])

            self.assertTrue(np.allclose(out, ref, atol=1e-9, rtol=0.0))

    def test_mixed_batch_matches_reference(self):
        deep_1 = dsgp4.tle.TLE([DEEP_SPACE_TLES[0][0], DEEP_SPACE_TLES[0][1]])
        deep_2 = dsgp4.tle.TLE([DEEP_SPACE_TLES[1][0], DEEP_SPACE_TLES[1][1]])
        leo = dsgp4.tle.TLE([LEO_TLE[0], LEO_TLE[1]])

        tles = [deep_1, leo, deep_2]
        tsinces = torch.tensor([45.0, 15.0, 360.0])

        _, initialized = dsgp4.initialize_tle(tles, gravity_constant_name="wgs-84")
        out = dsgp4.propagate_batch(initialized, tsinces, initialized=True).detach().numpy()

        ref = np.stack(
            [
                _reference_state(DEEP_SPACE_TLES[0][0], DEEP_SPACE_TLES[0][1], tsinces[0]),
                _reference_state(LEO_TLE[0], LEO_TLE[1], tsinces[1]),
                _reference_state(DEEP_SPACE_TLES[1][0], DEEP_SPACE_TLES[1][1], tsinces[2]),
            ]
        )

        self.assertTrue(np.allclose(out, ref, atol=1e-8, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
