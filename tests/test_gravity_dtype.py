import unittest

import torch

import dsgp4
from dsgp4.util import get_gravity_constants


class GravityConstantsDtypeTestCase(unittest.TestCase):
    def tearDown(self):
        torch.set_default_dtype(torch.float64)

    def test_constants_follow_default_dtype(self):
        for dtype in (torch.float32, torch.float64):
            torch.set_default_dtype(dtype)
            for name in ('wgs-72old', 'wgs-72', 'wgs-84'):
                consts = get_gravity_constants(name)
                self.assertEqual(len(consts), 8)
                for c in consts:
                    self.assertEqual(c.dtype, dtype, f'{name} with default {dtype}')

    def test_float64_values_unchanged(self):
        # Under the default dtype set on import (float64), values are unchanged.
        torch.set_default_dtype(torch.float64)
        tumin, mu, r, xke, j2, j3, j4, j3oj2 = get_gravity_constants('wgs-84')
        self.assertEqual(mu.item(), 398600.5)
        self.assertAlmostEqual((tumin * xke).item(), 1.0, places=15)

    def test_float32_propagation_stays_float32(self):
        line1 = '1 25544U 98067A   20190.11055656  .00000774  00000-0  21957-4 0  9990'
        line2 = '2 25544  51.6440 189.4914 0001940  86.6457  14.5000 15.49517290235698'
        torch.set_default_dtype(torch.float32)
        tle = dsgp4.tle.TLE([line1, line2])
        dsgp4.initialize_tle(tle)
        state = dsgp4.propagate(tle, torch.tensor([0.0, 60.0]))
        self.assertEqual(state.dtype, torch.float32)
