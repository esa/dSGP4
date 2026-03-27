import unittest
import importlib
from types import SimpleNamespace
from unittest.mock import patch

import dsgp4
import dsgp4.deep_space as deep_space
import numpy as np
import sgp4.earth_gravity
import sgp4.io
import sgp4.propagation
import torch


dsgp4_sgp4_module = importlib.import_module("dsgp4.sgp4")


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


class DeepSpaceCoverageBranchesTestCase(unittest.TestCase):
    @staticmethod
    def _tensor(v):
        return torch.tensor(float(v))

    def _make_dpper_satellite(self):
        z = self._tensor(0.0)
        return SimpleNamespace(
            _zmos=z,
            _zmol=z,
            _t=z,
            _se2=z,
            _se3=z,
            _si2=z,
            _si3=z,
            _sl2=z,
            _sl3=z,
            _sl4=z,
            _sgh2=z,
            _sgh3=z,
            _sgh4=z,
            _sh2=z,
            _sh3=z,
            _ee2=z,
            _e3=z,
            _xi2=z,
            _xi3=z,
            _xl2=z,
            _xl3=z,
            _xl4=z,
            _xgh2=z,
            _xgh3=z,
            _xgh4=z,
            _xh2=z,
            _xh3=z,
            _peo=z,
            _pinco=z,
            _plo=z,
            _pgho=z,
            _pho=self._tensor(0.1),
        )

    def _make_dsinit_satellite(self):
        return SimpleNamespace(
            _t=self._tensor(60.0),
            _xke=self._tensor(0.0743669161),
            _gsto=self._tensor(0.2),
            _mo=self._tensor(0.3),
            _nodeo=self._tensor(0.4),
            _argpo=self._tensor(0.1),
            _mdot=self._tensor(0.001),
            _nodedot=self._tensor(0.002),
            _no_unkozai=self._tensor(0.0086),
        )

    def test_private_helpers_and_dpper_low_inclination_wrap(self):
        x = torch.tensor(1.23)
        self.assertIs(deep_space._to_tensor(x, x), x)
        self.assertTrue(deep_space._to_bool(True))

        sat = self._make_dpper_satellite()
        ep, inclp, nodep, argpp, mp = deep_space.dpper(
            satellite=sat,
            inclo=self._tensor(0.1),
            init="n",
            ep=self._tensor(0.1),
            inclp=self._tensor(0.1),
            nodep=self._tensor(0.1),
            argpp=self._tensor(0.0),
            mp=self._tensor(0.0),
            opsmode="a",
        )

        self.assertLess(float(nodep), 0.0)
        self.assertTrue(torch.isfinite(ep))
        self.assertTrue(torch.isfinite(inclp))
        self.assertTrue(torch.isfinite(argpp))
        self.assertTrue(torch.isfinite(mp))

    def test_dpper_negative_node_wraps_for_afspc_mode(self):
        sat = self._make_dpper_satellite()
        _, _, nodep, _, _ = deep_space.dpper(
            satellite=sat,
            inclo=self._tensor(0.1),
            init="n",
            ep=self._tensor(0.1),
            inclp=self._tensor(0.1),
            nodep=self._tensor(-0.1),
            argpp=self._tensor(0.0),
            mp=self._tensor(0.0),
            opsmode="a",
        )

        self.assertGreaterEqual(float(nodep), 0.0)

    def _call_dsinit(self, ecco):
        sat = self._make_dsinit_satellite()
        one = self._tensor(1.0)
        half = self._tensor(0.5)
        em = self._tensor(ecco)
        eccsq = self._tensor(ecco * ecco)
        inclm = self._tensor(0.35)
        sinim = torch.sin(inclm)
        cosim = torch.cos(inclm)

        return deep_space.dsinit(
            satellite=sat,
            cosim=cosim,
            emsq=eccsq,
            argpo=self._tensor(0.1),
            s1=one,
            s2=one,
            s3=one,
            s4=one,
            s5=one,
            sinim=sinim,
            ss1=one,
            ss2=one,
            ss3=one,
            ss4=one,
            ss5=one,
            sz1=half,
            sz3=half,
            sz11=half,
            sz13=half,
            sz21=half,
            sz23=half,
            sz31=half,
            sz33=half,
            z1=half,
            z3=half,
            z11=half,
            z13=half,
            z21=half,
            z23=half,
            z31=half,
            z33=half,
            ecco=em,
            eccsq=eccsq,
            em=em,
            argpm=self._tensor(0.2),
            inclm=inclm,
            mm=self._tensor(0.3),
            nm=self._tensor(0.0086),
            nodem=self._tensor(0.4),
            xpidot=self._tensor(0.001),
        )

    def test_dsinit_irez2_low_eccentricity_branch(self):
        out = self._call_dsinit(ecco=0.6)
        self.assertEqual(out["irez"], 2)
        self.assertNotEqual(float(out["d2201"]), 0.0)
        self.assertNotEqual(float(out["d5232"]), 0.0)

    def test_dsinit_irez2_high_eccentricity_branch(self):
        out = self._call_dsinit(ecco=0.8)
        self.assertEqual(out["irez"], 2)
        self.assertNotEqual(float(out["d5421"]), 0.0)
        self.assertNotEqual(float(out["d5433"]), 0.0)

    def test_dsinit_irez2_mid_high_eccentricity_branch(self):
        out = self._call_dsinit(ecco=0.7)
        self.assertEqual(out["irez"], 2)
        self.assertNotEqual(float(out["d5220"]), 0.0)

    def test_dspace_irez2_integration_branch(self):
        sat = SimpleNamespace(
            _gsto=self._tensor(0.3),
            _t=self._tensor(1000.0),
            _dedt=self._tensor(0.0),
            _didt=self._tensor(0.0),
            _domdt=self._tensor(0.0),
            _dnodt=self._tensor(0.0),
            _dmdt=self._tensor(0.0),
            _irez=2,
            _atime=self._tensor(0.0),
            _xni=self._tensor(0.0086),
            _xli=self._tensor(0.2),
            _no_unkozai=self._tensor(0.0086),
            _xlamo=self._tensor(0.2),
            _xfact=self._tensor(1.0e-4),
            _argpo=self._tensor(0.1),
            _argpdot=self._tensor(2.0e-4),
            _d2201=self._tensor(1.0e-6),
            _d2211=self._tensor(1.0e-6),
            _d3210=self._tensor(1.0e-6),
            _d3222=self._tensor(1.0e-6),
            _d4410=self._tensor(1.0e-6),
            _d4422=self._tensor(1.0e-6),
            _d5220=self._tensor(1.0e-6),
            _d5232=self._tensor(1.0e-6),
            _d5421=self._tensor(1.0e-6),
            _d5433=self._tensor(1.0e-6),
        )

        em, argpm, inclm, mm, nodem, nm = deep_space.dspace(
            satellite=sat,
            em=self._tensor(0.7),
            argpm=self._tensor(0.2),
            inclm=self._tensor(0.3),
            mm=self._tensor(0.4),
            nodem=self._tensor(0.5),
            nm=self._tensor(0.0086),
        )

        self.assertNotEqual(float(sat._atime), 0.0)
        self.assertTrue(torch.isfinite(em))
        self.assertTrue(torch.isfinite(argpm))
        self.assertTrue(torch.isfinite(inclm))
        self.assertTrue(torch.isfinite(mm))
        self.assertTrue(torch.isfinite(nodem))
        self.assertTrue(torch.isfinite(nm))

    def test_sgp4_deep_space_edge_guards(self):
        tle = dsgp4.tle.TLE([DEEP_SPACE_TLES[0][0], DEEP_SPACE_TLES[0][1]])
        dsgp4.initialize_tle(tle, gravity_constant_name="wgs-84")
        self.assertEqual(tle._method, "d")

        with patch.object(
            dsgp4_sgp4_module,
            "dpper",
            return_value=(
                torch.tensor([1.1]),
                torch.tensor([-np.pi]),
                torch.tensor([0.2]),
                torch.tensor([0.1]),
                torch.tensor([0.3]),
            ),
        ):
            dsgp4_sgp4_module.sgp4(tle, torch.tensor([10.0]))

        self.assertEqual(int(tle._error), 3)
        self.assertTrue(torch.isfinite(tle._xlcof).all())


if __name__ == "__main__":
    unittest.main()
