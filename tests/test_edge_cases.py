import contextlib
import importlib
import io
import os
import tempfile
import types
import unittest
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import dsgp4
from dsgp4 import initl as initl_fn
from dsgp4 import util
from dsgp4.mldsgp4 import mldsgp4
from dsgp4.plot import plot_orbit, plot_tles
from dsgp4.sgp4 import sgp4
from dsgp4.sgp4_batched import sgp4_batched
from dsgp4.sgp4init import sgp4init
from dsgp4.sgp4init_batch import initl_batch, sgp4init_batch
from dsgp4.tle import TLE, load, load_from_lines, read_satellite_catalog_number


SAMPLE_LINE1 = "1 43437U 18100A   20143.90384230  .00041418  00000-0  10000-3 0 99968"
SAMPLE_LINE2 = "2 43437  97.8268 249.9127 0221000 123.9136 259.1144 15.12608579563539"

INITL_MODULE = importlib.import_module("dsgp4.initl")
NEWTON_MODULE = importlib.import_module("dsgp4.newton_method")
SGP4INIT_BATCH_MODULE = importlib.import_module("dsgp4.sgp4init_batch")


def _sample_tle():
    return TLE([SAMPLE_LINE1, SAMPLE_LINE2])


class CoverageEdgesTestCase(unittest.TestCase):
    def test_initl_opsmode_a_negative_gsto_branch(self):
        tle = _sample_tle()
        whichconst = util.get_gravity_constants("wgs-84")
        _, _, _, xke, j2, _, _, _ = whichconst
        with mock.patch.object(torch.Tensor, "__lt__", return_value=torch.tensor(True)):
            out = initl_fn(
                xke,
                j2,
                tle._ecco,
                (tle._jdsatepoch + tle._jdsatepochF) - 2433281.5,
                tle._inclo,
                tle._no_kozai,
                "a",
                "n",
            )
        self.assertTrue(torch.isfinite(out[-1]))

    def test_mldsgp4_load_model_sets_eval(self):
        model = mldsgp4(hidden_size=8)
        model.train()
        self.assertTrue(model.training)
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            torch.save(model.state_dict(), tmp_path)
            model.load_model(tmp_path)
            self.assertFalse(model.training)
        finally:
            os.unlink(tmp_path)

    def test_newton_verbose_converged(self):
        dummy = types.SimpleNamespace(
            _ecco=0.2,
            _argpo=0.1,
            _inclo=0.3,
            _mo=0.4,
            _no_kozai=0.05,
            _nodeo=0.6,
            _epoch=util.from_string_to_datetime("2020-01-01 00:00:00"),
        )
        target_state = torch.tensor([[0.2, 0.1, 0.3], [0.4, 0.05, 0.6]])

        def fake_propagate(x, *_args, **_kwargs):
            return torch.stack((x[:3], x[3:6]))

        with mock.patch.object(NEWTON_MODULE, "initial_guess_tle", return_value=dummy), mock.patch.object(
            NEWTON_MODULE, "_propagate", side_effect=fake_propagate
        ), mock.patch.object(NEWTON_MODULE, "update_TLE", return_value=dummy):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                dsgp4.newton_method(
                    tle0=dummy,
                    time_mjd=59000.0,
                    max_iter=2,
                    verbose=True,
                    target_state=target_state,
                )
        self.assertIn("converged", buf.getvalue())

    def test_newton_eccentricity_lower_bound(self):
        dummy = types.SimpleNamespace(
            _ecco=0.1,
            _argpo=0.1,
            _inclo=0.3,
            _mo=0.4,
            _no_kozai=0.05,
            _nodeo=0.6,
            _epoch=util.from_string_to_datetime("2020-01-01 00:00:00"),
        )

        def fake_propagate(x, *_args, **_kwargs):
            return torch.stack((x[:3], x[3:6]))

        with mock.patch.object(NEWTON_MODULE, "initial_guess_tle", return_value=dummy), mock.patch.object(
            NEWTON_MODULE, "_propagate", side_effect=fake_propagate
        ), mock.patch.object(NEWTON_MODULE, "update_TLE", return_value=dummy), mock.patch.object(
            NEWTON_MODULE.np.linalg, "solve", return_value=np.array([-10.0, 0, 0, 0, 0, 0])
        ):
            _, y = dsgp4.newton_method(
                tle0=dummy,
                time_mjd=59000.0,
                max_iter=1,
                target_state=torch.zeros((2, 3)),
            )
        self.assertGreater(float(y[0]), 0.0)

    def test_newton_eccentricity_upper_bound_and_max_iter_verbose(self):
        dummy = types.SimpleNamespace(
            _ecco=0.9,
            _argpo=0.1,
            _inclo=0.3,
            _mo=0.4,
            _no_kozai=0.05,
            _nodeo=0.6,
            _epoch=util.from_string_to_datetime("2020-01-01 00:00:00"),
        )

        def fake_propagate(x, *_args, **_kwargs):
            return torch.stack((x[:3], x[3:6]))

        with mock.patch.object(NEWTON_MODULE, "initial_guess_tle", return_value=dummy), mock.patch.object(
            NEWTON_MODULE, "_propagate", side_effect=fake_propagate
        ), mock.patch.object(NEWTON_MODULE, "update_TLE", return_value=dummy), mock.patch.object(
            NEWTON_MODULE.np.linalg, "solve", return_value=np.array([10.0, 0, 0, 0, 0, 0])
        ):
            _, y = dsgp4.newton_method(
                tle0=dummy,
                time_mjd=59000.0,
                max_iter=1,
                target_state=torch.zeros((2, 3)),
            )
        self.assertLess(float(y[0]), 2.0)

        with mock.patch.object(NEWTON_MODULE, "initial_guess_tle", return_value=dummy):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                dsgp4.newton_method(
                    tle0=dummy,
                    time_mjd=59000.0,
                    max_iter=0,
                    verbose=True,
                    target_state=torch.zeros((2, 3)),
                )
        self.assertIn("Solution not found", buf.getvalue())

    def test_plot_functions(self):
        states = torch.zeros((5, 2, 3))
        states[:, 0, 0] = torch.linspace(0.0, 1000.0, 5)
        states[:, 0, 1] = torch.linspace(0.0, 500.0, 5)
        states[:, 0, 2] = torch.linspace(0.0, 200.0, 5)
        ax = plot_orbit(states, elevation_azimuth=(20, 40), color="red", label="orb")
        self.assertIsNotNone(ax)
        fig = plt.figure()
        ax2 = fig.add_subplot(111, projection="3d")
        ax2 = plot_orbit(states, ax=ax2, color="blue", label="orb2")
        self.assertIsNotNone(ax2)

        tle1 = _sample_tle()
        tle2 = tle1.copy()
        tle2.update({"mean_motion": tle1.mean_motion * 1.01, "eccentricity": tle1.eccentricity * 1.1})
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plot_path = tmp.name
        try:
            axs = plot_tles(
                [tle1, tle2],
                file_name=plot_path,
                show=False,
                return_axs=True,
                log_yscale=True,
                color="green",
            )
            self.assertEqual(axs.shape, (3, 3))
            self.assertTrue(os.path.exists(plot_path))
            with mock.patch("matplotlib.pyplot.show") as mocked_show:
                plot_tles([tle1, tle2], show=True, return_axs=False)
                mocked_show.assert_called_once()
        finally:
            if os.path.exists(plot_path):
                os.unlink(plot_path)
        plt.close("all")

    def test_sgp4_type_and_attribute_errors(self):
        with self.assertRaises(TypeError):
            sgp4(object(), torch.tensor([0.0]))
        with self.assertRaises(AttributeError):
            sgp4(_sample_tle(), torch.tensor([0.0]))

    def test_sgp4_batched_value_errors(self):
        tle = _sample_tle()
        with self.assertRaises(ValueError):
            sgp4_batched(object(), torch.tensor([0.0]))
        with self.assertRaises(ValueError):
            sgp4_batched(tle, [0.0])
        with self.assertRaises(ValueError):
            sgp4_batched(tle, torch.tensor([[0.0]]))
        tle._argpo = torch.tensor([float(tle._argpo)])
        with self.assertRaises(ValueError):
            sgp4_batched(tle, torch.tensor([0.0, 1.0]))
        with self.assertRaises(AttributeError):
            sgp4_batched(tle, torch.tensor([0.0]))

    def test_sgp4init_perigee_and_cosio_edge(self):
        tle = _sample_tle()
        whichconst = util.get_gravity_constants("wgs-84")
        sgp4init(
            whichconst=whichconst,
            opsmode="i",
            satn=tle.satellite_catalog_number,
            epoch=(tle._jdsatepoch + tle._jdsatepochF) - 2433281.5,
            xbstar=tle._bstar,
            xndot=tle._ndot,
            xnddot=tle._nddot,
            xecco=torch.tensor(0.03),
            xargpo=tle._argpo,
            xinclo=tle._inclo,
            xmo=tle._mo,
            xno_kozai=torch.tensor(0.08),
            xnodeo=tle._nodeo,
            satellite=tle,
        )
        self.assertTrue(hasattr(tle, "_cc1"))

        tle2 = _sample_tle()
        sgp4init(
            whichconst=whichconst,
            opsmode="i",
            satn=tle2.satellite_catalog_number,
            epoch=(tle2._jdsatepoch + tle2._jdsatepochF) - 2433281.5,
            xbstar=tle2._bstar,
            xndot=tle2._ndot,
            xnddot=tle2._nddot,
            xecco=tle2._ecco,
            xargpo=tle2._argpo,
            xinclo=torch.tensor(np.pi),
            xmo=tle2._mo,
            xno_kozai=tle2._no_kozai,
            xnodeo=tle2._nodeo,
            satellite=tle2,
        )
        self.assertTrue(torch.isfinite(torch.tensor(float(tle2._xlcof))))

    def test_sgp4init_batch_opsmode_and_deep_space(self):
        tle = _sample_tle()
        whichconst = util.get_gravity_constants("wgs-84")

        with mock.patch.object(SGP4INIT_BATCH_MODULE, "numpy", types.SimpleNamespace(pi=np.pi), create=True), mock.patch.object(
            torch.Tensor, "__lt__", return_value=torch.tensor(True)
        ):
            out = initl_batch(
                whichconst[3],
                whichconst[4],
                torch.tensor([tle._ecco]),
                (tle._jdsatepoch + tle._jdsatepochF) - 2433281.5,
                torch.tensor([tle._inclo]),
                torch.tensor([tle._no_kozai]),
                "a",
                1,
                "n",
            )
        self.assertEqual(len(out[1]), 1)

        batch = tle.copy()
        with self.assertRaises(RuntimeError):
            sgp4init_batch(
                whichconst=whichconst,
                opsmode="i",
                satn=tle.satellite_catalog_number,
                epoch=(tle._jdsatepoch + tle._jdsatepochF) - 2433281.5,
                xbstar=torch.tensor([tle._bstar]),
                xndot=torch.tensor([tle._ndot]),
                xnddot=torch.tensor([tle._nddot]),
                xecco=torch.tensor([tle._ecco]),
                xargpo=torch.tensor([tle._argpo]),
                xinclo=torch.tensor([tle._inclo]),
                xmo=torch.tensor([tle._mo]),
                xno_kozai=torch.tensor([0.01]),
                xnodeo=torch.tensor([tle._nodeo]),
                satellite_batch=batch,
            )

    def test_tle_missing_paths(self):
        self.assertEqual(read_satellite_catalog_number("A1234"), 101234)

        with self.assertRaises(ValueError):
            load_from_lines([123, "x"])  # list non-string
        with self.assertRaises(ValueError):
            load_from_lines(12)  # invalid type
        with self.assertRaises(ValueError):
            load_from_lines([SAMPLE_LINE1])  # wrong length
        with self.assertRaises(ValueError):
            load_from_lines(["bad", SAMPLE_LINE2])

        bad_l2 = SAMPLE_LINE2.replace("43437", "43438", 1)
        with self.assertRaises(ValueError):
            load_from_lines([SAMPLE_LINE1, bad_l2])

        with self.assertRaises(ValueError):
            load_from_lines([SAMPLE_LINE1, "2 bad"])

        line1_1999 = SAMPLE_LINE1[:18] + "99" + SAMPLE_LINE1[20:]
        lines, data = load_from_lines([line1_1999, SAMPLE_LINE2])
        self.assertEqual(len(lines), 2)
        self.assertGreaterEqual(data["epoch_year"], 1999)

        lines_from_string, _ = load_from_lines(SAMPLE_LINE1 + "\n" + SAMPLE_LINE2)
        self.assertEqual(len(lines_from_string), 2)

        with mock.patch("dsgp4.tle.util.days2mdhms", return_value=(2, 31, 0, 0, 0.0)), mock.patch(
            "dsgp4.tle.util.invjday", return_value=(2020, 5, 22, 0, 0, 0.0)
        ):
            _, data_fallback = load_from_lines([SAMPLE_LINE1, SAMPLE_LINE2])
            self.assertIn("_epoch", data_fallback)

        tle = _sample_tle()
        with mock.patch("dsgp4.tle.util.days2mdhms", return_value=(2, 31, 0, 0, 0.0)), mock.patch(
            "dsgp4.tle.util.invjday", return_value=(2020, 5, 22, 0, 0, 0.0)
        ):
            tle2 = TLE(dict(tle._data))
            self.assertIn("_epoch", tle2._data)

        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write("0 NAME\n" + SAMPLE_LINE1 + "\n" + SAMPLE_LINE2 + "\n")
            tmp.write(SAMPLE_LINE1 + "\n" + SAMPLE_LINE2 + "\n")
            tmp_path = tmp.name
        try:
            tles = load(tmp_path)
            self.assertEqual(len(tles), 2)
        finally:
            os.unlink(tmp_path)

        with self.assertRaises(RuntimeError):
            TLE(3)

        d = dict(tle._data)
        d["epoch_year"] = 1999
        lines_99, _ = importlib.import_module("dsgp4.tle").load_from_data(d)
        self.assertEqual(len(lines_99), 2)

        with mock.patch("dsgp4.tle.compute_checksum", return_value=10):
            with self.assertRaises(RuntimeError):
                importlib.import_module("dsgp4.tle").load_from_data(dict(tle._data))

        with mock.patch("dsgp4.tle.compute_checksum", side_effect=[1, 10]):
            with self.assertRaises(RuntimeError):
                importlib.import_module("dsgp4.tle").load_from_data(dict(tle._data))

        old_epoch_days = tle.epoch_days
        tle.set_time(tle.date_mjd + 1.0)
        self.assertNotEqual(old_epoch_days, tle.epoch_days)

        old_mo = float(tle._mo)
        tle.update({"mean_anomaly": float(tle.mean_anomaly) + 0.01})
        self.assertNotEqual(old_mo, float(tle._mo))

        self.assertTrue(np.isfinite(tle.perigee_alt()))
        self.assertTrue(np.isfinite(tle.apogee_alt()))
        self.assertIn("TLE(", repr(tle))
        self.assertEqual(tle["line1"], tle.line1)
        self.assertEqual(tle.mean_motion, tle.__getattr__("mean_motion"))
        with self.assertRaises(AttributeError):
            tle.__getattr__("_data")
        with self.assertRaises(AttributeError):
            tle.__getattr__("does_not_exist")

    def test_util_missing_paths(self):
        with self.assertRaises(RuntimeError):
            util.get_gravity_constants("bad")

        tle = _sample_tle()
        err = "Error: deep space propagation not supported (yet). The provided satellite has an orbital period above 225 minutes. If you want to let us know you need it or you want to contribute to implement it, open a PR or raise an issue at: https://github.com/esa/dSGP4."
        with mock.patch.object(SGP4INIT_BATCH_MODULE, "sgp4init_batch", side_effect=Exception(err)):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                util.initialize_tle([tle])
            self.assertIn("were not initialized", buf.getvalue())

        with mock.patch.object(SGP4INIT_BATCH_MODULE, "sgp4init_batch", side_effect=Exception("other")):
            with self.assertRaises(Exception):
                util.initialize_tle([tle])

        d = util.from_year_day_to_date(2020, 32)
        self.assertEqual(d.month, 2)

        y, *_ = util.invjday(2415384.8)
        self.assertLessEqual(y, 1900)

        dt = util.from_string_to_datetime("2020-01-01 00:00:00")
        self.assertEqual(dt.year, 2020)

        days = util.from_mjd_to_epoch_days_after_1_jan(59000.0)
        self.assertTrue(days > 0)

        with self.assertRaises(ValueError):
            util.get_non_empty_lines(["a"])  # wrong input type
        self.assertEqual(util.get_non_empty_lines("a\n\n b\n"), ["a", " b"])

        mu = 1.0
        r_par = np.array([1.0, 0.0, 0.0])
        v_par = np.array([0.0, np.sqrt(2.0), 0.0])
        kep_par = util.from_cartesian_to_keplerian(r_par, v_par, mu)
        self.assertTrue(np.isinf(kep_par[0]))

        r_hyp = np.array([1.0, 0.0, 0.0])
        v_hyp = np.array([0.0, 2.0, 0.0])
        kep_hyp = util.from_cartesian_to_keplerian(r_hyp, v_hyp, mu)
        self.assertTrue(kep_hyp[1] > 1.0)

        with mock.patch("dsgp4.util.np.linalg.norm", side_effect=[1.0, np.sqrt(2.0), 1.0, 1.0, 1.0]):
            kep_parabolic = util.from_cartesian_to_keplerian(r_par, v_par, mu)
        self.assertTrue(np.isnan(kep_parabolic[5]))


if __name__ == "__main__":
    unittest.main()