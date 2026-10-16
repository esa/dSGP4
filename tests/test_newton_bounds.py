import unittest
from importlib import import_module
from unittest.mock import patch

import dsgp4
import torch

from dsgp4.newton_method import _bounded_eccentricity_step


class NewtonEccentricityBoundsTestCase(unittest.TestCase):
    def test_nominal_step_is_unchanged(self):
        self.assertAlmostEqual(
            _bounded_eccentricity_step(0.2, 0.05),
            0.05,
        )

    def test_negative_candidate_is_clamped_to_positive_domain(self):
        current = 0.2
        step = _bounded_eccentricity_step(current, -0.5)
        candidate = current + step

        self.assertGreater(candidate, 0.0)
        self.assertAlmostEqual(candidate, 1e-10)

    def test_superunit_candidate_is_clamped_below_one(self):
        current = 0.8
        step = _bounded_eccentricity_step(current, 0.5)
        candidate = current + step

        self.assertLess(candidate, 1.0)
        self.assertAlmostEqual(candidate, 1.0 - 1e-10)

    def test_newton_method_uses_the_bound_on_actual_solver_updates(self):
        module = import_module("dsgp4.newton_method")
        tle = dsgp4.TLE([
            "1 43437U 18100A   20143.90384230  .00041418  00000-0  10000-3 0 99968",
            "2 43437  97.8268 249.9127 0221000 123.9136 259.1144 15.12608579563539",
        ])
        dsgp4.initialize_tle(tle)
        epoch = dsgp4.util.from_datetime_to_mjd(tle._epoch)

        # An identity Jacobian makes the Newton update request exactly the
        # target eccentricity. No propagation or TLE serialization is needed.
        with (
            patch.object(module, "initial_guess_tle", return_value=tle),
            patch.object(module, "_propagate", side_effect=lambda values, *a, **kw: values.reshape(2, 3)),
            patch.object(module, "update_TLE", return_value=tle),
        ):
            for requested in (-0.5, 1.5):
                with self.subTest(requested=requested):
                    target = torch.zeros((2, 3))
                    target[0, 0] = requested
                    _, elements = module.newton_method(
                        tle, epoch, max_iter=1, new_tol=1e-20, target_state=target,
                    )
                    eccentricity = float(elements[0])
                    self.assertGreater(eccentricity, 0.0)
                    self.assertLess(eccentricity, 1.0)
                    bound = 1e-10 if requested < 0 else 1.0 - 1e-10
                    self.assertAlmostEqual(eccentricity, bound, places=8)

    def test_guard_corrects_existing_upper_bound_bug(self):
        current = 0.8

        # The previous implementation replaced a too-large Newton step with
        # 1 - 1e-10. Adding that as a step would yield an invalid e ~= 1.8.
        old_candidate = current + (1.0 - 1e-10)
        self.assertGreater(old_candidate, 1.0)

        corrected_step = _bounded_eccentricity_step(current, 0.5)
        self.assertLess(current + corrected_step, 1.0)


if __name__ == "__main__":
    unittest.main()
