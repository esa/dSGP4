import unittest

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
