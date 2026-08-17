import datetime
import unittest

import dsgp4


class DatetimePrecisionTestCase(unittest.TestCase):
    def test_leading_zero_microseconds_are_scaled_as_microseconds(self):
        # Leading zeroes are significant: 123 microseconds is 0.000123 s,
        # not 0.123 s.
        dt = datetime.datetime(2026, 8, 17, 12, 34, 56, 123)

        actual = dsgp4.util.from_datetime_to_jd(dt)
        expected = sum(
            dsgp4.util.jday(
                year=2026,
                mon=8,
                day=17,
                hr=12,
                minute=34,
                sec=56.000123,
            )
        )

        self.assertAlmostEqual(actual, expected, places=11)

    def test_fractional_seconds_follow_datetime_microsecond_scale(self):
        base = datetime.datetime(2026, 8, 17, 0, 0, 0, 0)
        dt = base.replace(microsecond=123)

        delta_days = (
            dsgp4.util.from_datetime_to_mjd(dt)
            - dsgp4.util.from_datetime_to_mjd(base)
        )
        expected_days = 123e-6 / 86400.0

        # Julian-date floats at modern epochs have finite resolution, so use
        # a tolerance far below the previous 0.123-second scaling error.
        self.assertAlmostEqual(delta_days, expected_days, delta=1e-9)

    def test_jd_round_trip_preserves_subsecond_epoch(self):
        original = datetime.datetime(2024, 2, 29, 12, 0, 0, 12345)
        jd = dsgp4.util.from_datetime_to_jd(original)
        recovered = dsgp4.util.from_jd_to_datetime(jd)

        error_seconds = abs((recovered - original).total_seconds())
        self.assertLess(error_seconds, 1e-4)


if __name__ == "__main__":
    unittest.main()
