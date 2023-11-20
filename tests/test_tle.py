import datetime
import numpy as np
import torch
import unittest

import dsgp4.tle

class UtilTestCase(unittest.TestCase):
    def test_tle(self):
        data = dict(
            satellite_catalog_number=43437,
            classification='U',
            international_designator='18100A',
            epoch_year=2020,
            epoch_days=143.90384230,
            ephemeris_type=0,
            element_number=9996,
            revolution_number_at_epoch=56353,
            mean_motion=0.0011,
            mean_motion_first_derivative=6.9722e-13,
            mean_motion_second_derivative=.0,
            eccentricity=0.0221,
            inclination=1.7074,
            argument_of_perigee=2.1627,
            raan=4.3618,
            mean_anomaly=4.5224,
            b_star=0.0001)

        tle = dsgp4.tle.TLE(data)
        line1, line2 = tle.line1, tle.line2
        line1Correct = '1 43437U 18100A   20143.90384230  .00041418  00000-0  10000-3 0 99968'
        line2Correct = '2 43437  97.8268 249.9127 0221000 123.9136 259.1144 15.12608579563539'

        self.assertEqual(line1Correct, line1)
        self.assertEqual(line2Correct, line2)

        data = dict(
            satellite_catalog_number=43437,
            classification='U',
            international_designator='18100A',
            epoch_year=2020,
            epoch_days=143.90384230,
            ephemeris_type=0,
            element_number=9996,
            revolution_number_at_epoch=56353,
            mean_motion=0.0010,
            mean_motion_first_derivative=1.6261e-13,
            mean_motion_second_derivative=1e-5,
            eccentricity=0.0274,
            inclination=1.5086,
            argument_of_perigee=4.9757,
            raan=3.8719,
            mean_anomaly=6.0775,
            b_star=-0.0010)

        tle = dsgp4.tle.TLE(data)
        line1, line2 = tle.line1, tle.line2
        line1Correct = '1 43437U 18100A   20143.90384230  .00009660  17108+9 -10000-2 0 99966'
        line2Correct = '2 43437  86.4364 221.8435 0274000 285.0866 348.2151 13.75098708563531'

        self.assertEqual(line1Correct, line1)
        self.assertEqual(line2Correct, line2)

    def test_read_tle(self):
        #Sentinel-1A:
        tle_line1 = '1 39634U 14016A   22059.08188563  .00000057  00000+0  21846-4 0  9990'
        tle_line2 = '2 39634  98.1819  68.1874 0001341  82.4703 277.6657 14.59199732421074'
        tle = dsgp4.tle.TLE([tle_line1, tle_line2])
        datetime_object = datetime.datetime.strptime(tle['date_string'], '%Y-%m-%d %H:%M:%S.%f')

        #Now the values to validate:
        satellite_catalog_number = 39634
        classification = 'U'
        international_designator = '14016A'
        mean_motion_first_derivative = 9.595270771959568e-16
        mean_motion_second_derivative = 0.0
        b_star = 2.1846e-05
        year = 2022
        month = 2
        day = 28
        hour = 1
        minute = 57
        second = 54
        microsecond = 918432
        date_string = '2022-02-28 01:57:54.918432'
        ephemeris_type = 0
        element_number = 999
        inclination = 1.7135974208638207
        raan = 1.1900946383743813
        eccentricity = 0.0001341
        argument_of_perigee = 1.4393782701074795
        mean_anomaly = 4.8461806848548195
        mean_motion = 0.0010611599903174525
        revolution_number_at_epoch = 42107

        self.assertEqual(tle['international_designator'], international_designator)
        self.assertEqual(tle['classification'], classification)
        self.assertEqual(tle['satellite_catalog_number'], satellite_catalog_number)
        self.assertEqual(datetime_object.year, year)
        self.assertEqual(datetime_object.month, month)
        self.assertEqual(datetime_object.day, day)
        self.assertEqual(datetime_object.hour, hour)
        self.assertEqual(datetime_object.minute, minute)
        self.assertEqual(datetime_object.second, second)
        self.assertEqual(datetime_object.microsecond, microsecond)
        self.assertEqual(tle['date_string'], date_string)
        self.assertEqual(tle['ephemeris_type'], ephemeris_type)
        self.assertEqual(tle['element_number'], element_number)
        self.assertEqual(tle['revolution_number_at_epoch'], revolution_number_at_epoch)
        self.assertAlmostEqual(tle['b_star'], b_star, places = 10)
        self.assertAlmostEqual(tle['mean_motion_first_derivative'], mean_motion_first_derivative, places = 10)
        self.assertAlmostEqual(tle['mean_motion_second_derivative'], mean_motion_second_derivative, places = 10)
        self.assertAlmostEqual(tle['inclination'], inclination, places = 10)
        self.assertAlmostEqual(tle['raan'], raan, places = 10)
        self.assertAlmostEqual(tle['eccentricity'], eccentricity, places = 10)
        self.assertAlmostEqual(tle['argument_of_perigee'], argument_of_perigee, places = 10)
        self.assertAlmostEqual(tle['mean_anomaly'], mean_anomaly, places = 10)
        self.assertAlmostEqual(tle['mean_motion'], mean_motion, places = 10)

    def test_read_tle_make_tle(self):

        #we now test that from a TLE, I can read the elements and construct a new one:
        lines_in = ['1 00046U 60007B   22054.60355309  .00000135  00000-0  49836-4 0  9990',
                 '2 00046  66.6904 181.4288 0206178  29.9665 331.3056 14.49532408228403']

        tle = dsgp4.tle.TLE(lines_in)

        datetime_object = datetime.datetime.strptime(tle['date_string'], '%Y-%m-%d %H:%M:%S.%f')
        epoch_year = datetime_object.year
        epoch_days = dsgp4.util.from_datetime_to_fractional_day(datetime_object)

        tle2 = dsgp4.tle.TLE(data = tle._data)
        lines_out = [tle2.line1, tle2.line2]

        self.assertEqual(lines_in[0], lines_out[0])
        self.assertEqual(lines_in[1], lines_out[1])

    def test_load_tles(self):
        tle_1 = ['1 00046U 60007B   22054.60355309  .00000135  00000-0  49836-4 0  9990',
                 '2 00046  66.6904 181.4288 0206178  29.9665 331.3056 14.49532408228403']
        lines_1, data_1 = dsgp4.tle.load_from_lines(tle_1)
        lines_2, data_2 = dsgp4.tle.load_from_data(data_1)
        self.assertEqual(tle_1, lines_1)
        self.assertEqual(lines_1, lines_2)
        self.assertEqual(data_1, data_2)
        
    def test_tle_from_vallado(self):
        #this test is taken from page 107 of Dr. David Vallado's book: Fundamentals of Astrodynamics and Applications, 4th Edition
        tle_lines=['1 16609U 86017A   93352.53502934  .00007889  00000-0  10529-3 0   342',
                   '2 16609  51.6190  13.3340 0005770 102.5680 257.5950 15.59114070447869']
        tle=dsgp4.tle.TLE(tle_lines)
        #TLE elements:
        xpdotp   =  1440.0 / (2.0 *np.pi);
        self.assertAlmostEqual(float(tle._bstar),0.00010529, places = 8)
        self.assertAlmostEqual(float(tle._ndot)*(xpdotp*1440.0),7.889*1e-5, places = 8)
        self.assertAlmostEqual(float(tle._nddot),0.0, places = 8)
        self.assertAlmostEqual(float(tle._ecco), 0.0005770, places = 8)
        self.assertAlmostEqual(float(tle._nodeo)*180.0/np.pi, 13.3340, places = 8)
        self.assertAlmostEqual(float(tle._inclo)*180.0/np.pi, 51.6190, places = 8)
        self.assertAlmostEqual(float(tle._mo)*180.0/np.pi, 257.5950, places = 8)
        self.assertAlmostEqual(float(tle._argpo)*180.0/np.pi, 102.5680, places = 8)
        self.assertAlmostEqual(round(float(tle._no_kozai)*xpdotp,8), 15.59114070, places = 8)

        #epoch:
        self.assertAlmostEqual(tle._epoch.month,12, places = 8)
        self.assertAlmostEqual(tle._epoch.year,1993, places = 8)
        self.assertAlmostEqual(tle._epoch.day,18, places = 8)
        self.assertAlmostEqual(tle._epoch.hour,12, places = 8)
        self.assertAlmostEqual(tle._epoch.minute,50, places = 8)
        self.assertAlmostEqual(tle._epoch.second,26, places = 8)
        self.assertAlmostEqual(round(tle._epoch.microsecond*1e-6,4),0.5350, places = 8)
