import datetime
import numpy as np
import os
import tempfile
import torch
import unittest

import dsgp4
import dsgp4.omm

#Sentinel-1A, as distributed by Space-Track in the OMM (JSON) format:
OMM_FIELDS = {
    "CCSDS_OMM_VERS": "3.0",
    "CREATION_DATE": "2022-02-28T09:16:12",
    "ORIGINATOR": "18 SPCS",
    "OBJECT_NAME": "SENTINEL-1A",
    "OBJECT_ID": "2014-016A",
    "CENTER_NAME": "EARTH",
    "REF_FRAME": "TEME",
    "TIME_SYSTEM": "UTC",
    "MEAN_ELEMENT_THEORY": "SGP4",
    "EPOCH": "2022-02-28T01:57:54.918432",
    "MEAN_MOTION": "14.59199732",
    "ECCENTRICITY": "0.0001341",
    "INCLINATION": "98.1819",
    "RA_OF_ASC_NODE": "68.1874",
    "ARG_OF_PERICENTER": "82.4703",
    "MEAN_ANOMALY": "277.6657",
    "EPHEMERIS_TYPE": "0",
    "CLASSIFICATION_TYPE": "U",
    "NORAD_CAT_ID": "39634",
    "ELEMENT_SET_NO": "999",
    "REV_AT_EPOCH": "42107",
    "BSTAR": "0.000021846",
    "MEAN_MOTION_DOT": "0.00000057",
    "MEAN_MOTION_DDOT": "0",
}

#the very same elements, in the TLE format:
TLE_LINES = ['1 39634U 14016A   22059.08188563  .00000057  00000+0  21846-4 0  9990',
             '2 39634  98.1819  68.1874 0001341  82.4703 277.6657 14.59199732421074']


class OMMTestCase(unittest.TestCase):
    def test_read_omm(self):
        omm = dsgp4.omm.OMM(OMM_FIELDS)
        datetime_object = datetime.datetime.strptime(omm['date_string'], '%Y-%m-%d %H:%M:%S.%f')

        #Now the values to validate:
        satellite_catalog_number = 39634
        classification = 'U'
        international_designator = '14016A'
        mean_motion_first_derivative = 9.595270771959568e-16
        mean_motion_second_derivative = 0.0
        b_star = 2.1846e-05
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

        self.assertEqual(omm['name'], 'SENTINEL-1A')
        self.assertEqual(omm['international_designator'], international_designator)
        self.assertEqual(omm['classification'], classification)
        self.assertEqual(omm['satellite_catalog_number'], satellite_catalog_number)
        self.assertEqual(omm['date_string'], date_string)
        self.assertEqual(datetime_object.year, 2022)
        self.assertEqual(datetime_object.microsecond, 918432)
        self.assertEqual(omm['ephemeris_type'], ephemeris_type)
        self.assertEqual(omm['element_number'], element_number)
        self.assertEqual(omm['revolution_number_at_epoch'], revolution_number_at_epoch)
        self.assertEqual(omm['creation_date'], '2022-02-28T09:16:12')
        self.assertEqual(omm['originator'], '18 SPCS')
        self.assertAlmostEqual(omm['b_star'], b_star, places = 10)
        self.assertAlmostEqual(omm['mean_motion_first_derivative'], mean_motion_first_derivative, places = 10)
        self.assertAlmostEqual(omm['mean_motion_second_derivative'], mean_motion_second_derivative, places = 10)
        self.assertAlmostEqual(omm['inclination'], inclination, places = 10)
        self.assertAlmostEqual(omm['raan'], raan, places = 10)
        self.assertAlmostEqual(omm['eccentricity'], eccentricity, places = 10)
        self.assertAlmostEqual(omm['argument_of_perigee'], argument_of_perigee, places = 10)
        self.assertAlmostEqual(omm['mean_anomaly'], mean_anomaly, places = 10)
        self.assertAlmostEqual(omm['mean_motion'], mean_motion, places = 10)

    def test_omm_matches_tle(self):
        #the OMM and the TLE of the same object must produce the very same SGP4 elements,
        #and therefore the very same states:
        omm = dsgp4.omm.OMM(OMM_FIELDS)
        tle = dsgp4.tle.TLE(TLE_LINES)
        for element in ['_bstar', '_ndot', '_nddot', '_ecco', '_argpo', '_inclo', '_mo',
                        '_no_kozai', '_nodeo', '_jdsatepoch', '_jdsatepochF']:
            self.assertEqual(float(omm[element]), float(tle[element]))

        tsinces = torch.tensor([-120., 0., 33.5, 720.])
        dsgp4.initialize_tle(omm)
        dsgp4.initialize_tle(tle)
        self.assertTrue(torch.equal(dsgp4.propagate(omm, tsinces), dsgp4.propagate(tle, tsinces)))

    def test_omm_is_a_tle(self):
        #an OMM must be usable everywhere a TLE is expected:
        omm = dsgp4.omm.OMM(OMM_FIELDS)
        self.assertIsInstance(omm, dsgp4.tle.TLE)
        r_earth = float(dsgp4.util.get_gravity_constants('wgs-84')[2])*1e3
        self.assertAlmostEqual(float(omm.semi_major_axis)*1e-3, 7073.8965, places=3)
        self.assertAlmostEqual(float(omm.perigee_alt(r_earth))*1e-3, 694.8109, places=3)
        self.assertAlmostEqual(float(omm.apogee_alt(r_earth))*1e-3, 696.7081, places=3)

        states = dsgp4.propagate_batch([omm, dsgp4.tle.TLE(TLE_LINES)],
                                       torch.tensor([10., 20.]),
                                       initialized=False)
        self.assertEqual(states.shape, torch.Size([2, 2, 3]))

    def test_omm_has_no_tle_lines(self):
        #an OMM carries the same elements of a TLE, but not the two lines: asking for them must
        #point at the conversion that does produce them
        omm = dsgp4.omm.OMM(OMM_FIELDS)
        for attribute in ['line0', 'line1', 'line2', '_lines']:
            self.assertFalse(hasattr(omm, attribute))
            with self.assertRaises(AttributeError) as context:
                getattr(omm, attribute)
            self.assertIn('to_tle()', str(context.exception))
        #the lines of a TLE that is converted into an OMM must not linger either:
        self.assertFalse(hasattr(dsgp4.tle.TLE(TLE_LINES).to_omm(), 'line1'))
        #while the TLE representation of course has them:
        self.assertEqual(omm.to_tle().line2, TLE_LINES[1])

    def test_inherited_tle_api(self):
        #the OMM object inherits the whole public API of the TLE class: this test is here so
        #that a method added to `dsgp4.tle.TLE` (e.g. one writing out the two lines, which an
        #OMM may not have) is checked against the OMM object as well, instead of silently
        #breaking it. If it fails, make sure the new method works here, then add it below.
        inherited = {name for name in vars(dsgp4.tle.TLE) if not name.startswith('_')}
        self.assertEqual(inherited, {'copy', 'to_omm', 'set_time', 'update',
                                     'perigee_alt', 'apogee_alt'},
                         'the public methods of `dsgp4.tle.TLE` changed: an OMM inherits them, '
                         'so make sure the new one does not assume the two lines (which an OMM '
                         'does not have), then update this list')
        omm = dsgp4.omm.OMM(OMM_FIELDS)
        for name in inherited:
            self.assertTrue(callable(getattr(omm, name)))
        #`to_omm` is the only one that is not exercised elsewhere in this file:
        self.assertEqual(float(omm.to_omm()._no_kozai), float(omm._no_kozai))

    def test_all_formats(self):
        #every serialization of the standard must yield the same elements:
        omm = dsgp4.omm.OMM(OMM_FIELDS)
        for file_format in ['json', 'xml', 'kvn', 'csv']:
            text = dsgp4.omm.dumps([omm, omm], file_format=file_format)
            self.assertEqual(dsgp4.omm.detect_format(text), file_format)
            records = dsgp4.omm.loads(text)
            self.assertEqual(len(records), 2)
            for record in records:
                other = dsgp4.omm.OMM(record)
                for element in ['_bstar', '_ndot', '_nddot', '_ecco', '_argpo', '_inclo',
                                '_mo', '_no_kozai', '_nodeo', '_jdsatepoch', '_jdsatepochF']:
                    self.assertEqual(float(other[element]), float(omm[element]))
                self.assertEqual(other.name, omm.name)
                self.assertEqual(other.satellite_catalog_number, omm.satellite_catalog_number)

    def test_omm_from_string(self):
        omm = dsgp4.omm.OMM(OMM_FIELDS)
        for file_format in ['json', 'xml', 'kvn', 'csv']:
            other = dsgp4.omm.OMM(dsgp4.omm.dumps(omm, file_format=file_format))
            self.assertEqual(float(other._no_kozai), float(omm._no_kozai))

    def test_kvn_with_units_and_comments(self):
        #the KVN format allows comments, and the units to be appended to the values:
        text = '\n'.join(['CCSDS_OMM_VERS = 3.0',
                          'COMMENT this line is to be ignored',
                          'CREATION_DATE = 2022-02-28T09:16:12',
                          'EPOCH = 2022-02-28T01:57:54.918432',
                          'MEAN_MOTION = 14.59199732 [rev/day]',
                          'ECCENTRICITY = 0.0001341',
                          'INCLINATION = 98.1819 [deg]',
                          'RA_OF_ASC_NODE = 68.1874 [deg]',
                          'ARG_OF_PERICENTER = 82.4703 [deg]',
                          'MEAN_ANOMALY = 277.6657 [deg]',
                          'NORAD_CAT_ID = 39634',
                          'BSTAR = 0.000021846 [1/ER]'])
        omm = dsgp4.omm.OMM(text)
        self.assertEqual(float(omm._no_kozai), float(dsgp4.omm.OMM(OMM_FIELDS)._no_kozai))
        self.assertEqual(float(omm._inclo), float(dsgp4.omm.OMM(OMM_FIELDS)._inclo))
        self.assertEqual(omm.satellite_catalog_number, 39634)

    def test_tle_omm_conversion(self):
        tle = dsgp4.tle.TLE(TLE_LINES)
        omm = tle.to_omm()
        self.assertIsInstance(omm, dsgp4.omm.OMM)
        self.assertEqual(omm._fields['NORAD_CAT_ID'], '39634')
        self.assertEqual(omm._fields['OBJECT_ID'], '2014-016A')
        self.assertEqual(omm._fields['MEAN_ELEMENT_THEORY'], 'SGP4')
        #and back: going through the OMM must give the very same lines that the TLE object
        #writes out of its own elements:
        back = omm.to_tle()
        rewritten = dsgp4.tle.TLE(dsgp4.tle.copy_data(tle._data))
        self.assertIsInstance(back, dsgp4.tle.TLE)
        self.assertEqual([back.line1, back.line2], [rewritten.line1, rewritten.line2])
        self.assertEqual(back.line2, TLE_LINES[1])

    def test_load_from_file(self):
        omm = dsgp4.omm.OMM(OMM_FIELDS)
        for file_format, extension in [('json', '.json'), ('xml', '.xml'),
                                       ('kvn', '.kvn'), ('csv', '.csv')]:
            with tempfile.TemporaryDirectory() as folder:
                file_name = os.path.join(folder, 'omm'+extension)
                with open(file_name, 'w') as f:
                    f.write(dsgp4.omm.dumps([omm, omm, omm], file_format=file_format))
                omms = dsgp4.omm.load(file_name)
                self.assertEqual(len(omms), 3)
                for other in omms:
                    self.assertIsInstance(other, dsgp4.omm.OMM)
                    self.assertEqual(float(other._no_kozai), float(omm._no_kozai))

    def test_catalog_number_beyond_the_tle_format(self):
        #objects whose catalog number is above 339999 cannot be written as TLEs, but the OMM
        #format handles them without any issue:
        fields = dict(OMM_FIELDS, NORAD_CAT_ID='700123')
        omm = dsgp4.omm.OMM(fields)
        self.assertEqual(omm.satellite_catalog_number, 700123)
        dsgp4.initialize_tle(omm)
        state = dsgp4.propagate(omm, torch.tensor([0., 10.]))
        self.assertEqual(state.shape, torch.Size([2, 2, 3]))
        self.assertEqual(omm.copy().satellite_catalog_number, 700123)
        with self.assertRaises(ValueError):
            omm.to_tle()

    def test_copy_set_time_and_update(self):
        omm = dsgp4.omm.OMM(OMM_FIELDS)
        copied = omm.copy()
        self.assertIsInstance(copied, dsgp4.omm.OMM)
        self.assertEqual(copied.date_mjd, omm.date_mjd)

        copied.set_time(omm.date_mjd+0.5)
        self.assertAlmostEqual(copied.date_mjd, omm.date_mjd+0.5, places=6)
        self.assertTrue(copied._fields['EPOCH'].startswith('2022-02-28T13:57:54'))
        #the original object must be left untouched:
        self.assertEqual(omm._fields['EPOCH'], OMM_FIELDS['EPOCH'])

        copied.update({'b_star': 0.001})
        self.assertAlmostEqual(float(copied._bstar), 0.001, places=10)
        self.assertEqual(copied._fields['BSTAR'], '0.001')

    def test_newton_method_with_omm(self):
        omm = dsgp4.omm.OMM(OMM_FIELDS)
        found, _ = dsgp4.newton_method(omm, omm.date_mjd+0.1)
        self.assertIsInstance(found, dsgp4.omm.OMM)

    def test_object_id_and_epoch_conversions(self):
        self.assertEqual(dsgp4.omm.from_object_id_to_international_designator('1998-067A'), '98067A')
        self.assertEqual(dsgp4.omm.from_object_id_to_international_designator('2014-016A'), '14016A')
        self.assertEqual(dsgp4.omm.from_object_id_to_international_designator('2024-123ABC'), '24123ABC')
        #unknown or already converted identifiers are returned as they are:
        self.assertEqual(dsgp4.omm.from_object_id_to_international_designator('UNKNOWN'), 'UNKNOWN')
        self.assertEqual(dsgp4.omm.from_object_id_to_international_designator('98067A'), '98067A')
        self.assertEqual(dsgp4.omm.from_international_designator_to_object_id('98067A'), '1998-067A')
        self.assertEqual(dsgp4.omm.from_international_designator_to_object_id('14016A'), '2014-016A')
        self.assertEqual(dsgp4.omm.from_international_designator_to_object_id('56001A'), '2056-001A')
        self.assertEqual(dsgp4.omm.from_international_designator_to_object_id('57001A'), '1957-001A')
        self.assertEqual(dsgp4.omm.from_international_designator_to_object_id(''), '')

        epoch = datetime.datetime(2024, 2, 29, 12, 34, 56, 789012)
        for text in ['2024-02-29T12:34:56.789012', '2024-02-29T12:34:56.789012Z',
                     '2024-02-29 12:34:56.789012', '2024-060T12:34:56.789012']:
            self.assertEqual(dsgp4.omm.from_omm_epoch_to_datetime(text), epoch)
        self.assertEqual(dsgp4.omm.from_omm_epoch_to_datetime(epoch), epoch)
        self.assertEqual(dsgp4.omm.from_datetime_to_omm_epoch(epoch), '2024-02-29T12:34:56.789012')
        self.assertEqual(dsgp4.omm.from_omm_epoch_to_datetime('2024-02-29'), datetime.datetime(2024, 2, 29))
        with self.assertRaises(ValueError):
            dsgp4.omm.from_omm_epoch_to_datetime('2024')

    def test_optional_fields(self):
        #everything but the mean elements is optional:
        fields = {key: OMM_FIELDS[key] for key in
                  ['EPOCH', 'MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION', 'RA_OF_ASC_NODE',
                   'ARG_OF_PERICENTER', 'MEAN_ANOMALY']}
        omm = dsgp4.omm.OMM(fields)
        self.assertEqual(omm.satellite_catalog_number, 0)
        self.assertEqual(omm.classification, 'U')
        self.assertEqual(omm.b_star, 0.)
        self.assertEqual(omm.element_number, 0)
        self.assertEqual(omm.revolution_number_at_epoch, 0)
        self.assertNotIn('name', omm._data)
        #the CSV format leaves the fields that do not apply to an object empty:
        omm = dsgp4.omm.OMM(dict(OMM_FIELDS, BSTAR='', REV_AT_EPOCH=''))
        self.assertEqual(omm.b_star, 0.)
        self.assertEqual(omm.revolution_number_at_epoch, 0)
        #the keywords of the standard are upper case, but they are read no matter their case:
        _, data = dsgp4.omm.load_from_omm({key.title(): value for key, value in OMM_FIELDS.items()})
        self.assertEqual(data['satellite_catalog_number'], 39634)

    def test_errors(self):
        #dSGP4 can only propagate SGP4 mean elements:
        with self.assertRaises(ValueError):
            dsgp4.omm.OMM(dict(OMM_FIELDS, MEAN_ELEMENT_THEORY='DSST'))
        #the mean elements are mandatory:
        with self.assertRaises(ValueError):
            dsgp4.omm.OMM({key: value for key, value in OMM_FIELDS.items() if key != 'MEAN_MOTION'})
        with self.assertRaises(ValueError):
            dsgp4.omm.load_from_omm(['not', 'a', 'dictionary'])
        #a string must contain one and only one message:
        with self.assertRaises(ValueError):
            dsgp4.omm.OMM(dsgp4.omm.dumps([dsgp4.omm.OMM(OMM_FIELDS)]*2, file_format='kvn'))
        with self.assertRaises(RuntimeError):
            dsgp4.omm.OMM(42)
        #unknown or undetectable formats:
        with self.assertRaises(ValueError):
            dsgp4.omm.loads('CCSDS_OMM_VERS = 3.0', file_format='yaml')
        with self.assertRaises(ValueError):
            dsgp4.omm.dumps(dsgp4.omm.OMM(OMM_FIELDS), file_format='yaml')
        with self.assertRaises(ValueError):
            dsgp4.omm.loads(['not', 'a', 'string'])
        with self.assertRaises(ValueError):
            dsgp4.omm.detect_format('COMMENT nothing else here')

    def test_against_python_sgp4(self):
        #cross-check with the OMM support of python-sgp4 (which initializes the propagator
        #with the WGS-72 constants):
        import sgp4.api
        import sgp4.omm

        satellite = sgp4.api.Satrec()
        sgp4.omm.initialize(satellite, dict(OMM_FIELDS))
        omm = dsgp4.omm.OMM(OMM_FIELDS)
        dsgp4.initialize_tle(omm, gravity_constant_name="wgs-72")
        for element, reference in [('_bstar', 'bstar'), ('_ndot', 'ndot'), ('_nddot', 'nddot'),
                                   ('_ecco', 'ecco'), ('_argpo', 'argpo'), ('_inclo', 'inclo'),
                                   ('_mo', 'mo'), ('_no_kozai', 'no_kozai'), ('_nodeo', 'nodeo')]:
            self.assertEqual(float(omm[element]), getattr(satellite, reference))

        for tsince in [-100., 0., 42.7, 600.]:
            _, position, velocity = satellite.sgp4_tsince(tsince)
            state = dsgp4.propagate(omm, torch.tensor([tsince]))
            for i in range(3):
                self.assertAlmostEqual(position[i], float(state[0][i]), places=8)
                self.assertAlmostEqual(velocity[i], float(state[1][i]), places=8)
