import unittest

import torch

import dsgp4

LEO_TLES = [
    ["1 25544U 98067A   24087.49097222  .00016717  00000+0  10270-3 0  9993",
     "2 25544  51.6400  82.2420 0006290  58.9900  53.5550 15.50000000000000"],
    ["1 39634U 14016A   22059.08188563  .00000057  00000+0  21846-4 0  9990",
     "2 39634  98.1819  68.1874 0001341  82.4703 277.6657 14.59199732421074"],
]
DEEP_SPACE_TLE = ["1 14128U 83058A   06176.02844893 -.00000158  00000-0  10000-3 0  9627",
                  "2 14128  11.4384  35.2134 0011562  26.4582 333.5652  0.98870114 46093"]
#Sentinel-1A (the same elements as the second LEO TLE), in the OMM format:
OMM_FIELDS = {
    "CCSDS_OMM_VERS": "3.0", "CREATION_DATE": "2022-02-28T09:16:12", "ORIGINATOR": "18 SPCS",
    "OBJECT_NAME": "SENTINEL-1A", "OBJECT_ID": "2014-016A", "CENTER_NAME": "EARTH", "REF_FRAME": "TEME",
    "TIME_SYSTEM": "UTC", "MEAN_ELEMENT_THEORY": "SGP4", "EPOCH": "2022-02-28T01:57:54.918432",
    "MEAN_MOTION": "14.59199732", "ECCENTRICITY": "0.0001341", "INCLINATION": "98.1819",
    "RA_OF_ASC_NODE": "68.1874", "ARG_OF_PERICENTER": "82.4703", "MEAN_ANOMALY": "277.6657",
    "EPHEMERIS_TYPE": "0", "CLASSIFICATION_TYPE": "U", "NORAD_CAT_ID": "39634", "ELEMENT_SET_NO": "999",
    "REV_AT_EPOCH": "42107", "BSTAR": "0.000021846", "MEAN_MOTION_DOT": "0.00000057", "MEAN_MOTION_DDOT": "0",
}


def float64_tensors(tle):
    """Names of the float64 tensors a TLE carries: parsed elements (`_data`) and, once
    initialized, the SGP4 attributes set on the object itself."""
    tensors = dict(tle._data)
    tensors.update(vars(tle))
    return sorted(k for k, v in tensors.items() if isinstance(v, torch.Tensor) and v.dtype == torch.float64)


class Float32DefaultDtypeTestCase(unittest.TestCase):
    """Under a float32 default, every tensor dsgp4 creates must be float32."""

    def setUp(self):
        self.previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)

    def tearDown(self):
        torch.set_default_dtype(self.previous_dtype)

    def assertFloat32(self, tle):
        self.assertEqual(float64_tensors(tle), [], f"float64 tensors in {tle.satellite_catalog_number}")

    def test_parsed_elements(self):
        self.assertFloat32(dsgp4.tle.TLE(LEO_TLES[0]))
        self.assertFloat32(dsgp4.omm.OMM(OMM_FIELDS))

    def test_gravity_constants(self):
        for name in ("wgs-72", "wgs-84", "wgs-72old"):
            for constant in dsgp4.util.get_gravity_constants(name):
                self.assertEqual(constant.dtype, torch.float32, name)

    def test_single_tle(self):
        for lines in (LEO_TLES[0], DEEP_SPACE_TLE):
            tle = dsgp4.tle.TLE(lines)
            elements = dsgp4.initialize_tle(tle, with_grad=True)
            self.assertEqual(elements.dtype, torch.float32)
            self.assertFloat32(tle)
            state = dsgp4.propagate(tle, torch.linspace(0.0, 1440.0, 7))
            self.assertEqual(state.dtype, torch.float32)

    def test_batch(self):
        tles = [dsgp4.tle.TLE(lines) for lines in LEO_TLES]
        elements, batch = dsgp4.initialize_tle(tles)
        self.assertEqual(torch.stack(elements).dtype, torch.float32)
        self.assertFloat32(batch)
        state = dsgp4.propagate_batch(batch, torch.tensor([10.0, 20.0]))
        self.assertEqual(state.dtype, torch.float32)

    def test_batch_with_deep_space_fallback(self):
        tles = [dsgp4.tle.TLE(DEEP_SPACE_TLE), dsgp4.tle.TLE(LEO_TLES[0])]
        _, initialized = dsgp4.initialize_tle(tles)
        self.assertIsInstance(initialized, list)
        for tle in initialized:
            self.assertFloat32(tle)
        state = dsgp4.propagate_batch(initialized, torch.tensor([10.0, 20.0]))
        self.assertEqual(state.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
