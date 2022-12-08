import dsgp4
import sgp4
import sgp4.earth_gravity
import unittest


class UtilTestCase(unittest.TestCase):
    def test_earth_gravity_constants(self):
        out=dsgp4.util.get_gravity_constants("wgs-72")
        tumin,mu,radiusearthkm,xke,j2,j3,j4,j3oj2=out
        self.assertAlmostEqual(float(tumin),sgp4.earth_gravity.wgs72.tumin,places=10)
        self.assertAlmostEqual(float(mu),sgp4.earth_gravity.wgs72.mu,places=10)
        self.assertAlmostEqual(float(radiusearthkm),sgp4.earth_gravity.wgs72.radiusearthkm,places=10)
        self.assertAlmostEqual(float(xke),sgp4.earth_gravity.wgs72.xke,places=10)
        self.assertAlmostEqual(float(j2),sgp4.earth_gravity.wgs72.j2,places=10)
        self.assertAlmostEqual(float(j3),sgp4.earth_gravity.wgs72.j3,places=10)
        self.assertAlmostEqual(float(j4),sgp4.earth_gravity.wgs72.j4,places=10)
        self.assertAlmostEqual(float(j3oj2),sgp4.earth_gravity.wgs72.j3oj2,places=10)

        out=dsgp4.util.get_gravity_constants("wgs-72old")
        tumin,mu,radiusearthkm,xke,j2,j3,j4,j3oj2=out
        self.assertAlmostEqual(float(tumin),sgp4.earth_gravity.wgs72old.tumin,places=10)
        self.assertAlmostEqual(float(mu),sgp4.earth_gravity.wgs72old.mu,places=10)
        self.assertAlmostEqual(float(radiusearthkm),sgp4.earth_gravity.wgs72old.radiusearthkm,places=10)
        self.assertAlmostEqual(float(xke),sgp4.earth_gravity.wgs72old.xke,places=10)
        self.assertAlmostEqual(float(j2),sgp4.earth_gravity.wgs72old.j2,places=10)
        self.assertAlmostEqual(float(j3),sgp4.earth_gravity.wgs72old.j3,places=10)
        self.assertAlmostEqual(float(j4),sgp4.earth_gravity.wgs72old.j4,places=10)
        self.assertAlmostEqual(float(j3oj2),sgp4.earth_gravity.wgs72old.j3oj2,places=10)

        out=dsgp4.util.get_gravity_constants("wgs-84")
        tumin,mu,radiusearthkm,xke,j2,j3,j4,j3oj2=out
        self.assertAlmostEqual(float(tumin),sgp4.earth_gravity.wgs84.tumin,places=10)
        self.assertAlmostEqual(float(mu),sgp4.earth_gravity.wgs84.mu,places=10)
        self.assertAlmostEqual(float(radiusearthkm),sgp4.earth_gravity.wgs84.radiusearthkm,places=10)
        self.assertAlmostEqual(float(xke),sgp4.earth_gravity.wgs84.xke,places=10)
        self.assertAlmostEqual(float(j2),sgp4.earth_gravity.wgs84.j2,places=10)
        self.assertAlmostEqual(float(j3),sgp4.earth_gravity.wgs84.j3,places=10)
        self.assertAlmostEqual(float(j4),sgp4.earth_gravity.wgs84.j4,places=10)
        self.assertAlmostEqual(float(j3oj2),sgp4.earth_gravity.wgs84.j3oj2,places=10)
