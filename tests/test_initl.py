import dsgp4
import kessler
import numpy as np
import sgp4
import sgp4.io
import unittest

class UtilTestCase(unittest.TestCase):
    def test_initl(self):
        error_string="Error: deep space propagation not supported (yet). The provided satellite has\
        an orbital period above 225 minutes. If you want to let us know you need it or you want to \
        contribute to implement it, open a PR or raise an issue at: https://github.com/kesslerlib/dSGP4."
        whichconst=dsgp4.util.get_gravity_constants("wgs-72")
        tles=kessler.tle.load(file_name="tests/TLEs_catalog_tests.txt")
        tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2=whichconst
        satrec_tumin, satrec_mu, satrec_radiusearthkm, satrec_xke, satrec_j2, satrec_j3, satrec_j4, satrec_j3oj2=sgp4.earth_gravity.wgs72
        self.assertAlmostEqual(satrec_tumin,float(tumin))
        self.assertAlmostEqual(satrec_mu,float(mu))
        self.assertAlmostEqual(satrec_radiusearthkm,float(radiusearthkm))
        self.assertAlmostEqual(satrec_xke,float(xke))
        self.assertAlmostEqual(satrec_j2,float(j2))
        self.assertAlmostEqual(satrec_j3,float(j3))
        self.assertAlmostEqual(satrec_j4,float(j4))
        self.assertAlmostEqual(satrec_j3oj2,float(j3oj2))
        for tle_sat in tles:
            satrec=sgp4.io.twoline2rv(tle_sat.line1,tle_sat.line2, whichconst=sgp4.earth_gravity.wgs72)
            (
            satrec_no_unkozai,
            satrec_method,
            satrec_ainv,  satrec_ao,    satrec_con41,  satrec_con42, satrec_cosio,
            satrec_cosio2, satrec_eccsq, satrec_omeosq, satrec_posq,
            satrec_rp,    satrec_rteosq, satrec_sinio , satrec_gsto,
            ) = sgp4.propagation._initl(
               float(xke), float(j2), satrec.ecco, (satrec.jdsatepoch+satrec.jdsatepochF)-2433281.5, satrec.inclo, satrec.no_kozai, satrec.method,
               satrec.operationmode
             )
            try:

                (
                tle_sat_no_unkozai,
                tle_sat_method,
                tle_sat_ainv,  tle_sat_ao,    tle_sat_con41,  tle_sat_con42, tle_sat_cosio,
                tle_sat_cosio2, tle_sat_eccsq, tle_sat_omeosq, tle_sat_posq,
                tle_sat_rp,    tle_sat_rteosq, tle_sat_sinio , tle_sat_gsto,
                ) = dsgp4.initl(
                   xke, j2, tle_sat._ecco, (tle_sat._jdsatepoch+tle_sat._jdsatepochF)-2433281.5, tle_sat._inclo, tle_sat._no_kozai, 'n',
                   'i'
                 );
                self.assertAlmostEqual(satrec_no_unkozai, float(tle_sat_no_unkozai))
                self.assertTrue(satrec_method==tle_sat_method)
                self.assertAlmostEqual(satrec_ainv, float(tle_sat_ainv))
                self.assertAlmostEqual(satrec_ao, float(tle_sat_ao))
                self.assertAlmostEqual(satrec_con41, float(tle_sat_con41))
                self.assertAlmostEqual(satrec_con42, float(tle_sat_con42))
                self.assertAlmostEqual(satrec_cosio, float(tle_sat_cosio))
                self.assertAlmostEqual(satrec_cosio2, float(tle_sat_cosio2))
                self.assertAlmostEqual(satrec_eccsq, float(tle_sat_eccsq))
                self.assertAlmostEqual(satrec_omeosq, float(tle_sat_omeosq))
                self.assertAlmostEqual(satrec_posq, float(tle_sat_posq))
                self.assertAlmostEqual(satrec_rp, float(tle_sat_rp))
                self.assertAlmostEqual(satrec_rteosq, float(tle_sat_rteosq))
                self.assertAlmostEqual(satrec_sinio, float(tle_sat_sinio))
                self.assertAlmostEqual(satrec_gsto, float(tle_sat_gsto))
            except Exception as e:
                self.assertTrue((str(e).split()==error_string.split()))
