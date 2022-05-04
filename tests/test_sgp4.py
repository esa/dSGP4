import dsgp4
import kessler
import numpy as np
import random
import sgp4
import sgp4.io
import torch
import unittest

class UtilTestCase(unittest.TestCase):
    def test_sgp4(self):
        error_string="Error: deep space propagation not supported (yet). The provided satellite has\
        an orbital period above 225 minutes. If you want to let us know you need it or you want to \
        contribute to implement it, open a PR or raise an issue at: https://github.com/kesslerlib/dSGP4."
        whichconst=dsgp4.util.get_gravity_constants("wgs-72")
        tles=kessler.tle.load(file_name="tests/TLEs_catalog_tests.txt")
        for tle_sat in tles:
            satrec=sgp4.io.twoline2rv(tle_sat.line1,tle_sat.line2, whichconst=sgp4.earth_gravity.wgs72)
            sgp4.propagation.sgp4init(whichconst=sgp4.earth_gravity.wgs72,
                                opsmode='i',
                                satn=tles[0].satellite_catalog_number,
                                epoch=(satrec.jdsatepoch+satrec.jdsatepochF)-2433281.5,
                                xbstar=satrec.bstar,
                                xndot=satrec.ndot,
                                xnddot=satrec.nddot,
                                xecco=satrec.ecco,
                                xargpo=satrec.argpo,
                                xinclo=satrec.inclo,
                                xmo=satrec.mo,
                                xno_kozai=satrec.no_kozai,
                                xnodeo=satrec.nodeo,
                                satrec=satrec)
            try:
                whichconst=dsgp4.util.get_gravity_constants("wgs-72")
                dsgp4.sgp4init(whichconst=whichconst,
                                    opsmode=tle_sat._opsmode,
                                    satn=tle_sat.satellite_catalog_number,
                                    epoch=(tle_sat._jdsatepoch+tle_sat._jdsatepochF)-2433281.5,
                                    xbstar=tle_sat._bstar,
                                    xndot=tle_sat._ndot,
                                    xnddot=tle_sat._nddot,
                                    xecco=tle_sat._ecco,
                                    xargpo=tle_sat._argpo,
                                    xinclo=tle_sat._inclo,
                                    xmo=tle_sat._mo,
                                    xno_kozai=tle_sat._no_kozai,
                                    xnodeo=tle_sat._nodeo,
                                    satrec=tle_sat)
                for _ in range(50):
                    tsince=random.uniform(-100.,100.)
                    satrec_state=sgp4.propagation.sgp4(satrec, tsince)
                    tle_sat_state=dsgp4.sgp4(tle_sat, tsince*torch.ones(1,1))
                    self.assertAlmostEqual(satrec_state[0][0], float(tle_sat_state[0][0][0]))
                    self.assertAlmostEqual(satrec_state[0][1], float(tle_sat_state[0][0][1]))
                    self.assertAlmostEqual(satrec_state[0][2], float(tle_sat_state[0][0][2]))
                    self.assertAlmostEqual(satrec_state[1][0], float(tle_sat_state[0][1][0]))
                    self.assertAlmostEqual(satrec_state[1][1], float(tle_sat_state[0][1][1]))
                    self.assertAlmostEqual(satrec_state[1][2], float(tle_sat_state[0][1][2]))
            except Exception as e:
                self.assertTrue((str(e).split()==error_string.split()))
