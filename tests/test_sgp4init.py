import kessler
import unittest
import dsgp4
import sgp4
import sgp4.io
import numpy as np

class UtilTestCase(unittest.TestCase):
    def test_sgp4init(self):
        whichconst=dsgp4.util.get_gravity_constants("wgs-72")
        import os
        tles=kessler.tle.load(file_name="tests/TLEs_catalog_tests.txt")
        for tle_sat in tles:
            satrec=sgp4.io.twoline2rv(tle_sat.line1,tle_sat.line2, whichconst=sgp4.earth_gravity.wgs72)
            sgp4.propagation.sgp4init(whichconst=sgp4.earth_gravity.wgs72,
                                opsmode='i',
                                satn=satrec.satnum,
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
            #I need a try excpet: to exclude the deep space cases
            error_string="Error: deep space propagation not supported (yet). The provided satellite has\
            an orbital period above 225 minutes. If you want to let us know you need it or you want to \
            contribute to implement it, open a PR or raise an issue at: https://github.com/kesslerlib/dSGP4."
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
                self.assertAlmostEqual(satrec.isimp,float(tle_sat._isimp))
                self.assertTrue(satrec.method==tle_sat._method)
                self.assertAlmostEqual(satrec.aycof , float(tle_sat._aycof))
                self.assertAlmostEqual(satrec.con41 , float(tle_sat._con41))
                self.assertAlmostEqual(satrec.cc1   , float(tle_sat._cc1))
                self.assertAlmostEqual(satrec.cc4   , float(tle_sat._cc4))
                self.assertAlmostEqual(satrec.cc5   , float(tle_sat._cc5))
                self.assertAlmostEqual(satrec.d2    , float(tle_sat._d2))
                self.assertAlmostEqual(satrec.d3    , float(tle_sat._d3))
                self.assertAlmostEqual(satrec.d4    , float(tle_sat._d4))
                self.assertAlmostEqual(satrec.delmo , float(tle_sat._delmo))
                self.assertAlmostEqual(satrec.eta   , float(tle_sat._eta))
                self.assertAlmostEqual(satrec.argpdot,   float(tle_sat._argpdot))
                self.assertAlmostEqual(satrec.omgcof, float(tle_sat._omgcof))
                self.assertAlmostEqual(satrec.sinmao, float(tle_sat._sinmao))
                self.assertAlmostEqual(satrec.t     , float(tle_sat._t))
                self.assertAlmostEqual(satrec.t2cof , float(tle_sat._t2cof))
                self.assertAlmostEqual(satrec.t3cof , float(tle_sat._t3cof))
                self.assertAlmostEqual(satrec.t4cof , float(tle_sat._t4cof))
                self.assertAlmostEqual(satrec.t5cof  , float(tle_sat._t5cof))
                self.assertAlmostEqual(satrec.x1mth2,  float(tle_sat._x1mth2))
                self.assertAlmostEqual(satrec.x7thm1  , float(tle_sat._x7thm1))
                self.assertAlmostEqual(satrec.mdot , float(tle_sat._mdot))
                self.assertAlmostEqual(satrec.nodedot , float(tle_sat._nodedot))
                self.assertAlmostEqual(satrec.xlcof , float(tle_sat._xlcof))
                self.assertAlmostEqual(satrec.xmcof  , float(tle_sat._xmcof))
                self.assertAlmostEqual(satrec.nodecf, float(tle_sat._nodecf))
                self.assertAlmostEqual(satrec.tumin, float(tle_sat._tumin))
                self.assertAlmostEqual(satrec.mu, float(tle_sat._mu))
                self.assertAlmostEqual(satrec.radiusearthkm, float(tle_sat._radiusearthkm))
                self.assertAlmostEqual(satrec.xke, float(tle_sat._xke))
                self.assertAlmostEqual(satrec.j2, float(tle_sat._j2))
                self.assertAlmostEqual(satrec.j3, float(tle_sat._j3))
                self.assertAlmostEqual(satrec.j4, float(tle_sat._j4))
                self.assertAlmostEqual(satrec.j3oj2, float(tle_sat._j3oj2))
                self.assertTrue(satrec.error==int(tle_sat._error))
                self.assertTrue(satrec.operationmode==tle_sat._operationmode)
                self.assertAlmostEqual(satrec.satnum, float(tle_sat._satnum))
                self.assertAlmostEqual(satrec.bstar, float(tle_sat._bstar))
                self.assertAlmostEqual(satrec.ndot , float(tle_sat._ndot))
                self.assertAlmostEqual(satrec.nddot, float(tle_sat._nddot))
                self.assertAlmostEqual(satrec.ecco , float(tle_sat._ecco))
                self.assertAlmostEqual(satrec.argpo, float(tle_sat._argpo))
                self.assertAlmostEqual(satrec.inclo, float(tle_sat._inclo))
                self.assertAlmostEqual(satrec.mo , float(tle_sat._mo))
                self.assertAlmostEqual(satrec.no_kozai, float(tle_sat._no_kozai))
                self.assertAlmostEqual(satrec.nodeo , float(tle_sat._nodeo))
                self.assertAlmostEqual(satrec.am, float(tle_sat._am))
                self.assertAlmostEqual(satrec.em, float(tle_sat._em))
                self.assertAlmostEqual(satrec.im, float(tle_sat._im))
                self.assertAlmostEqual(satrec.Om, float(tle_sat._Om))
                self.assertAlmostEqual(satrec.mm, float(tle_sat._mm))
                self.assertAlmostEqual(satrec.nm, float(tle_sat._nm))
                self.assertTrue(satrec.init==tle_sat._init)
                self.assertAlmostEqual(satrec.t,tle_sat._t[0][0])
                self.assertAlmostEqual(satrec.no_unkozai, float(tle_sat._no_unkozai))
                self.assertAlmostEqual(satrec.gsto, float(tle_sat._gsto))
                self.assertAlmostEqual(satrec.a, float(tle_sat._a))
                self.assertAlmostEqual(satrec.alta, float(tle_sat._alta))
                self.assertAlmostEqual(satrec.altp, float(tle_sat._altp))
                self.assertAlmostEqual(satrec.eta, float(tle_sat._eta))
                self.assertTrue(satrec.init==tle_sat._init)
            except Exception as e:
                self.assertTrue((str(e).split()==error_string.split()))
