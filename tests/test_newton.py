import datetime
import dsgp4
import numpy as np
import random
import torch
import unittest

class UtilTestCase(unittest.TestCase):
    def test_newton_method_1(self):
        lines=[]
        lines.append('1 43437U 18100A   20143.90384230  .00041418  00000-0  10000-3 0 99968')
        lines.append('2 43437  97.8268 249.9127 0221000 123.9136 259.1144 15.12608579563539')
        my_tle=dsgp4.tle.TLE(lines)
        whichconst=dsgp4.util.get_gravity_constants("wgs-72")
        dsgp4.sgp4init(whichconst=whichconst,
                opsmode=my_tle._opsmode,
                satn=my_tle.satellite_catalog_number,
                epoch=(my_tle._jdsatepoch+my_tle._jdsatepochF)-2433281.5,
                xbstar=my_tle._bstar,
                xndot=my_tle._ndot,
                xnddot=my_tle._nddot,
                xecco=my_tle._ecco,
                xargpo=my_tle._argpo,
                xinclo=my_tle._inclo,
                xmo=my_tle._mo,
                xno_kozai=my_tle._no_kozai,
                xnodeo=my_tle._nodeo,
                satellite=my_tle)

        found_tle,_=dsgp4.newton_method(tle_0=my_tle, time_mjd=dsgp4.util.from_datetime_to_mjd(my_tle._epoch))
        self.assertEqual(my_tle.line1, found_tle.line1)
        self.assertEqual(my_tle.line2, found_tle.line2)
        #I now propagate for 1000 minutes and make sure that the found TLE w Newton
        #actually satisfies the tolerance
        tsince_2=1000.
        new_tol=1e-13
        target_state=dsgp4.sgp4(my_tle,tsince_2*torch.ones(1,1,requires_grad=True))
        time_mjd_0=dsgp4.util.from_datetime_to_mjd(my_tle._epoch)
        time_mjd=time_mjd_0+tsince_2/24/60
        found_tle_2,_=dsgp4.newton_method(tle_0=my_tle, time_mjd=time_mjd, new_tol=new_tol, target_state=target_state)
        found_state=dsgp4.sgp4(found_tle_2,torch.tensor((time_mjd-dsgp4.util.from_datetime_to_mjd(found_tle_2._epoch))*1440.))#(dsgp4.util.from_datetime_to_mjd(found_tle_2._epoch)-dsgp4.util.from_datetime_to_mjd(new_date))*1440.*torch.ones(1,1,requires_grad=True))
        self.assertTrue(torch.norm(found_state-target_state)<new_tol)
