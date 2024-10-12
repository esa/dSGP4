import numpy
import torch
from .initl import initl
from .sgp4 import sgp4

def sgp4init(
      whichconst,   opsmode,  satn, epoch,
      xbstar,   xndot,   xnddot,   xecco,   xargpo,
      xinclo,   xmo,   xno_kozai,
      xnodeo,  satellite,
      ):
    """
    This function initializes the sgp4 propagator.

    Parameters:
    ----------------
    whichconst (``tuple``): this contains all the necessary constants (tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2)), according to the chosen gravitational model (wgs-72, wgs-84, wgs-72old are three possible choices
    opsmode (``str``): mode of operation (possibilities: afspc or improved, 'a' and 'i', respectively
    satn (``str``): satellite catalog numbe
    epoch (``torch.float``): TLE days since 1949 December 31 00:00 U
    xbstar (``torch.float``): drag coefficient (in 1/earth_radii
    xndot (``torch.float``): first derivative of mean motion (in revs/day**2
    xnddot (``torch.float``): second derivative mean motion (in revs/day**3
    xecco (``torch.float``): eccentricit
    xargpo (``torch.float``): argument of perigee [rad
    xinclo (``torch.float``): inclination [rad
    xmo (``torch.float``): mean anomaly [rad
    xno_kozai (``torch.float``): mean motion (radians/minute
    xnodeo (``torch.float``): right ascension of the ascending node [rad
    satellite (``dsgp4.tle.TLE``): TLE object

    ..Note:
    If satellite._error is different than 0 once this routine is called, then the satellite propagation
    has had an error. In particular, the following error values hold for `satellite._error`:
    * `1` -> eccentricity `>=1.` or `<-0.001` or semi-major axis (in Earth radii) `<0.95`
    * `2` -> mean_motion `<0.`
    * `3` -> eccentricity `<0.` or `>1.`
    * `4` -> semi-latus rectum `<0.`
    * `5` -> epoch elements are sub-orbital
    * `6` -> satellite has decayed
    """
    temp4    =   torch.tensor(1.5e-12)

    #  ----------- set all near earth variables to zero ------------
    satellite._isimp   = torch.tensor(0);   satellite._method = 'n';               satellite._aycof    = torch.tensor(0.0);
    satellite._con41   = torch.tensor(0.0); satellite._cc1    = torch.tensor(0.0); satellite._cc4      = torch.tensor(0.0);
    satellite._cc5     = torch.tensor(0.0); satellite._d2     = torch.tensor(0.0); satellite._d3       = torch.tensor(0.0);
    satellite._d4      = torch.tensor(0.0); satellite._delmo  = torch.tensor(0.0); satellite._eta      = torch.tensor(0.0);
    satellite._argpdot = torch.tensor(0.0); satellite._omgcof = torch.tensor(0.0); satellite._sinmao   = torch.tensor(0.0);
    satellite._t       = torch.tensor(0.0); satellite._t2cof  = torch.tensor(0.0); satellite._t3cof    = torch.tensor(0.0);
    satellite._t4cof   = torch.tensor(0.0); satellite._t5cof  = torch.tensor(0.0); satellite._x1mth2   = torch.tensor(0.0);
    satellite._x7thm1  = torch.tensor(0.0); satellite._mdot   = torch.tensor(0.0); satellite._nodedot  = torch.tensor(0.0);
    satellite._xlcof   = torch.tensor(0.0); satellite._xmcof  = torch.tensor(0.0); satellite._nodecf   = torch.tensor(0.0);

    #  ------------------------ earth constants -----------------------
    #  sgp4fix identify constants and allow alternate values
    #  this is now the only call for the constants
    (satellite._tumin, satellite._mu, satellite._radiusearthkm, satellite._xke,
    satellite._j2, satellite._j3, satellite._j4, satellite._j3oj2) = whichconst

    # -------------------------------------------------------------------------

    satellite._error = torch.tensor(0)
    satellite._operationmode = opsmode
    satellite._satnum = satn

    satellite._bstar   = xbstar.clone()
    # sgp4fix allow additional parameters in the struct
    satellite._ndot    = xndot.clone()
    satellite._nddot   = xnddot.clone()
    satellite._ecco    = xecco.clone()
    satellite._argpo   = xargpo.clone()
    satellite._inclo   = xinclo.clone()
    satellite._mo	    = xmo.clone()
    # sgp4fix rename variables to clarify which mean motion is intended
    satellite._no_kozai= xno_kozai.clone()
    satellite._nodeo   = xnodeo.clone()

    # single averaged mean elements
    satellite._am = torch.tensor(0.0)
    satellite._em = torch.tensor(0.0)
    satellite._im = torch.tensor(0.0)
    satellite._Om = torch.tensor(0.0)
    satellite._mm = torch.tensor(0.0)
    satellite._nm = torch.tensor(0.0)

    ss     = 78.0 / satellite._radiusearthkm + 1.0

    qzms2ttemp = (120.0 - 78.0) / satellite._radiusearthkm
    qzms2t = qzms2ttemp * qzms2ttemp * qzms2ttemp * qzms2ttemp
    x2o3   =  2.0 / 3.0

    satellite._init = 'y'
    satellite._t	 = 0.0

    # sgp4fix remove satn as it is not needed in initl
    (
    satellite._no_unkozai,
    method,
    ainv,  ao,    satellite._con41,  con42, cosio,
    cosio2,eccsq, omeosq, posq,
    rp,    rteosq,sinio , satellite._gsto,
    ) = initl(
        satellite._xke, satellite._j2, satellite._ecco, epoch, satellite._inclo, satellite._no_kozai,
        satellite._operationmode, satellite._method
        )
    satellite._a    = torch.pow( satellite._no_unkozai*satellite._tumin , (-2.0/3.0) )
    satellite._alta = satellite._a*(1.0 + satellite._ecco) - 1.0
    satellite._altp = satellite._a*(1.0 - satellite._ecco) - 1.0

    if omeosq >= 0.0 or satellite._no_unkozai >= 0.0:

        satellite._isimp = 0
        if rp < 220.0 / satellite._radiusearthkm + 1.0:
            satellite._isimp = 1
        sfour  = ss
        qzms24 = qzms2t
        perige = (rp - 1.0) * satellite._radiusearthkm

        if perige < 156.0:
            sfour = perige - 78.0
            if perige < 98.0:
                sfour = 20.0
            qzms24temp =  (120.0 - sfour) / satellite._radiusearthkm
            qzms24 = qzms24temp * qzms24temp * qzms24temp * qzms24temp
            sfour  = sfour / satellite._radiusearthkm + 1.0

        pinvsq = 1.0 / posq

        tsi  = 1.0 / (ao - sfour)
        satellite._eta  = ao * satellite._ecco * tsi
        etasq = satellite._eta * satellite._eta
        eeta  = satellite._ecco * satellite._eta
        psisq = torch.abs(1.0 - etasq)
        coef  = qzms24 * torch.pow(tsi, 4.0)
        coef1 = coef / torch.pow(psisq, 3.5)
        cc2   = coef1 * satellite._no_unkozai * (ao * (1.0 + 1.5 * etasq + eeta *
                    (4.0 + etasq)) + 0.375 * satellite._j2 * tsi / psisq * satellite._con41 *
                    (8.0 + 3.0 * etasq * (8.0 + etasq)))
        satellite._cc1   = satellite._bstar * cc2
        cc3   = 0.0
        if satellite._ecco > 1.0e-4:
            cc3 = -2.0 * coef * tsi * satellite._j3oj2 * satellite._no_unkozai * sinio / satellite._ecco
        satellite._x1mth2 = 1.0 - cosio2
        satellite._cc4    = 2.0* satellite._no_unkozai * coef1 * ao * omeosq * \
                        (satellite._eta * (2.0 + 0.5 * etasq) + satellite._ecco *
                        (0.5 + 2.0 * etasq) - satellite._j2 * tsi / (ao * psisq) *
                        (-3.0 * satellite._con41 * (1.0 - 2.0 * eeta + etasq *
                        (1.5 - 0.5 * eeta)) + 0.75 * satellite._x1mth2 *
                        (2.0 * etasq - eeta * (1.0 + etasq)) * (2.0 * satellite._argpo).cos()))
        satellite._cc5 = 2.0 * coef1 * ao * omeosq * (1.0 + 2.75 *
                    (etasq + eeta) + eeta * etasq)
        cosio4 = cosio2 * cosio2
        temp1  = 1.5 * satellite._j2 * pinvsq * satellite._no_unkozai
        temp2  = 0.5 * temp1 * satellite._j2 * pinvsq
        temp3  = -0.46875 * satellite._j4 * pinvsq * pinvsq * satellite._no_unkozai
        satellite._mdot     = satellite._no_unkozai + 0.5 * temp1 * rteosq * satellite._con41 + 0.0625 * \
                        temp2 * rteosq * (13.0 - 78.0 * cosio2 + 137.0 * cosio4)
        satellite._argpdot  = (-0.5 * temp1 * con42 + 0.0625 * temp2 *
                            (7.0 - 114.0 * cosio2 + 395.0 * cosio4) +
                            temp3 * (3.0 - 36.0 * cosio2 + 49.0 * cosio4))
        xhdot1            = -temp1 * cosio
        satellite._nodedot = xhdot1 + (0.5 * temp2 * (4.0 - 19.0 * cosio2) +
                            2.0 * temp3 * (3.0 - 7.0 * cosio2)) * cosio
        xpidot            =  satellite._argpdot+ satellite._nodedot
        satellite._omgcof   = satellite._bstar * cc3 * satellite._argpo.cos()
        if satellite._ecco > 1.0e-4:
            satellite._xmcof = -x2o3 * coef * satellite._bstar / eeta
        satellite._nodecf = 3.5 * omeosq * xhdot1 * satellite._cc1
        satellite._t2cof   = 1.5 * satellite._cc1
        if torch.abs(cosio+1.0) > 1.5e-12:
            satellite._xlcof = -0.25 * satellite._j3oj2 * sinio * (3.0 + 5.0 * cosio) / (1.0 + cosio)
        else:
            satellite._xlcof = -0.25 * satellite._j3oj2 * sinio * (3.0 + 5.0 * cosio) / temp4
        satellite._aycof   = -0.5 * satellite._j3oj2 * sinio
        delmotemp = 1.0 + satellite._eta * satellite._mo.cos()
        satellite._delmo   = delmotemp * delmotemp * delmotemp
        satellite._sinmao  = satellite._mo.sin()
        satellite._x7thm1  = 7.0 * cosio2 - 1.0

        #  --------------- deep space initialization -------------
        if 2*numpy.pi / satellite._no_unkozai >= 225.0:
            raise RuntimeError("Error: deep space propagation not supported (yet). The provided satellite has\
            an orbital period above 225 minutes. If you want to let us know you need it or you want to \
            contribute to implement it, open a PR or raise an issue at: https://github.com/esa/dSGP4.")

        if satellite._isimp != 1:
            cc1sq          = satellite._cc1 * satellite._cc1
            satellite._d2    = 4.0 * ao * tsi * cc1sq
            temp           = satellite._d2 * tsi * satellite._cc1 / 3.0
            satellite._d3    = (17.0 * ao + sfour) * temp
            satellite._d4    = 0.5 * temp * ao * tsi * (221.0 * ao + 31.0 * sfour) * \
                            satellite._cc1
            satellite._t3cof = satellite._d2 + 2.0 * cc1sq
            satellite._t4cof = 0.25 * (3.0 * satellite._d3 + satellite._cc1 *
                            (12.0 * satellite._d2 + 10.0 * cc1sq))
            satellite._t5cof = 0.2 * (3.0 * satellite._d4 +
                            12.0 * satellite._cc1 * satellite._d3 +
                            6.0 * satellite._d2 * satellite._d2 +
                            15.0 * cc1sq * (2.0 * satellite._d2 + cc1sq))
    sgp4(satellite, torch.zeros(1,1));

    satellite._init = 'n'