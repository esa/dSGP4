import numpy
import torch
from .initl import initl
from .sgp4 import sgp4
torch.set_default_dtype(torch.float64)

def sgp4init(
       whichconst,   opsmode,  satn, epoch,
       xbstar,   xndot,   xnddot,   xecco,   xargpo,
       xinclo,   xmo,   xno_kozai,
       xnodeo,  satrec,
       ):
     """
     This function initializes the sgp4 propagator.
     Args:
        - whichconst (``tuple``): this contains all the necessary constants (tumin, mu, radiusearthkm, xke,
                        j2, j3, j4, j3oj2)), according to the chosen gravitational model (wgs-72, wgs-84,
                        wgs-72old are three possible choices)
        - opsmode (``str``): mode of operation (possibilities: afspc or improved, 'a' and 'i', respectively)
        - satn (``str``): satellite catalog number
        - epoch (``torch.float``): TLE days since 1949 December 31 00:00 UT
        - xbstar (``torch.float``): drag coefficient (in 1/earth_radii)
        - xndot (``torch.float``): first derivative of mean motion (in revs/day**2)
        - xnddot (``torch.float``): second derivative mean motion (in revs/day**3)
        - xecco (``torch.float``): eccentricity
        - xargpo (``torch.float``): argument of perigee [rad]
        - xinclo (``torch.float``): inclination [rad]
        - xmo (``torch.float``): mean anomaly [rad]
        - xno_kozai (``torch.float``): mean motion (radians/minute)
        - xnodeo (``torch.float``): right ascension of the ascending node [rad]
        - satrec (``kessler.tle.TLE``): TLE object

    ..Note:
        If satrec._error is different than 0 once this routine is called, then the satellite propagation
        has had an error. In particular, the following error values hold for `satrec._error`:
        * `1` -> eccentricity `>=1.` or `<-0.001` or semi-major axis (in Earth radii) `<0.95`
        * `2` -> mean_motion `<0.`
        * `3` -> eccentricity `<0.` or `>1.`
        * `4` -> semi-latus rectum `<0.`
        * `5` -> epoch elements are sub-orbital
        * `6` -> satellite has decayed
     """
     temp4    =   torch.tensor(1.5e-12);

     #  ----------- set all near earth variables to zero ------------
     satrec._isimp   = torch.tensor(0);   satrec._method = 'n';                 satrec._aycof    = torch.tensor(0.0);
     satrec._con41   = torch.tensor(0.0); satrec._cc1    = torch.tensor(0.0); satrec._cc4      = torch.tensor(0.0);
     satrec._cc5     = torch.tensor(0.0); satrec._d2     = torch.tensor(0.0); satrec._d3       = torch.tensor(0.0);
     satrec._d4      = torch.tensor(0.0); satrec._delmo  = torch.tensor(0.0); satrec._eta      = torch.tensor(0.0);
     satrec._argpdot = torch.tensor(0.0); satrec._omgcof = torch.tensor(0.0); satrec._sinmao   = torch.tensor(0.0);
     satrec._t       = torch.tensor(0.0); satrec._t2cof  = torch.tensor(0.0); satrec._t3cof    = torch.tensor(0.0);
     satrec._t4cof   = torch.tensor(0.0); satrec._t5cof  = torch.tensor(0.0); satrec._x1mth2   = torch.tensor(0.0);
     satrec._x7thm1  = torch.tensor(0.0); satrec._mdot   = torch.tensor(0.0); satrec._nodedot  = torch.tensor(0.0);
     satrec._xlcof   = torch.tensor(0.0); satrec._xmcof  = torch.tensor(0.0); satrec._nodecf   = torch.tensor(0.0);

     #  ------------------------ earth constants -----------------------
     #  sgp4fix identify constants and allow alternate values
     #  this is now the only call for the constants
     (satrec._tumin, satrec._mu, satrec._radiusearthkm, satrec._xke,
       satrec._j2, satrec._j3, satrec._j4, satrec._j3oj2) = whichconst;

 	 # -------------------------------------------------------------------------

     satrec._error = torch.tensor(0);
     satrec._operationmode = opsmode;
     satrec._satnum = satn;

     satrec._bstar   = xbstar.clone();
    # sgp4fix allow additional parameters in the struct
     satrec._ndot    = xndot.clone();
     satrec._nddot   = xnddot.clone();
     satrec._ecco    = xecco.clone();
     satrec._argpo   = xargpo.clone();
     satrec._inclo   = xinclo.clone();
     satrec._mo	    = xmo.clone();
	# sgp4fix rename variables to clarify which mean motion is intended
     satrec._no_kozai= xno_kozai.clone();
     satrec._nodeo   = xnodeo.clone();

    # single averaged mean elements
     satrec._am = torch.tensor(0.0)
     satrec._em = torch.tensor(0.0)
     satrec._im = torch.tensor(0.0)
     satrec._Om = torch.tensor(0.0)
     satrec._mm = torch.tensor(0.0)
     satrec._nm = torch.tensor(0.0)

     ss     = 78.0 / satrec._radiusearthkm + 1.0;

     qzms2ttemp = (120.0 - 78.0) / satrec._radiusearthkm;
     qzms2t = qzms2ttemp * qzms2ttemp * qzms2ttemp * qzms2ttemp;
     x2o3   =  2.0 / 3.0;

     satrec._init = 'y';
     satrec._t	 = 0.0;

    # sgp4fix remove satn as it is not needed in initl
     (
       satrec._no_unkozai,
       method,
       ainv,  ao,    satrec._con41,  con42, cosio,
       cosio2,eccsq, omeosq, posq,
       rp,    rteosq,sinio , satrec._gsto,
       ) = initl(
           satrec._xke, satrec._j2, satrec._ecco, epoch, satrec._inclo, satrec._no_kozai, satrec._method,
           satrec._operationmode
         );
     satrec._a    = torch.pow( satrec._no_unkozai*satrec._tumin , (-2.0/3.0) );
     satrec._alta = satrec._a*(1.0 + satrec._ecco) - 1.0;
     satrec._altp = satrec._a*(1.0 - satrec._ecco) - 1.0;

     if omeosq >= 0.0 or satrec._no_unkozai >= 0.0:

         satrec._isimp = 0;
         if rp < 220.0 / satrec._radiusearthkm + 1.0:
             satrec._isimp = 1;
         sfour  = ss;
         qzms24 = qzms2t;
         perige = (rp - 1.0) * satrec._radiusearthkm;

         if perige < 156.0:
             sfour = perige - 78.0;
             if perige < 98.0:
                 sfour = 20.0;
             qzms24temp =  (120.0 - sfour) / satrec._radiusearthkm;
             qzms24 = qzms24temp * qzms24temp * qzms24temp * qzms24temp;
             sfour  = sfour / satrec._radiusearthkm + 1.0;

         pinvsq = 1.0 / posq;

         tsi  = 1.0 / (ao - sfour);
         satrec._eta  = ao * satrec._ecco * tsi;
         etasq = satrec._eta * satrec._eta;
         eeta  = satrec._ecco * satrec._eta;
         psisq = torch.abs(1.0 - etasq);
         coef  = qzms24 * torch.pow(tsi, 4.0);
         coef1 = coef / torch.pow(psisq, 3.5);
         cc2   = coef1 * satrec._no_unkozai * (ao * (1.0 + 1.5 * etasq + eeta *
                        (4.0 + etasq)) + 0.375 * satrec._j2 * tsi / psisq * satrec._con41 *
                        (8.0 + 3.0 * etasq * (8.0 + etasq)));
         satrec._cc1   = satrec._bstar * cc2;
         cc3   = 0.0;
         if satrec._ecco > 1.0e-4:
             cc3 = -2.0 * coef * tsi * satrec._j3oj2 * satrec._no_unkozai * sinio / satrec._ecco;
         satrec._x1mth2 = 1.0 - cosio2;
         satrec._cc4    = 2.0* satrec._no_unkozai * coef1 * ao * omeosq * \
                           (satrec._eta * (2.0 + 0.5 * etasq) + satrec._ecco *
                           (0.5 + 2.0 * etasq) - satrec._j2 * tsi / (ao * psisq) *
                           (-3.0 * satrec._con41 * (1.0 - 2.0 * eeta + etasq *
                           (1.5 - 0.5 * eeta)) + 0.75 * satrec._x1mth2 *
                           (2.0 * etasq - eeta * (1.0 + etasq)) * (2.0 * satrec._argpo).cos()));
         satrec._cc5 = 2.0 * coef1 * ao * omeosq * (1.0 + 2.75 *
                        (etasq + eeta) + eeta * etasq);
         cosio4 = cosio2 * cosio2;
         temp1  = 1.5 * satrec._j2 * pinvsq * satrec._no_unkozai;
         temp2  = 0.5 * temp1 * satrec._j2 * pinvsq;
         temp3  = -0.46875 * satrec._j4 * pinvsq * pinvsq * satrec._no_unkozai;
         satrec._mdot     = satrec._no_unkozai + 0.5 * temp1 * rteosq * satrec._con41 + 0.0625 * \
                            temp2 * rteosq * (13.0 - 78.0 * cosio2 + 137.0 * cosio4);
         satrec._argpdot  = (-0.5 * temp1 * con42 + 0.0625 * temp2 *
                             (7.0 - 114.0 * cosio2 + 395.0 * cosio4) +
                             temp3 * (3.0 - 36.0 * cosio2 + 49.0 * cosio4));
         xhdot1            = -temp1 * cosio;
         satrec._nodedot = xhdot1 + (0.5 * temp2 * (4.0 - 19.0 * cosio2) +
                              2.0 * temp3 * (3.0 - 7.0 * cosio2)) * cosio;
         xpidot            =  satrec._argpdot+ satrec._nodedot;
         satrec._omgcof   = satrec._bstar * cc3 * satrec._argpo.cos();
         if satrec._ecco > 1.0e-4:
             satrec._xmcof = -x2o3 * coef * satrec._bstar / eeta;
         satrec._nodecf = 3.5 * omeosq * xhdot1 * satrec._cc1;
         satrec._t2cof   = 1.5 * satrec._cc1;
         if torch.abs(cosio+1.0) > 1.5e-12:
             satrec._xlcof = -0.25 * satrec._j3oj2 * sinio * (3.0 + 5.0 * cosio) / (1.0 + cosio);
         else:
             satrec._xlcof = -0.25 * satrec._j3oj2 * sinio * (3.0 + 5.0 * cosio) / temp4;
         satrec._aycof   = -0.5 * satrec._j3oj2 * sinio;
         delmotemp = 1.0 + satrec._eta * satrec._mo.cos();
         satrec._delmo   = delmotemp * delmotemp * delmotemp;
         satrec._sinmao  = satrec._mo.sin();
         satrec._x7thm1  = 7.0 * cosio2 - 1.0;

         #  --------------- deep space initialization -------------
         if 2*numpy.pi / satrec._no_unkozai >= 225.0:
             raise RuntimeError("Error: deep space propagation not supported (yet). The provided satellite has\
             an orbital period above 225 minutes. If you want to let us know you need it or you want to \
             contribute to implement it, open a PR or raise an issue at: https://github.com/kesslerlib/dSGP4.")

         if satrec._isimp != 1:
           cc1sq          = satrec._cc1 * satrec._cc1;
           satrec._d2    = 4.0 * ao * tsi * cc1sq;
           temp           = satrec._d2 * tsi * satrec._cc1 / 3.0;
           satrec._d3    = (17.0 * ao + sfour) * temp;
           satrec._d4    = 0.5 * temp * ao * tsi * (221.0 * ao + 31.0 * sfour) * \
                            satrec._cc1;
           satrec._t3cof = satrec._d2 + 2.0 * cc1sq;
           satrec._t4cof = 0.25 * (3.0 * satrec._d3 + satrec._cc1 *
                            (12.0 * satrec._d2 + 10.0 * cc1sq));
           satrec._t5cof = 0.2 * (3.0 * satrec._d4 +
                            12.0 * satrec._cc1 * satrec._d3 +
                            6.0 * satrec._d2 * satrec._d2 +
                            15.0 * cc1sq * (2.0 * satrec._d2 + cc1sq));
     sgp4(satrec, torch.zeros(1,1));

     satrec._init = 'n';
