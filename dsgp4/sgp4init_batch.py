import numpy as np
import torch
from .sgp4_batched import sgp4_batched
from .util import gstime

def initl_batch(
      xke, j2,
      ecco,   epoch,  inclo,   no,
      opsmode, batch_size, method='n',
      ):

    x2o3   = torch.full((batch_size,),2.0 / 3.0);

    eccsq  = ecco * ecco;
    omeosq = 1.0 - eccsq;
    rteosq = omeosq.sqrt();
    cosio  = inclo.cos();
    cosio2 = cosio * cosio;

    ak    = torch.pow(xke / no, x2o3);
    d1    = 0.75 * j2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq);
    del_  = d1 / (ak * ak);
    adel  = ak * (1.0 - del_ * del_ - del_ *
            (1.0 / 3.0 + 134.0 * del_ * del_ / 81.0));
    del_  = d1/(adel * adel);
    no    = no / (1.0 + del_);

    ao    = torch.pow(xke / no, x2o3);
    sinio = inclo.sin();
    po    = ao * omeosq;
    con42 = 1.0 - 5.0 * cosio2;
    con41 = -con42-cosio2-cosio2;
    ainv  = 1.0 / ao;
    posq  = po * po;
    rp    = ao * (1.0 - ecco);
    method = [method]*batch_size;

    if opsmode == 'a':
        #  gst time
        ts70  = epoch - 7305.0;
        ds70 = torch.floor_divide(ts70 + 1.0e-8,1);
        tfrac = ts70 - ds70;
        #  find greenwich location at epoch
        c1    = torch.tensor(1.72027916940703639e-2);
        thgr70= torch.tensor(1.7321343856509374);
        fk5r  = torch.tensor(5.07551419432269442e-15);
        c1p2p = c1 + (2*numpy.pi);
        gsto  = (thgr70 + c1*ds70 + c1p2p*tfrac + ts70*ts70*fk5r) % (2*numpy.pi)
        if gsto < 0.0:
            gsto = gsto + (2*numpy.pi);

    else:
        gsto = gstime(epoch + 2433281.5);

    return (
      no,
      method,
      ainv,  ao,    con41,  con42, cosio,
      cosio2,eccsq, omeosq, posq,
      rp,    rteosq,sinio , gsto,
      )


def sgp4init_batch(
      whichconst,   opsmode,  satn, epoch,
      xbstar,   xndot,   xnddot,   xecco,   xargpo,
      xinclo,   xmo,   xno_kozai,
      xnodeo,  satellite_batch
      ):
    """
    This function initializes the sgp4 propagator.

    Parameters:
    ----------------
    whichconst (``tuple``): this contains all the necessary constants (tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2)), according to the chosen gravitational model (wgs-72, wgs-84, wgs-72old are three possible choices)
    opsmode (``str``): mode of operation (possibilities: afspc or improved, 'a' and 'i', respectively)
    satn (``str``): satellite catalog number
    epoch (``torch.float``): TLE days since 1949 December 31 00:00 UT
    xbstar (``torch.float``): drag coefficient (in 1/earth_radii)
    xndot (``torch.float``): first derivative of mean motion (in revs/day**2)
    xnddot (``torch.float``): second derivative mean motion (in revs/day**3)
    xecco (``torch.float``): eccentricity
    xargpo (``torch.float``): argument of perigee [rad]
    xinclo (``torch.float``): inclination [rad]
    xmo (``torch.float``): mean anomaly [rad]
    xno_kozai (``torch.float``): mean motion (radians/minute)
    xnodeo (``torch.float``): right ascension of the ascending node [rad]
    satellite_batch (``dsgp4.tle.TLE``): TLE object, that represents the batch of TLEs (each element is an N-dimensional tensor)

    ..Note:
        If satellite_batch._error is different than 0 once this routine is called, then the satellite propagation
        has had an error. In particular, the following error values hold for `satellite_batch._error`:
        * `1` -> eccentricity `>=1.` or `<-0.001` or semi-major axis (in Earth radii) `<0.95`
        * `2` -> mean_motion `<0.`
        * `3` -> eccentricity `<0.` or `>1.`
        * `4` -> semi-latus rectum `<0.`
        * `5` -> epoch elements are sub-orbital
        * `6` -> satellite has decayed
    """
    batch_size = len(xecco)
    temp4    =   torch.tensor(1.5e-12)

    #  ----------- set all near earth variables to zero ------------
    satellite_batch._isimp   = torch.full((batch_size,),0);   satellite_batch._method = ["n"]*batch_size;              satellite_batch._aycof    = torch.full((batch_size,),0.0);
    satellite_batch._con41   = torch.full((batch_size,),0.0); satellite_batch._cc1    = torch.full((batch_size,),0.0); satellite_batch._cc4      = torch.full((batch_size,),0.0);
    satellite_batch._cc5     = torch.full((batch_size,),0.0); satellite_batch._d2     = torch.full((batch_size,),0.0); satellite_batch._d3       = torch.full((batch_size,),0.0);
    satellite_batch._d4      = torch.full((batch_size,),0.0); satellite_batch._delmo  = torch.full((batch_size,),0.0); satellite_batch._eta      = torch.full((batch_size,),0.0);
    satellite_batch._argpdot = torch.full((batch_size,),0.0); satellite_batch._omgcof = torch.full((batch_size,),0.0); satellite_batch._sinmao   = torch.full((batch_size,),0.0);
    satellite_batch._t       = torch.full((batch_size,),0.0); satellite_batch._t2cof  = torch.full((batch_size,),0.0); satellite_batch._t3cof    = torch.full((batch_size,),0.0);
    satellite_batch._t4cof   = torch.full((batch_size,),0.0); satellite_batch._t5cof  = torch.full((batch_size,),0.0); satellite_batch._x1mth2   = torch.full((batch_size,),0.0);
    satellite_batch._x7thm1  = torch.full((batch_size,),0.0); satellite_batch._mdot   = torch.full((batch_size,),0.0); satellite_batch._nodedot  = torch.full((batch_size,),0.0);
    satellite_batch._xlcof   = torch.full((batch_size,),0.0); satellite_batch._xmcof  = torch.full((batch_size,),0.0); satellite_batch._nodecf   = torch.full((batch_size,),0.0);

    #  ------------------------ earth constants -----------------------
    #  sgp4fix identify constants and allow alternate values
    #  this is now the only call for the constants
    (satellite_batch._tumin, satellite_batch._mu, satellite_batch._radiusearthkm, satellite_batch._xke,
      satellite_batch._j2, satellite_batch._j3, satellite_batch._j4, satellite_batch._j3oj2) = whichconst

  # -------------------------------------------------------------------------

    satellite_batch._error = torch.full((batch_size,),0)
    satellite_batch._operationmode = opsmode
    satellite_batch._satnum = satn

    satellite_batch._bstar   = xbstar.clone()
    # sgp4fix allow additional parameters in the struct
    satellite_batch._ndot    = xndot.clone()
    satellite_batch._nddot   = xnddot.clone()
    satellite_batch._ecco    = xecco.clone()
    satellite_batch._argpo   = xargpo.clone()
    satellite_batch._inclo   = xinclo.clone()
    satellite_batch._mo	    = xmo.clone()
  # sgp4fix rename variables to clarify which mean motion is intended
    satellite_batch._no_kozai= xno_kozai.clone()
    satellite_batch._nodeo   = xnodeo.clone()

    # single averaged mean elements
    satellite_batch._am = torch.full((batch_size,),0.0)
    satellite_batch._em = torch.full((batch_size,),0.0)
    satellite_batch._im = torch.full((batch_size,),0.0)
    satellite_batch._Om = torch.full((batch_size,),0.0)
    satellite_batch._mm = torch.full((batch_size,),0.0)
    satellite_batch._nm = torch.full((batch_size,),0.0)

    ss     = 78.0 / satellite_batch._radiusearthkm + 1.0

    qzms2ttemp = (120.0 - 78.0) / satellite_batch._radiusearthkm
    qzms2t = qzms2ttemp * qzms2ttemp * qzms2ttemp * qzms2ttemp
    x2o3   =  2.0 / 3.0

    satellite_batch._init = 'y'
    satellite_batch._t	 = 0.0

    # sgp4fix remove satn as it is not needed in initl
    (
      satellite_batch._no_unkozai,
      method,
      ainv,  ao,    satellite_batch._con41,  con42, cosio,
      cosio2,eccsq, omeosq, posq,
      rp,    rteosq,sinio , satellite_batch._gsto,
      ) = initl_batch(
          satellite_batch._xke, satellite_batch._j2, satellite_batch._ecco, epoch, satellite_batch._inclo, satellite_batch._no_kozai,
          satellite_batch._operationmode, batch_size, satellite_batch._method
        )
    satellite_batch._a    = torch.pow( satellite_batch._no_unkozai*satellite_batch._tumin , (-2.0/3.0) )
    satellite_batch._alta = satellite_batch._a*(1.0 + satellite_batch._ecco) - 1.0
    satellite_batch._altp = satellite_batch._a*(1.0 - satellite_batch._ecco) - 1.0

    # Assuming all relevant variables are tensors.
    condition1 = (omeosq >= 0.0) | (satellite_batch._no_unkozai >= 0.0)

    # Initialize all variables that will be conditionally assigned
    satellite_batch._isimp = torch.where(condition1, torch.tensor(0), satellite_batch._isimp)
    condition2 = condition1 & (rp < 220.0 / satellite_batch._radiusearthkm + 1.0)
    satellite_batch._isimp = torch.where(condition2, torch.tensor(1), satellite_batch._isimp)

    sfour = torch.where(condition1, ss, torch.zeros_like(ss))
    qzms24 = torch.where(condition1, qzms2t, torch.zeros_like(qzms2t))
    perige = torch.where(condition1, (rp - 1.0) * satellite_batch._radiusearthkm, torch.zeros_like(rp))

    condition3 = condition1 & (perige < 156.0)
    sfour_temp1 = torch.where(perige < 98.0, torch.tensor(20.0), perige - 78.0)
    sfour = torch.where(condition3, sfour_temp1, sfour)
    qzms24temp = (120.0 - sfour) / satellite_batch._radiusearthkm
    qzms24 = torch.where(condition3, qzms24temp ** 4, qzms24)
    sfour = torch.where(condition3, sfour / satellite_batch._radiusearthkm + 1.0, sfour)

    pinvsq = 1.0 / posq
    tsi = 1.0 / (ao - sfour)
    satellite_batch._eta = ao * satellite_batch._ecco * tsi
    etasq = satellite_batch._eta ** 2
    eeta = satellite_batch._ecco * satellite_batch._eta
    psisq = torch.abs(1.0 - etasq)
    coef = qzms24 * (tsi ** 4)
    coef1 = coef / (psisq ** 3.5)
    cc2 = coef1 * satellite_batch._no_unkozai * (ao * (1.0 + 1.5 * etasq + eeta *
                    (4.0 + etasq)) + 0.375 * satellite_batch._j2 * tsi / psisq * satellite_batch._con41 *
                    (8.0 + 3.0 * etasq * (8.0 + etasq)))
    satellite_batch._cc1 = satellite_batch._bstar * cc2
    cc3 = torch.where(satellite_batch._ecco > 1.0e-4,
                    -2.0 * coef * tsi * satellite_batch._j3oj2 * satellite_batch._no_unkozai * sinio / satellite_batch._ecco,
                    torch.tensor(0.0))
    satellite_batch._x1mth2 = 1.0 - cosio2
    satellite_batch._cc4 = 2.0 * satellite_batch._no_unkozai * coef1 * ao * omeosq * \
                        (satellite_batch._eta * (2.0 + 0.5 * etasq) + satellite_batch._ecco *
                            (0.5 + 2.0 * etasq) - satellite_batch._j2 * tsi / (ao * psisq) *
                            (-3.0 * satellite_batch._con41 * (1.0 - 2.0 * eeta + etasq *
                            (1.5 - 0.5 * eeta)) + 0.75 * satellite_batch._x1mth2 *
                            (2.0 * etasq - eeta * (1.0 + etasq)) * (2.0 * satellite_batch._argpo).cos()))
    satellite_batch._cc5 = 2.0 * coef1 * ao * omeosq * (1.0 + 2.75 *
                        (etasq + eeta) + eeta * etasq)
    cosio4 = cosio2 ** 2
    temp1 = 1.5 * satellite_batch._j2 * pinvsq * satellite_batch._no_unkozai
    temp2 = 0.5 * temp1 * satellite_batch._j2 * pinvsq
    temp3 = -0.46875 * satellite_batch._j4 * pinvsq * pinvsq * satellite_batch._no_unkozai
    satellite_batch._mdot = satellite_batch._no_unkozai + 0.5 * temp1 * rteosq * satellite_batch._con41 + 0.0625 * \
                            temp2 * rteosq * (13.0 - 78.0 * cosio2 + 137.0 * cosio4)
    satellite_batch._argpdot = (-0.5 * temp1 * con42 + 0.0625 * temp2 *
                                (7.0 - 114.0 * cosio2 + 395.0 * cosio4) +
                                temp3 * (3.0 - 36.0 * cosio2 + 49.0 * cosio4))
    xhdot1 = -temp1 * cosio
    satellite_batch._nodedot = xhdot1 + (0.5 * temp2 * (4.0 - 19.0 * cosio2) +
                                        2.0 * temp3 * (3.0 - 7.0 * cosio2)) * cosio
    xpidot = satellite_batch._argpdot + satellite_batch._nodedot
    satellite_batch._omgcof = satellite_batch._bstar * cc3 * satellite_batch._argpo.cos()
    satellite_batch._xmcof = torch.where(satellite_batch._ecco > 1.0e-4,
                                        -x2o3 * coef * satellite_batch._bstar / eeta,
                                        satellite_batch._xmcof)
    satellite_batch._nodecf = 3.5 * omeosq * xhdot1 * satellite_batch._cc1
    satellite_batch._t2cof = 1.5 * satellite_batch._cc1
    satellite_batch._xlcof = torch.where(torch.abs(cosio + 1.0) > 1.5e-12,
                                        -0.25 * satellite_batch._j3oj2 * sinio * (3.0 + 5.0 * cosio) / (1.0 + cosio),
                                        -0.25 * satellite_batch._j3oj2 * sinio * (3.0 + 5.0 * cosio) / temp4)
    satellite_batch._aycof = -0.5 * satellite_batch._j3oj2 * sinio
    delmotemp = 1.0 + satellite_batch._eta * satellite_batch._mo.cos()
    satellite_batch._delmo = delmotemp ** 3
    satellite_batch._sinmao = satellite_batch._mo.sin()
    satellite_batch._x7thm1 = 7.0 * cosio2 - 1.0

    # Deep space initialization
    deep_space_condition = (2 * np.pi / satellite_batch._no_unkozai >= 225.0)
    if torch.any(deep_space_condition):
        raise RuntimeError("Error: deep space propagation not supported (yet). One of the provided satellites has"
                        "an orbital period above 225 minutes. If you want to let us know you need it or you want to "
                        "contribute to implement it, open a PR or raise an issue at: https://github.com/esa/dSGP4.")

    isimp_condition = (satellite_batch._isimp != 1)
    if torch.any(isimp_condition):
        cc1sq = satellite_batch._cc1 ** 2
        d2 = 4.0 * ao * tsi * cc1sq
        temp = d2 * tsi * satellite_batch._cc1 / 3.0
        d3 = (17.0 * ao + sfour) * temp
        d4 = 0.5 * temp * ao * tsi * (221.0 * ao + 31.0 * sfour) * satellite_batch._cc1
        t3cof = d2 + 2.0 * cc1sq
        t4cof = 0.25 * (3.0 * d3 + satellite_batch._cc1 * (12.0 * d2 + 10.0 * cc1sq))
        t5cof = 0.2 * (3.0 * d4 + 12.0 * satellite_batch._cc1 * d3 +
                    6.0 * d2 ** 2 + 15.0 * cc1sq * (2.0 * d2 + cc1sq))
        satellite_batch._d2 = torch.where(isimp_condition, d2, satellite_batch._d2)
        satellite_batch._d3 = torch.where(isimp_condition, d3, satellite_batch._d3)
        satellite_batch._d4 = torch.where(isimp_condition, d4, satellite_batch._d4)
        satellite_batch._t3cof = torch.where(isimp_condition, t3cof, satellite_batch._t3cof)
        satellite_batch._t4cof = torch.where(isimp_condition, t4cof, satellite_batch._t4cof)
        satellite_batch._t5cof = torch.where(isimp_condition, t5cof, satellite_batch._t5cof)
    sgp4_batched(satellite_batch, torch.zeros((batch_size,)))

    satellite_batch._init = 'n'
