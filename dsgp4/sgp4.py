import numpy
import torch
from .tle import TLE

#@torch.jit.script
def sgp4(satellite, tsince):
    """
    This function represents the SGP4 propagator. Having created the TLE object, and
    initialized the propagator (using `dsgp4.sgp4.sgp4init`), one can use this method
    to propagate the TLE at future times. The method returns the satellite position and velocity
    in km and km/s, respectively, after `tsince` minutes.

    Parameters:
    ----------------
    satellite (``dsgp4.tle.TLE``): a TLE object.
    tsince (``torch.tensor``): a torch.tensor of times since the TLE epoch in minutes.

    Returns:
    ----------------
    (``torch.tensor``): a tensor of len(tsince)x2x3 representing the satellite position and velocity in km and km/s, respectively.
    """
    #quick check to see if the satellite has been initialized
    if not isinstance(satellite, TLE):
        raise TypeError('The satellite object should be a dsgp4.tle.TLE object.')
    if not hasattr(satellite, '_radiusearthkm'):
        raise AttributeError('It looks like the satellite has not been initialized. Please use the `initialize_tle` method or directly `sgp4init` to initialize the satellite. Otherwise, if you are propagating, another option is to use `dsgp4.propagate` and pass `initialized=True` in the arguments.')
    #in case an int, float or list are passed, convert them to torch.tensor
    if isinstance(tsince, (int, float, list)):
        tsince = torch.tensor(tsince)
    mrt = torch.zeros(tsince.size())
    temp4 = torch.ones(tsince.size())*1.5e-12
    x2o3  = torch.tensor(2.0 / 3.0)

    vkmpersec    = torch.ones(tsince.size())*(satellite._radiusearthkm * satellite._xke/60.0)

    # sgp4 error flag
    satellite._t    = tsince.clone()
    satellite._error = torch.tensor(0)

    #  secular gravity and atmospheric drag
    xmdf    = satellite._mo + satellite._mdot * satellite._t
    argpdf  = satellite._argpo + satellite._argpdot * satellite._t
    nodedf  = satellite._nodeo + satellite._nodedot * satellite._t
    argpm   = argpdf
    mm     = xmdf
    t2     = satellite._t * satellite._t
    nodem   = nodedf + satellite._nodecf * t2
    tempa   = 1.0 - satellite._cc1 * satellite._t
    tempe   = satellite._bstar * satellite._cc4 * satellite._t
    templ   = satellite._t2cof * t2

    if satellite._isimp != 1:
        delomg = satellite._omgcof * satellite._t
        delmtemp =  1.0 + satellite._eta * xmdf.cos()
        delm   = satellite._xmcof * \
                  (delmtemp * delmtemp * delmtemp -
                  satellite._delmo)
        temp   = delomg + delm
        mm     = xmdf + temp
        argpm  = argpdf - temp
        t3     = t2 * satellite._t
        t4     = t3 * satellite._t
        tempa  = tempa - satellite._d2 * t2 - satellite._d3 * t3 - \
                          satellite._d4 * t4
        tempe  = tempe + satellite._bstar * satellite._cc5 * (mm.sin() -
                          satellite._sinmao)
        templ  = templ + satellite._t3cof * t3 + t4 * (satellite._t4cof +
                          satellite._t * satellite._t5cof)

    nm    = satellite._no_unkozai.clone()
    em    = satellite._ecco.clone()
    inclm = satellite._inclo.clone()

    satellite._error=torch.any(nm<=0)*2

    am = torch.pow((satellite._xke / nm),x2o3) * tempa * tempa
    nm = satellite._xke / torch.pow(am, 1.5)
    em = em - tempe

    if satellite._error==0.:
        satellite._error=torch.any((em>=1.0) | (em<-0.001))*1

    #  sgp4fix fix tolerance to avoid a divide by zero
    em=torch.where(em<1.0e-6,1.0e-6,em)
    mm     = mm + satellite._no_unkozai * templ
    xlm    = mm + argpm + nodem
    emsq   = em * em
    temp   = 1.0 - emsq

    nodem = torch.fmod(nodem, torch.tensor(2*numpy.pi))

    argpm  = argpm % (2*numpy.pi)
    xlm    = xlm % (2*numpy.pi)
    mm     = (xlm - argpm - nodem) % (2*numpy.pi)

    satellite._am = am.clone()
    satellite._em = em.clone()
    satellite._im = inclm.clone()
    satellite._Om = nodem.clone()
    satellite._om = argpm.clone()
    satellite._mm = mm.clone()
    satellite._nm = nm.clone()

    # compute extra mean quantities
    sinim = inclm.sin()
    cosim = inclm.cos()

    # add lunar-solar periodics
    ep     = em
    xincp  = inclm
    argpp  = argpm
    nodep  = nodem
    mp     = mm
    sinip  = sinim
    cosip  = cosim

    axnl = ep * argpp.cos()
    temp = 1.0 / (am * (1.0 - ep * ep))
    aynl = ep* argpp.sin() + temp * satellite._aycof
    xl   = mp + argpp + nodep + temp * satellite._xlcof * axnl

    # solve kepler's equation
    u    = (xl - nodep) % (2*numpy.pi)
    eo1  = u
    tem5 = torch.ones(tsince.size())
    # kepler iteration
    for _ in range(10):
        coseo1=eo1.cos()
        sineo1=eo1.sin()
        tem5   = 1.0 - coseo1 * axnl - sineo1 * aynl
        tem5   = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5
        tem5=torch.where(tem5>=0.95, 0.95, tem5)
        tem5=torch.where(tem5<=-0.95, -0.95, tem5)
        #we need to break if abs value of tem5 is less than 1e-12:
        eo1    = eo1 + tem5
        if torch.all(torch.abs(tem5) < 1e-12):
            break

    #  short period preliminary quantities
    ecose = axnl*coseo1 + aynl*sineo1
    esine = axnl*sineo1 - aynl*coseo1
    el2   = axnl*axnl + aynl*aynl
    pl    = am*(1.0-el2)
    if satellite._error==0.:
        satellite._error=torch.any(pl<0.)*4

    rl     = am * (1.0 - ecose)
    rdotl  = am.sqrt() * esine/rl
    rvdotl = pl.sqrt() / rl
    betal  = (1.0 - el2).sqrt()
    temp   = esine / (1.0 + betal)
    sinu   = am / rl * (sineo1 - aynl - axnl * temp)
    cosu   = am / rl * (coseo1 - axnl + aynl * temp)
    su     = torch.atan2(sinu, cosu)
    sin2u  = (cosu + cosu) * sinu
    cos2u  = 1.0 - 2.0 * sinu * sinu
    temp   = 1.0 / pl
    temp1  = 0.5 * satellite._j2 * temp
    temp2  = temp1 * temp

    mrt   = rl * (1.0 - 1.5 * temp2 * betal * satellite._con41) + \
             0.5 * temp1 * satellite._x1mth2 * cos2u
    su    = su - 0.25 * temp2 * satellite._x7thm1 * sin2u
    xnode = nodep + 1.5 * temp2 * cosip * sin2u
    xinc  = xincp + 1.5 * temp2 * cosip * sinip * cos2u
    mvt   = rdotl - nm * temp1 * satellite._x1mth2 * sin2u / satellite._xke
    rvdot = rvdotl + nm * temp1 * (satellite._x1mth2 * cos2u +
             1.5 * satellite._con41) / satellite._xke

    # orientation vectors
    sinsu =  su.sin()
    cossu =  su.cos()
    snod  =  xnode.sin()
    cnod  =  xnode.cos()
    sini  =  xinc.sin()
    cosi  =  xinc.cos()
    xmx   = -snod * cosi
    xmy   =  cnod * cosi
    ux    =  xmx * sinsu + cnod * cossu
    uy    =  xmy * sinsu + snod * cossu
    uz    =  sini * sinsu
    vx    =  xmx * cossu - cnod * sinsu
    vy    =  xmy * cossu - snod * sinsu
    vz    =  sini * cossu

    # position and velocity (in km and km/sec)
    _mr = mrt * satellite._radiusearthkm

    r = torch.stack((_mr * ux, _mr * uy, _mr * uz))
    v = torch.stack(((mvt * ux + rvdot * vx) * vkmpersec,
          (mvt * uy + rvdot * vy) * vkmpersec,
          (mvt * uz + rvdot * vz) * vkmpersec))

    # decaying satellites
    if satellite._error==0.:
        satellite._error=torch.any(mrt<1.0)*6
    return torch.transpose(torch.stack((r.squeeze(),v.squeeze()),1),0,-1)#torch.cat((r.swapaxes(0,2),v.swapaxes(0,2)),1)#torch.stack(list(r)+list(v)).reshape(2,3)
