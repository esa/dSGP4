import torch
import numpy 
from .tle import TLE

def sgp4_batched(satellite_batch, tsince):
    """
    This function represents the batch SGP4 propagator. 
    It resembles `sgp4`, but accepts batches of TLEs.
    Having created the TLE object, and initialized the propagator (using `dsgp4.sgp4.sgp4init`), 
    one can use this method to propagate the TLE at future times. 
    The method returns the satellite position and velocity
    in km and km/s, respectively, after `tsince` minutes.

    Parameters:
    ----------------
    satellite_batch (``dsgp4.tle.TLE``): a TLE object.
    tsince (``torch.tensor``): a torch.tensor of times since the TLE epoch in minutes.

    Returns:
    ----------------
    batch_state (``torch.tensor``): a batch of 2x3 tensors, where the first row represents the spacecraft
                                    position (in km) and the second the spacecraft velocity (in km/s)
    """
    if not isinstance(satellite_batch, TLE):
        raise ValueError("satellite_batch should be a TLE object.")
    if not torch.is_tensor(tsince):
        raise ValueError("tsince must be a tensor.")
    if tsince.ndim!=1:
        raise ValueError("tsince should be a one dimensional tensor.")
    if len(tsince)!=len(satellite_batch._argpo):
        raise ValueError(f"in batch mode, tsince and satellite_batch shall have attributes of same length. Instead {len(tsince)} for time, and {len(satellite_batch._argpo)} for satellites' attributes found")
    if not hasattr(satellite_batch, '_radiusearthkm'):
        raise AttributeError('It looks like the satellite_batch has not been initialized. Please use the `initialize_tle` method or directly `sgp4init` to initialize the satellite_batch. Otherwise, if you are propagating, another option is to use `dsgp4.propagate` and pass `initialized=True` in the arguments.')
    
    batch_size = len(tsince)
    mrt = torch.zeros(batch_size)
    x2o3  = torch.tensor(2.0 / 3.0)

    vkmpersec    = torch.ones(batch_size)*(satellite_batch._radiusearthkm * satellite_batch._xke/60.0)
    #  sgp4 error flag
    satellite_batch._t    = tsince.clone()
    satellite_batch._error = torch.zeros(batch_size)

    # update for secular gravity and atmospheric drag
    xmdf    = satellite_batch._mo + satellite_batch._mdot * satellite_batch._t
    argpdf  = satellite_batch._argpo + satellite_batch._argpdot * satellite_batch._t
    nodedf  = satellite_batch._nodeo + satellite_batch._nodedot * satellite_batch._t
    argpm1   = argpdf
    mm1     = xmdf
    t2     = satellite_batch._t * satellite_batch._t
    nodem   = nodedf + satellite_batch._nodecf * t2
    tempa1   = 1.0 - satellite_batch._cc1 * satellite_batch._t
    tempe1   = satellite_batch._bstar * satellite_batch._cc4 * satellite_batch._t
    templ1   = satellite_batch._t2cof * t2


    delomg = satellite_batch._omgcof * satellite_batch._t

    delmtemp =  1.0 + satellite_batch._eta * xmdf.cos()
    delm   = satellite_batch._xmcof * \
                (delmtemp * delmtemp * delmtemp -
                satellite_batch._delmo)
    temp   = delomg + delm
    mm0     = xmdf + temp
    argpm0  = argpdf - temp
    t3     = t2 * satellite_batch._t
    t4     = t3 * satellite_batch._t
    tempa0  = tempa1 - satellite_batch._d2 * t2 - satellite_batch._d3 * t3 - \
                        satellite_batch._d4 * t4
    tempe0  = tempe1 + satellite_batch._bstar * satellite_batch._cc5 * (mm0.sin() -
                        satellite_batch._sinmao)
    templ0  = templ1 + satellite_batch._t3cof * t3 + t4 * (satellite_batch._t4cof +
                        satellite_batch._t * satellite_batch._t5cof)
    
    mm    = torch.where(satellite_batch._isimp==0,mm0,mm1)
    argpm = torch.where(satellite_batch._isimp==0,argpm0,argpm1)
    tempa = torch.where(satellite_batch._isimp==0,tempa0,tempa1)
    tempe = torch.where(satellite_batch._isimp==0,tempe0,tempe1)
    templ = torch.where(satellite_batch._isimp==0,templ0,templ1)

    nm    = satellite_batch._no_unkozai.clone()
    em    = satellite_batch._ecco.clone()
    inclm = satellite_batch._inclo.clone()

    satellite_batch._error = torch.full(nm.size(),2) * (nm <= 0.)

    am = torch.pow((satellite_batch._xke / nm),x2o3) * tempa * tempa
    nm = satellite_batch._xke / torch.pow(am, 1.5)
    em = em - tempe
    
    satellite_batch._error = torch.full(em.size(), 1) * ((em>=1.0) | (em<-0.001))

    em=torch.clamp(em, min=1.0e-6)
    mm     = mm + satellite_batch._no_unkozai * templ
    xlm    = mm + argpm + nodem
    emsq   = em * em
    temp   = 1.0 - emsq

    nodem = torch.fmod(nodem, torch.tensor(2*numpy.pi))

    argpm  = argpm % (2*numpy.pi)
    xlm    = xlm % (2*numpy.pi)
    mm     = (xlm - argpm - nodem) % (2*numpy.pi)

    satellite_batch._am = am.clone()
    satellite_batch._em = em.clone()
    satellite_batch._im = inclm.clone()
    satellite_batch._Om = nodem.clone()
    satellite_batch._om = argpm.clone()
    satellite_batch._mm = mm.clone()
    satellite_batch._nm = nm.clone()

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
    aynl = ep* argpp.sin() + temp * satellite_batch._aycof
    xl   = mp + argpp + nodep + temp * satellite_batch._xlcof * axnl

    #  solve kepler's equation 
    u    = (xl - nodep) % (2*numpy.pi)
    eo1  = u
    tem5 = torch.ones(tsince.size())

    for _ in range(10):
        coseo1=eo1.cos()
        sineo1=eo1.sin()
        tem5   = 1.0 - coseo1 * axnl - sineo1 * aynl
        tem5   = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5
        tem5=torch.where(tem5>=0.95, 0.95, tem5)
        tem5=torch.where(tem5<=-0.95, -0.95, tem5)
        eo1    = eo1 + tem5
        if torch.all(torch.abs(tem5) < 1e-12):
            break

    # short period preliminary quantities
    ecose = axnl*coseo1 + aynl*sineo1
    esine = axnl*sineo1 - aynl*coseo1
    el2   = axnl*axnl + aynl*aynl
    pl    = am*(1.0-el2)
    satellite_batch._error=torch.where(satellite_batch._error==0.,torch.any(pl<0.)*4,1)

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
    temp1  = 0.5 * satellite_batch._j2 * temp
    temp2  = temp1 * temp

    mrt   = rl * (1.0 - 1.5 * temp2 * betal * satellite_batch._con41) + \
             0.5 * temp1 * satellite_batch._x1mth2 * cos2u
    su    = su - 0.25 * temp2 * satellite_batch._x7thm1 * sin2u
    xnode = nodep + 1.5 * temp2 * cosip * sin2u
    xinc  = xincp + 1.5 * temp2 * cosip * sinip * cos2u
    mvt   = rdotl - nm * temp1 * satellite_batch._x1mth2 * sin2u / satellite_batch._xke
    rvdot = rvdotl + nm * temp1 * (satellite_batch._x1mth2 * cos2u +
             1.5 * satellite_batch._con41) / satellite_batch._xke

    #  orientation vectors
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
    _mr = mrt * satellite_batch._radiusearthkm

    r = torch.stack((_mr * ux, _mr * uy, _mr * uz))
    v = torch.stack(((mvt * ux + rvdot * vx) * vkmpersec,
          (mvt * uy + rvdot * vy) * vkmpersec,
          (mvt * uz + rvdot * vz) * vkmpersec))

    satellite_batch._error=torch.where(satellite_batch._error==0.,torch.any(mrt<1.0)*6,satellite_batch._error)
    return torch.transpose(torch.stack((r.squeeze(),v.squeeze()),1),0,-1)#torch.cat((r.swapaxes(0,2),v.swapaxes(0,2)),1)#torch.stack(list(r)+list(v)).reshape(2,3)
