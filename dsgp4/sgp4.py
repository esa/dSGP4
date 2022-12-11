import numpy
import torch
torch.set_default_dtype(torch.float64)

#@torch.jit.script
def sgp4(satrec, tsince):
    """
    This function represents the SGP4 propagator. Having created the TLE object, and
    initialized the propagator (using `kessler.sgp4.sgp4init`), one can use this method
    to propagate the TLE at future times. The method returns the satellite position and velocity
    in km and km/s, respectively, after `tsince` minutes.

    Args:
        - satrec (``kessler.tle.TLE``): TLE object
        - tsince (``torch.tensor``): time to propagate, since the TLE epoch, in minutes

    Returns:
        - state (``torch.tensor``): a 2x3 tensor, where the first row represents the spacecraft
                                    position (in km) and the second the spacecraft velocity (in km/s)
    """
    mrt = torch.zeros(tsince.size());
    temp4 = torch.ones(tsince.size())*1.5e-12;
    x2o3  = torch.tensor(2.0 / 3.0);
    #  sgp4fix identify constants and allow alternate values
    # tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 = whichconst
    vkmpersec    = torch.ones(tsince.size())*(satrec._radiusearthkm * satrec._xke/60.0);

    #  --------------------- clear sgp4 error flag -----------------
    satrec._t    = tsince.clone();
    satrec._error = torch.tensor(0);
#    satrec._error_message = torch.ones(tsince.size())*torch.nan;

    #  ------- update for secular gravity and atmospheric drag -----
    xmdf    = satrec._mo + satrec._mdot * satrec._t;
    argpdf  = satrec._argpo + satrec._argpdot * satrec._t;
    nodedf  = satrec._nodeo + satrec._nodedot * satrec._t;
    argpm   = argpdf;
    mm     = xmdf;
    t2     = satrec._t * satrec._t;
    nodem   = nodedf + satrec._nodecf * t2;
    tempa   = 1.0 - satrec._cc1 * satrec._t;
    tempe   = satrec._bstar * satrec._cc4 * satrec._t;
    templ   = satrec._t2cof * t2;

    if satrec._isimp != 1:
        delomg = satrec._omgcof * satrec._t;
        #  sgp4fix use mutliply for speed instead of pow
        delmtemp =  1.0 + satrec._eta * xmdf.cos();
        delm   = satrec._xmcof * \
                  (delmtemp * delmtemp * delmtemp -
                  satrec._delmo);
        temp   = delomg + delm;
        mm     = xmdf + temp;
        argpm  = argpdf - temp;
        t3     = t2 * satrec._t;
        t4     = t3 * satrec._t;
        tempa  = tempa - satrec._d2 * t2 - satrec._d3 * t3 - \
                          satrec._d4 * t4;
        tempe  = tempe + satrec._bstar * satrec._cc5 * (mm.sin() -
                          satrec._sinmao);
        templ  = templ + satrec._t3cof * t3 + t4 * (satrec._t4cof +
                          satrec._t * satrec._t5cof);

    nm    = satrec._no_unkozai.clone();
    em    = satrec._ecco.clone();
    inclm = satrec._inclo.clone();
#     satrec._error=torch.where(nm<=0,2,0)
    #satrec._error_message=torch.where(nm<=0., ('mean motion {0:f} is less than zero'.format(nm)), )
    satrec._error=torch.any(nm<=0)*2
    #if nm <= 0.0:
    #    satrec._error_message = ('mean motion {0:f} is less than zero'
    #                             .format(nm))
    #    satrec._error = 2;
         #  sgp4fix add return
    #    return false, false;

    am = torch.pow((satrec._xke / nm),x2o3) * tempa * tempa;
    nm = satrec._xke / torch.pow(am, 1.5);
    em = em - tempe;
     #  fix tolerance for error recognition
     #  sgp4fix am is fixed from the previous nm check
    if satrec._error==0.:
        satrec._error=torch.any((em>=1.0) | (em<-0.001))*1
#     if em >= 1.0 or em < -0.001:  # || (am < 0.95)
#         satrec._error_message = ('mean eccentricity {0:f} not within'
#                                 ' range 0.0 <= e < 1.0'.format(em))
#         satrec._error = 1;
         #  sgp4fix to return if there is an error in eccentricity
#         return false, false;

     #  sgp4fix fix tolerance to avoid a divide by zero
    em=torch.where(em<1.0e-6,1.0e-6,em)
#     if em < 1.0e-6:
#         em  = 1.0e-6;
    mm     = mm + satrec._no_unkozai * templ;
    xlm    = mm + argpm + nodem;
    emsq   = em * em;
    temp   = 1.0 - emsq;

    nodem = torch.fmod(nodem, torch.tensor(2*numpy.pi))
#     nodem  = nodem % (2*numpy.pi) if nodem >= 0.0 else -(-nodem % (2*numpy.pi))

    argpm  = argpm % (2*numpy.pi)
    xlm    = xlm % (2*numpy.pi)
    mm     = (xlm - argpm - nodem) % (2*numpy.pi)

     # sgp4fix recover singly averaged mean elements
    satrec._am = am.clone();
    satrec._em = em.clone();
    satrec._im = inclm.clone();
    satrec._Om = nodem.clone();
    satrec._om = argpm.clone();
    satrec._mm = mm.clone();
    satrec._nm = nm.clone();

     #  ----------------- compute extra mean quantities -------------
    sinim = inclm.sin();
    cosim = inclm.cos();

     #  -------------------- add lunar-solar periodics --------------
    ep     = em;
    xincp  = inclm;
    argpp  = argpm;
    nodep  = nodem;
    mp     = mm;
    sinip  = sinim;
    cosip  = cosim;

    axnl = ep * argpp.cos();
    temp = 1.0 / (am * (1.0 - ep * ep));
    aynl = ep* argpp.sin() + temp * satrec._aycof;
    xl   = mp + argpp + nodep + temp * satrec._xlcof * axnl;

     #  --------------------- solve kepler's equation ---------------
    u    = (xl - nodep) % (2*numpy.pi)
    eo1  = u;
    tem5 = torch.ones(tsince.size());
     #    sgp4fix for kepler iteration
     #    the following iteration needs better limits on corrections
#    while tem5.abs() >= 1.0e-12 and ktr <= 10:
    for _ in range(10):
        coseo1=eo1.cos()
        sineo1=eo1.sin()
        tem5   = 1.0 - coseo1 * axnl - sineo1 * aynl;
        tem5   = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5;
        tem5=torch.where(tem5>=0.95, 0.95, tem5)
        tem5=torch.where(tem5<=-0.95, -0.95, tem5)
        eo1    = eo1 + tem5;
#        tem5, eo1=kepler_iteration(tem5, eo1, axnl, aynl, u)
        #tem5, eo1=torch.where(torch.abs(tem5)>=1.0e-12, kepler_iteration(tem5, eo1, axnl, aynl, u)[0], (tem5,eo1))

     #  ------------- short period preliminary quantities -----------
    ecose = axnl*coseo1 + aynl*sineo1;
    esine = axnl*sineo1 - aynl*coseo1;
    el2   = axnl*axnl + aynl*aynl;
    pl    = am*(1.0-el2);
    if satrec._error==0.:
        satrec._error=torch.any(pl<0.)*4
     #if pl < 0.0:

    # satrec._error_message = ('semilatus rectum {0:f} is less than zero'
    #                             .format(pl))
    #     satrec._error = 4;
         #  sgp4fix add return
    #     return false, false;

     #else:
    rl     = am * (1.0 - ecose);
    rdotl  = am.sqrt() * esine/rl;
    rvdotl = pl.sqrt() / rl;
    betal  = (1.0 - el2).sqrt();
    temp   = esine / (1.0 + betal);
    sinu   = am / rl * (sineo1 - aynl - axnl * temp);
    cosu   = am / rl * (coseo1 - axnl + aynl * temp);
    su     = torch.atan2(sinu, cosu);
    sin2u  = (cosu + cosu) * sinu;
    cos2u  = 1.0 - 2.0 * sinu * sinu;
    temp   = 1.0 / pl;
    temp1  = 0.5 * satrec._j2 * temp;
    temp2  = temp1 * temp;

    mrt   = rl * (1.0 - 1.5 * temp2 * betal * satrec._con41) + \
             0.5 * temp1 * satrec._x1mth2 * cos2u;
    su    = su - 0.25 * temp2 * satrec._x7thm1 * sin2u;
    xnode = nodep + 1.5 * temp2 * cosip * sin2u;
    xinc  = xincp + 1.5 * temp2 * cosip * sinip * cos2u;
    mvt   = rdotl - nm * temp1 * satrec._x1mth2 * sin2u / satrec._xke;
    rvdot = rvdotl + nm * temp1 * (satrec._x1mth2 * cos2u +
             1.5 * satrec._con41) / satrec._xke;

     #  --------------------- orientation vectors -------------------
    sinsu =  su.sin();
    cossu =  su.cos();
    snod  =  xnode.sin();
    cnod  =  xnode.cos();
    sini  =  xinc.sin();
    cosi  =  xinc.cos();
    xmx   = -snod * cosi;
    xmy   =  cnod * cosi;
    ux    =  xmx * sinsu + cnod * cossu;
    uy    =  xmy * sinsu + snod * cossu;
    uz    =  sini * sinsu;
    vx    =  xmx * cossu - cnod * sinsu;
    vy    =  xmy * cossu - snod * sinsu;
    vz    =  sini * cossu;

     #  --------- position and velocity (in km and km/sec) ----------
    _mr = mrt * satrec._radiusearthkm

    r = torch.stack((_mr * ux, _mr * uy, _mr * uz))
    v = torch.stack(((mvt * ux + rvdot * vx) * vkmpersec,
          (mvt * uy + rvdot * vy) * vkmpersec,
          (mvt * uz + rvdot * vz) * vkmpersec))

     #  sgp4fix for decaying satellites
    if satrec._error==0.:
        satrec._error=torch.any(mrt<1.0)*6
     #if mrt < 1.0:

    #     satrec._error_message = ('mrt {0:f} is less than 1.0 indicating'
    #                             ' the satellite has decayed'.format(mrt))
    #     satrec._error = 6;
    #     return false, false
    return torch.transpose(torch.stack((r.squeeze(),v.squeeze()),1),0,-1)#torch.cat((r.swapaxes(0,2),v.swapaxes(0,2)),1)#torch.stack(list(r)+list(v)).reshape(2,3);
