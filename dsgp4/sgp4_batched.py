import torch
import numpy 

# This should be the default and renamed to sgp4. When we are sure it works, we can remove the old sgp4.
def sgp4_batched(satrec, tsince):
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

    if not isinstance(satrec, list):
        raise ValueError("satrec should be a list of TLE objects.")
    if not torch.is_tensor(tsince):
        raise ValueError("tsince must be a tensor.")
    if tsince.ndim!=1:
        raise ValueError("tsince should be a one dimensional tensor.")
    if len(tsince)!=len(satrec):
        raise ValueError("tsince and satrec shall be of same length.")
    
    batch_size = len(satrec)
        
    satrec_batch=satrec[0].copy()    
    satrec_batch._bstar=torch.stack([s._bstar for s in satrec])
    satrec_batch._ndot=torch.stack([s._ndot for s in satrec])
    satrec_batch._nddot=torch.stack([s._nddot for s in satrec])
    satrec_batch._ecco=torch.stack([s._ecco for s in satrec])
    satrec_batch._argpo=torch.stack([s._argpo for s in satrec])
    satrec_batch._inclo=torch.stack([s._inclo for s in satrec])
    satrec_batch._mo=torch.stack([s._mo for s in satrec])

    satrec_batch._no_kozai=torch.stack([s._no_kozai for s in satrec])
    satrec_batch._nodeo=torch.stack([s._nodeo for s in satrec])
    satrec_batch.satellite_catalog_number=torch.tensor([s.satellite_catalog_number for s in satrec])
    satrec_batch._jdsatepoch=torch.stack([s._jdsatepoch for s in satrec])
    satrec_batch._jdsatepochF=torch.stack([s._jdsatepochF for s in satrec])
    satrec_batch._isimp=torch.tensor([s._isimp for s in satrec])
    satrec_batch._method=[s._method for s in satrec]

    satrec_batch._mdot=torch.stack([s._mdot for s in satrec])
    satrec_batch._argpdot=torch.stack([s._argpdot for s in satrec])
    satrec_batch._nodedot=torch.stack([s._nodedot for s in satrec])
    satrec_batch._nodecf=torch.stack([s._nodecf for s in satrec])
    satrec_batch._cc1=torch.stack([s._cc1 for s in satrec])
    satrec_batch._cc4=torch.stack([s._cc4 for s in satrec])
    satrec_batch._cc5=torch.stack([s._cc5 for s in satrec])
    satrec_batch._t2cof=torch.stack([s._t2cof for s in satrec])

    satrec_batch._omgcof=torch.stack([s._omgcof for s in satrec])
    satrec_batch._eta=torch.stack([s._eta for s in satrec])
    satrec_batch._xmcof=torch.stack([s._xmcof for s in satrec])
    satrec_batch._delmo=torch.stack([s._delmo for s in satrec])
    satrec_batch._d2=torch.stack([s._d2 for s in satrec])
    satrec_batch._d3=torch.stack([s._d3 for s in satrec])
    satrec_batch._d4=torch.stack([s._d4 for s in satrec])
    satrec_batch._cc5=torch.stack([s._cc5 for s in satrec])
    satrec_batch._sinmao=torch.stack([s._sinmao for s in satrec])
    satrec_batch._t3cof=torch.stack([s._t3cof for s in satrec])
    satrec_batch._t4cof=torch.stack([s._t4cof for s in satrec])
    satrec_batch._t5cof=torch.stack([s._t5cof for s in satrec])
    
    satrec_batch._xke=torch.stack([s._xke for s in satrec])
    satrec_batch._radiusearthkm=torch.stack([s._radiusearthkm for s in satrec])
    satrec_batch._t=torch.stack([s._t for s in satrec])
    satrec_batch._aycof=torch.stack([s._aycof for s in satrec])
    satrec_batch._x1mth2=torch.stack([s._x1mth2 for s in satrec])
    satrec_batch._con41=torch.stack([s._con41 for s in satrec])
    satrec_batch._x7thm1=torch.stack([s._x7thm1 for s in satrec])
    satrec_batch._xlcof=torch.stack([s._xlcof for s in satrec])
    satrec_batch._tumin=torch.stack([s._tumin for s in satrec])
    satrec_batch._mu=torch.stack([s._mu for s in satrec])
    satrec_batch._j2=torch.stack([s._j2 for s in satrec])
    satrec_batch._j3=torch.stack([s._j3 for s in satrec])
    satrec_batch._j4=torch.stack([s._j4 for s in satrec])
    satrec_batch._j3oj2=torch.stack([s._j3oj2 for s in satrec])
    satrec_batch._error=torch.stack([s._error for s in satrec])
    satrec_batch._operationmode=[s._operationmode for s in satrec]
    satrec_batch._satnum=torch.tensor([s._satnum for s in satrec])
    satrec_batch._am=torch.stack([s._am for s in satrec])
    satrec_batch._em=torch.stack([s._em for s in satrec])
    satrec_batch._im=torch.stack([s._im for s in satrec])
    satrec_batch._Om=torch.stack([s._Om for s in satrec])
    satrec_batch._mm=torch.stack([s._mm for s in satrec])
    satrec_batch._nm=torch.stack([s._nm for s in satrec])
    satrec_batch._init=[s._init for s in satrec]
            
    satrec_batch._no_unkozai=torch.stack([s._no_unkozai for s in satrec])
    satrec_batch._a=torch.stack([s._a for s in satrec])
    satrec_batch._alta=torch.stack([s._altp for s in satrec])

    mrt = torch.zeros(batch_size);
    # temp4 = torch.ones(batch_size)*1.5e-12;
    x2o3  = torch.tensor(2.0 / 3.0);
    #  sgp4fix identify constants and allow alternate values
    # tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 = whichconst
    vkmpersec    = torch.ones(batch_size)*(satrec_batch._radiusearthkm * satrec_batch._xke/60.0);

    #  --------------------- clear sgp4 error flag -----------------
    satrec_batch._t    = tsince.clone();
    satrec_batch._error = torch.zeros(batch_size);
#    satrec._error_message = torch.ones(tsince.size())*torch.nan;

    #  ------- update for secular gravity and atmospheric drag -----
    xmdf    = satrec_batch._mo + satrec_batch._mdot * satrec_batch._t;
    argpdf  = satrec_batch._argpo + satrec_batch._argpdot * satrec_batch._t;
    nodedf  = satrec_batch._nodeo + satrec_batch._nodedot * satrec_batch._t;
    argpm   = argpdf;
    mm     = xmdf;
    t2     = satrec_batch._t * satrec_batch._t;
    nodem   = nodedf + satrec_batch._nodecf * t2;
    tempa   = 1.0 - satrec_batch._cc1 * satrec_batch._t;
    tempe   = satrec_batch._bstar * satrec_batch._cc4 * satrec_batch._t;
    templ   = satrec_batch._t2cof * t2;

    # START: ASSUME satrec._isimp IS ALWAYS 0
    for sat in satrec:
        if sat._isimp == 1:
            raise ValueError("isimp == 1 not supported.")

    delomg = satrec_batch._omgcof * satrec_batch._t;
    #  sgp4fix use mutliply for speed instead of pow
    delmtemp =  1.0 + satrec_batch._eta * xmdf.cos();
    delm   = satrec_batch._xmcof * \
                (delmtemp * delmtemp * delmtemp -
                satrec_batch._delmo);
    temp   = delomg + delm;
    mm     = xmdf + temp;
    argpm  = argpdf - temp;
    t3     = t2 * satrec_batch._t;
    t4     = t3 * satrec_batch._t;
    tempa  = tempa - satrec_batch._d2 * t2 - satrec_batch._d3 * t3 - \
                        satrec_batch._d4 * t4;
    tempe  = tempe + satrec_batch._bstar * satrec_batch._cc5 * (mm.sin() -
                        satrec_batch._sinmao);
    templ  = templ + satrec_batch._t3cof * t3 + t4 * (satrec_batch._t4cof +
                        satrec_batch._t * satrec_batch._t5cof);
    # END: ASSUME satrec._isimp IS ALWAYS 0

    nm    = satrec_batch._no_unkozai.clone();
    em    = satrec_batch._ecco.clone();
    inclm = satrec_batch._inclo.clone();
#     satrec._error=torch.where(nm<=0,2,0)
    #satrec._error_message=torch.where(nm<=0., ('mean motion {0:f} is less than zero'.format(nm)), )
    satrec_batch._error = torch.full(nm.size(),2) * (nm <= 0.)
    #if nm <= 0.0:
    #    satrec._error_message = ('mean motion {0:f} is less than zero'
    #                             .format(nm))
    #    satrec._error = 2;
         #  sgp4fix add return
    #    return false, false;

    am = torch.pow((satrec_batch._xke / nm),x2o3) * tempa * tempa;
    nm = satrec_batch._xke / torch.pow(am, 1.5);
    em = em - tempe;
     #  fix tolerance for error recognition
     #  sgp4fix am is fixed from the previous nm check

    satrec_batch._error = torch.full(em.size(), 1) * ((em>=1.0) | (em<-0.001))

    # The if statement above is the non-batched version of what we had before, where there was only one satrec._error value (a scalar). Now instead we have have batch of errors in satrec._error given in a tensor. So the equivalent of the above is:

    # if torch.any((em>=1.0) | (em<-0.001)):
    #     satrec._error=torch.ones(tsince.size())*1
    # else:
    #     satrec._error=torch.zeros(tsince.size())





    # em=torch.where(em<1.0e-6,1.0e-6,em)
    em=torch.clamp(em, min=1.0e-6)
#     if em < 1.0e-6:
#         em  = 1.0e-6;
    mm     = mm + satrec_batch._no_unkozai * templ;
    xlm    = mm + argpm + nodem;
    emsq   = em * em;
    temp   = 1.0 - emsq;

    nodem = torch.fmod(nodem, torch.tensor(2*numpy.pi))
#     nodem  = nodem % (2*numpy.pi) if nodem >= 0.0 else -(-nodem % (2*numpy.pi))

    argpm  = argpm % (2*numpy.pi)
    xlm    = xlm % (2*numpy.pi)
    mm     = (xlm - argpm - nodem) % (2*numpy.pi)

     # sgp4fix recover singly averaged mean elements
    satrec_batch._am = am.clone();
    satrec_batch._em = em.clone();
    satrec_batch._im = inclm.clone();
    satrec_batch._Om = nodem.clone();
    satrec_batch._om = argpm.clone();
    satrec_batch._mm = mm.clone();
    satrec_batch._nm = nm.clone();

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
    aynl = ep* argpp.sin() + temp * satrec_batch._aycof;
    xl   = mp + argpp + nodep + temp * satrec_batch._xlcof * axnl;

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
    satrec_batch._error=torch.where(satrec_batch._error==0.,torch.any(pl<0.)*4,1)
#    if satrec_batch._error==0.:
#        satrec_batch._error=torch.any(pl<0.)*4
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
    temp1  = 0.5 * satrec_batch._j2 * temp;
    temp2  = temp1 * temp;

    mrt   = rl * (1.0 - 1.5 * temp2 * betal * satrec_batch._con41) + \
             0.5 * temp1 * satrec_batch._x1mth2 * cos2u;
    su    = su - 0.25 * temp2 * satrec_batch._x7thm1 * sin2u;
    xnode = nodep + 1.5 * temp2 * cosip * sin2u;
    xinc  = xincp + 1.5 * temp2 * cosip * sinip * cos2u;
    mvt   = rdotl - nm * temp1 * satrec_batch._x1mth2 * sin2u / satrec_batch._xke;
    rvdot = rvdotl + nm * temp1 * (satrec_batch._x1mth2 * cos2u +
             1.5 * satrec_batch._con41) / satrec_batch._xke;

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
    _mr = mrt * satrec_batch._radiusearthkm

    r = torch.stack((_mr * ux, _mr * uy, _mr * uz))
    v = torch.stack(((mvt * ux + rvdot * vx) * vkmpersec,
          (mvt * uy + rvdot * vy) * vkmpersec,
          (mvt * uz + rvdot * vz) * vkmpersec))

     #  sgp4fix for decaying satellites
    satrec_batch._error=torch.where(satrec_batch._error==0.,torch.any(mrt<1.0)*6,satrec_batch._error)
#    if satrec_batch._error==0.:
#        satrec_batch._error=torch.any(mrt<1.0)*6
     #if mrt < 1.0:

    #     satrec._error_message = ('mrt {0:f} is less than 1.0 indicating'
    #                             ' the satellite has decayed'.format(mrt))
    #     satrec._error = 6;
    #     return false, false
    return torch.transpose(torch.stack((r.squeeze(),v.squeeze()),1),0,-1)#torch.cat((r.swapaxes(0,2),v.swapaxes(0,2)),1)#torch.stack(list(r)+list(v)).reshape(2,3);
