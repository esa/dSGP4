"""
/* -----------------------------------------------------------------------------
*
*                           procedure dpper
*
*  this procedure provides deep space long period periodic contributions
*    to the mean elements.  by design, these periodics are zero at epoch.
*    this used to be dscom which included initialization, but it's really a
*    recurring function.
*
*  author        : david vallado                  719-573-2600   28 jun 2005
*
*  inputs        :
*    e3          -
*    ee2         -
*    peo         -
*    pgho        -
*    pho         -
*    pinco       -
*    plo         -
*    se2 , se3 , sgh2, sgh3, sgh4, sh2, sh3, si2, si3, sl2, sl3, sl4 -
*    t           -
*    xh2, xh3, xi2, xi3, xl2, xl3, xl4 -
*    zmol        -
*    zmos        -
*    ep          - eccentricity                           0.0 - 1.0
*    inclo       - inclination - needed for lyddane modification
*    nodep       - right ascension of ascending node
*    argpp       - argument of perigee
*    mp          - mean anomaly
*
*  outputs       :
*    ep          - eccentricity                           0.0 - 1.0
*    inclp       - inclination
*    nodep        - right ascension of ascending node
*    argpp       - argument of perigee
*    mp          - mean anomaly
*
*  locals        :
*    alfdp       -
*    betdp       -
*    cosip  , sinip  , cosop  , sinop  ,
*    dalf        -
*    dbet        -
*    dls         -
*    f2, f3      -
*    pe          -
*    pgh         -
*    ph          -
*    pinc        -
*    pl          -
*    sel   , ses   , sghl  , sghs  , shl   , shs   , sil   , sinzf , sis   ,
*    sll   , sls
*    xls         -
*    xnoh        -
*    zf          -
*    zm          -
*
*  coupling      :
*    none.
*
*  references    :
*    hoots, roehrich, norad spacetrack report #3 1980
*    hoots, norad spacetrack report #6 1986
*    hoots, schumacher and glover 2004
*    vallado, crawford, hujsak, kelso  2006
  ----------------------------------------------------------------------------*/
"""



def _dpper(satrec, inclo, init, ep, inclp, nodep, argpp, mp, opsmode):

     # Copy satellite attributes into local variables for convenience
     # and symmetry in writing formulae.
    
     e3 = satrec.e3
     ee2 = satrec.ee2
     peo = satrec.peo
     pgho = satrec.pgho
     pho = satrec.pho
     pinco = satrec.pinco
     plo = satrec.plo
     se2 = satrec.se2
     se3 = satrec.se3
     sgh2 = satrec.sgh2
     sgh3 = satrec.sgh3
     sgh4 = satrec.sgh4
     sh2 = satrec.sh2
     sh3 = satrec.sh3
     si2 = satrec.si2
     si3 = satrec.si3
     sl2 = satrec.sl2
     sl3 = satrec.sl3
     sl4 = satrec.sl4
     t = satrec.t
     xgh2 = satrec.xgh2
     xgh3 = satrec.xgh3
     xgh4 = satrec.xgh4
     xh2 = satrec.xh2
     xh3 = satrec.xh3
     xi2 = satrec.xi2
     xi3 = satrec.xi3
     xl2 = satrec.xl2
     xl3 = satrec.xl3
     xl4 = satrec.xl4
     zmol = satrec.zmol
     zmos = satrec.zmos

     #  ---------------------- constants -----------------------------
     zns   = 1.19459e-5;
     zes   = 0.01675;
     znl   = 1.5835218e-4;
     zel   = 0.05490;

     #  --------------- calculate time varying periodics -----------
     zm    = zmos + zns * t;
     # be sure that the initial call has time set to zero
     if init == 'y':
         zm = zmos;
     zf    = zm + 2.0 * zes * sin(zm);
     sinzf = sin(zf);
     f2    =  0.5 * sinzf * sinzf - 0.25;
     f3    = -0.5 * sinzf * cos(zf);
     ses   = se2* f2 + se3 * f3;
     sis   = si2 * f2 + si3 * f3;
     sls   = sl2 * f2 + sl3 * f3 + sl4 * sinzf;
     sghs  = sgh2 * f2 + sgh3 * f3 + sgh4 * sinzf;
     shs   = sh2 * f2 + sh3 * f3;
     zm    = zmol + znl * t;
     if init == 'y':
         zm = zmol;
     zf    = zm + 2.0 * zel * sin(zm);
     sinzf = sin(zf);
     f2    =  0.5 * sinzf * sinzf - 0.25;
     f3    = -0.5 * sinzf * cos(zf);
     sel   = ee2 * f2 + e3 * f3;
     sil   = xi2 * f2 + xi3 * f3;
     sll   = xl2 * f2 + xl3 * f3 + xl4 * sinzf;
     sghl  = xgh2 * f2 + xgh3 * f3 + xgh4 * sinzf;
     shll  = xh2 * f2 + xh3 * f3;
     pe    = ses + sel;
     pinc  = sis + sil;
     pl    = sls + sll;
     pgh   = sghs + sghl;
     ph    = shs + shll;

     if init == 'n':

       pe    = pe - peo;
       pinc  = pinc - pinco;
       pl    = pl - plo;
       pgh   = pgh - pgho;
       ph    = ph - pho;
       inclp = inclp + pinc;
       ep    = ep + pe;
       sinip = sin(inclp);
       cosip = cos(inclp);

       """
       /* ----------------- apply periodics directly ------------ */
       //  sgp4fix for lyddane choice
       //  strn3 used original inclination - this is technically feasible
       //  gsfc used perturbed inclination - also technically feasible
       //  probably best to readjust the 0.2 limit value and limit discontinuity
       //  0.2 rad = 11.45916 deg
       //  use next line for original strn3 approach and original inclination
       //  if (inclo >= 0.2)
       //  use next line for gsfc version and perturbed inclination
       """

       if inclp >= 0.2:

           ph /= sinip
           pgh -= cosip * ph
           argpp += pgh
           nodep += ph
           mp += pl

       else:

           #  ---- apply periodics with lyddane modification ----
           sinop  = sin(nodep);
           cosop  = cos(nodep);
           alfdp  = sinip * sinop;
           betdp  = sinip * cosop;
           dalf   =  ph * cosop + pinc * cosip * sinop;
           dbet   = -ph * sinop + pinc * cosip * cosop;
           alfdp  = alfdp + dalf;
           betdp  = betdp + dbet;
           nodep  = nodep % twopi if nodep >= 0.0 else -(-nodep % twopi)
           #   sgp4fix for afspc written intrinsic functions
           #  nodep used without a trigonometric function ahead
           if nodep < 0.0 and opsmode == 'a':
               nodep = nodep + twopi;
           xls = mp + argpp + pl + pgh + (cosip - pinc * sinip) * nodep
           xnoh   = nodep;
           nodep  = atan2(alfdp, betdp);
           #   sgp4fix for afspc written intrinsic functions
           #  nodep used without a trigonometric function ahead
           if nodep < 0.0 and opsmode == 'a':
               nodep = nodep + twopi;
           if fabs(xnoh - nodep) > pi:
             if nodep < xnoh:
                nodep = nodep + twopi;
             else:
                nodep = nodep - twopi;
           mp += pl
           argpp = xls - mp - cosip * nodep;

     return ep, inclp, nodep, argpp, mp
