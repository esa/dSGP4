"""
Batched (vectorized) versions of the SDP4 deep-space routines in `deep_space.py`.

Every function here operates element-wise on 1-D tensors (one entry per satellite).
Scalar Python branches of the reference implementation are replaced by `torch.where`,
so a batch can mix resonance classes (irez = 0, 1, 2), inclination regimes and
propagation directions, and stays differentiable. Denominators of branches that are
not selected are replaced by safe values so that no NaN leaks into gradients.

`dscom` from `deep_space.py` contains no data-dependent branches and is reused as-is.
"""
import numpy as np
import torch

from .deep_space import _to_tensor, _mod2pi, _mod2pi_signed

TWOPI = 2.0 * np.pi


def _safe(den, mask):
    """Return `den` where `mask` holds, 1 elsewhere (avoids NaN in unused branches)."""
    return torch.where(mask, den, torch.ones_like(den))


def dpper_batched(satellite, t, ep, inclp, nodep, argpp, mp, opsmode):
    """
    Batched lunar-solar periodics (the `init == 'n'` path of `deep_space.dpper`).

    Parameters:
    ----------------
    satellite (``dsgp4.tle.TLE``): batch-initialized TLE object (deep-space coefficients as 1-D tensors)
    t (``torch.tensor``): times since epoch [min], one per satellite
    ep, inclp, nodep, argpp, mp (``torch.tensor``): mean elements, one per satellite
    opsmode (``str``): 'a' or 'i'

    Returns:
    ----------------
    (ep, inclp, nodep, argpp, mp) with the periodic corrections applied.
    """
    zns = _to_tensor(1.19459e-5, ep)
    zes = _to_tensor(0.01675, ep)
    znl = _to_tensor(1.5835218e-4, ep)
    zel = _to_tensor(0.05490, ep)
    two_pi = _to_tensor(TWOPI, ep)

    zm = satellite._zmos + zns * t
    zf = zm + 2.0 * zes * torch.sin(zm)
    sinzf = torch.sin(zf)
    f2 = 0.5 * sinzf * sinzf - 0.25
    f3 = -0.5 * sinzf * torch.cos(zf)
    ses = satellite._se2 * f2 + satellite._se3 * f3
    sis = satellite._si2 * f2 + satellite._si3 * f3
    sls = satellite._sl2 * f2 + satellite._sl3 * f3 + satellite._sl4 * sinzf
    sghs = satellite._sgh2 * f2 + satellite._sgh3 * f3 + satellite._sgh4 * sinzf
    shs = satellite._sh2 * f2 + satellite._sh3 * f3

    zm = satellite._zmol + znl * t
    zf = zm + 2.0 * zel * torch.sin(zm)
    sinzf = torch.sin(zf)
    f2 = 0.5 * sinzf * sinzf - 0.25
    f3 = -0.5 * sinzf * torch.cos(zf)
    sel = satellite._ee2 * f2 + satellite._e3 * f3
    sil = satellite._xi2 * f2 + satellite._xi3 * f3
    sll = satellite._xl2 * f2 + satellite._xl3 * f3 + satellite._xl4 * sinzf
    sghl = satellite._xgh2 * f2 + satellite._xgh3 * f3 + satellite._xgh4 * sinzf
    shll = satellite._xh2 * f2 + satellite._xh3 * f3

    pe = ses + sel - satellite._peo
    pinc = sis + sil - satellite._pinco
    pl = sls + sll - satellite._plo
    pgh = sghs + sghl - satellite._pgho
    ph = shs + shll - satellite._pho

    inclp = inclp + pinc
    ep = ep + pe
    sinip = torch.sin(inclp)
    cosip = torch.cos(inclp)

    high = inclp >= 0.2

    # --- inclination >= 0.2 rad: apply periodics directly
    ph_h = ph / _safe(sinip, high)
    pgh_h = pgh - cosip * ph_h
    argpp_h = argpp + pgh_h
    nodep_h = nodep + ph_h
    mp_h = mp + pl

    # --- inclination < 0.2 rad: Lyddane modification
    sinop = torch.sin(nodep)
    cosop = torch.cos(nodep)
    alfdp = sinip * sinop + (ph * cosop + pinc * cosip * sinop)
    betdp = sinip * cosop + (-ph * sinop + pinc * cosip * cosop)
    nodep_l = _mod2pi_signed(nodep)
    if opsmode == 'a':
        nodep_l = torch.where(nodep_l < 0.0, nodep_l + two_pi, nodep_l)
    xls = mp + argpp + pl + pgh + (cosip - pinc * sinip) * nodep_l
    xnoh = nodep_l
    # atan2(0, 0) (e.g. an equatorial near-earth member of a mixed batch) has an undefined
    # gradient; give that unused degenerate case a safe denominator so NaNs cannot leak
    degenerate = (alfdp == 0.0) & (betdp == 0.0)
    nodep_l = torch.atan2(alfdp, torch.where(degenerate, torch.ones_like(betdp), betdp))
    if opsmode == 'a':
        nodep_l = torch.where(nodep_l < 0.0, nodep_l + two_pi, nodep_l)
    wrap = torch.abs(xnoh - nodep_l) > np.pi
    nodep_l = torch.where(wrap & (nodep_l < xnoh), nodep_l + two_pi,
                          torch.where(wrap, nodep_l - two_pi, nodep_l))
    mp_l = mp + pl
    argpp_l = xls - mp_l - cosip * nodep_l

    argpp = torch.where(high, argpp_h, argpp_l)
    nodep = torch.where(high, nodep_h, nodep_l)
    mp = torch.where(high, mp_h, mp_l)
    return ep, inclp, nodep, argpp, mp


def dsinit_batched(satellite, cosim, emsq, s1, s2, s3, s4, s5, sinim, ss1, ss2, ss3, ss4, ss5,
                   sz1, sz3, sz11, sz13, sz21, sz23, sz31, sz33, z1, z3, z11, z13, z21, z23, z31, z33,
                   ecco, eccsq, em, inclm, nm, xpidot):
    """
    Batched deep-space initialization (resonance coefficients), equivalent to
    `deep_space.dsinit` evaluated at t = 0 for every satellite of the batch.

    Returns a dict of 1-D tensors; `irez` is an integer tensor (0: none, 1: 24h synchronous, 2: 12h half-day).
    """
    q22 = _to_tensor(1.7891679e-6, em)
    q31 = _to_tensor(2.1460748e-6, em)
    q33 = _to_tensor(2.2123015e-7, em)
    root22 = _to_tensor(1.7891679e-6, em)
    root44 = _to_tensor(7.3636953e-9, em)
    root54 = _to_tensor(2.1765803e-9, em)
    rptim = _to_tensor(4.37526908801129966e-3, em)
    root32 = _to_tensor(3.7393792e-7, em)
    root52 = _to_tensor(1.1428639e-7, em)
    x2o3 = _to_tensor(2.0 / 3.0, em)
    znl = _to_tensor(1.5835218e-4, em)
    zns = _to_tensor(1.19459e-5, em)
    zero = torch.zeros_like(em)

    irez = torch.zeros(em.shape, dtype=torch.long, device=em.device)
    irez = torch.where((nm > 0.0034906585) & (nm < 0.0052359877), 1, irez)
    irez = torch.where((nm >= 8.26e-3) & (nm <= 9.24e-3) & (em >= 0.5), 2, irez)

    polar_or_equatorial = (inclm < 5.2359877e-2) | (inclm > np.pi - 5.2359877e-2)
    nz = sinim != 0.0
    sinim_s = _safe(sinim, nz)

    ses = ss1 * zns * ss5
    sis = ss2 * zns * (sz11 + sz13)
    sls = -zns * ss3 * (sz1 + sz3 - 14.0 - 6.0 * emsq)
    sghs = ss4 * zns * (sz31 + sz33 - 6.0)
    shs = -zns * ss2 * (sz21 + sz23)
    shs = torch.where(polar_or_equatorial, zero, shs)
    shs = torch.where(nz, shs / sinim_s, shs)
    sgs = sghs - cosim * shs

    dedt = ses + s1 * znl * s5
    didt = sis + s2 * znl * (z11 + z13)
    dmdt = sls - znl * s3 * (z1 + z3 - 14.0 - 6.0 * emsq)
    sghl = s4 * znl * (z31 + z33 - 6.0)
    shll = -znl * s2 * (z21 + z23)
    shll = torch.where(polar_or_equatorial, zero, shll)
    domdt = sgs + sghl
    dnodt = shs
    domdt = torch.where(nz, domdt - cosim / sinim_s * shll, domdt)
    dnodt = torch.where(nz, dnodt + shll / sinim_s, dnodt)

    aonv = torch.pow(nm / satellite._xke, x2o3)
    theta = _mod2pi(satellite._gsto + zero * rptim)

    # ---------------- 12h resonance (irez == 2) ----------------
    cosisq = cosim * cosim
    e2 = ecco
    e2sq = eccsq
    eoc = e2 * e2sq
    g201 = -0.306 - (e2 - 0.64) * 0.440
    lo = e2 <= 0.65
    g211 = torch.where(lo, 3.616 - 13.2470 * e2 + 16.2900 * e2sq,
                       -72.099 + 331.819 * e2 - 508.738 * e2sq + 266.724 * eoc)
    g310 = torch.where(lo, -19.302 + 117.3900 * e2 - 228.4190 * e2sq + 156.5910 * eoc,
                       -346.844 + 1582.851 * e2 - 2415.925 * e2sq + 1246.113 * eoc)
    g322 = torch.where(lo, -18.9068 + 109.7927 * e2 - 214.6334 * e2sq + 146.5816 * eoc,
                       -342.585 + 1554.908 * e2 - 2366.899 * e2sq + 1215.972 * eoc)
    g410 = torch.where(lo, -41.122 + 242.6940 * e2 - 471.0940 * e2sq + 313.9530 * eoc,
                       -1052.797 + 4758.686 * e2 - 7193.992 * e2sq + 3651.957 * eoc)
    g422 = torch.where(lo, -146.407 + 841.8800 * e2 - 1629.014 * e2sq + 1083.4350 * eoc,
                       -3581.690 + 16178.110 * e2 - 24462.770 * e2sq + 12422.520 * eoc)
    g520 = torch.where(lo, -532.114 + 3017.977 * e2 - 5740.032 * e2sq + 3708.2760 * eoc,
                       torch.where(e2 > 0.715,
                                   -5149.66 + 29936.92 * e2 - 54087.36 * e2sq + 31324.56 * eoc,
                                   1464.74 - 4664.75 * e2 + 3763.64 * e2sq))
    lo7 = e2 < 0.7
    g533 = torch.where(lo7, -919.22770 + 4988.6100 * e2 - 9064.7700 * e2sq + 5542.21 * eoc,
                       -37995.780 + 161616.52 * e2 - 229838.20 * e2sq + 109377.94 * eoc)
    g521 = torch.where(lo7, -822.71072 + 4568.6173 * e2 - 8491.4146 * e2sq + 5337.524 * eoc,
                       -51752.104 + 218913.95 * e2 - 309468.16 * e2sq + 146349.42 * eoc)
    g532 = torch.where(lo7, -853.66600 + 4690.2500 * e2 - 8624.7700 * e2sq + 5341.4 * eoc,
                       -40023.880 + 170470.89 * e2 - 242699.48 * e2sq + 115605.82 * eoc)

    sini2 = sinim * sinim
    f220 = 0.75 * (1.0 + 2.0 * cosim + cosisq)
    f221 = 1.5 * sini2
    f321 = 1.875 * sinim * (1.0 - 2.0 * cosim - 3.0 * cosisq)
    f322 = -1.875 * sinim * (1.0 + 2.0 * cosim - 3.0 * cosisq)
    f441 = 35.0 * sini2 * f220
    f442 = 39.3750 * sini2 * sini2
    f522 = 9.84375 * sinim * (sini2 * (1.0 - 2.0 * cosim - 5.0 * cosisq)
                              + 0.33333333 * (-2.0 + 4.0 * cosim + 6.0 * cosisq))
    f523 = sinim * (4.92187512 * sini2 * (-2.0 - 4.0 * cosim + 10.0 * cosisq)
                    + 6.56250012 * (1.0 + 2.0 * cosim - 3.0 * cosisq))
    f542 = 29.53125 * sinim * (2.0 - 8.0 * cosim + cosisq * (-12.0 + 8.0 * cosim + 10.0 * cosisq))
    f543 = 29.53125 * sinim * (-2.0 - 8.0 * cosim + cosisq * (12.0 + 8.0 * cosim - 10.0 * cosisq))
    xno2 = nm * nm
    ainv2 = aonv * aonv
    temp1 = 3.0 * xno2 * ainv2
    temp = temp1 * root22
    d2201 = temp * f220 * g201
    d2211 = temp * f221 * g211
    temp1 = temp1 * aonv
    temp = temp1 * root32
    d3210 = temp * f321 * g310
    d3222 = temp * f322 * g322
    temp1 = temp1 * aonv
    temp = 2.0 * temp1 * root44
    d4410 = temp * f441 * g410
    d4422 = temp * f442 * g422
    temp1 = temp1 * aonv
    temp = temp1 * root52
    d5220 = temp * f522 * g520
    d5232 = temp * f523 * g532
    temp = 2.0 * temp1 * root54
    d5421 = temp * f542 * g521
    d5433 = temp * f543 * g533
    xlamo2 = _mod2pi(satellite._mo + satellite._nodeo + satellite._nodeo - theta - theta)
    xfact2 = satellite._mdot + dmdt + 2.0 * (satellite._nodedot + dnodt - rptim) - satellite._no_unkozai

    # ---------------- 24h resonance (irez == 1) ----------------
    g200 = 1.0 + emsq * (-2.5 + 0.8125 * emsq)
    g310_1 = 1.0 + 2.0 * emsq
    g300 = 1.0 + emsq * (-6.0 + 6.60937 * emsq)
    f220_1 = 0.75 * (1.0 + cosim) * (1.0 + cosim)
    f311 = 0.9375 * sinim * sinim * (1.0 + 3.0 * cosim) - 0.75 * (1.0 + cosim)
    f330 = 1.0 + cosim
    f330 = 1.875 * f330 * f330 * f330
    del1 = 3.0 * nm * nm * aonv * aonv
    del2 = 2.0 * del1 * f220_1 * g200 * q22
    del3 = 3.0 * del1 * f330 * g300 * q33 * aonv
    del1 = del1 * f311 * g310_1 * q31 * aonv
    xlamo1 = _mod2pi(satellite._mo + satellite._nodeo + satellite._argpo - theta)
    xfact1 = satellite._mdot + xpidot - rptim + dmdt + domdt + dnodt - satellite._no_unkozai

    is1 = irez == 1
    is2 = irez == 2
    d2201, d2211, d3210, d3222, d4410, d4422, d5220, d5232, d5421, d5433 = [
        torch.where(is2, d, zero) for d in (d2201, d2211, d3210, d3222, d4410, d4422,
                                            d5220, d5232, d5421, d5433)]
    del1, del2, del3 = [torch.where(is1, d, zero) for d in (del1, del2, del3)]
    xlamo = torch.where(is2, xlamo2, torch.where(is1, xlamo1, zero))
    xfact = torch.where(is2, xfact2, torch.where(is1, xfact1, zero))

    return {
        "irez": irez,
        "d2201": d2201, "d2211": d2211, "d3210": d3210, "d3222": d3222,
        "d4410": d4410, "d4422": d4422, "d5220": d5220, "d5232": d5232,
        "d5421": d5421, "d5433": d5433,
        "dedt": dedt, "didt": didt, "dmdt": dmdt, "dnodt": dnodt, "domdt": domdt,
        "del1": del1, "del2": del2, "del3": del3,
        "xfact": xfact, "xlamo": xlamo,
    }


def dspace_batched(satellite, t, em, argpm, inclm, mm, nodem, nm):
    """
    Batched deep-space secular effects and resonance integration (`deep_space.dspace`).

    The resonance integrator is stateless: each call restarts from epoch and takes the
    same fixed 720-minute steps as the reference implementation, with a per-satellite
    mask that stops each element once it is within one step of its own `t`.
    Satellites with `irez == 0` are left untouched by the integrator.
    """
    fasx2 = _to_tensor(0.13130908, em)
    fasx4 = _to_tensor(2.8843198, em)
    fasx6 = _to_tensor(0.37448087, em)
    g22 = _to_tensor(5.7686396, em)
    g32 = _to_tensor(0.95240898, em)
    g44 = _to_tensor(1.8014998, em)
    g52 = _to_tensor(1.0508330, em)
    g54 = _to_tensor(4.4108898, em)
    rptim = _to_tensor(4.37526908801129966e-3, em)
    stepp = _to_tensor(720.0, em)
    step2 = _to_tensor(259200.0, em)

    theta = _mod2pi(satellite._gsto + t * rptim)
    em = em + satellite._dedt * t
    inclm = inclm + satellite._didt * t
    argpm = argpm + satellite._domdt * t
    nodem = nodem + satellite._dnodt * t
    mm = mm + satellite._dmdt * t

    irez = satellite._irez
    res = irez != 0
    if not bool(torch.any(res)):
        return em, argpm, inclm, mm, nodem, nm
    is2 = irez == 2

    delt = torch.where(t > 0.0, stepp, -stepp)
    atime = torch.zeros_like(t)
    xni = satellite._no_unkozai + torch.zeros_like(t)
    xli = satellite._xlamo + torch.zeros_like(t)

    def _derivs(xli, xni, atime):
        # 24h synchronous
        xndt1 = (satellite._del1 * torch.sin(xli - fasx2)
                 + satellite._del2 * torch.sin(2.0 * (xli - fasx4))
                 + satellite._del3 * torch.sin(3.0 * (xli - fasx6)))
        xnddt1 = (satellite._del1 * torch.cos(xli - fasx2)
                  + 2.0 * satellite._del2 * torch.cos(2.0 * (xli - fasx4))
                  + 3.0 * satellite._del3 * torch.cos(3.0 * (xli - fasx6)))
        # 12h half-day
        xomi = satellite._argpo + satellite._argpdot * atime
        x2omi = xomi + xomi
        x2li = xli + xli
        xndt2 = (satellite._d2201 * torch.sin(x2omi + xli - g22)
                 + satellite._d2211 * torch.sin(xli - g22)
                 + satellite._d3210 * torch.sin(xomi + xli - g32)
                 + satellite._d3222 * torch.sin(-xomi + xli - g32)
                 + satellite._d4410 * torch.sin(x2omi + x2li - g44)
                 + satellite._d4422 * torch.sin(x2li - g44)
                 + satellite._d5220 * torch.sin(xomi + xli - g52)
                 + satellite._d5232 * torch.sin(-xomi + xli - g52)
                 + satellite._d5421 * torch.sin(xomi + x2li - g54)
                 + satellite._d5433 * torch.sin(-xomi + x2li - g54))
        xnddt2 = (satellite._d2201 * torch.cos(x2omi + xli - g22)
                  + satellite._d2211 * torch.cos(xli - g22)
                  + satellite._d3210 * torch.cos(xomi + xli - g32)
                  + satellite._d3222 * torch.cos(-xomi + xli - g32)
                  + satellite._d5220 * torch.cos(xomi + xli - g52)
                  + satellite._d5232 * torch.cos(-xomi + xli - g52)
                  + 2.0 * (satellite._d4410 * torch.cos(x2omi + x2li - g44)
                           + satellite._d4422 * torch.cos(x2li - g44)
                           + satellite._d5421 * torch.cos(xomi + x2li - g54)
                           + satellite._d5433 * torch.cos(-xomi + x2li - g54)))
        xndt = torch.where(is2, xndt2, xndt1)
        xldot = xni + satellite._xfact
        xnddt = torch.where(is2, xnddt2, xnddt1) * xldot
        return xndt, xldot, xnddt

    xndt, xldot, xnddt = _derivs(xli, xni, atime)
    active = res & (torch.abs(t - atime) >= stepp)
    while bool(torch.any(active)):
        xli = torch.where(active, xli + xldot * delt + xndt * step2, xli)
        xni = torch.where(active, xni + xndt * delt + xnddt * step2, xni)
        atime = torch.where(active, atime + delt, atime)
        xndt, xldot, xnddt = _derivs(xli, xni, atime)
        active = res & (torch.abs(t - atime) >= stepp)

    ft = t - atime
    nm_r = xni + xndt * ft + xnddt * ft * ft * 0.5
    xl = xli + xldot * ft + xndt * ft * ft * 0.5
    mm_r = torch.where(irez != 1, xl - 2.0 * nodem + 2.0 * theta, xl - nodem - argpm + theta)

    nm = torch.where(res, nm_r, nm)
    mm = torch.where(res, mm_r, mm)
    return em, argpm, inclm, mm, nodem, nm
