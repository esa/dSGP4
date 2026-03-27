import numpy as np
import torch

TWOPI = 2.0 * np.pi


def _to_tensor(value, like):
    if torch.is_tensor(value):
        return value
    return torch.tensor(value, dtype=like.dtype, device=like.device)


def _to_bool(value):
    if torch.is_tensor(value):
        return bool(value.detach().item())
    return bool(value)


def _mod2pi(x):
    two_pi = _to_tensor(TWOPI, x)
    return torch.remainder(x, two_pi)


def _mod2pi_signed(x):
    two_pi = _to_tensor(TWOPI, x)
    rp = torch.remainder(torch.abs(x), two_pi)
    return torch.where(x >= 0.0, rp, -rp)


def dpper(satellite, inclo, init, ep, inclp, nodep, argpp, mp, opsmode):
    zns = _to_tensor(1.19459e-5, ep)
    zes = _to_tensor(0.01675, ep)
    znl = _to_tensor(1.5835218e-4, ep)
    zel = _to_tensor(0.05490, ep)

    zm = satellite._zmos + zns * satellite._t
    if init == 'y':
        zm = satellite._zmos
    zf = zm + 2.0 * zes * torch.sin(zm)
    sinzf = torch.sin(zf)
    f2 = 0.5 * sinzf * sinzf - 0.25
    f3 = -0.5 * sinzf * torch.cos(zf)
    ses = satellite._se2 * f2 + satellite._se3 * f3
    sis = satellite._si2 * f2 + satellite._si3 * f3
    sls = satellite._sl2 * f2 + satellite._sl3 * f3 + satellite._sl4 * sinzf
    sghs = satellite._sgh2 * f2 + satellite._sgh3 * f3 + satellite._sgh4 * sinzf
    shs = satellite._sh2 * f2 + satellite._sh3 * f3

    zm = satellite._zmol + znl * satellite._t
    if init == 'y':
        zm = satellite._zmol
    zf = zm + 2.0 * zel * torch.sin(zm)
    sinzf = torch.sin(zf)
    f2 = 0.5 * sinzf * sinzf - 0.25
    f3 = -0.5 * sinzf * torch.cos(zf)
    sel = satellite._ee2 * f2 + satellite._e3 * f3
    sil = satellite._xi2 * f2 + satellite._xi3 * f3
    sll = satellite._xl2 * f2 + satellite._xl3 * f3 + satellite._xl4 * sinzf
    sghl = satellite._xgh2 * f2 + satellite._xgh3 * f3 + satellite._xgh4 * sinzf
    shll = satellite._xh2 * f2 + satellite._xh3 * f3

    pe = ses + sel
    pinc = sis + sil
    pl = sls + sll
    pgh = sghs + sghl
    ph = shs + shll

    if init == 'n':
        pe = pe - satellite._peo
        pinc = pinc - satellite._pinco
        pl = pl - satellite._plo
        pgh = pgh - satellite._pgho
        ph = ph - satellite._pho

        inclp = inclp + pinc
        ep = ep + pe
        sinip = torch.sin(inclp)
        cosip = torch.cos(inclp)

        if _to_bool(inclp >= 0.2):
            ph = ph / sinip
            pgh = pgh - cosip * ph
            argpp = argpp + pgh
            nodep = nodep + ph
            mp = mp + pl
        else:
            sinop = torch.sin(nodep)
            cosop = torch.cos(nodep)
            alfdp = sinip * sinop
            betdp = sinip * cosop
            dalf = ph * cosop + pinc * cosip * sinop
            dbet = -ph * sinop + pinc * cosip * cosop
            alfdp = alfdp + dalf
            betdp = betdp + dbet
            nodep = _mod2pi_signed(nodep)
            if _to_bool((nodep < 0.0) & (opsmode == 'a')):
                nodep = nodep + _to_tensor(TWOPI, nodep)
            xls = mp + argpp + pl + pgh + (cosip - pinc * sinip) * nodep
            xnoh = nodep
            nodep = torch.atan2(alfdp, betdp)
            if _to_bool((nodep < 0.0) & (opsmode == 'a')):
                nodep = nodep + _to_tensor(TWOPI, nodep)
            if _to_bool(torch.abs(xnoh - nodep) > np.pi):
                if _to_bool(nodep < xnoh):
                    nodep = nodep + _to_tensor(TWOPI, nodep)
                else:
                    nodep = nodep - _to_tensor(TWOPI, nodep)
            mp = mp + pl
            argpp = xls - mp - cosip * nodep

    return ep, inclp, nodep, argpp, mp


def dscom(epoch, ep, argpp, tc, inclp, nodep, np_in):
    zes = _to_tensor(0.01675, ep)
    zel = _to_tensor(0.05490, ep)
    c1ss = _to_tensor(2.9864797e-6, ep)
    c1l = _to_tensor(4.7968065e-7, ep)
    zsinis = _to_tensor(0.39785416, ep)
    zcosis = _to_tensor(0.91744867, ep)
    zcosgs = _to_tensor(0.1945905, ep)
    zsings = _to_tensor(-0.98088458, ep)

    nm = np_in
    em = ep
    snodm = torch.sin(nodep)
    cnodm = torch.cos(nodep)
    sinomm = torch.sin(argpp)
    cosomm = torch.cos(argpp)
    sinim = torch.sin(inclp)
    cosim = torch.cos(inclp)
    emsq = em * em
    betasq = 1.0 - emsq
    rtemsq = torch.sqrt(betasq)

    peo = _to_tensor(0.0, ep)
    pinco = _to_tensor(0.0, ep)
    plo = _to_tensor(0.0, ep)
    pgho = _to_tensor(0.0, ep)
    pho = _to_tensor(0.0, ep)

    day = epoch + 18261.5 + tc / 1440.0
    xnodce = _mod2pi(4.5236020 - 9.2422029e-4 * day)
    stem = torch.sin(xnodce)
    ctem = torch.cos(xnodce)
    zcosil = 0.91375164 - 0.03568096 * ctem
    zsinil = torch.sqrt(1.0 - zcosil * zcosil)
    zsinhl = 0.089683511 * stem / zsinil
    zcoshl = torch.sqrt(1.0 - zsinhl * zsinhl)
    gam = 5.8351514 + 0.0019443680 * day
    zx = zsinis * stem / zsinil
    zy = zcoshl * ctem + zcosis * zsinhl * stem
    zx = torch.atan2(zx, zy)
    zx = gam + zx - xnodce
    zcosgl = torch.cos(zx)
    zsingl = torch.sin(zx)

    zcosg = zcosgs
    zsing = zsings
    zcosi = zcosis
    zsini = zsinis
    zcosh = cnodm
    zsinh = snodm
    cc = c1ss
    xnoi = 1.0 / nm

    for lsflg in (1, 2):
        a1 = zcosg * zcosh + zsing * zcosi * zsinh
        a3 = -zsing * zcosh + zcosg * zcosi * zsinh
        a7 = -zcosg * zsinh + zsing * zcosi * zcosh
        a8 = zsing * zsini
        a9 = zsing * zsinh + zcosg * zcosi * zcosh
        a10 = zcosg * zsini
        a2 = cosim * a7 + sinim * a8
        a4 = cosim * a9 + sinim * a10
        a5 = -sinim * a7 + cosim * a8
        a6 = -sinim * a9 + cosim * a10

        x1 = a1 * cosomm + a2 * sinomm
        x2 = a3 * cosomm + a4 * sinomm
        x3 = -a1 * sinomm + a2 * cosomm
        x4 = -a3 * sinomm + a4 * cosomm
        x5 = a5 * sinomm
        x6 = a6 * sinomm
        x7 = a5 * cosomm
        x8 = a6 * cosomm

        z31 = 12.0 * x1 * x1 - 3.0 * x3 * x3
        z32 = 24.0 * x1 * x2 - 6.0 * x3 * x4
        z33 = 12.0 * x2 * x2 - 3.0 * x4 * x4
        z1 = 3.0 * (a1 * a1 + a2 * a2) + z31 * emsq
        z2 = 6.0 * (a1 * a3 + a2 * a4) + z32 * emsq
        z3 = 3.0 * (a3 * a3 + a4 * a4) + z33 * emsq
        z11 = -6.0 * a1 * a5 + emsq * (-24.0 * x1 * x7 - 6.0 * x3 * x5)
        z12 = -6.0 * (a1 * a6 + a3 * a5) + emsq * (
            -24.0 * (x2 * x7 + x1 * x8) - 6.0 * (x3 * x6 + x4 * x5)
        )
        z13 = -6.0 * a3 * a6 + emsq * (-24.0 * x2 * x8 - 6.0 * x4 * x6)
        z21 = 6.0 * a2 * a5 + emsq * (24.0 * x1 * x5 - 6.0 * x3 * x7)
        z22 = 6.0 * (a4 * a5 + a2 * a6) + emsq * (
            24.0 * (x2 * x5 + x1 * x6) - 6.0 * (x4 * x7 + x3 * x8)
        )
        z23 = 6.0 * a4 * a6 + emsq * (24.0 * x2 * x6 - 6.0 * x4 * x8)
        z1 = z1 + z1 + betasq * z31
        z2 = z2 + z2 + betasq * z32
        z3 = z3 + z3 + betasq * z33
        s3 = cc * xnoi
        s2 = -0.5 * s3 / rtemsq
        s4 = s3 * rtemsq
        s1 = -15.0 * em * s4
        s5 = x1 * x3 + x2 * x4
        s6 = x2 * x3 + x1 * x4
        s7 = x2 * x4 - x1 * x3

        if lsflg == 1:
            ss1 = s1
            ss2 = s2
            ss3 = s3
            ss4 = s4
            ss5 = s5
            ss6 = s6
            ss7 = s7
            sz1 = z1
            sz2 = z2
            sz3 = z3
            sz11 = z11
            sz12 = z12
            sz13 = z13
            sz21 = z21
            sz22 = z22
            sz23 = z23
            sz31 = z31
            sz32 = z32
            sz33 = z33
            zcosg = zcosgl
            zsing = zsingl
            zcosi = zcosil
            zsini = zsinil
            zcosh = zcoshl * cnodm + zsinhl * snodm
            zsinh = snodm * zcoshl - cnodm * zsinhl
            cc = c1l

    zmol = _mod2pi(4.7199672 + 0.22997150 * day - gam)
    zmos = _mod2pi(6.2565837 + 0.017201977 * day)

    se2 = 2.0 * ss1 * ss6
    se3 = 2.0 * ss1 * ss7
    si2 = 2.0 * ss2 * sz12
    si3 = 2.0 * ss2 * (sz13 - sz11)
    sl2 = -2.0 * ss3 * sz2
    sl3 = -2.0 * ss3 * (sz3 - sz1)
    sl4 = -2.0 * ss3 * (-21.0 - 9.0 * emsq) * zes
    sgh2 = 2.0 * ss4 * sz32
    sgh3 = 2.0 * ss4 * (sz33 - sz31)
    sgh4 = -18.0 * ss4 * zes
    sh2 = -2.0 * ss2 * sz22
    sh3 = -2.0 * ss2 * (sz23 - sz21)

    ee2 = 2.0 * s1 * s6
    e3 = 2.0 * s1 * s7
    xi2 = 2.0 * s2 * z12
    xi3 = 2.0 * s2 * (z13 - z11)
    xl2 = -2.0 * s3 * z2
    xl3 = -2.0 * s3 * (z3 - z1)
    xl4 = -2.0 * s3 * (-21.0 - 9.0 * emsq) * zel
    xgh2 = 2.0 * s4 * z32
    xgh3 = 2.0 * s4 * (z33 - z31)
    xgh4 = -18.0 * s4 * zel
    xh2 = -2.0 * s2 * z22
    xh3 = -2.0 * s2 * (z23 - z21)

    return {
        "e3": e3,
        "ee2": ee2,
        "peo": peo,
        "pgho": pgho,
        "pho": pho,
        "pinco": pinco,
        "plo": plo,
        "se2": se2,
        "se3": se3,
        "sgh2": sgh2,
        "sgh3": sgh3,
        "sgh4": sgh4,
        "sh2": sh2,
        "sh3": sh3,
        "si2": si2,
        "si3": si3,
        "sl2": sl2,
        "sl3": sl3,
        "sl4": sl4,
        "xgh2": xgh2,
        "xgh3": xgh3,
        "xgh4": xgh4,
        "xh2": xh2,
        "xh3": xh3,
        "xi2": xi2,
        "xi3": xi3,
        "xl2": xl2,
        "xl3": xl3,
        "xl4": xl4,
        "zmol": zmol,
        "zmos": zmos,
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s4,
        "s5": s5,
        "ss1": ss1,
        "ss2": ss2,
        "ss3": ss3,
        "ss4": ss4,
        "ss5": ss5,
        "sz1": sz1,
        "sz3": sz3,
        "sz11": sz11,
        "sz13": sz13,
        "sz21": sz21,
        "sz23": sz23,
        "sz31": sz31,
        "sz33": sz33,
        "z1": z1,
        "z3": z3,
        "z11": z11,
        "z13": z13,
        "z21": z21,
        "z23": z23,
        "z31": z31,
        "z33": z33,
        "sinim": sinim,
        "cosim": cosim,
        "emsq": emsq,
        "em": em,
        "nm": nm,
    }


def dsinit(satellite, cosim, emsq, argpo, s1, s2, s3, s4, s5, sinim, ss1, ss2, ss3, ss4, ss5,
           sz1, sz3, sz11, sz13, sz21, sz23, sz31, sz33, z1, z3, z11, z13, z21, z23, z31, z33,
           ecco, eccsq, em, argpm, inclm, mm, nm, nodem, xpidot):
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

    irez = 0
    if _to_bool((nm > 0.0034906585) & (nm < 0.0052359877)):
        irez = 1
    if _to_bool((nm >= 8.26e-3) & (nm <= 9.24e-3) & (em >= 0.5)):
        irez = 2

    ses = ss1 * zns * ss5
    sis = ss2 * zns * (sz11 + sz13)
    sls = -zns * ss3 * (sz1 + sz3 - 14.0 - 6.0 * emsq)
    sghs = ss4 * zns * (sz31 + sz33 - 6.0)
    shs = -zns * ss2 * (sz21 + sz23)
    if _to_bool((inclm < 5.2359877e-2) | (inclm > np.pi - 5.2359877e-2)):
        shs = _to_tensor(0.0, shs)
    if _to_bool(sinim != 0.0):
        shs = shs / sinim
    sgs = sghs - cosim * shs

    dedt = ses + s1 * znl * s5
    didt = sis + s2 * znl * (z11 + z13)
    dmdt = sls - znl * s3 * (z1 + z3 - 14.0 - 6.0 * emsq)
    sghl = s4 * znl * (z31 + z33 - 6.0)
    shll = -znl * s2 * (z21 + z23)
    if _to_bool((inclm < 5.2359877e-2) | (inclm > np.pi - 5.2359877e-2)):
        shll = _to_tensor(0.0, shll)
    domdt = sgs + sghl
    dnodt = shs
    if _to_bool(sinim != 0.0):
        domdt = domdt - cosim / sinim * shll
        dnodt = dnodt + shll / sinim

    dndt = _to_tensor(0.0, em)
    em = em + dedt * satellite._t
    inclm = inclm + didt * satellite._t
    argpm = argpm + domdt * satellite._t
    nodem = nodem + dnodt * satellite._t
    mm = mm + dmdt * satellite._t

    d2201 = _to_tensor(0.0, em)
    d2211 = _to_tensor(0.0, em)
    d3210 = _to_tensor(0.0, em)
    d3222 = _to_tensor(0.0, em)
    d4410 = _to_tensor(0.0, em)
    d4422 = _to_tensor(0.0, em)
    d5220 = _to_tensor(0.0, em)
    d5232 = _to_tensor(0.0, em)
    d5421 = _to_tensor(0.0, em)
    d5433 = _to_tensor(0.0, em)
    del1 = _to_tensor(0.0, em)
    del2 = _to_tensor(0.0, em)
    del3 = _to_tensor(0.0, em)
    xlamo = _to_tensor(0.0, em)
    xfact = _to_tensor(0.0, em)

    if irez != 0:
        aonv = torch.pow(nm / satellite._xke, x2o3)

        if irez == 2:
            cosisq = cosim * cosim
            emo = em
            em = ecco
            emsqo = emsq
            emsq = eccsq
            eoc = em * emsq
            g201 = -0.306 - (em - 0.64) * 0.440

            if _to_bool(em <= 0.65):
                g211 = 3.616 - 13.2470 * em + 16.2900 * emsq
                g310 = -19.302 + 117.3900 * em - 228.4190 * emsq + 156.5910 * eoc
                g322 = -18.9068 + 109.7927 * em - 214.6334 * emsq + 146.5816 * eoc
                g410 = -41.122 + 242.6940 * em - 471.0940 * emsq + 313.9530 * eoc
                g422 = -146.407 + 841.8800 * em - 1629.014 * emsq + 1083.4350 * eoc
                g520 = -532.114 + 3017.977 * em - 5740.032 * emsq + 3708.2760 * eoc
            else:
                g211 = -72.099 + 331.819 * em - 508.738 * emsq + 266.724 * eoc
                g310 = -346.844 + 1582.851 * em - 2415.925 * emsq + 1246.113 * eoc
                g322 = -342.585 + 1554.908 * em - 2366.899 * emsq + 1215.972 * eoc
                g410 = -1052.797 + 4758.686 * em - 7193.992 * emsq + 3651.957 * eoc
                g422 = -3581.690 + 16178.110 * em - 24462.770 * emsq + 12422.520 * eoc
                if _to_bool(em > 0.715):
                    g520 = -5149.66 + 29936.92 * em - 54087.36 * emsq + 31324.56 * eoc
                else:
                    g520 = 1464.74 - 4664.75 * em + 3763.64 * emsq

            if _to_bool(em < 0.7):
                g533 = -919.22770 + 4988.6100 * em - 9064.7700 * emsq + 5542.21 * eoc
                g521 = -822.71072 + 4568.6173 * em - 8491.4146 * emsq + 5337.524 * eoc
                g532 = -853.66600 + 4690.2500 * em - 8624.7700 * emsq + 5341.4 * eoc
            else:
                g533 = -37995.780 + 161616.52 * em - 229838.20 * emsq + 109377.94 * eoc
                g521 = -51752.104 + 218913.95 * em - 309468.16 * emsq + 146349.42 * eoc
                g532 = -40023.880 + 170470.89 * em - 242699.48 * emsq + 115605.82 * eoc

            sini2 = sinim * sinim
            f220 = 0.75 * (1.0 + 2.0 * cosim + cosisq)
            f221 = 1.5 * sini2
            f321 = 1.875 * sinim * (1.0 - 2.0 * cosim - 3.0 * cosisq)
            f322 = -1.875 * sinim * (1.0 + 2.0 * cosim - 3.0 * cosisq)
            f441 = 35.0 * sini2 * f220
            f442 = 39.3750 * sini2 * sini2
            f522 = 9.84375 * sinim * (
                sini2 * (1.0 - 2.0 * cosim - 5.0 * cosisq)
                + 0.33333333 * (-2.0 + 4.0 * cosim + 6.0 * cosisq)
            )
            f523 = sinim * (
                4.92187512 * sini2 * (-2.0 - 4.0 * cosim + 10.0 * cosisq)
                + 6.56250012 * (1.0 + 2.0 * cosim - 3.0 * cosisq)
            )
            f542 = 29.53125 * sinim * (
                2.0 - 8.0 * cosim + cosisq * (-12.0 + 8.0 * cosim + 10.0 * cosisq)
            )
            f543 = 29.53125 * sinim * (
                -2.0 - 8.0 * cosim + cosisq * (12.0 + 8.0 * cosim - 10.0 * cosisq)
            )
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
            theta = _mod2pi(satellite._gsto + satellite._t * _to_tensor(4.37526908801129966e-3, em))
            xlamo = _mod2pi(satellite._mo + satellite._nodeo + satellite._nodeo - theta - theta)
            xfact = satellite._mdot + dmdt + 2.0 * (satellite._nodedot + dnodt - rptim) - satellite._no_unkozai
            em = emo
            emsq = emsqo

        if irez == 1:
            g200 = 1.0 + emsq * (-2.5 + 0.8125 * emsq)
            g310 = 1.0 + 2.0 * emsq
            g300 = 1.0 + emsq * (-6.0 + 6.60937 * emsq)
            f220 = 0.75 * (1.0 + cosim) * (1.0 + cosim)
            f311 = 0.9375 * sinim * sinim * (1.0 + 3.0 * cosim) - 0.75 * (1.0 + cosim)
            f330 = 1.0 + cosim
            f330 = 1.875 * f330 * f330 * f330
            del1 = 3.0 * nm * nm * aonv * aonv
            del2 = 2.0 * del1 * f220 * g200 * q22
            del3 = 3.0 * del1 * f330 * g300 * q33 * aonv
            del1 = del1 * f311 * g310 * q31 * aonv
            theta = _mod2pi(satellite._gsto + satellite._t * _to_tensor(4.37526908801129966e-3, em))
            xlamo = _mod2pi(satellite._mo + satellite._nodeo + satellite._argpo - theta)
            xfact = satellite._mdot + xpidot - rptim + dmdt + domdt + dnodt - satellite._no_unkozai

        xli = xlamo
        xni = satellite._no_unkozai
        atime = _to_tensor(0.0, em)
        nm = satellite._no_unkozai + dndt
    else:
        xli = _to_tensor(0.0, em)
        xni = satellite._no_unkozai
        atime = _to_tensor(0.0, em)

    return {
        "em": em,
        "argpm": argpm,
        "inclm": inclm,
        "mm": mm,
        "nm": nm,
        "nodem": nodem,
        "irez": irez,
        "atime": atime,
        "d2201": d2201,
        "d2211": d2211,
        "d3210": d3210,
        "d3222": d3222,
        "d4410": d4410,
        "d4422": d4422,
        "d5220": d5220,
        "d5232": d5232,
        "d5421": d5421,
        "d5433": d5433,
        "dedt": dedt,
        "didt": didt,
        "dmdt": dmdt,
        "dnodt": dnodt,
        "domdt": domdt,
        "del1": del1,
        "del2": del2,
        "del3": del3,
        "xfact": xfact,
        "xlamo": xlamo,
        "xli": xli,
        "xni": xni,
    }


def dspace(satellite, em, argpm, inclm, mm, nodem, nm):
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
    stepn = _to_tensor(-720.0, em)
    step2 = _to_tensor(259200.0, em)

    dndt = _to_tensor(0.0, em)
    theta = _mod2pi(satellite._gsto + satellite._t * rptim)
    em = em + satellite._dedt * satellite._t
    inclm = inclm + satellite._didt * satellite._t
    argpm = argpm + satellite._domdt * satellite._t
    nodem = nodem + satellite._dnodt * satellite._t
    mm = mm + satellite._dmdt * satellite._t

    ft = _to_tensor(0.0, em)
    if satellite._irez != 0:
        atime = satellite._atime
        xni = satellite._xni
        xli = satellite._xli

        if _to_bool((atime == 0.0) | (satellite._t * atime <= 0.0) | (torch.abs(satellite._t) < torch.abs(atime))):
            atime = _to_tensor(0.0, em)
            xni = satellite._no_unkozai
            xli = satellite._xlamo

        delt = stepp if _to_bool(satellite._t > 0.0) else stepn
        iretn = 381
        while iretn == 381:
            if satellite._irez != 2:
                xndt = (
                    satellite._del1 * torch.sin(xli - fasx2)
                    + satellite._del2 * torch.sin(2.0 * (xli - fasx4))
                    + satellite._del3 * torch.sin(3.0 * (xli - fasx6))
                )
                xldot = xni + satellite._xfact
                xnddt = (
                    satellite._del1 * torch.cos(xli - fasx2)
                    + 2.0 * satellite._del2 * torch.cos(2.0 * (xli - fasx4))
                    + 3.0 * satellite._del3 * torch.cos(3.0 * (xli - fasx6))
                )
                xnddt = xnddt * xldot
            else:
                xomi = satellite._argpo + satellite._argpdot * atime
                x2omi = xomi + xomi
                x2li = xli + xli
                xndt = (
                    satellite._d2201 * torch.sin(x2omi + xli - g22)
                    + satellite._d2211 * torch.sin(xli - g22)
                    + satellite._d3210 * torch.sin(xomi + xli - g32)
                    + satellite._d3222 * torch.sin(-xomi + xli - g32)
                    + satellite._d4410 * torch.sin(x2omi + x2li - g44)
                    + satellite._d4422 * torch.sin(x2li - g44)
                    + satellite._d5220 * torch.sin(xomi + xli - g52)
                    + satellite._d5232 * torch.sin(-xomi + xli - g52)
                    + satellite._d5421 * torch.sin(xomi + x2li - g54)
                    + satellite._d5433 * torch.sin(-xomi + x2li - g54)
                )
                xldot = xni + satellite._xfact
                xnddt = (
                    satellite._d2201 * torch.cos(x2omi + xli - g22)
                    + satellite._d2211 * torch.cos(xli - g22)
                    + satellite._d3210 * torch.cos(xomi + xli - g32)
                    + satellite._d3222 * torch.cos(-xomi + xli - g32)
                    + satellite._d5220 * torch.cos(xomi + xli - g52)
                    + satellite._d5232 * torch.cos(-xomi + xli - g52)
                    + 2.0
                    * (
                        satellite._d4410 * torch.cos(x2omi + x2li - g44)
                        + satellite._d4422 * torch.cos(x2li - g44)
                        + satellite._d5421 * torch.cos(xomi + x2li - g54)
                        + satellite._d5433 * torch.cos(-xomi + x2li - g54)
                    )
                )
                xnddt = xnddt * xldot

            if _to_bool(torch.abs(satellite._t - atime) >= stepp):
                iretn = 381
            else:
                ft = satellite._t - atime
                iretn = 0

            if iretn == 381:
                xli = xli + xldot * delt + xndt * step2
                xni = xni + xndt * delt + xnddt * step2
                atime = atime + delt

        nm = xni + xndt * ft + xnddt * ft * ft * 0.5
        xl = xli + xldot * ft + xndt * ft * ft * 0.5

        if satellite._irez != 1:
            mm = xl - 2.0 * nodem + 2.0 * theta
            dndt = nm - satellite._no_unkozai
        else:
            mm = xl - nodem - argpm + theta
            dndt = nm - satellite._no_unkozai

        nm = satellite._no_unkozai + dndt

        satellite._atime = atime
        satellite._xni = xni
        satellite._xli = xli

    return em, argpm, inclm, mm, nodem, nm
