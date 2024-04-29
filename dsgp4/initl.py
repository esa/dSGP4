import numpy
import torch

from . import util

def initl(
        xke, j2,
        ecco, epoch, inclo, no,
        method,
        opsmode,
):
    # Initialize a tensor for the fraction 2/3, used in Kepler's third law calculations.
    # Kepler third law is: The ratio of the square of an object's orbital period with the cube of the semi-major axis
    # of its orbit is the same for all objects orbiting the same primary.
    x2o3 = torch.tensor(2.0 / 3.0)

    # Calculate the square of the eccentricity to evaluate orbit shape and perturbation effects
    # ecco = eccentricity
    eccsq = ecco * ecco

    # Compute one minus the square of eccentricity. This is used to describe how much an object's orbit differs from
    # a perfect circle. If the eccentricity were zero, the value would be 1, IE: a perfect circle.
    omeosq = 1.0 - eccsq

    # Calculate the square root of (1 - eccsq), used in later orbital calculations. Ex: calculating orbital radius,
    # and other orbital geometry
    rteosq = omeosq.sqrt()

    # Compute the cosine of the inclination, determines the object's orientation which we can use later to calculate
    # how the object will move along the orbit
    cosio = inclo.cos()

    # Square the cosine of the inclination, used in perturbation calculations.
    cosio2 = cosio * cosio

    # Compute the semi-major axis (ak - half the diameter of an ellipse between its two furthest points) from Kepler's
    # third law, essential for orbit scaling.
    ak = torch.pow(xke / no, x2o3)

    # Calculate the first part of the drag term, important for orbital decay predictions. The equation effectively
    # calculates the drag induced by the "oblateness" of the earth. IE: the fact that it is flatter at the poles than
    # in the middle.
    # j2: represents the flattening of the Earth at the poles and its effect on the gravitational field.
    d1 = 0.75 * j2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq)

    # Estimate the perturbation factor to adjust the semi-major axis.
    del_ = d1 / (ak * ak)

    # Adjust the semi-major axis with the perturbation factor. This improves the orbit accuracy.
    adel = ak * (1.0 - del_ * del_ - del_ *
                 (1.0 / 3.0 + 134.0 * del_ * del_ / 81.0))

    # Recalculate the perturbation factor with the adjusted semi-major axis.
    del_ = d1 / (adel * adel)

    # Correct the mean motion with the recalculated perturbation
    no = no / (1.0 + del_)

    # Recompute the semi-major axis
    ao = torch.pow(xke / no, x2o3)

    # Calculate the sine of the inclination, used in orientation and perturbation computations.
    sinio = inclo.sin()

    # Compute the orbital period
    po = ao * omeosq

    # Define constants related to the inclination, used in gravitational perturbation corrections.
    con42 = 1.0 - 5.0 * cosio2
    con41 = -con42 - cosio2 - cosio2

    # Calculate the inverse of the semi-major axis, important for orbit adjustments.
    ainv = 1.0 / ao

    # Compute the square of the orbital period, used in timing and synchronization.
    posq = po * po

    # Determine the perigee radius, essential for closest approach calculations.
    rp = ao * (1.0 - ecco)

    # Set the method to 'n' - standard processing mode.
    method = 'n'

    if opsmode == 'a':
        # Perform calculations for the Greenwich Sidereal Time in alternative mode.
        ts70 = epoch - 7305.0
        ds70 = torch.floor_divide(ts70 + 1.0e-8, 1)
        tfrac = ts70 - ds70
        c1 = torch.tensor(1.72027916940703639e-2)
        thgr70 = torch.tensor(1.7321343856509374)
        fk5r = torch.tensor(5.07551419432269442e-15)
        c1p2p = c1 + (2 * numpy.pi)
        gsto = (thgr70 + c1 * ds70 + c1p2p * tfrac + ts70 * ts70 * fk5r) % (2 * numpy.pi)
        if gsto < 0.0:
            gsto += 2 * numpy.pi
    else:
        # Compute the Greenwich Sidereal Time for standard processing.
        gsto = util.gstime(epoch + 2433281.5)

    # Return a tuple of updated orbital elements and computed variables
    return (
        no,
        method,
        ainv, ao, con41, con42, cosio,
        cosio2, eccsq, omeosq, posq,
        rp, rteosq, sinio, gsto,
    )
