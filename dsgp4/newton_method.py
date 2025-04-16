import numpy as np
import torch
from .sgp4 import sgp4
from .sgp4init import sgp4init
from . import util
from .tle import TLE

def update_TLE(old_tle, y0):
    xpdotp = 1440.0 / (2.0 * np.pi)
    mean_motion = float(y0[4]) * xpdotp * (np.pi / 43200.0)

    tle_elements = {
        'b_star': old_tle._bstar,
        'raan': float(y0[5]),
        'eccentricity': float(y0[0]),
        'argument_of_perigee': float(y0[1]),
        'inclination': float(y0[2]),
        'mean_anomaly': float(y0[3]),
        'mean_motion': mean_motion,
        'mean_motion_first_derivative': old_tle.mean_motion_first_derivative,
        'mean_motion_second_derivative': old_tle.mean_motion_second_derivative,
        'epoch_days': old_tle.epoch_days,
        'epoch_year': old_tle.epoch_year,
        'classification': old_tle.classification,
        'satellite_catalog_number': old_tle.satellite_catalog_number,
        'ephemeris_type': old_tle.ephemeris_type,
        'international_designator': old_tle.international_designator,
        'revolution_number_at_epoch': old_tle.revolution_number_at_epoch,
        'element_number': old_tle.element_number,
    }

    return TLE(tle_elements)

def initial_guess_tle(time_mjd, tle_object, gravity_constant_name="wgs-84"):
    #first, let's decompose the time into -> epoch of the year and days 
    datetime_obj=util.from_mjd_to_datetime(time_mjd)
    epoch_days=util.from_datetime_to_fractional_day(datetime_obj)
    #then we need to propagate the state, and extract the keplerian elements:
    util.initialize_tle(tle_object)
    tsince=(time_mjd-util.from_datetime_to_mjd(tle_object._epoch))*1440.
    target_state=util.propagate(tle_object, tsince).detach().numpy()*1e3
    _,mu_earth,_,_,_,_,_,_=util.get_gravity_constants(gravity_constant_name)
    mu_earth=float(mu_earth)*1e9
    kepl_el=util.from_cartesian_to_keplerian(target_state[0],target_state[1],mu_earth)
    #we need to convert the keplerian elements to TLE elements:
    data = dict(
                satellite_catalog_number=tle_object.satellite_catalog_number,
                classification=tle_object.classification,
                international_designator=tle_object.international_designator,
                epoch_year=datetime_obj.year,
                epoch_days=epoch_days,
                ephemeris_type=tle_object.ephemeris_type,
                element_number=tle_object.element_number,
                revolution_number_at_epoch=tle_object.revolution_number_at_epoch,
                mean_motion=np.sqrt(mu_earth/((kepl_el[0])**(3.0))),
                mean_motion_first_derivative=tle_object.mean_motion_first_derivative,
                mean_motion_second_derivative=tle_object.mean_motion_second_derivative,
                eccentricity=kepl_el[1],
                inclination=kepl_el[2],
                argument_of_perigee=kepl_el[4],
                raan=kepl_el[3],
                mean_anomaly=kepl_el[5],
                b_star=tle_object.b_star)
    return TLE(data)

def _propagate(x, tle_sat, tsince, gravity_constant_name="wgs-84"):
    whichconst=util.get_gravity_constants(gravity_constant_name)
    sgp4init(whichconst=whichconst,
                        opsmode='i',
                        satn=tle_sat.satellite_catalog_number,
                        epoch=(tle_sat._jdsatepoch+tle_sat._jdsatepochF)-2433281.5,
                        xbstar=tle_sat._bstar,
                        xndot=tle_sat._ndot,
                        xnddot=tle_sat._nddot,
                        xecco=x[0],
                        xargpo=x[1],
                        xinclo=x[2],
                        xmo=x[3],
                        xno_kozai=x[4],
                        xnodeo=x[5],
                        satellite=tle_sat)
    state=sgp4(tle_sat, tsince*torch.ones(1,1))
    return state

def newton_method(tle0, time_mjd, max_iter=50, new_tol=1e-12, verbose=False, target_state=None):
    if target_state is None:
        util.initialize_tle(tle0)
        target_state=util.propagate(tle0, (time_mjd-util.from_datetime_to_mjd(tle0._epoch))*1440.)

    i,tol=0,1e9
    next_tle=initial_guess_tle(time_mjd, tle0)
    y0 = torch.tensor([
                        next_tle._ecco,
                        next_tle._argpo,
                        next_tle._inclo,
                        next_tle._mo,
                        next_tle._no_kozai,
                        next_tle._nodeo,
                    ], requires_grad=True)
    def propagate_fn(x):
        tsince=(time_mjd-util.from_datetime_to_mjd(next_tle._epoch))*1440.
        return _propagate(x,next_tle,tsince)
    while i<max_iter and tol>new_tol:
        grads=[]
        F=[]
        for idx, (row,col) in enumerate([(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]):
            y=util.clone_w_grad(y0)
            val=propagate_fn(y)[row][col]
            val.backward()
            grads.append(y.grad)
            F.append((val-target_state[row][col]).item())
        tol=np.linalg.norm(F)
        if tol<new_tol:
            if verbose:
                print(f'Solution F(y) = {tol}, converged in {i} iterations')
            return next_tle, y0
        DF=np.stack(grads)
        #dY=-np.linalg.pinv(DF.T@DF)@DF.T@F
        dY=np.linalg.solve(DF, -np.array(F))
        dY=dY#/np.linalg.norm(dY)
        #avoid negative eccentricity:
        if y0[0]+dY[0]<0:
            dY[0]=1e-10
        if y0[0]+dY[0]>1.:
            dY[0]=1-1e-10
        dY=torch.tensor(dY, requires_grad=True)
        #update the state:
        y0 = torch.tensor([float(a) + float(b) for a, b in zip(y0, dY)], requires_grad=True)
        next_tle = update_TLE(next_tle, y0)
        i+=1
    if verbose:
        print("Solution not found, returning best found so far")
        print(f"F(y): {tol:.3e}")
    return next_tle, y0