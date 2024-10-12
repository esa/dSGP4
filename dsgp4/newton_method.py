import numpy as np
import torch
from .sgp4 import sgp4
from .sgp4init import sgp4init
from . import util
from .tle import TLE

def _propagate(x, tle_sat, tsince, gravity_constant_name="wgs-84"):
    from .sgp4init import sgp4init
    from .sgp4 import sgp4
    whichconst=util.get_gravity_constants(gravity_constant_name)
    sgp4init(whichconst=whichconst,
                        opsmode='i',
                        satn=tle_sat.satellite_catalog_number,
                        epoch=(tle_sat._jdsatepoch+tle_sat._jdsatepochF)-2433281.5,
                        xbstar=x[0],
                        xndot=x[1],
                        xnddot=x[2],
                        xecco=x[3],
                        xargpo=x[4],
                        xinclo=x[5],
                        xmo=x[6],
                        xno_kozai=x[7],
                        xnodeo=x[8],
                        satellite=tle_sat)
    state=sgp4(tle_sat, tsince*torch.ones(1,1))
    return state

def initial_guess(tle_0, time_mjd, target_state=None):
    """
    This method takes an initial TLE and the time at which we want to propagate it, and returns
    a set of parameter related to an initial guess useful to find the TLE observation that corresponds to the propagated state

    Parameters:
    ----------------
    tle_0 (``dsgp4.tle.TLE``): starting TLE at time
    new_date (``datetime.datetime``): new date of the TLE, as a datetime objec
    target_state (``torch.tensor``): a 2x3 tensor for the position and velocity of the state. If None, then this is computed 
                                            by propagating `tle_0` at the `new_date`.

    Returns:
    ----------------
    y0 (``torch.tensor``): initial guess for the TLE elements. In particular, y0 contains
                                 the following elements (see SGP4 for a thorough description of these parameters):
                                * y0[0]: bstar (B* parameter)
                                * y0[1]: ndot (mean motion first derivative)
                                * y0[2]: nddot (mean motion second derivative)
                                * y0[3]: ecco (eccentricity)
                                * y0[4]: argpo (argument of perigee)
                                * y0[5]: inclo (inclination)
                                * y0[6]: mo (mean anomaly)
                                * y0[7]: no_kozai (mean anomaly in certain units)
                                * y0[8]: nodeo (right ascension of the ascending node)
                        
                        target_state (``torch.tensor``): this expresses the cartesian state as position & velocity in km & km/s, at the propagated time.
                                            The objective of the Newton method is to find the TLE observation that corresponds to that, at the propagated
                                            time.
                                    
                                    new_tle (``dsgp4.tle.TLE``): TLE constructed with `y0`, in order to find the TLE that corresponds to `target_state`
                                    
                                    tle_elements_0 (``dict``): dictionary used to construct `new_tle`

    """
    new_date=util.from_mjd_to_datetime(time_mjd)
    if target_state is None:
        whichconst=util.get_gravity_constants("wgs-84")
        sgp4init(whichconst=whichconst,
                            opsmode=tle_0._opsmode,
                            satn=tle_0.satellite_catalog_number,
                            epoch=(tle_0._jdsatepoch+tle_0._jdsatepochF)-2433281.5,
                            xbstar=tle_0._bstar,
                            xndot=tle_0._ndot,
                            xnddot=tle_0._nddot,
                            xecco=tle_0._ecco,
                            xargpo=tle_0._argpo,
                            xinclo=tle_0._inclo,
                            xmo=tle_0._mo,
                            xno_kozai=tle_0._no_kozai,
                            xnodeo=tle_0._nodeo,
                            satellite=tle_0)
        tsince=(time_mjd-util.from_datetime_to_mjd(tle_0._epoch))*1440.
        x=tsince*torch.ones(1,1,requires_grad=True)#torch.rand(1,1, requires_grad=True)
        target_state=sgp4(tle_0, x)
    #print(target_state.size(), target_state)
    tle_elements_0=util.from_cartesian_to_tle_elements(target_state.detach().numpy()*1e3)
    #you need to transform it to the appropriate units
    xpdotp=1440.0 / (2.0 *np.pi)
    tle_elements_0['epoch_year']=new_date.year
    tle_elements_0['epoch_days']=util.from_datetime_to_fractional_day(new_date)
    #tle_elements_0['no_kozai']=tle_elements_0['mean_motion']/(np.pi/43200.0)/xpdotp
    #tle_elements_0['inclo']=tle_elements_0['inclination']
    #tle_elements_0['nodeo']=tle_elements_0['raan']
    #tle_elements_0['argpo']=tle_elements_0['argument_of_perigee']
    #tle_elements_0['mo']=tle_elements_0['mean_anomaly']
    #tle_elements_0['ecco']=tle_elements_0['eccentricity']
    tle_elements_0['mean_motion_first_derivative']=tle_0.mean_motion_first_derivative
    tle_elements_0['mean_motion_second_derivative']=tle_0.mean_motion_second_derivative
    tle_elements_0['classification']=tle_0.classification
    tle_elements_0['international_designator']=tle_0.international_designator
    tle_elements_0['satellite_catalog_number']=tle_0.satellite_catalog_number
    tle_elements_0['ephemeris_type']=tle_0.ephemeris_type
    tle_elements_0['element_number']=tle_0.element_number
    tle_elements_0['b_star']=tle_0.b_star
    tle_elements_0['revolution_number_at_epoch']=tle_0.revolution_number_at_epoch
    new_tle=TLE(tle_elements_0)
    y0=torch.tensor([new_tle._bstar,
                     new_tle._ndot,
                     new_tle._nddot,
                     new_tle._ecco,
                     new_tle._argpo,
                     new_tle._inclo,
                     new_tle._mo,
                     new_tle._no_kozai,
                     new_tle._nodeo,
                     0.],requires_grad=True)
    return y0, target_state, new_tle, tle_elements_0

def update_TLE(old_tle,y0):
    """
    This method takes a TLE and an initial guess, and returns a TLE updated accordingly. It
    is useful while doing Newton iterations.

    Parameters:
    ----------------
    old_tle (``dsgp4.tle.TLE``): TLE corresponding to previous guess
    y0 (``torch.tensor``): initial guess (see the docstrings for `initial_guess` to know the content of `y0`)

    Returns:
    ----------------
    new_tle (``dsgp4.tle.TLE``): updated TLE

    """
    tle_elements={}
    xpdotp=1440.0/(2.0 *np.pi)
    #I need to convert no_kozai to consistent units for what is expected by TLE class:
    no_kozai=float(y0[7])*xpdotp
    tle_elements['mean_motion']=no_kozai*np.pi/43200.0

    #the rest is consistent:
    tle_elements['b_star']=float(y0[0])
    tle_elements['raan']=float(y0[8])
    tle_elements['eccentricity']=float(y0[3])
    tle_elements['argument_of_perigee']=float(y0[4])
    tle_elements['inclination']=float(y0[5])
    tle_elements['mean_anomaly']=float(y0[6])

    #now the ones that stay the same:
    tle_elements['mean_motion_first_derivative']=old_tle.mean_motion_first_derivative
    tle_elements['mean_motion_second_derivative']=old_tle.mean_motion_second_derivative
    tle_elements['epoch_days']=old_tle.epoch_days
    tle_elements['epoch_year']=old_tle.epoch_year
    tle_elements['classification']=old_tle.classification
    tle_elements['satellite_catalog_number']=old_tle.satellite_catalog_number
    tle_elements['ephemeris_type']=old_tle.ephemeris_type
    tle_elements['international_designator']=old_tle.international_designator
    tle_elements['revolution_number_at_epoch']=old_tle.revolution_number_at_epoch
    tle_elements['element_number']=old_tle.element_number
    return TLE(tle_elements)

def newton_method(tle_0, time_mjd, target_state=None, new_tol=1e-12,max_iter=50):
    """
    This method performs Newton method starting from an initial TLE and a given propagation time. The objective 
    is to find a TLE that accurately reconstructs the propagated state, at observation time.

    Parameters:
    ----------------
    tle_0 (``dsgp4.tle.TLE``): starting TLE (i.e., TLE at a given initial time)
    time_mjd (``float``): time at which we want to propagate the TLE (in Modified Julian Date)
    target_state (``torch.tensor``): a 2x3 tensor for the position and velocity of the state. If None, then this is computed by propagating `tle_0` at the `new_date`.
    new_tol (``float``): tolerance for the Newton method
    max_iter (``int``): maximum number of iterations for the Newton method

    Returns:
    ----------------
    next_tle (``dsgp4.tle.TLE``): TLE corresponding to the propagated state
    y0 (``torch.tensor``): initial guess for the TLE elements. In particular, y0 contains the following elements (see SGP4 for a thorough description of these parameters):

    """
    i=0
    tol=1e9
    y0, state_target, next_tle, tle_elements_0=initial_guess(tle_0=tle_0,time_mjd=time_mjd,target_state=target_state)
    #print(y0,states)
    #newton iterations:
    while i<max_iter and tol>new_tol:
        #print(f"y0: {y0}")
        propagate=lambda x: _propagate(x,next_tle,(time_mjd-util.from_datetime_to_mjd(next_tle._epoch))*1440.)
        y1=util.clone_w_grad(y0)
        y2=util.clone_w_grad(y0)
        y3=util.clone_w_grad(y0)
        y4=util.clone_w_grad(y0)
        y5=util.clone_w_grad(y0)
        r_x=propagate(y0)[0][0]
        r_x.backward()
        gradient_rx=y0.grad
        r_y=propagate(y1)[0][1]
        r_y.backward()
        gradient_ry=y1.grad
        r_z=propagate(y2)[0][2]
        r_z.backward()
        gradient_rz=y2.grad
        v_x=propagate(y3)[1][0]
        v_x.backward()
        gradient_vx=y3.grad
        v_y=propagate(y4)[1][1]
        v_y.backward()
        gradient_vy=y4.grad
        v_z=propagate(y5)[1][2]
        v_z.backward()
        gradient_vz=y5.grad
        F=np.array([float(r_x)-float(state_target[0][0]),float(r_y)-float(state_target[0][1]), float(r_z)-float(state_target[0][2]), float(v_x)-float(state_target[1][0]), float(v_y)-float(state_target[1][1]), float(v_z)-float(state_target[1][2])])
        tol=np.linalg.norm(F)
        #print(f"|F|: {tol}")
        DF=np.stack((gradient_rx, gradient_ry, gradient_rz, gradient_vx, gradient_vy, gradient_vz))
        #we remove the first three columns (they are all zeros):
        DF=DF[:,3:]
        DF=DF[:,:-1]
        dY=-np.matmul(np.matmul(np.linalg.pinv(np.matmul(DF.T,DF)),DF.T),F)
        #we make sure the eccentricity does not go to negative values:
        if float(y0[3])+dY[0]<0:
            dY[0]=-float(y0[3])*0.9999
        dY=torch.tensor([0.,0.,0.]+list(dY)+[0.], requires_grad=True)
        if tol<new_tol:
            print(f"F(y): {np.linalg.norm(F)}")
            print(f"Solution found, at iter: {i}")
            return next_tle, y0#+dY
        else:
            #Newton update:
            #y0=y0+dY
            y0=torch.tensor([float(el1)+float(el2) for el1, el2 in zip(list(y0),list(dY))],requires_grad=True)
            next_tle=update_TLE(next_tle, y0)
        i+=1
    print("Solution not found, returning best found so far")
    print(f"F(y): {np.linalg.norm(F)}")
    return next_tle, y0
