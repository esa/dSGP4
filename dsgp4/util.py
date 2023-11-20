import datetime
import numpy as np
import torch

#torch.set_default_dtype(torch.float64)

def get_gravity_constants(gravity_constant_name):
    if gravity_constant_name == 'wgs-72old':
        mu     = 398600.79964        #  in km3 / s2
        radiusearthkm = 6378.135     #  km
        xke    = 0.0743669161
        tumin  = 1.0 / xke
        j2     =   0.001082616
        j3     =  -0.00000253881
        j4     =  -0.00000165597
        j3oj2  =  j3 / j2
    elif gravity_constant_name == 'wgs-72':
       mu     = 398600.8            #  in km3 / s2
       radiusearthkm = 6378.135     #  km
       xke    = 60.0 / np.sqrt(radiusearthkm*radiusearthkm*radiusearthkm/mu)
       tumin  = 1.0 / xke
       j2     =   0.001082616
       j3     =  -0.00000253881
       j4     =  -0.00000165597
       j3oj2  =  j3 / j2
    elif gravity_constant_name=="wgs-84":
       mu     = 398600.5            #  in km3 / s2
       radiusearthkm = 6378.137     #  km
       xke    = 60.0 / np.sqrt(radiusearthkm*radiusearthkm*radiusearthkm/mu)
       tumin  = 1.0 / xke
       j2     =   0.00108262998905
       j3     =  -0.00000253215306
       j4     =  -0.00000161098761
       j3oj2  =  j3 / j2
    else:
       raise RuntimeError("Supported gravity constant names: wgs-72, wgs-84, wgs-72old while "+gravity_constant_name+" was provided")

    return torch.tensor(tumin), torch.tensor(mu), torch.tensor(radiusearthkm), torch.tensor(xke), torch.tensor(j2), torch.tensor(j3), torch.tensor(j4), torch.tensor(j3oj2)

def propagate(x, tle_sat, tsince, gravity_constant_name="wgs-84"):
    """
    This function takes a tensor of inputs and a TLE, and returns the corresponding state.
    It can be used to take the gradient of the state w.r.t. the inputs.

    Args:
        - x (``torch.tensor``): input of tensors, with the following values (x[0:9] have the same units as the ones in the TLE):
                                    - x[0]: bstar
                                    - x[1]: ndot
                                    - x[2]: nddot
                                    - x[3]: ecco
                                    - x[4]: argpo
                                    - x[5]: inclo
                                    - x[6]: mo
                                    - x[7]: kozai
                                    - x[8]: nodeo
        - tle_sat (``dsgp4.tle.TLE``): TLE object to be propagated
        - tsince (``float``): propagation time in minutes

    Returns:
        - state (``torch.tensor``): (2x3) tensor representing position and velocity in km and km/s.
    """
    from .sgp4init import sgp4init
    from .sgp4 import sgp4
    whichconst=get_gravity_constants(gravity_constant_name)
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

def from_year_day_to_date(y,d):
    return (datetime.datetime(y, 1, 1) + datetime.timedelta(d - 1))

def gstime(jdut1):
    deg2rad=np.pi/180.
    tut1 = (jdut1 - 2451545.0) / 36525.0
    temp = -6.2e-6* tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 + \
         (876600.0*3600 + 8640184.812866) * tut1 + 67310.54841  #  sec
    temp = (temp*(np.pi/180.0) / 240.0) % (2*np.pi) # 360/86400 = 1/240, to deg, to rad

     #  ------------------------ check quadrants ---------------------
    temp=torch.where(temp<0., temp+(2*np.pi), temp)
    return temp

def clone_w_grad(y):
    return y.clone().detach().requires_grad_(True)

def jday(year, mon, day, hr, minute, sec):
    """
    Converts a date and time to a Julian Date. The Julian Date is the number of days since noon on January 1st, 4713 BC.

    Args:
        year (`int`): year
        mon (`int`): month
        day (`int`): day
        hr (`int`): hour
        minute (`int`): minute
        sec (`float`): second

    Returns:
        `float`: Julian Date
    """
    jd=(367.0 * year -
            7.0 * (year + ((mon + 9.0) // 12.0)) * 0.25 // 1.0 +
            275.0 * mon // 9.0 +
            day + 1721013.5)
    fr=(sec + minute * 60.0 + hr * 3600.0) / 86400.0
    return jd,fr

def invjday(jd):
    """
    Converts a Julian Date to a date and time. The Julian Date is the number of days since noon on January 1st, 4713 BC.

    Args:
        jd (`float`): Julian Date

    Returns:
        `tuple`: (year, month, day, hour, minute, second)
    """
    temp    = jd - 2415019.5
    tu      = temp / 365.25
    year    = 1900 + int(tu // 1.0)
    leapyrs = int(((year - 1901) * 0.25) // 1.0)
    days    = temp - ((year - 1900) * 365.0 + leapyrs) + 0.00000000001
    if (days < 1.0):
        year    = year - 1
        leapyrs = int(((year - 1901) * 0.25) // 1.0)
        days    = temp - ((year - 1900) * 365.0 + leapyrs)
    mon, day, hr, minute, sec = days2mdhms(year, days)
    sec = sec - 0.00000086400
    return year, mon, day, hr, minute, sec

def days2mdhms(year, fractional_day):
    """
    Converts a number of days to months, days, hours, minutes, and seconds.

    Args:
        year (`int`): year
        fractional_day (`float`): number of days

    Returns:
        `tuple`: (month, day, hour, minute, second)
    """
    d=datetime.timedelta(days=fractional_day)
    datetime_obj=datetime.datetime(year-1,12,31)+d
    return datetime_obj.month, datetime_obj.day, datetime_obj.hour, datetime_obj.minute, datetime_obj.second+datetime_obj.microsecond/1e6

def from_string_to_datetime(string):
    """
    Converts a string to a datetime object.

    Args:
        string (`str`): string to convert

    Returns:
        `datetime.datetime`: datetime object
    """
    if string.find('.')!=-1:
        return datetime.datetime.strptime(string, '%Y-%m-%d %H:%M:%S.%f')
    else:
        return datetime.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')

def from_mjd_to_epoch_days_after_1_jan(mjd_date):
    """
    Converts a Modified Julian Date to the number of days after 1 Jan 2000.

    Args:
        mjd_date (`float`): Modified Julian Date

    Returns:
        `float`: number of days after 1 Jan 2000
    """
    d = from_mjd_to_datetime(mjd_date)
    dd = d - datetime.datetime(d.year-1, 12, 31)
    days = dd.days
    days_fraction = (dd.seconds + dd.microseconds/1e6) / (60*60*24)
    return days + days_fraction

def from_mjd_to_datetime(mjd_date):
    """
    Converts a Modified Julian Date to a datetime object. The Modified Julian Date is the number of days since midnight on November 17, 1858.

    Args:
        mjd_date (`float`): Modified Julian Date

    Returns:
        `datetime.datetime`: datetime object
    """
    jd_date=mjd_date+2400000.5
    return from_jd_to_datetime(jd_date)

def from_jd_to_datetime(jd_date):
    """
    Converts a Julian Date to a datetime object. The Julian Date is the number of days since noon on January 1st, 4713 BC.

    Args:
        jd_date (`float`): Julian Date

    Returns:
        `datetime.datetime`: datetime object
    """
    year, month, day, hour, minute, seconds=invjday(jd_date)
    e_1=datetime.datetime(year=int(year), month=int(month), day=int(day), hour=int(hour), minute=int(minute), second=0)
    return e_1+datetime.timedelta(seconds=seconds)

def get_non_empty_lines(lines):
    """
    This function returns the non-empty lines of a list of lines.

    Args:
        lines (`list`): list of lines

    Returns:
        `list`: non-empty lines
    """
    if not isinstance(lines, str):
        raise ValueError('Expecting a string')
    lines = lines.splitlines()
    lines = [line for line in lines if line.strip()]
    return lines

def from_datetime_to_fractional_day(datetime_object):
    """
    Converts a datetime object to a fractional day. The fractional day is the number of days since the beginning of the year. For example, January 1st is 0.0, January 2nd is 1.0, etc.

    Args:
        datetime_object (`datetime.datetime`): datetime object to convert

    Returns:
        `float`: fractional day
    """
    d = datetime_object-datetime.datetime(datetime_object.year-1, 12, 31)
    fractional_day = d.days + d.seconds/60./60./24 + d.microseconds/60./60./24./1e6
    return fractional_day

def from_datetime_to_mjd(datetime_obj):
    """
    Converts a datetime object to a Modified Julian Date. The Modified Julian Date is the number of days since midnight on November 17, 1858.

    Args:
        datetime_obj (`datetime.datetime`): datetime object to convert

    Returns:
        `float`: Modified Julian Date
    """
    return from_datetime_to_jd(datetime_obj)-2400000.5

def from_datetime_to_jd(datetime_obj):
    """
    Converts a datetime object to a Julian Date. The Julian Date is the number of days since noon on January 1, 4713 BC.

    Args:
        datetime_obj (`datetime.datetime`): datetime object to convert

    Returns:
        `float`: Julian Date
    """
    return sum(jday(year=datetime_obj.year, mon=datetime_obj.month, day=datetime_obj.day, hr=datetime_obj.hour, minute=datetime_obj.minute, sec=datetime_obj.second+float('0.'+str(datetime_obj.microsecond))))

def from_cartesian_to_tle_elements(state, gravity_constant_name='wgs-72'):
    """
    This function converts the provided state from Cartesian to TLE elements.

    Args:
        state (`np.ndarray`): state to convert
        gravity_constant_name (`str`): name of the central body (default: 'wgs-72')

    Returns:
        tuple: tuple containing: - `float`: semi-major axis - `float`: eccentricity - `float`: inclination - `float`: right ascension of the ascending node - `float`: argument of perigee - `float`: mean anomaly
    """
    _,mu_earth,_,_,_,_,_,_=get_gravity_constants(gravity_constant_name)
    mu_earth=float(mu_earth)*1e9
    kepl_el = from_cartesian_to_keplerian(state, mu_earth)
    tle_elements={}
    tle_elements['mean_motion']         = np.sqrt(mu_earth/((kepl_el[0])**(3.0)))
    tle_elements['eccentricity']        = kepl_el[1]
    tle_elements['inclination']         = kepl_el[2]
    tle_elements['raan']                = kepl_el[3]
    tle_elements['argument_of_perigee'] = kepl_el[4]
    mean_anomaly = kepl_el[5] - kepl_el[1]*np.sin(kepl_el[5])
    tle_elements['mean_anomaly']        = mean_anomaly%(2*np.pi)
    return tle_elements

def from_cartesian_to_keplerian(state, mu):
    """
    This function takes the state in cartesian coordinates and the gravitational
    parameter of the central body, and returns the state in Keplerian elements.

    Args:
        state (`np.array`): numpy array of 2 rows and 3 columns, where
                                    the first row represents position, and the second velocity.
        mu (`float`): gravitational parameter of the central body

    Returns:
        `np.array`: numpy array of the six keplerian elements: (a,e,i,omega,Omega,mean_anomaly)
                                             (i.e., semi major axis, eccentricity, inclination,
                                             right ascension of ascending node, argument of perigee,
                                             mean anomaly). All the angles are in radiants, eccentricity is unitless
                                             and semi major axis is in SI.
    """
    h_bar = np.cross(np.array([state[0,0], state[0,1], state[0,2]]), np.array([state[1,0], state[1,1], state[1,2]]))
    h = np.linalg.norm(h_bar)
    r = np.linalg.norm(np.array([state[0,0], state[0,1], state[0,2]]))
    v = np.linalg.norm(np.array([state[1,0], state[1,1], state[1,2]]))
    E = 0.5*(v**2)-mu/r
    a = -mu/(2*E)
    e = np.sqrt(1-(h**2)/(a*mu))
    i = np.arccos(h_bar[2]/h)
    Omega = np.arctan2(h_bar[0],-h_bar[1])

    lat = np.arctan2(np.divide(state[0,2],(np.sin(i))), (state[0,0]*np.cos(Omega) + state[0,1]*np.sin(Omega)))
    p = a*(1-e**2)
    nu = np.arctan2(np.sqrt(p/mu)*np.dot(np.array([state[0,0], state[0,1], state[0,2]]),np.array([state[1,0], state[1,1], state[1,2]])), p-r)
    omega = (lat-nu)
    eccentric_anomaly = 2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(nu/2))
    n = np.sqrt(mu/(a**3))
    mean_anomaly=eccentric_anomaly-e*np.sin(eccentric_anomaly)
    #I make sure they are always in 0,2pi
    if mean_anomaly<0:
        mean_anomaly = 2*np.pi-abs(mean_anomaly)
    if omega<0:
        omega=2*np.pi-abs(omega)
    if Omega<0:
        Omega=2*np.pi-abs(Omega)
    if abs(mean_anomaly)>2*np.pi:
        mean_anomaly=mean_anomaly%(2*np.pi)
    if abs(omega)>2*np.pi:
        omega=omega%(2*np.pi)
    if abs(Omega)>2*np.pi:
        Omega=Omega%(2*np.pi)
    return np.array([a, e, i, Omega, omega, mean_anomaly])

def from_cartesian_to_keplerian_torch(state, mu):
    """
    Same as from_cartesian_to_keplerian, but for torch tensors.

    Args:
        state (`np.array`): numpy array of 2 rows and 3 columns, where
                                    the first row represents position, and the second velocity.
        mu (`float`): gravitational parameter of the central body

    Returns:
        `np.array`: numpy array of the six keplerian elements: (a,e,i,omega,Omega,mean_anomaly)
                                             (i.e., semi major axis, eccentricity, inclination,
                                             right ascension of ascending node, argument of perigee,
                                             mean anomaly). All the angles are in radiants, eccentricity is unitless
                                             and semi major axis is in SI.
    """
    h_bar = torch.cross(state[0], state[1])
    h = h_bar.norm()
    r = state[0].norm()
    v = torch.norm(state[1])
    E = 0.5*(v**2)-mu/r
    a = -mu/(2*E)
    e = torch.sqrt(1-(h**2)/(a*mu))
    i = torch.arccos(h_bar[2]/h)
    Omega = torch.arctan2(h_bar[0],-h_bar[1])
    lat = torch.arctan2(torch.divide(state[0,2],(torch.sin(i))), (state[0,0]*torch.cos(Omega) + state[0,1]*torch.sin(Omega)))
    p = a*(1-e**2)
    nu = torch.arctan2(torch.sqrt(p/mu)*torch.dot(state[0],state[1]), p-r)
    omega = (lat-nu)
    eccentric_anomaly = 2*torch.arctan(torch.sqrt((1-e)/(1+e))*torch.tan(nu/2))
    n = torch.sqrt(mu/(a**3))
    mean_anomaly=eccentric_anomaly-e*torch.sin(eccentric_anomaly)
    #I make sure they are always in 0,2pi
    mean_motion=torch.sqrt(mu/((a)**(3.0)))
    xpdotp   =  1440.0 / (2.0 *np.pi)
    no_kozai_conversion_factor=xpdotp/43200.0* np.pi
    no_kozai=mean_motion/no_kozai_conversion_factor
    return [no_kozai, e, i, Omega, omega, mean_anomaly]
