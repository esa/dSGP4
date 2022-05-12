import numpy
import torch

torch.set_default_dtype(torch.float64)

def propagate(x, tle_sat):
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
                                    - x[9]: propagation time, in minutes
        - tle_sat (``kessler.tle.TLE``): TLE object to be propagated

    Returns:
        - state (``torch.tensor``): (2x3) tensor representing position and velocity in km and km/s.
    """
    whichconst=get_gravity_constants("wgs-72")
    from .sgp4init import sgp4init
    from .sgp4 import sgp4
    sgp4init(whichconst=whichconst,
                        opsmode='i',
                        satn=tle_sat.satellite_catalog_number,
                        epoch=(tle_sat._jdsatepoch+tle_sat._jdsatepochF)-2433281.,
                        xbstar=x[0],
                        xndot=x[1],
                        xnddot=x[2],
                        xecco=x[3],
                        xargpo=x[4],
                        xinclo=x[5],
                        xmo=x[6],
                        xno_kozai=x[7],
                        xnodeo=x[8],
                        satrec=tle_sat)
    state=sgp4(tle_sat, x[9]*torch.ones(1,1))
    return state

def gstime(jdut1):
    deg2rad=numpy.pi/180.
    tut1 = (jdut1 - 2451545.0) / 36525.0;
    temp = -6.2e-6* tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 + \
         (876600.0*3600 + 8640184.812866) * tut1 + 67310.54841;  #  sec
    temp = (temp*(numpy.pi/180.0) / 240.0) % (2*numpy.pi) # 360/86400 = 1/240, to deg, to rad

     #  ------------------------ check quadrants ---------------------
    temp=torch.where(temp<0., temp+(2*numpy.pi), temp)
    return temp;

def get_gravity_constants(gravity_constant_name):
    if gravity_constant_name == 'wgs-72old':
        mu     = 398600.79964;        #  in km3 / s2
        radiusearthkm = 6378.135;     #  km
        xke    = 0.0743669161;
        tumin  = 1.0 / xke;
        j2     =   0.001082616;
        j3     =  -0.00000253881;
        j4     =  -0.00000165597;
        j3oj2  =  j3 / j2;
    elif gravity_constant_name == 'wgs-72':
       mu     = 398600.8;            #  in km3 / s2
       radiusearthkm = 6378.135;     #  km
       xke    = 60.0 / numpy.sqrt(radiusearthkm*radiusearthkm*radiusearthkm/mu);
       tumin  = 1.0 / xke;
       j2     =   0.001082616;
       j3     =  -0.00000253881;
       j4     =  -0.00000165597;
       j3oj2  =  j3 / j2;
    elif gravity_constant_name=="wgs-84":
       mu     = 398600.5;            #  in km3 / s2
       radiusearthkm = 6378.137;     #  km
       xke    = 60.0 / numpy.sqrt(radiusearthkm*radiusearthkm*radiusearthkm/mu);
       tumin  = 1.0 / xke;
       j2     =   0.00108262998905;
       j3     =  -0.00000253215306;
       j4     =  -0.00000161098761;
       j3oj2  =  j3 / j2;
    else:
       raise RuntimeError("Supported gravity constant names: wgs-72, wgs-84, wgs-72old; while "+gravity_constant_name+" was provided")

    return torch.tensor(tumin), torch.tensor(mu), torch.tensor(radiusearthkm), torch.tensor(xke), torch.tensor(j2), torch.tensor(j3), torch.tensor(j4), torch.tensor(j3oj2)
