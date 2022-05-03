import numpy
import torch
torch.set_default_dtype(torch.float64)

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
