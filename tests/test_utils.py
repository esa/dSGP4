import dsgp4
import torch
import sgp4
import sgp4.earth_gravity
import unittest
import numpy as np

class UtilTestCase(unittest.TestCase):
    def test_earth_gravity_constants(self):
        out=dsgp4.util.get_gravity_constants("wgs-72")
        tumin,mu,radiusearthkm,xke,j2,j3,j4,j3oj2=out
        self.assertAlmostEqual(float(tumin),sgp4.earth_gravity.wgs72.tumin,places=10)
        self.assertAlmostEqual(float(mu),sgp4.earth_gravity.wgs72.mu,places=10)
        self.assertAlmostEqual(float(radiusearthkm),sgp4.earth_gravity.wgs72.radiusearthkm,places=10)
        self.assertAlmostEqual(float(xke),sgp4.earth_gravity.wgs72.xke,places=10)
        self.assertAlmostEqual(float(j2),sgp4.earth_gravity.wgs72.j2,places=10)
        self.assertAlmostEqual(float(j3),sgp4.earth_gravity.wgs72.j3,places=10)
        self.assertAlmostEqual(float(j4),sgp4.earth_gravity.wgs72.j4,places=10)
        self.assertAlmostEqual(float(j3oj2),sgp4.earth_gravity.wgs72.j3oj2,places=10)

        out=dsgp4.util.get_gravity_constants("wgs-72old")
        tumin,mu,radiusearthkm,xke,j2,j3,j4,j3oj2=out
        self.assertAlmostEqual(float(tumin),sgp4.earth_gravity.wgs72old.tumin,places=10)
        self.assertAlmostEqual(float(mu),sgp4.earth_gravity.wgs72old.mu,places=10)
        self.assertAlmostEqual(float(radiusearthkm),sgp4.earth_gravity.wgs72old.radiusearthkm,places=10)
        self.assertAlmostEqual(float(xke),sgp4.earth_gravity.wgs72old.xke,places=10)
        self.assertAlmostEqual(float(j2),sgp4.earth_gravity.wgs72old.j2,places=10)
        self.assertAlmostEqual(float(j3),sgp4.earth_gravity.wgs72old.j3,places=10)
        self.assertAlmostEqual(float(j4),sgp4.earth_gravity.wgs72old.j4,places=10)
        self.assertAlmostEqual(float(j3oj2),sgp4.earth_gravity.wgs72old.j3oj2,places=10)

        out=dsgp4.util.get_gravity_constants("wgs-84")
        tumin,mu,radiusearthkm,xke,j2,j3,j4,j3oj2=out
        self.assertAlmostEqual(float(tumin),sgp4.earth_gravity.wgs84.tumin,places=10)
        self.assertAlmostEqual(float(mu),sgp4.earth_gravity.wgs84.mu,places=10)
        self.assertAlmostEqual(float(radiusearthkm),sgp4.earth_gravity.wgs84.radiusearthkm,places=10)
        self.assertAlmostEqual(float(xke),sgp4.earth_gravity.wgs84.xke,places=10)
        self.assertAlmostEqual(float(j2),sgp4.earth_gravity.wgs84.j2,places=10)
        self.assertAlmostEqual(float(j3),sgp4.earth_gravity.wgs84.j3,places=10)
        self.assertAlmostEqual(float(j4),sgp4.earth_gravity.wgs84.j4,places=10)
        self.assertAlmostEqual(float(j3oj2),sgp4.earth_gravity.wgs84.j3oj2,places=10)
    
    def test_keplerian_cartesian(self):
        lines=[]
        lines.append("0 COSMOS 2251 DEB")
        lines.append("1 34427U 93036RU  22068.94647328  .00008100  00000-0  11455-2 0  9999")
        lines.append("2 34427  74.0145 306.8269 0033346  13.0723 347.1308 14.76870515693886")
        tle=dsgp4.TLE(lines)
        dsgp4.initialize_tle(tle)
        # Extract the state vector from the dSGP4 output
        st=dsgp4.sgp4(tle,torch.tensor(0.0)).detach().numpy()*1e3  # Convert to meters and meters/second
        r_vec = st[0]  # Position vector in meters
        v_vec = st[1]  # Velocity vector in m/s
        # code to generate the poliastro outputs:
        #now let's retrieve the poliastro gravitational parameter of the Earth:
        mu = 398600441800000.0000000000000000#Earth.k.to(u.m**3 / u.s**2).value
        # Let's then convert Cartesian -> Keplerian using our function
        a, e, i, Omega, omega, M = dsgp4.util.from_cartesian_to_keplerian(r_vec, v_vec, mu)

        # # Use poliastro to compute the same transformations
        # # Create an orbit from Cartesian state vectors
        # orbit = Orbit.from_vectors(Earth, 
        #                             r_vec * u.m, 
        #                             v_vec * u.m / u.s)
        # # Extract Keplerian elements from poliastro
        # a_poliastro = orbit.a.to(u.m).value
        # e_poliastro = orbit.ecc.value
        # i_poliastro = orbit.inc.to(u.rad).value
        # Omega_poliastro = orbit.raan.to(u.rad).value
        # omega_poliastro = orbit.argp.to(u.rad).value
        # nu_poliastro = orbit.nu.to(u.rad).value  # True anomaly
        # # Compute mean anomaly (M) from true anomaly (ν)
        # if e < 1.0:  # Elliptical orbit
        #     E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu_poliastro / 2))
        #     M_poliastro = E - e * np.sin(E)
        # else:  # Hyperbolic orbit
        #     F = 2 * np.arctanh(np.sqrt((e - 1) / (e + 1)) * np.tanh(nu_poliastro / 2))
        #     M_poliastro = e * np.sinh(F) - F     
        # Omega_poliastro %= 2 * np.pi
        # omega_poliastro %= 2 * np.pi
        # M_poliastro %= 2 * np.pi
        a_poliastro=7023679.5817881366237998
        e_poliastro=0.0041649630912143
        i_poliastro=1.2919744331609118
        Omega_poliastro=5.3551396410293757
        omega_poliastro=0.4409272281996022
        M_poliastro=5.8458093777349722

        # Compare the results
        self.assertAlmostEqual(a, a_poliastro, places=6)
        self.assertAlmostEqual(e, e_poliastro, places=6)
        self.assertAlmostEqual(i, i_poliastro, places=6)
        self.assertAlmostEqual(Omega, Omega_poliastro, places=6)
        self.assertAlmostEqual(omega, omega_poliastro, places=6)
        self.assertAlmostEqual(M, M_poliastro, places=6)   
