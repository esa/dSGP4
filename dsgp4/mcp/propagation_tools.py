"""
MCP tools of the `propagation` domain: plain (non-ML) dSGP4 propagation of
single objects and batches, plus the state and time conversions that go with it.
"""
from typing import List, Optional

import numpy as np
import torch

from .. import util
from ..util import initialize_tle, propagate as propagate_state, propagate_batch as propagate_states_batch
from .common import (friendly_errors, MAX_TIME_POINTS, describe_element_set, epoch_mjd, label_of, parse_datetime,
                     parse_element_sets, parse_single_element_set, propagation_warnings,
                     resolve_times, states_to_entries)


def register(server):
    """Registers the tools of the `propagation` domain on the given `MCPServer`."""

    @server.tool()
    @friendly_errors
    def propagate(element_set: str,
                  minutes_since_epoch: Optional[List[float]] = None,
                  dates_utc: Optional[List[str]] = None,
                  gravity_constants: str = 'wgs-84') -> dict:
        """
        Propagate a single element set (a TLE or an OMM message) with dSGP4 and return
        the position (km) and velocity (km/s) in the TEME frame at the requested times.
        Times are given either as `minutes_since_epoch` (minutes from the element set
        epoch, the native SGP4 variable, negative values allowed) or as `dates_utc`
        (ISO 8601 UTC dates); provide exactly one of the two. At most 1000 time points
        per call. `gravity_constants` is one of 'wgs-72old', 'wgs-72', 'wgs-84'.
        """
        satellite = parse_single_element_set(element_set)
        tsince = resolve_times(satellite, minutes_since_epoch, dates_utc)
        initialize_tle(satellite, gravity_constant_name=gravity_constants)
        states = propagate_state(satellite, tsince)
        return {
            'satellite': label_of(satellite),
            'satellite_catalog_number': int(satellite.satellite_catalog_number),
            'frame': 'TEME',
            'units': {'position': 'km', 'velocity': 'km/s'},
            'states': states_to_entries(satellite, states, tsince),
            'warnings': propagation_warnings(satellite),
        }

    @server.tool()
    @friendly_errors
    def propagate_batch(element_sets: str,
                        minutes_since_epoch: Optional[List[float]] = None,
                        dates_utc: Optional[List[str]] = None) -> dict:
        """
        Propagate several element sets (TLEs or an OMM document with several messages)
        at once, in a single batched dSGP4 evaluation. Times are given either as
        `minutes_since_epoch` (one value per object, or a single value applied to all)
        or as `dates_utc` (ISO 8601 UTC dates, one per object or a single common date:
        with a common date every object is propagated to the same instant, regardless
        of its own epoch). Returns, per object, the TEME position (km) and velocity
        (km/s).
        """
        satellites = parse_element_sets(element_sets)
        if (minutes_since_epoch is None) == (dates_utc is None):
            raise ValueError("Provide exactly one of 'minutes_since_epoch' or 'dates_utc'.")
        if minutes_since_epoch is not None:
            values = [float(value) for value in minutes_since_epoch]
            if len(values) == 1:
                values = values * len(satellites)
            if len(values) != len(satellites):
                raise ValueError('Expecting one time per object ({}) or a single common value, '
                                 'while {} were provided.'.format(len(satellites), len(values)))
            tsinces = torch.tensor(values)
        else:
            dates = list(dates_utc)
            if len(dates) == 1:
                dates = dates * len(satellites)
            if len(dates) != len(satellites):
                raise ValueError('Expecting one date per object ({}) or a single common date, '
                                 'while {} were provided.'.format(len(satellites), len(dates)))
            mjds = [util.from_datetime_to_mjd(parse_datetime(date)) for date in dates]
            tsinces = torch.tensor([(mjd - epoch_mjd(satellite)) * 1440.0
                                    for satellite, mjd in zip(satellites, mjds)])
        if len(satellites) > MAX_TIME_POINTS:
            raise ValueError('Too many objects ({}): this tool accepts at most {} per call.'.format(
                len(satellites), MAX_TIME_POINTS))
        states = propagate_states_batch(satellites, tsinces, initialized=False)
        array = states.detach().numpy().reshape(-1, 2, 3)
        results = []
        for satellite, values, minutes in zip(satellites, array, tsinces.tolist()):
            results.append({
                'satellite': label_of(satellite),
                'satellite_catalog_number': int(satellite.satellite_catalog_number),
                'tsince_minutes': float(minutes),
                'date_utc': util.from_mjd_to_datetime(epoch_mjd(satellite) + minutes / 1440.0).isoformat(),
                'position_km': [float(value) for value in values[0]],
                'velocity_km_s': [float(value) for value in values[1]],
            })
        return {'frame': 'TEME', 'units': {'position': 'km', 'velocity': 'km/s'}, 'states': results}

    @server.tool()
    @friendly_errors
    def cartesian_to_keplerian(position_km: List[float], velocity_km_s: List[float]) -> dict:
        """
        Convert a Cartesian state (position in km, velocity in km/s, e.g. a TEME state
        returned by the propagation tools) into osculating Keplerian elements around
        the Earth (WGS-84 gravitational parameter): semi-major axis, eccentricity,
        inclination, RAAN, argument of perigee, mean anomaly, orbital period and
        perigee/apogee altitudes.
        """
        if len(position_km) != 3 or len(velocity_km_s) != 3:
            raise ValueError('Expecting two 3-vectors: position in km and velocity in km/s.')
        _, mu_earth, radius_earth_km, _, _, _, _, _ = util.get_gravity_constants('wgs-84')
        mu_earth = float(mu_earth) * 1e9
        elements = util.from_cartesian_to_keplerian(np.array(position_km, dtype=np.float64) * 1e3,
                                                    np.array(velocity_km_s, dtype=np.float64) * 1e3,
                                                    mu_earth)
        semi_major_axis_m, eccentricity = float(elements[0]), float(elements[1])
        radius_earth_m = float(radius_earth_km) * 1e3
        return {
            'semi_major_axis_km': semi_major_axis_m / 1e3,
            'eccentricity': eccentricity,
            'inclination_deg': float(np.rad2deg(elements[2])),
            'raan_deg': float(np.rad2deg(elements[3])),
            'argument_of_perigee_deg': float(np.rad2deg(elements[4])),
            'mean_anomaly_deg': float(np.rad2deg(elements[5])),
            'orbital_period_minutes': float(2.0 * np.pi * np.sqrt(semi_major_axis_m**3 / mu_earth) / 60.0),
            'perigee_altitude_km': (semi_major_axis_m * (1.0 - eccentricity) - radius_earth_m) / 1e3,
            'apogee_altitude_km': (semi_major_axis_m * (1.0 + eccentricity) - radius_earth_m) / 1e3,
        }

    @server.tool()
    @friendly_errors
    def convert_time(date_utc: Optional[str] = None,
                     mjd: Optional[float] = None,
                     jd: Optional[float] = None,
                     element_set: Optional[str] = None) -> dict:
        """
        Convert between the time scales used by the library: UTC dates (ISO 8601),
        Modified Julian Dates and Julian Dates. Provide exactly one of `date_utc`,
        `mjd` or `jd`; the other representations are returned. When an `element_set`
        (TLE or OMM) is also given, the corresponding `tsince_minutes` (minutes from
        its epoch, i.e. the propagation time the other tools expect) is included.
        """
        provided = [value for value in (date_utc, mjd, jd) if value is not None]
        if len(provided) != 1:
            raise ValueError("Provide exactly one of 'date_utc', 'mjd' or 'jd'.")
        if date_utc is not None:
            date = parse_datetime(date_utc)
        elif mjd is not None:
            date = util.from_mjd_to_datetime(float(mjd))
        else:
            date = util.from_jd_to_datetime(float(jd))
        mjd_value = util.from_datetime_to_mjd(date)
        result = {
            'date_utc': date.isoformat(),
            'mjd': mjd_value,
            'jd': mjd_value + 2400000.5,
            'day_of_year': util.from_datetime_to_fractional_day(date),
        }
        if element_set:
            satellite = parse_single_element_set(element_set)
            result['tsince_minutes'] = (mjd_value - epoch_mjd(satellite)) * 1440.0
            result['element_set_epoch_utc'] = satellite._epoch.isoformat()
        return result

    @server.tool()
    @friendly_errors
    def orbital_elements_summary(element_sets: str) -> dict:
        """
        Parse one or more element sets (TLEs or an OMM document) and return, for each
        object, the compact summary of its orbit: catalog number, name, epoch, orbit
        class, period, perigee/apogee altitude, inclination and eccentricity. Useful
        to survey a whole catalog file at a glance.
        """
        satellites = parse_element_sets(element_sets)
        summaries = []
        for satellite in satellites:
            description = describe_element_set(satellite)
            summaries.append({
                'satellite': label_of(satellite),
                'satellite_catalog_number': description['satellite_catalog_number'],
                'epoch_utc': description['epoch_utc'],
                'orbit_class': description['derived']['orbit_class'],
                'orbital_period_minutes': description['derived']['orbital_period_minutes'],
                'perigee_altitude_km': description['derived']['perigee_altitude_km'],
                'apogee_altitude_km': description['derived']['apogee_altitude_km'],
                'inclination_deg': description['elements']['inclination_deg'],
                'eccentricity': description['elements']['eccentricity'],
            })
        return {'number_of_objects': len(summaries), 'objects': summaries}
