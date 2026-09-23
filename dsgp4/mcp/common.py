"""
Shared helpers for the dSGP4 MCP server.

All the tools of the server accept satellites in the form of an "element set" string,
which can either be a TLE (two or three lines) or a CCSDS OMM message (JSON, XML, KVN
or CSV, as supported by `dsgp4.omm`). Times are accepted either as minutes since the
element set epoch (the native SGP4 independent variable) or as UTC dates in ISO 8601
format; the helpers in this module resolve both to the tensors the propagator expects.
"""
import datetime
import functools
from typing import List, Optional

import numpy as np
import torch

from mcp.server.mcpserver.exceptions import ToolError

from .. import omm as omm_module
from .. import util
from ..tle import TLE

#largest number of time points a tool returns as numbers (plots use MAX_PLOT_POINTS):
MAX_TIME_POINTS = 1000
MAX_PLOT_POINTS = 20000
#largest number of time points for the Jacobian tools (each point costs six backward passes):
MAX_GRADIENT_POINTS = 20

#the nine parameters the SGP4 state is differentiated against, in the order used by
#`dsgp4.initialize_tle` (i.e. the order of the columns of the Jacobians), with the
#internal SGP4 units in which the derivatives are taken:
TLE_PARAMETERS = [
    'b_star',
    'mean_motion_first_derivative',
    'mean_motion_second_derivative',
    'eccentricity',
    'argument_of_perigee',
    'inclination',
    'mean_anomaly',
    'mean_motion',
    'raan',
]
TLE_PARAMETER_UNITS = {
    'b_star': '1/earth_radii',
    'mean_motion_first_derivative': 'rad/min**2',
    'mean_motion_second_derivative': 'rad/min**3',
    'eccentricity': '-',
    'argument_of_perigee': 'rad',
    'inclination': 'rad',
    'mean_anomaly': 'rad',
    'mean_motion': 'rad/min',
    'raan': 'rad',
}
STATE_COMPONENTS = ['x_km', 'y_km', 'z_km', 'vx_km_s', 'vy_km_s', 'vz_km_s']

#SGP4 error codes (`satellite._error` after a propagation):
SGP4_ERRORS = {
    1: 'mean eccentricity out of range',
    2: 'mean motion less than 0.0',
    3: 'perturbed eccentricity out of range',
    4: 'semi-latus rectum lower than 0.0',
    5: 'epoch elements are sub-orbital',
    6: 'satellite has decayed (radius below the surface of the Earth)',
}


def friendly_errors(fn):
    """
    Decorator for the MCP tools: converts the exceptions the library legitimately
    raises on bad inputs into `ToolError`s, whose message reaches the client (any
    other exception is masked by the server as a generic internal error).
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except (ValueError, RuntimeError, TypeError, AttributeError, KeyError, IndexError, OSError) as error:
            raise ToolError(str(error)) from error
    return wrapper


def parse_element_sets(text: str):
    """
    Parses a string carrying one or more element sets (TLEs, or an OMM document in
    JSON/XML/KVN/CSV format) and returns the corresponding list of `dsgp4.tle.TLE`
    (or `dsgp4.omm.OMM`) objects.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Empty element set: expecting a TLE (two or three lines) or an OMM document.')
    try:
        return _parse_tles(text)
    except ValueError as tle_error:
        try:
            records = omm_module.loads(text)
        except Exception:
            raise ValueError('The input could not be parsed, neither as a TLE ({}) nor as an '
                             'OMM document in JSON, XML, KVN or CSV format.'.format(tle_error))
        return [omm_module.OMM(record) for record in records]


def _parse_tles(text):
    """Parses a string with one or more TLEs (with or without the name line)."""
    lines = util.get_non_empty_lines(text)
    i = 0
    tles = []
    while i < len(lines):
        if not (lines[i].startswith('1 ') or lines[i].startswith('2 ')):
            tles.append(TLE(lines[i:i + 3]))
            i += 3
        else:
            tles.append(TLE(lines[i:i + 2]))
            i += 2
    if not tles:
        raise ValueError('No TLE lines found.')
    return tles


def parse_single_element_set(text: str):
    """Same as `parse_element_sets`, but expects (and returns) exactly one object."""
    element_sets = parse_element_sets(text)
    if len(element_sets) != 1:
        raise ValueError('Expecting a single element set, while {} were provided.'.format(len(element_sets)))
    return element_sets[0]


def parse_datetime(date_utc: str) -> datetime.datetime:
    """Parses an ISO 8601 UTC date (e.g. '2024-03-27T11:47:00', trailing 'Z' allowed)."""
    text = str(date_utc).strip()
    if text.endswith('Z') or text.endswith('z'):
        text = text[:-1]
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError:
        raise ValueError("Date '{}' is not a valid ISO 8601 UTC date (expected e.g. "
                         "'2024-03-27T11:47:00' or '2024-03-27 11:47:00.123456').".format(date_utc))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return parsed


def epoch_mjd(element_set) -> float:
    """Returns the epoch of an element set as a Modified Julian Date."""
    return float(element_set.date_mjd)


def resolve_times(element_set,
                  minutes_since_epoch: Optional[List[float]] = None,
                  dates_utc: Optional[List[str]] = None,
                  max_points: int = MAX_TIME_POINTS) -> torch.Tensor:
    """
    Resolves the times of a tool call into a tensor of minutes since the element set
    epoch. Exactly one of `minutes_since_epoch` and `dates_utc` must be provided.
    """
    if (minutes_since_epoch is None) == (dates_utc is None):
        raise ValueError("Provide exactly one of 'minutes_since_epoch' (minutes since the element "
                         "set epoch) or 'dates_utc' (ISO 8601 UTC dates).")
    if minutes_since_epoch is not None:
        values = [float(value) for value in minutes_since_epoch]
    else:
        epoch = epoch_mjd(element_set)
        values = [(util.from_datetime_to_mjd(parse_datetime(date)) - epoch) * 1440.0 for date in dates_utc]
    if not values:
        raise ValueError('At least one propagation time is required.')
    if len(values) > max_points:
        raise ValueError('Too many time points ({}): this tool accepts at most {} per call.'.format(
            len(values), max_points))
    return torch.tensor(values)


def tsince_to_date_utc(element_set, tsince_minutes: float) -> str:
    """Converts minutes since the element set epoch into an ISO 8601 UTC date."""
    mjd = epoch_mjd(element_set) + float(tsince_minutes) / 1440.0
    return util.from_mjd_to_datetime(mjd).isoformat()


def states_to_entries(element_set, states: torch.Tensor, tsince: torch.Tensor) -> List[dict]:
    """
    Serializes a `(N, 2, 3)` (or `(2, 3)`) tensor of TEME states into a list of
    JSON-friendly entries, one per time point.
    """
    array = states.detach().numpy().reshape(-1, 2, 3)
    times = tsince.detach().reshape(-1).tolist()
    entries = []
    for values, minutes in zip(array, times):
        entries.append({
            'tsince_minutes': float(minutes),
            'date_utc': tsince_to_date_utc(element_set, minutes),
            'position_km': [float(value) for value in values[0]],
            'velocity_km_s': [float(value) for value in values[1]],
        })
    return entries


def propagation_warnings(element_set) -> List[str]:
    """Returns the SGP4 error flag of a freshly propagated satellite as a list of warnings."""
    error = int(getattr(element_set, '_error', 0))
    if error == 0:
        return []
    return ['SGP4 error code {}: {}'.format(error, SGP4_ERRORS.get(error, 'unknown error'))]


def classify_orbit(perigee_altitude_km, apogee_altitude_km, eccentricity, period_minutes, inclination_deg) -> str:
    """Heuristic classification of the orbit regime out of the mean elements."""
    if apogee_altitude_km < 2000.0:
        return 'LEO'
    if abs(period_minutes - 1436.1) < 60.0 and eccentricity < 0.1:
        return 'GEO' if inclination_deg < 15.0 else 'inclined GSO'
    if eccentricity >= 0.25:
        return 'HEO'
    if apogee_altitude_km < 35586.0:
        return 'MEO'
    return 'high Earth orbit'


def describe_element_set(element_set) -> dict:
    """Returns a JSON-friendly description of a TLE/OMM object (elements and derived quantities)."""
    mean_motion_rad_s = float(element_set.mean_motion)
    period_minutes = 2.0 * np.pi / mean_motion_rad_s / 60.0
    perigee_altitude_km = float(element_set.perigee_alt()) / 1000.0
    apogee_altitude_km = float(element_set.apogee_alt()) / 1000.0
    eccentricity = float(element_set.eccentricity)
    inclination_deg = float(np.rad2deg(element_set.inclination))
    description = {
        'format': 'OMM' if isinstance(element_set, omm_module.OMM) else 'TLE',
        'satellite_catalog_number': int(element_set.satellite_catalog_number),
        'classification': str(element_set.classification),
        'international_designator': str(element_set.international_designator),
        'epoch_utc': element_set._epoch.isoformat(),
        'epoch_mjd': epoch_mjd(element_set),
        'element_number': int(element_set.element_number),
        'revolution_number_at_epoch': int(element_set.revolution_number_at_epoch),
        'elements': {
            'mean_motion_revs_per_day': mean_motion_rad_s * 86400.0 / (2.0 * np.pi),
            'mean_motion_rad_s': mean_motion_rad_s,
            'eccentricity': eccentricity,
            'inclination_deg': inclination_deg,
            'raan_deg': float(np.rad2deg(element_set.raan)),
            'argument_of_perigee_deg': float(np.rad2deg(element_set.argument_of_perigee)),
            'mean_anomaly_deg': float(np.rad2deg(element_set.mean_anomaly)),
            'b_star': float(element_set.b_star),
            'mean_motion_first_derivative_rad_s2': float(element_set.mean_motion_first_derivative),
            'mean_motion_second_derivative_rad_s3': float(element_set.mean_motion_second_derivative),
        },
        'derived': {
            'semi_major_axis_km': float(element_set.semi_major_axis) / 1000.0,
            'orbital_period_minutes': period_minutes,
            'perigee_altitude_km': perigee_altitude_km,
            'apogee_altitude_km': apogee_altitude_km,
            'orbit_class': classify_orbit(perigee_altitude_km, apogee_altitude_km,
                                          eccentricity, period_minutes, inclination_deg),
            'uses_deep_space_corrections': bool(period_minutes >= 225.0),
        },
    }
    name = element_set._data.get('name')
    if name:
        description['name'] = str(name)
    try:
        description['lines'] = list(element_set._lines)
    except AttributeError:
        #OMM objects carry no TLE lines
        pass
    return description


def orbital_period_minutes(element_set) -> float:
    """Returns the (Keplerian, mean-motion based) orbital period in minutes."""
    return 2.0 * np.pi / float(element_set.mean_motion) / 60.0


def label_of(element_set) -> str:
    """A short human-readable label for plots and reports."""
    name = element_set._data.get('name')
    if name:
        return str(name)
    return 'NORAD {}'.format(int(element_set.satellite_catalog_number))
