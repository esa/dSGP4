"""
MCP tools of the `gradients` domain: partial derivatives of the propagated state
(the very reason dSGP4 exists), obtained via PyTorch automatic differentiation, and
the covariance transformations built on top of them.
"""
from typing import List, Optional

import torch

from ..util import initialize_tle, propagate as propagate_state
from .common import (friendly_errors, MAX_GRADIENT_POINTS, STATE_COMPONENTS, TLE_PARAMETERS, TLE_PARAMETER_UNITS,
                     parse_single_element_set, resolve_times, states_to_entries, tsince_to_date_utc)

#the (row, column) pairs of the state tensor, in the order of STATE_COMPONENTS:
_STATE_INDICES = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def _state_jacobians(element_set, tsince):
    """
    Returns, for each time in `tsince`, the 6x9 Jacobian of the TEME state (x, y, z in
    km, vx, vy, vz in km/s) with respect to the nine TLE parameters (in the order and
    units of `TLE_PARAMETERS`/`TLE_PARAMETER_UNITS`), together with the states.
    """
    tle_elements = initialize_tle(element_set, with_grad=True)
    states = propagate_state(element_set, tsince).reshape(-1, 2, 3)
    jacobians = []
    for index in range(states.shape[0]):
        rows = []
        for row, column in _STATE_INDICES:
            gradient = torch.autograd.grad(states[index, row, column], tle_elements, retain_graph=True)[0]
            rows.append([float(value) for value in gradient])
        jacobians.append(rows)
    return jacobians, states


def register(server):
    """Registers the tools of the `gradients` domain on the given `MCPServer`."""

    @server.tool()
    @friendly_errors
    def state_partials_wrt_tle(element_set: str,
                               minutes_since_epoch: Optional[List[float]] = None,
                               dates_utc: Optional[List[str]] = None) -> dict:
        """
        Compute, via automatic differentiation, the Jacobian of the propagated TEME
        state with respect to the nine TLE parameters (b_star, mean motion and its two
        derivatives, eccentricity, argument of perigee, inclination, mean anomaly,
        RAAN). For each requested time a 6x9 matrix is returned: rows are x, y, z (km)
        and vx, vy, vz (km/s), columns are the TLE parameters in the internal SGP4
        units reported in `parameter_units`. This is the sensitivity matrix used for
        state transition matrices, covariance mapping and gradient-based estimation.
        At most 20 time points per call (each costs six backward passes).
        """
        satellite = parse_single_element_set(element_set)
        tsince = resolve_times(satellite, minutes_since_epoch, dates_utc, max_points=MAX_GRADIENT_POINTS)
        jacobians, states = _state_jacobians(satellite, tsince)
        entries = states_to_entries(satellite, states, tsince)
        for entry, jacobian in zip(entries, jacobians):
            entry['jacobian'] = jacobian
        return {
            'satellite_catalog_number': int(satellite.satellite_catalog_number),
            'rows': STATE_COMPONENTS,
            'columns': TLE_PARAMETERS,
            'parameter_units': TLE_PARAMETER_UNITS,
            'entries': entries,
        }

    @server.tool()
    @friendly_errors
    def state_partials_wrt_time(element_set: str,
                                minutes_since_epoch: Optional[List[float]] = None,
                                dates_utc: Optional[List[str]] = None) -> dict:
        """
        Compute, via automatic differentiation, the derivative of the propagated TEME
        state with respect to the propagation time: for each requested time a 6-vector
        is returned with d(x,y,z)/dt in km/min and d(vx,vy,vz)/dt in km/s/min (the
        first three components are the velocity expressed in km/min, the last three
        the acceleration seen by SGP4). At most 20 time points per call.
        """
        satellite = parse_single_element_set(element_set)
        tsince = resolve_times(satellite, minutes_since_epoch, dates_utc, max_points=MAX_GRADIENT_POINTS)
        tsince.requires_grad_(True)
        initialize_tle(satellite)
        states = propagate_state(satellite, tsince).reshape(-1, 2, 3)
        entries = states_to_entries(satellite, states.detach(), tsince.detach())
        for index, entry in enumerate(entries):
            derivative = []
            for row, column in _STATE_INDICES:
                gradient = torch.autograd.grad(states[index, row, column], tsince, retain_graph=True)[0]
                derivative.append(float(gradient.reshape(-1)[index]))
            entry['dstate_dt'] = derivative
        return {
            'satellite_catalog_number': int(satellite.satellite_catalog_number),
            'dstate_dt_components': ['dx_dt_km_min', 'dy_dt_km_min', 'dz_dt_km_min',
                                     'dvx_dt_km_s_min', 'dvy_dt_km_s_min', 'dvz_dt_km_s_min'],
            'entries': entries,
        }

    @server.tool()
    @friendly_errors
    def transform_covariance(element_set: str,
                             covariance: List[List[float]],
                             parameters: Optional[List[str]] = None,
                             minutes_since_epoch: Optional[List[float]] = None,
                             dates_utc: Optional[List[str]] = None) -> dict:
        """
        Map a covariance expressed in TLE parameter space into a 6x6 Cartesian TEME
        covariance (position in km, velocity in km/s) at the requested time(s), via
        the similarity transform C_xyz = J C J^T, where J is the automatic-
        differentiation Jacobian of the state. `parameters` names the k rows/columns
        of the k x k input `covariance` (subset of: b_star, mean_motion_first_derivative,
        mean_motion_second_derivative, eccentricity, argument_of_perigee, inclination,
        mean_anomaly, mean_motion, raan; default: the six mean orbital elements), in
        the internal SGP4 units reported by the `state_partials_wrt_tle` tool (angles
        in rad, mean motion in rad/min). Returns the Cartesian covariance and the
        standard deviations of its diagonal.
        """
        satellite = parse_single_element_set(element_set)
        tsince = resolve_times(satellite, minutes_since_epoch, dates_utc, max_points=MAX_GRADIENT_POINTS)
        if parameters is None:
            parameters = ['eccentricity', 'argument_of_perigee', 'inclination',
                          'mean_anomaly', 'mean_motion', 'raan']
        try:
            columns = [TLE_PARAMETERS.index(parameter) for parameter in parameters]
        except ValueError:
            raise ValueError('Unknown parameter name: supported names are {}.'.format(', '.join(TLE_PARAMETERS)))
        matrix = torch.tensor(covariance)
        if matrix.shape != (len(parameters), len(parameters)):
            raise ValueError('The covariance must be a {0}x{0} matrix matching the {0} '
                             'parameters.'.format(len(parameters)))
        jacobians, _ = _state_jacobians(satellite, tsince)
        entries = []
        for minutes, jacobian in zip(tsince.reshape(-1).tolist(), jacobians):
            selected = torch.tensor(jacobian)[:, columns]
            cartesian = selected @ matrix @ selected.T
            entries.append({
                'tsince_minutes': float(minutes),
                'date_utc': tsince_to_date_utc(satellite, minutes),
                'covariance_teme': [[float(value) for value in row] for row in cartesian],
                'sigmas': [float(value) for value in torch.sqrt(torch.clamp(torch.diagonal(cartesian), min=0.0))],
            })
        return {
            'satellite_catalog_number': int(satellite.satellite_catalog_number),
            'input_parameters': parameters,
            'parameter_units': {parameter: TLE_PARAMETER_UNITS[parameter] for parameter in parameters},
            'rows': STATE_COMPONENTS,
            'entries': entries,
        }
