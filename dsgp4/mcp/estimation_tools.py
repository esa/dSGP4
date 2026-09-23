"""
MCP tools of the `estimation` domain: gradient-based TLE determination via the
differentiable Newton-Raphson method shipped with the library (`dsgp4.newton_method`).
"""
from typing import List

import torch

from .. import util
from ..newton_method import newton_method
from .common import friendly_errors, describe_element_set, epoch_mjd, parse_datetime, parse_single_element_set


def _fit_report(fitted, target_state, time_mjd):
    """Propagates the fitted element set to `time_mjd` and reports the residuals vs the target."""
    check = fitted.copy()
    util.initialize_tle(check)
    tsince = (time_mjd - epoch_mjd(check)) * 1440.0
    achieved = util.propagate(check, torch.tensor([tsince])).detach().reshape(2, 3)
    residual = achieved - target_state.reshape(2, 3)
    report = describe_element_set(fitted)
    report['fit'] = {
        'position_residual_km': float(torch.norm(residual[0])),
        'velocity_residual_km_s': float(torch.norm(residual[1])),
        'achieved_position_km': [float(value) for value in achieved[0]],
        'achieved_velocity_km_s': [float(value) for value in achieved[1]],
    }
    return report


def register(server):
    """Registers the tools of the `estimation` domain on the given `MCPServer`."""

    @server.tool()
    @friendly_errors
    def update_tle_epoch(element_set: str,
                         new_epoch_utc: str,
                         max_iterations: int = 50,
                         tolerance: float = 1e-12) -> dict:
        """
        Move a TLE (or OMM) to a new epoch: the original element set is propagated to
        `new_epoch_utc` (ISO 8601) and a new TLE with that epoch is fitted to the
        propagated state with the library's differentiable Newton-Raphson method, so
        that the new element set reproduces the same trajectory around the new epoch.
        Returns the new TLE together with the position/velocity residuals of the fit
        (in km and km/s); a large residual means the iteration did not converge.
        """
        original = parse_single_element_set(element_set)
        time_mjd = util.from_datetime_to_mjd(parse_datetime(new_epoch_utc))
        target = original.copy()
        util.initialize_tle(target)
        target_state = util.propagate(target, torch.tensor([(time_mjd - epoch_mjd(target)) * 1440.0]))
        fitted, _ = newton_method(original.copy(), time_mjd, max_iter=int(max_iterations),
                                  new_tol=float(tolerance))
        return _fit_report(fitted, target_state.detach(), time_mjd)

    @server.tool()
    @friendly_errors
    def fit_tle_to_state(template_element_set: str,
                         position_km: List[float],
                         velocity_km_s: List[float],
                         date_utc: str,
                         max_iterations: int = 50,
                         tolerance: float = 1e-12) -> dict:
        """
        Determine the TLE whose SGP4 propagation matches a given Cartesian TEME state
        (position in km, velocity in km/s) at `date_utc` (ISO 8601), using the
        library's differentiable Newton-Raphson method. `template_element_set` is a
        TLE or OMM of the same object (or a nearby orbit) used as the starting guess
        and as the source of the fields that are not estimated (b_star, drag terms,
        identifiers). Returns the fitted TLE, whose epoch is `date_utc`, together with
        the position/velocity residuals of the fit (in km and km/s).
        """
        if len(position_km) != 3 or len(velocity_km_s) != 3:
            raise ValueError('Expecting two 3-vectors: position in km and velocity in km/s.')
        template = parse_single_element_set(template_element_set)
        time_mjd = util.from_datetime_to_mjd(parse_datetime(date_utc))
        target_state = torch.tensor([[float(value) for value in position_km],
                                     [float(value) for value in velocity_km_s]])
        fitted, _ = newton_method(template.copy(), time_mjd, max_iter=int(max_iterations),
                                  new_tol=float(tolerance), target_state=target_state)
        return _fit_report(fitted, target_state, time_mjd)
