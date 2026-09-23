"""
MCP tools of the `ml` domain: orbit forecasts with the hybrid ML-dSGP4 model
(`dsgp4.mldsgp4`), where neural networks correct the inputs and outputs of the
differentiable SGP4 propagator to better match higher-precision data.
"""
from typing import List, Optional

import torch

from ..mldsgp4 import mldsgp4
from .common import (friendly_errors, label_of, parse_element_sets, propagation_warnings, resolve_times,
                     states_to_entries)


def _build_model(model_path, hidden_size, input_correction, output_correction,
                 normalization_R, normalization_V):
    model = mldsgp4(normalization_R=normalization_R,
                    normalization_V=normalization_V,
                    hidden_size=hidden_size,
                    input_correction=input_correction,
                    output_correction=output_correction)
    if model_path:
        model.load_model(model_path)
    else:
        model.eval()
    return model


def register(server):
    """Registers the tools of the `ml` domain on the given `MCPServer`."""

    @server.tool()
    @friendly_errors
    def mldsgp4_propagate(element_set: str,
                          minutes_since_epoch: List[float],
                          model_path: Optional[str] = None,
                          hidden_size: int = 100,
                          input_correction: float = 1e-2,
                          output_correction: float = 0.8,
                          normalization_R: float = 6958.137,
                          normalization_V: float = 7.947155867983262) -> dict:
        """
        Propagate an element set (TLE or OMM) with the hybrid ML-dSGP4 model, in which
        neural networks correct the inputs and outputs of the differentiable SGP4
        propagator (e.g. after training against numerically-propagated or observed
        ephemerides). `model_path` is the path of a trained checkpoint saved with
        `torch.save(model.state_dict(), path)`; the architecture hyperparameters must
        match the checkpoint. WITHOUT `model_path` the network weights are random and
        the output is NOT a meaningful forecast (a warning is returned): use the plain
        `propagate` tool for uncorrected SGP4 states. Returns positions (km) and
        velocities (km/s) in the TEME frame at the given minutes since the epoch.
        """
        satellites = parse_element_sets(element_set)
        if len(satellites) != 1:
            raise ValueError('Expecting a single element set: for several objects call the tool once per object.')
        satellite = satellites[0]
        tsince = resolve_times(satellite, minutes_since_epoch, None)
        model = _build_model(model_path, int(hidden_size), float(input_correction),
                             float(output_correction), float(normalization_R), float(normalization_V))
        normalized = model(satellite, tsince)
        states = torch.cat((normalized[:, :3] * model.normalization_R,
                            normalized[:, 3:] * model.normalization_V), dim=1).reshape(-1, 2, 3)
        warnings = propagation_warnings(satellite)
        if not model_path:
            warnings.append('No model_path was provided: the neural corrections are UNTRAINED '
                            '(random weights), so these states are not a meaningful forecast.')
        return {
            'satellite': label_of(satellite),
            'satellite_catalog_number': int(satellite.satellite_catalog_number),
            'frame': 'TEME',
            'units': {'position': 'km', 'velocity': 'km/s'},
            'model': {'trained': bool(model_path), 'model_path': model_path,
                      'hidden_size': int(hidden_size)},
            'states': states_to_entries(satellite, states, tsince),
            'warnings': warnings,
        }

    @server.tool()
    @friendly_errors
    def mldsgp4_inspect(model_path: str) -> dict:
        """
        Inspect a trained ML-dSGP4 checkpoint (a state dict saved with `torch.save`):
        returns the tensors it contains with their shapes, the total number of
        parameters, the hidden size inferred from the first layer and the learned
        input/output correction scales. Use it to recover the `hidden_size` needed by
        the `mldsgp4_propagate` tool when it is not known.
        """
        state_dict = torch.load(model_path, map_location='cpu')
        tensors = {key: list(value.shape) for key, value in state_dict.items()}
        result = {
            'model_path': model_path,
            'tensors': tensors,
            'number_of_parameters': int(sum(value.numel() for value in state_dict.values())),
        }
        if 'fc1.weight' in state_dict:
            result['hidden_size'] = int(state_dict['fc1.weight'].shape[0])
        for key in ('input_correction', 'output_correction'):
            if key in state_dict:
                result['{}_mean'.format(key)] = float(state_dict[key].mean())
        return result
