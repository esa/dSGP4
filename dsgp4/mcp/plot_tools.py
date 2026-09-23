"""
MCP tools of the `plot` domain: rendered orbit and catalog visualizations, returned
to the client as PNG images (and optionally saved to disk).
"""
import io
from typing import Optional

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import torch

from mcp.server.mcpserver import Image

from ..plot import plot_orbit, plot_tles
from ..util import initialize_tle, propagate as propagate_state
from .common import friendly_errors, MAX_PLOT_POINTS, label_of, orbital_period_minutes, parse_element_sets

#at most this many orbits in a single 3D plot (readability, not performance):
MAX_PLOTTED_ORBITS = 20


def _figure_to_image(figure, file_path=None, dpi=100):
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    if file_path:
        figure.savefig(file_path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)
    return Image(data=buffer.getvalue(), format='png')


def register(server):
    """Registers the tools of the `plot` domain on the given `MCPServer`."""

    @server.tool()
    @friendly_errors
    def plot_orbits(element_sets: str,
                    duration_minutes: Optional[float] = None,
                    number_of_points: int = 300,
                    elevation_deg: Optional[float] = None,
                    azimuth_deg: Optional[float] = None,
                    file_path: Optional[str] = None) -> Image:
        """
        Propagate one or more element sets (TLEs or an OMM document, at most 20
        objects) with dSGP4 and render their 3D orbits around the Earth (TEME frame,
        km) as a PNG image. Each object is propagated from its epoch for
        `duration_minutes` (default: its own orbital period, i.e. one revolution)
        sampled with `number_of_points` points. `elevation_deg`/`azimuth_deg` set the
        3D view angles; `file_path` optionally also saves the PNG to disk.
        """
        satellites = parse_element_sets(element_sets)
        if len(satellites) > MAX_PLOTTED_ORBITS:
            raise ValueError('Too many objects ({}): this tool plots at most {} orbits.'.format(
                len(satellites), MAX_PLOTTED_ORBITS))
        number_of_points = int(number_of_points)
        if not 2 <= number_of_points <= MAX_PLOT_POINTS:
            raise ValueError('number_of_points must be between 2 and {}.'.format(MAX_PLOT_POINTS))
        elevation_azimuth = None
        if elevation_deg is not None or azimuth_deg is not None:
            elevation_azimuth = (float(elevation_deg or 0.0), float(azimuth_deg or 0.0))
        ax = None
        for satellite in satellites:
            duration = float(duration_minutes) if duration_minutes else orbital_period_minutes(satellite)
            tsince = torch.linspace(0.0, duration, number_of_points)
            initialize_tle(satellite)
            states = propagate_state(satellite, tsince).detach()
            ax = plot_orbit(states, elevation_azimuth=elevation_azimuth, ax=ax, label=label_of(satellite))
        return _figure_to_image(ax.get_figure(), file_path=file_path)

    @server.tool()
    @friendly_errors
    def plot_element_distributions(element_sets: str,
                                   log_yscale: bool = False,
                                   file_path: Optional[str] = None) -> Image:
        """
        Render, as a PNG image, the histograms of the orbital elements of a set of
        element sets (TLEs or an OMM document): mean motion, eccentricity,
        inclination, argument of perigee, RAAN, B*, mean anomaly and the mean motion
        derivatives. Useful to characterize a whole catalog or constellation at a
        glance. `file_path` optionally also saves the PNG to disk.
        """
        satellites = parse_element_sets(element_sets)
        if len(satellites) < 2:
            raise ValueError('At least two element sets are needed to plot distributions.')
        axs = plot_tles(satellites, figsize=(24, 12), show=False, return_axs=True,
                        log_yscale=bool(log_yscale))
        return _figure_to_image(axs[0, 0].get_figure(), file_path=file_path, dpi=80)
