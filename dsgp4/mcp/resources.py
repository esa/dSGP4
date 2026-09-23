"""
MCP resources of the dSGP4 server: reference documentation (conventions, formats,
differentiable parameters, gravity models) and example data that a model can read
before using the tools.
"""
from .. import util
from .common import TLE_PARAMETERS, TLE_PARAMETER_UNITS

#a few real element sets (from the library's own test data) that can be fed to any tool:
EXAMPLE_TLES = """\
ISS (ZARYA)
1 25544U 98067A   24087.49097222  .00016717  00000+0  10270-3 0  9990
2 25544  51.6400  82.2420 0006290  58.9900  53.5550 15.50000000000008
COSMOS 2251 DEB
1 34427U 93036RU  22068.94647328  .00008100  00000-0  11455-2 0  9999
2 34427  74.0145 306.8269 0033346  13.0723 347.1308 14.76870515693886
COSMOS 2251 DEB
1 34428U 93036RV  22068.90158861  .00002627  00000-0  63561-3 0  9999
2 34428  74.0386 139.1157 0038434 196.7068 279.9147 14.53118052686950
TDRS 1 (DEEP SPACE)
1 14128U 83058A   06176.02844893 -.00000158  00000-0  10000-3 0  9627
2 14128  11.4384  35.2134 0011562  26.4582 333.5652  0.98870114 46093
"""

OVERVIEW = """\
# dSGP4 MCP server

dSGP4 (https://github.com/esa/dSGP4) is a differentiable PyTorch implementation of
the SGP4/SDP4 orbit propagator: it propagates Earth-orbiting objects from their TLE
or CCSDS OMM element sets and, being differentiable, also provides the exact partial
derivatives of the propagated state (state transition matrices, covariance mapping,
gradient-based TLE determination and hybrid ML models).

## Conventions used by every tool

- Element sets: any tool parameter called `element_set`/`element_sets`/`tle` accepts
  a TLE as two or three lines in a single string (newline separated), or a CCSDS OMM
  message in JSON, XML, KVN or CSV format.
- Times: propagation times are given either as `minutes_since_epoch` (minutes from
  the element set epoch, the native SGP4 variable, negative values allowed) or as
  `dates_utc` (ISO 8601 UTC dates, e.g. '2024-03-27T11:47:00'); exactly one of the two.
- Output states are in the TEME (True Equator Mean Equinox) frame, positions in km
  and velocities in km/s.
- Angles in tool inputs/outputs are in degrees; the *internal* SGP4 parameters used
  by the gradient tools are in radians and radians/minute (their units are always
  reported in the responses).
- SGP4 is a general perturbations model: typical accuracy is on the order of a few
  km at epoch, degrading by ~1-3 km/day. It is not suitable for precision ephemeris.
- Element sets whose orbital period is >= 225 minutes are automatically handled with
  the deep-space (SDP4) corrections.

## Tool domains

- `tle`: parse/validate/build/convert element sets (TLE and OMM formats).
- `propagation`: dSGP4 propagation (single and batched), state and time conversions.
- `gradients`: automatic-differentiation Jacobians of the state with respect to the
  TLE parameters and to time, and covariance transformations.
- `estimation`: differentiable Newton-Raphson TLE determination (re-epoch a TLE,
  fit a TLE to a Cartesian state).
- `ml`: forecasts with the hybrid ML-dSGP4 model (neural input/output corrections
  around SGP4).
- `plot`: rendered PNG visualizations (3D orbits, element distributions).
"""

TLE_FORMAT = """\
# The Two-Line Element (TLE) format

A TLE describes the mean orbital state of an Earth-orbiting object for use with the
SGP4/SDP4 analytical propagators. It has an optional name line ("line 0") and two
69-character data lines. Fields are fixed-width; column ranges below are 1-based.

## Line 1

| Columns | Field |
|---------|-------|
| 1       | Line number ('1') |
| 3-7     | Satellite catalog number (Alpha-5: a leading letter encodes numbers above 99999, up to 339999) |
| 8       | Classification (U/C/S) |
| 10-17   | International designator (launch year, launch number, piece) |
| 19-32   | Epoch: two-digit year (57-99 => 19xx, 00-56 => 20xx) and fractional day of the year |
| 34-43   | First derivative of mean motion / 2 [rev/day^2] |
| 45-52   | Second derivative of mean motion / 6 [rev/day^3], implied decimal point and exponent |
| 54-61   | B* drag term [1/earth radii], implied decimal point and exponent |
| 63      | Ephemeris type |
| 65-68   | Element set number |
| 69      | Checksum (sum of digits, minus signs count 1, modulo 10) |

## Line 2

| Columns | Field |
|---------|-------|
| 1       | Line number ('2') |
| 3-7     | Satellite catalog number |
| 9-16    | Inclination [deg] |
| 18-25   | Right ascension of the ascending node (RAAN) [deg] |
| 27-33   | Eccentricity (implied leading decimal point) |
| 35-42   | Argument of perigee [deg] |
| 44-51   | Mean anomaly [deg] |
| 53-63   | Mean motion [rev/day] |
| 64-68   | Revolution number at epoch |
| 69      | Checksum |

The TLE elements are *mean* elements in the SGP4 sense: they are NOT osculating
Keplerian elements and only make sense together with the SGP4/SDP4 model that
generated them. Objects with a catalog number above 339999 cannot be encoded in a
TLE and require the OMM format.
"""

OMM_FORMAT = """\
# The CCSDS Orbit Mean-Elements Message (OMM)

The OMM (CCSDS 502.0-B-3, distributed e.g. by Space-Track) carries the same SGP4
mean elements of a TLE without the fixed-width constraints of the two lines: catalog
numbers beyond the Alpha-5 range and full-precision elements can be represented.
dSGP4 parses and writes the four standard serializations - JSON, XML, KVN and CSV -
and OMM objects can be used everywhere a TLE is expected (all the tools of this
server accept them).

Main fields: OBJECT_NAME, OBJECT_ID (international designator), EPOCH (ISO UTC),
MEAN_MOTION [rev/day], ECCENTRICITY, INCLINATION [deg], RA_OF_ASC_NODE [deg],
ARG_OF_PERICENTER [deg], MEAN_ANOMALY [deg], EPHEMERIS_TYPE, CLASSIFICATION_TYPE,
NORAD_CAT_ID, ELEMENT_SET_NO, REV_AT_EPOCH, BSTAR, MEAN_MOTION_DOT, MEAN_MOTION_DDOT,
plus the MEAN_ELEMENT_THEORY (must be SGP4 for this library).

Use the `convert_element_sets` tool to convert between TLE, JSON, XML, KVN and CSV.
"""

SGP4_PARAMETERS = """\
# The nine differentiable SGP4 parameters

The gradient tools differentiate the propagated TEME state with respect to the nine
free parameters of the SGP4 initialization, in this fixed order and in these
internal SGP4 units (NOT the human-friendly units of the TLE lines):

| # | Parameter | Internal unit | Note |
|---|-----------|---------------|------|
{rows}

Jacobians returned by `state_partials_wrt_tle` are 6x9 matrices: rows are
x, y, z [km], vx, vy, vz [km/s]; columns are the parameters above. To convert a
derivative to per-degree units multiply by pi/180; to convert the mean-motion
column to rev/day divide by (1440 / 2 pi) rad/min per rev/day.
"""

MLDSGP4 = """\
# ML-dSGP4 (`dsgp4.mldsgp4`)

ML-dSGP4 is the hybrid model of Acciarini, Baydin, Izzo, "Closing the gap between
SGP4 and high-precision propagation via differentiable programming", Acta
Astronautica (2025). Two small feed-forward networks correct the *inputs* (the six
mean orbital elements) and the *outputs* (the normalized Cartesian state) of the
differentiable SGP4 propagator; since SGP4 sits in the middle of the computational
graph, the whole pipeline trains end-to-end against higher-precision ephemerides
(numerical propagation or observations).

Architecture (hidden size H, default 100):
- input net: 6 -> H -> H -> 6 (LeakyReLU), applied as a relative correction scaled
  by the learned `input_correction` parameter;
- dSGP4 propagation of the corrected elements;
- output net: 6 -> H -> H -> 6 (LeakyReLU), applied as a relative correction scaled
  by the learned `output_correction` parameter, on the state normalized by
  `normalization_R` (position) and `normalization_V` (velocity).

The `mldsgp4_propagate` tool runs this model given a checkpoint trained with the
library (`torch.save(model.state_dict(), path)`); without a checkpoint the weights
are random and the output is not a meaningful forecast. Training itself is a
Python workflow (see the `mldsgp4_training_guide` prompt), not an MCP tool.
"""


def register(server):
    """Registers the resources on the given `MCPServer`."""

    @server.resource('dsgp4://reference/overview', mime_type='text/markdown',
                     description='What the dSGP4 MCP server does, its conventions (frames, units, '
                                 'time inputs) and how the tool domains are organized. Read this first.')
    def overview() -> str:
        return OVERVIEW

    @server.resource('dsgp4://reference/tle-format', mime_type='text/markdown',
                     description='Field-by-field description of the Two-Line Element (TLE) format.')
    def tle_format() -> str:
        return TLE_FORMAT

    @server.resource('dsgp4://reference/omm-format', mime_type='text/markdown',
                     description='The CCSDS Orbit Mean-Elements Message (OMM) format and its serializations.')
    def omm_format() -> str:
        return OMM_FORMAT

    @server.resource('dsgp4://reference/sgp4-parameters', mime_type='text/markdown',
                     description='The nine differentiable SGP4 parameters, their order and internal units, '
                                 'as used by the gradient tools.')
    def sgp4_parameters() -> str:
        notes = {
            'b_star': 'drag-like term',
            'mean_motion_first_derivative': 'not used by the SGP4 dynamics',
            'mean_motion_second_derivative': 'not used by the SGP4 dynamics',
            'mean_motion': 'Kozai convention',
        }
        rows = '\n'.join('| {} | {} | {} | {} |'.format(index + 1, name, TLE_PARAMETER_UNITS[name],
                                                        notes.get(name, ''))
                         for index, name in enumerate(TLE_PARAMETERS))
        return SGP4_PARAMETERS.format(rows=rows)

    @server.resource('dsgp4://reference/gravity-models/{model}', mime_type='text/markdown',
                     description="Constants of a gravity model ('wgs-72old', 'wgs-72' or 'wgs-84', "
                                 'the default of the tools).')
    def gravity_model(model: str) -> str:
        try:
            tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 = util.get_gravity_constants(model)
        except RuntimeError as error:
            from mcp.server.mcpserver.exceptions import ResourceError
            raise ResourceError(str(error)) from error
        lines = ['# Gravity model {}'.format(model), '',
                 '| Constant | Value | Unit |', '|---|---|---|',
                 '| mu | {} | km^3/s^2 |'.format(float(mu)),
                 '| Earth equatorial radius | {} | km |'.format(float(radiusearthkm)),
                 '| xke | {} | sqrt(earth radii^3/min^2) |'.format(float(xke)),
                 '| tumin | {} | min |'.format(float(tumin)),
                 '| J2 | {} | - |'.format(float(j2)),
                 '| J3 | {} | - |'.format(float(j3)),
                 '| J4 | {} | - |'.format(float(j4))]
        return '\n'.join(lines)

    @server.resource('dsgp4://reference/mldsgp4', mime_type='text/markdown',
                     description='The hybrid ML-dSGP4 model: architecture, training idea and how the '
                                 'ml tools use its checkpoints.')
    def mldsgp4_reference() -> str:
        return MLDSGP4

    @server.resource('dsgp4://examples/element-sets', mime_type='text/plain',
                     description='Real example TLEs (LEO, debris and a deep-space object) that can be '
                                 'pasted into any tool of this server.')
    def example_element_sets() -> str:
        return EXAMPLE_TLES
