"""
MCP tools of the `tle` domain: parsing, describing, validating, building and
converting element sets (TLEs and CCSDS OMM messages).
"""
import numpy as np

from .. import omm as omm_module
from .. import util
from ..tle import TLE, compute_checksum
from .common import friendly_errors, describe_element_set, parse_datetime, parse_element_sets, parse_single_element_set


def register(server):
    """Registers the tools of the `tle` domain on the given `MCPServer`."""

    @server.tool()
    @friendly_errors
    def parse_element_set(element_set: str) -> dict:
        """
        Parse a TLE (two or three lines) or a CCSDS OMM message (JSON/XML/KVN/CSV) and
        return its orbital elements together with derived quantities: semi-major axis,
        orbital period, perigee/apogee altitude, orbit class (LEO/MEO/GEO/HEO), epoch
        as a UTC date and whether SGP4 applies deep-space corrections to it. Angles are
        returned in degrees and distances in km.
        """
        return describe_element_set(parse_single_element_set(element_set))

    @server.tool()
    @friendly_errors
    def validate_tle(tle: str) -> dict:
        """
        Validate the format of a TLE: line lengths, field layout, checksums of both
        lines and consistency of the satellite catalog numbers. Returns whether the
        TLE is valid, together with the list of problems that were found (if any).
        """
        lines = util.get_non_empty_lines(tle)
        if len(lines) == 3:
            lines = lines[1:]
        if len(lines) != 2:
            return {'valid': False,
                    'errors': ['Expecting two TLE lines (or three, with the name line), '
                               'while {} were provided.'.format(len(lines))]}
        errors = []
        checksums = {}
        for label, line in zip(('line1', 'line2'), lines):
            line = line.rstrip()
            if len(line) != 69:
                errors.append('{} has {} characters instead of 69.'.format(label, len(line)))
            expected = compute_checksum(line)
            actual = line[68] if len(line) >= 69 else None
            checksums[label] = {'expected': expected, 'found': actual}
            if actual is None or not actual.isdigit():
                errors.append('{} has no checksum digit in column 69.'.format(label))
            elif int(actual) != expected:
                errors.append('{} checksum mismatch: found {}, expected {}.'.format(label, actual, expected))
        try:
            TLE(lines)
        except Exception as parse_error:
            errors.append('The TLE could not be parsed: {}'.format(parse_error))
        return {'valid': not errors, 'errors': errors, 'checksums': checksums}

    @server.tool()
    @friendly_errors
    def fix_tle_checksum(line: str) -> dict:
        """
        Compute the modulo-10 checksum of a single TLE line and return the line with
        the correct checksum appended in column 69.
        """
        line = line.rstrip()
        if len(line) < 68:
            raise ValueError('A TLE line has 68 characters plus the checksum, while {} were provided.'.format(len(line)))
        checksum = compute_checksum(line)
        return {'checksum': checksum, 'line': line[:68] + str(checksum)}

    @server.tool()
    @friendly_errors
    def build_tle(satellite_catalog_number: int,
                  epoch_utc: str,
                  mean_motion_revs_per_day: float,
                  eccentricity: float,
                  inclination_deg: float,
                  raan_deg: float,
                  argument_of_perigee_deg: float,
                  mean_anomaly_deg: float,
                  b_star: float = 0.0,
                  mean_motion_first_derivative_revs_per_day2: float = 0.0,
                  mean_motion_second_derivative_revs_per_day3: float = 0.0,
                  classification: str = 'U',
                  international_designator: str = '00000A',
                  element_number: int = 999,
                  revolution_number_at_epoch: int = 0,
                  ephemeris_type: int = 0,
                  name: str = '') -> dict:
        """
        Build a TLE out of its orbital elements (angles in degrees, mean motion and its
        derivatives in revolutions/day, revolutions/day^2 and revolutions/day^3, epoch
        as an ISO 8601 UTC date) and return the two formatted lines together with the
        parsed elements. The satellite catalog number must fit the TLE format (i.e. be
        at most 339999, encoded with the Alpha-5 convention above 99999).
        """
        epoch = parse_datetime(epoch_utc)
        data = {
            'satellite_catalog_number': int(satellite_catalog_number),
            'classification': classification,
            'international_designator': international_designator,
            'epoch_year': epoch.year,
            'epoch_days': util.from_datetime_to_fractional_day(epoch),
            'mean_motion': float(mean_motion_revs_per_day) * 2.0 * np.pi / 86400.0,
            'mean_motion_first_derivative': float(mean_motion_first_derivative_revs_per_day2) * 2.0 * np.pi / 86400.0**2,
            'mean_motion_second_derivative': float(mean_motion_second_derivative_revs_per_day3) * 2.0 * np.pi / 86400.0**3,
            'eccentricity': float(eccentricity),
            'inclination': float(np.deg2rad(inclination_deg)),
            'raan': float(np.deg2rad(raan_deg)),
            'argument_of_perigee': float(np.deg2rad(argument_of_perigee_deg)),
            'mean_anomaly': float(np.deg2rad(mean_anomaly_deg)),
            'b_star': float(b_star),
            'element_number': int(element_number),
            'revolution_number_at_epoch': int(revolution_number_at_epoch),
            'ephemeris_type': int(ephemeris_type),
        }
        if name:
            data['name'] = name
        return describe_element_set(TLE(data))

    @server.tool()
    @friendly_errors
    def convert_element_sets(element_sets: str, output_format: str) -> dict:
        """
        Convert one or more element sets between the TLE format and the CCSDS OMM
        serializations. The input can be TLEs or an OMM document; `output_format` must
        be one of 'tle', 'json', 'xml', 'kvn' or 'csv'. Note that objects with a
        satellite catalog number above 339999 cannot be converted to the TLE format.
        """
        objects = parse_element_sets(element_sets)
        output_format = output_format.strip().lower()
        if output_format == 'tle':
            lines = []
            for element_set in objects:
                if isinstance(element_set, omm_module.OMM):
                    element_set = element_set.to_tle()
                lines.extend(element_set._lines)
            content = '\n'.join(lines)
        elif output_format in ('json', 'xml', 'kvn', 'csv'):
            content = omm_module.dumps(objects, file_format=output_format)
        else:
            raise ValueError("Supported output formats: tle, json, xml, kvn, csv "
                             "while '{}' was provided.".format(output_format))
        return {'output_format': output_format,
                'number_of_objects': len(objects),
                'content': content}
