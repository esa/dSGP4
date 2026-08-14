"""
This module implements the support for the OMM (Orbit Mean-Elements Message) format, as
defined by the CCSDS Recommended Standard 502.0-B-3 and distributed by Space-Track. The OMM
carries the very same SGP4 mean elements of a TLE, but without the fixed-width constraints of
the two lines: this makes it possible to represent objects whose catalog number does not fit
the TLE format, and to store the elements at their full precision.

The four serializations defined by the standard (JSON, XML, KVN and CSV) are supported, and
the resulting `dsgp4.omm.OMM` objects can be used everywhere a `dsgp4.tle.TLE` object is
expected (e.g. `dsgp4.initialize_tle`, `dsgp4.propagate`, `dsgp4.propagate_batch`).
"""
import csv
import datetime
import io
import json
import numpy as np
import os
import torch
from xml.etree import ElementTree

from . import util
from .tle import TLE, add_derived_quantities, copy_data

#version of the CCSDS OMM standard that is written out:
CCSDS_OMM_VERS = '3.0'
#mean element theories that dSGP4 can propagate (SGP4-XP, for instance, is not supported):
SUPPORTED_MEAN_ELEMENT_THEORIES = ('SGP4', 'SGP/SGP4', 'SGP')
#reference context assumed by the SGP4 implementation:
SUPPORTED_CENTER_NAME = 'EARTH'
SUPPORTED_REF_FRAME = 'TEME'
SUPPORTED_TIME_SYSTEM = 'UTC'
#OMM fields that make up the header of the message:
OMM_HEADER = ('CCSDS_OMM_VERS', 'CREATION_DATE', 'ORIGINATOR')
#OMM fields that make up the metadata section of the message:
OMM_METADATA = ('OBJECT_NAME', 'OBJECT_ID', 'CENTER_NAME', 'REF_FRAME', 'TIME_SYSTEM', 'MEAN_ELEMENT_THEORY')
#OMM fields that make up the mean elements section of the message:
OMM_MEAN_ELEMENTS = ('EPOCH', 'MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION', 'RA_OF_ASC_NODE', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY')
#OMM fields that make up the TLE parameters section of the message:
OMM_TLE_PARAMETERS = ('EPHEMERIS_TYPE', 'CLASSIFICATION_TYPE', 'NORAD_CAT_ID', 'ELEMENT_SET_NO', 'REV_AT_EPOCH', 'BSTAR', 'MEAN_MOTION_DOT', 'MEAN_MOTION_DDOT')
#file extensions that are recognized by `load`:
FORMAT_BY_EXTENSION = {'.json': 'json', '.xml': 'xml', '.kvn': 'kvn', '.omm': 'kvn', '.txt': 'kvn', '.csv': 'csv'}


def from_omm_epoch_to_datetime(epoch):
    """
    This function takes the epoch of an OMM (i.e. a CCSDS ASCII time code, for instance
    '2024-02-29T12:00:00.000000' or, in its day of the year form, '2024-060T12:00:00.000000')
    and returns the corresponding datetime object.

    Parameters:
    ----------------
    epoch (``str``): OMM epoch

    Returns:
    ----------------
    ``datetime.datetime``: datetime object
    """
    if isinstance(epoch, datetime.datetime):
        return epoch
    text = str(epoch).strip().replace(' ', 'T')
    if text.endswith('Z'):
        text = text[:-1]
    calendar_date, _, time_of_day = text.partition('T')
    fields = calendar_date.split('-')
    if len(fields) == 2:
        #the standard also allows the day of the year form (e.g. '2024-060'):
        date_datetime = datetime.datetime(int(fields[0]), 1, 1) + datetime.timedelta(days=int(fields[1]) - 1)
    elif len(fields) == 3:
        date_datetime = datetime.datetime(int(fields[0]), int(fields[1]), int(fields[2]))
    else:
        raise ValueError('Epoch not compatible with the OMM format: {}'.format(epoch))
    if time_of_day:
        hr, minute, sec = time_of_day.split(':')
        date_datetime += datetime.timedelta(hours=int(hr), minutes=int(minute), seconds=float(sec))
    return date_datetime


def from_datetime_to_omm_epoch(date_datetime):
    """
    This function takes a datetime object and returns the corresponding OMM epoch.

    Parameters:
    ----------------
    date_datetime (``datetime.datetime``): datetime object

    Returns:
    ----------------
    ``str``: OMM epoch
    """
    return date_datetime.strftime(format='%Y-%m-%dT%H:%M:%S.%f')


def from_object_id_to_international_designator(object_id):
    """
    This function takes the object identifier of an OMM (e.g. '1998-067A') and returns the
    corresponding international designator, in the form used by the TLE format (e.g. '98067A').

    Parameters:
    ----------------
    object_id (``str``): OMM object identifier

    Returns:
    ----------------
    ``str``: international designator
    """
    text = str(object_id).strip()
    if len(text) < 6 or text[4] != '-' or not text[:4].isdigit():
        #the identifier is either already in the TLE form, or unknown (e.g. 'UNKNOWN'):
        return text
    return text[2:4] + text[5:]


def from_international_designator_to_object_id(international_designator):
    """
    This function takes an international designator, in the form used by the TLE format
    (e.g. '98067A'), and returns the corresponding OMM object identifier (e.g. '1998-067A').

    Parameters:
    ----------------
    international_designator (``str``): international designator

    Returns:
    ----------------
    ``str``: OMM object identifier
    """
    text = str(international_designator).strip()
    if len(text) < 5 or not text[:5].isdigit():
        #the designator is either already in the OMM form, or unknown:
        return text
    two_digit_year = int(text[:2])
    if two_digit_year < 57:
        year = two_digit_year + 2000
    else:
        year = two_digit_year + 1900
    return '{}-{}'.format(year, text[2:])


def parse_json(text):
    """
    This function parses OMM data in JSON format (i.e. either a single message or a list of
    messages, as returned by the Space-Track API), and returns the corresponding list of
    dictionaries of OMM fields.

    Parameters:
    ----------------
    text (``str``): OMM data in JSON format

    Returns:
    ----------------
    ``list``: list of dictionaries of OMM fields
    """
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        parsed = [parsed]
    return [dict(fields) for fields in parsed]


def parse_xml(text):
    """
    This function parses OMM data in XML format, and returns the corresponding list of
    dictionaries of OMM fields.

    Parameters:
    ----------------
    text (``str``): OMM data in XML format

    Returns:
    ----------------
    ``list``: list of dictionaries of OMM fields
    """
    root = ElementTree.fromstring(text)
    messages = [element for element in root.iter() if _local_name(element.tag) == 'omm']
    if not messages:
        messages = [root]
    records = []
    for message in messages:
        header = {}
        #the version is stored as an attribute of the message, while the rest of the header
        #(i.e. creation date and originator) is stored as a separate element:
        if message.get('version') is not None:
            header['CCSDS_OMM_VERS'] = message.get('version')
        for element in message.iter():
            if _local_name(element.tag) == 'header':
                header.update(_leaf_fields(element))
        for element in message.iter():
            if _local_name(element.tag) == 'segment':
                fields = dict(header)
                fields.update(_leaf_fields(element))
                records.append(fields)
    return records


def parse_kvn(text):
    """
    This function parses OMM data in KVN (i.e. keyword-value notation) format, and returns the
    corresponding list of dictionaries of OMM fields.

    Parameters:
    ----------------
    text (``str``): OMM data in KVN format

    Returns:
    ----------------
    ``list``: list of dictionaries of OMM fields
    """
    records, fields = [], {}
    for line in util.get_non_empty_lines(text):
        line = line.strip()
        if line.startswith('COMMENT'):
            continue
        key, separator, value = line.partition('=')
        if not separator:
            continue
        key, value = key.strip().upper(), value.strip()
        #the standard allows the units to be appended to the value (e.g. '15.5 [rev/day]'):
        if value.endswith(']') and value.rfind('[') != -1:
            value = value[:value.rfind('[')].strip()
        if key == 'CCSDS_OMM_VERS' and fields:
            #each message of a multi-message file starts with the version of the standard:
            records.append(fields)
            fields = {}
        fields[key] = value
    if fields:
        records.append(fields)
    return records


def parse_csv(text):
    """
    This function parses OMM data in CSV format (i.e. one header line with the OMM keywords,
    followed by one line per object), and returns the corresponding list of dictionaries of OMM fields.

    Parameters:
    ----------------
    text (``str``): OMM data in CSV format

    Returns:
    ----------------
    ``list``: list of dictionaries of OMM fields
    """
    return [dict(row) for row in csv.DictReader(io.StringIO(text))]


def loads(text, file_format=None):
    """
    This function parses OMM data and returns the corresponding list of dictionaries of OMM
    fields.

    Parameters:
    ----------------
    text (``str``): OMM data
    file_format (``str``): format of the data, either 'json', 'xml', 'kvn' or 'csv' (if None, it is detected from the content of the data)

    Returns:
    ----------------
    ``list``: list of dictionaries of OMM fields
    """
    if not isinstance(text, str):
        raise ValueError('Expecting a string')
    if file_format is None:
        file_format = detect_format(text)
    parsers = {'json': parse_json, 'xml': parse_xml, 'kvn': parse_kvn, 'csv': parse_csv}
    if file_format not in parsers:
        raise ValueError("Supported OMM formats: {} while {} was provided".format(', '.join(sorted(parsers)), file_format))
    return parsers[file_format](text)


def dumps(omms, file_format='json'):
    """
    This function takes an OMM object (or a TLE object, or a list of them, or the corresponding
    dictionaries of OMM fields), and returns the corresponding text in the requested format.

    Parameters:
    ----------------
    omms (``dsgp4.omm.OMM``, ``dsgp4.tle.TLE``, ``dict``, ``list``): OMM data to be written
    file_format (``str``): format of the output, either 'json', 'xml', 'kvn' or 'csv'

    Returns:
    ----------------
    ``str``: OMM data in the requested format
    """
    is_single = isinstance(omms, (TLE, dict))
    records = [omms] if is_single else list(omms)
    records = [record._fields if isinstance(record, OMM) else
               record.to_omm()._fields if isinstance(record, TLE) else dict(record) for record in records]
    if file_format == 'json':
        return json.dumps(records[0] if is_single else records, indent=2)
    elif file_format == 'kvn':
        lines = []
        for record in records:
            lines.extend('{} = {}'.format(key, value) for key, value in record.items())
            lines.append('')
        return '\n'.join(lines)
    elif file_format == 'csv':
        keys = list(dict.fromkeys(key for record in records for key in record))
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=keys, lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
        return output.getvalue()
    elif file_format == 'xml':
        ndm = ElementTree.Element('ndm')
        for record in records:
            message = ElementTree.SubElement(ndm, 'omm', {'id': 'CCSDS_OMM_VERS',
                                                          'version': record.get('CCSDS_OMM_VERS', CCSDS_OMM_VERS)})
            header = ElementTree.SubElement(message, 'header')
            _add_xml_fields(header, record, OMM_HEADER[1:])
            segment = ElementTree.SubElement(ElementTree.SubElement(message, 'body'), 'segment')
            _add_xml_fields(ElementTree.SubElement(segment, 'metadata'), record, OMM_METADATA)
            data = ElementTree.SubElement(segment, 'data')
            _add_xml_fields(ElementTree.SubElement(data, 'meanElements'), record, OMM_MEAN_ELEMENTS)
            _add_xml_fields(ElementTree.SubElement(data, 'tleParameters'), record, OMM_TLE_PARAMETERS)
        return ElementTree.tostring(ndm, encoding='unicode')
    else:
        raise ValueError("Supported OMM formats: csv, json, kvn, xml while {} was provided".format(file_format))


def detect_format(text):
    """
    This function takes OMM data and returns the format it is written in.

    Parameters:
    ----------------
    text (``str``): OMM data

    Returns:
    ----------------
    ``str``: format of the data, either 'json', 'xml', 'kvn' or 'csv'
    """
    stripped = text.lstrip()
    if stripped.startswith('{') or stripped.startswith('['):
        return 'json'
    if stripped.startswith('<'):
        return 'xml'
    for line in util.get_non_empty_lines(stripped):
        line = line.strip()
        if line.startswith('COMMENT'):
            continue
        if '=' in line:
            return 'kvn'
        if ',' in line:
            return 'csv'
        break
    raise ValueError('Could not detect the OMM format: pass it explicitly via the `file_format` argument.')


def to_omm_fields(data):
    """
    This function takes the dictionary of elements used by `dsgp4.tle.TLE` and `dsgp4.omm.OMM`,
    and returns the corresponding dictionary of OMM fields.

    Parameters:
    ----------------
    data (``dict``): elements in the form of a dictionary

    Returns:
    ----------------
    ``dict``: OMM data in the form of a dictionary of OMM fields
    """
    epoch_year, epoch_days = int(data['epoch_year']), float(data['epoch_days'])
    date_datetime = datetime.datetime(epoch_year-1, 12, 31, 0, 0, 0, 0)+datetime.timedelta(days=epoch_days)
    fields = {}
    fields['CCSDS_OMM_VERS'] = CCSDS_OMM_VERS
    if 'creation_date' in data:
        fields['CREATION_DATE'] = str(data['creation_date'])
    if 'originator' in data:
        fields['ORIGINATOR'] = str(data['originator'])
    if 'name' in data:
        fields['OBJECT_NAME'] = str(data['name'])
    fields['OBJECT_ID'] = from_international_designator_to_object_id(data['international_designator'])
    fields['CENTER_NAME'] = SUPPORTED_CENTER_NAME
    fields['REF_FRAME'] = SUPPORTED_REF_FRAME
    fields['TIME_SYSTEM'] = SUPPORTED_TIME_SYSTEM
    fields['MEAN_ELEMENT_THEORY'] = 'SGP4'
    fields['EPOCH'] = from_datetime_to_omm_epoch(date_datetime)
    #the OMM stores the mean motion in rev/day and the angles in degrees:
    fields['MEAN_MOTION'] = _format_value(float(data['mean_motion'])*43200.0/np.pi)
    fields['ECCENTRICITY'] = _format_value(data['eccentricity'])
    fields['INCLINATION'] = _format_value(np.rad2deg(float(data['inclination'])))
    fields['RA_OF_ASC_NODE'] = _format_value(np.rad2deg(float(data['raan'])%(2*np.pi)))
    fields['ARG_OF_PERICENTER'] = _format_value(np.rad2deg(float(data['argument_of_perigee'])%(2*np.pi)))
    fields['MEAN_ANOMALY'] = _format_value(np.rad2deg(float(data['mean_anomaly'])%(2*np.pi)))
    fields['EPHEMERIS_TYPE'] = str(int(data['ephemeris_type']))
    fields['CLASSIFICATION_TYPE'] = str(data['classification'])[0]
    fields['NORAD_CAT_ID'] = str(int(data['satellite_catalog_number']))
    fields['ELEMENT_SET_NO'] = str(int(data['element_number']))
    fields['REV_AT_EPOCH'] = str(int(data['revolution_number_at_epoch']))
    fields['BSTAR'] = _format_value(data['b_star'])
    #the OMM stores the derivatives of the mean motion in rev/day**2 and rev/day**3, halved
    #and divided by six respectively, exactly as the TLE format does:
    fields['MEAN_MOTION_DOT'] = _format_value(float(data['mean_motion_first_derivative'])*1.86624e9/np.pi)
    fields['MEAN_MOTION_DDOT'] = _format_value(float(data['mean_motion_second_derivative'])*5.3747712e13/np.pi)
    return fields


def load_from_omm(fields, opsmode='i'):
    """
    This function takes an OMM as a dictionary of OMM fields, and returns both itself and its
    representation as a dictionary of elements (i.e. the same dictionary used by
    `dsgp4.tle.TLE`).

    Parameters:
    ----------------
    fields (``dict``): OMM data in the form of a dictionary of OMM fields
    opsmode (``str``): operation mode, either 'i' or 'a'

    Returns:
    ----------------
    ``dict``: OMM data in the form of a dictionary of OMM fields
    ``dict``: elements in the form of a dictionary
    """
    if not isinstance(fields, dict):
        raise ValueError('Expecting a dictionary of OMM fields')
    #the keywords of the standard are upper case, but the parsers are lenient about it:
    fields = {str(key).strip().upper(): value for key, value in fields.items()}

    theory = str(_field(fields, 'MEAN_ELEMENT_THEORY', 'SGP4')).strip().upper()
    if theory not in SUPPORTED_MEAN_ELEMENT_THEORIES:
        raise ValueError('Supported mean element theories: {} while {} was provided'.format(
            ', '.join(SUPPORTED_MEAN_ELEMENT_THEORIES), theory))

    reference_context = (
        ('CENTER_NAME', SUPPORTED_CENTER_NAME),
        ('REF_FRAME', SUPPORTED_REF_FRAME),
        ('TIME_SYSTEM', SUPPORTED_TIME_SYSTEM),
    )
    for key, supported in reference_context:
        value = str(_field(fields, key, supported)).strip().upper()
        if value != supported:
            raise ValueError('{} must be {} for dSGP4 propagation, while {} was provided'.format(
                key, supported, value))

    missing = [key for key in OMM_MEAN_ELEMENTS if _field(fields, key) is None]
    if missing:
        raise ValueError('The following mandatory OMM fields are missing: {}'.format(', '.join(missing)))

    #for SGP4:
    xpdotp   =  1440.0 / (2.0 *np.pi);
    #we initialize the elements dictionary:
    data = {}
    date_datetime = from_omm_epoch_to_datetime(fields['EPOCH'])
    year = date_datetime.year
    epochdays = util.from_datetime_to_fractional_day(date_datetime)
    date_string = date_datetime.strftime(format='%Y-%m-%d %H:%M:%S.%f')

    data['satellite_catalog_number'] = int(float(_field(fields, 'NORAD_CAT_ID', 0)))
    data['classification'] = str(_field(fields, 'CLASSIFICATION_TYPE', 'U'))[0]
    data['international_designator'] = from_object_id_to_international_designator(_field(fields, 'OBJECT_ID', ''))
    data['epoch_year'] = year
    data['epoch_days'] = epochdays
    data['date_string'] = date_string
    data['date_mjd'] = util.from_datetime_to_mjd(util.from_string_to_datetime(date_string))
    data['mean_motion_first_derivative'] = float(_field(fields, 'MEAN_MOTION_DOT', 0.))*np.pi/1.86624e9
    data['mean_motion_second_derivative'] = float(_field(fields, 'MEAN_MOTION_DDOT', 0.))*np.pi/5.3747712e13
    data['b_star'] = float(_field(fields, 'BSTAR', 0.))
    data['ephemeris_type'] = int(float(_field(fields, 'EPHEMERIS_TYPE', 0)))
    data['element_number'] = int(float(_field(fields, 'ELEMENT_SET_NO', 0)))
    data['inclination'] = np.deg2rad(float(fields['INCLINATION']))
    data['raan'] = np.deg2rad(float(fields['RA_OF_ASC_NODE']))
    data['eccentricity'] = float(fields['ECCENTRICITY'])
    data['argument_of_perigee'] = np.deg2rad(float(fields['ARG_OF_PERICENTER']))
    data['mean_anomaly'] = np.deg2rad(float(fields['MEAN_ANOMALY']))
    data['mean_motion'] = float(fields['MEAN_MOTION'])*np.pi/43200.0
    data['revolution_number_at_epoch'] = int(float(_field(fields, 'REV_AT_EPOCH', 0)))
    #for SGP4:
    data['_epochdays'] = epochdays
    data['_bstar'] = torch.tensor(float(_field(fields, 'BSTAR', 0.)))
    data['_ndot'] = torch.tensor(float(_field(fields, 'MEAN_MOTION_DOT', 0.))/(xpdotp*1440.0))
    data['_nddot'] = torch.tensor(float(_field(fields, 'MEAN_MOTION_DDOT', 0.))/(xpdotp*1440.0*1440))
    data['_inclo'] = torch.tensor(np.deg2rad(float(fields['INCLINATION'])))
    data['_nodeo'] = torch.tensor(np.deg2rad(float(fields['RA_OF_ASC_NODE'])))
    data['_ecco'] = torch.tensor(float(fields['ECCENTRICITY']))
    data['_argpo'] = torch.tensor(np.deg2rad(float(fields['ARG_OF_PERICENTER'])))
    data['_mo'] = torch.tensor(np.deg2rad(float(fields['MEAN_ANOMALY'])))
    data['_no_kozai'] = torch.tensor(float(fields['MEAN_MOTION']) / xpdotp);

    add_derived_quantities(data, year, epochdays, opsmode)
    # Process the optional descriptive fields
    if _field(fields, 'OBJECT_NAME') is not None:
        data['name'] = str(fields['OBJECT_NAME']).strip()
    if _field(fields, 'CREATION_DATE') is not None:
        data['creation_date'] = str(fields['CREATION_DATE']).strip()
    if _field(fields, 'ORIGINATOR') is not None:
        data['originator'] = str(fields['ORIGINATOR']).strip()
    return fields, data


def load_from_data(data, opsmode='i'):
    """
    This function takes a set of elements as a dictionary (i.e. the same dictionary used by
    `dsgp4.tle.TLE`), and returns both its representation as a dictionary of OMM fields and
    the dictionary itself.

    Parameters:
    ----------------
    data (``dict``): elements in the form of a dictionary
    opsmode (``str``): operation mode, either 'i' or 'a'

    Returns:
    ----------------
    ``dict``: OMM data in the form of a dictionary of OMM fields
    ``dict``: elements in the form of a dictionary
    """
    fields = to_omm_fields(data)
    #the elements are then read back from the OMM fields, so that the two representations are
    #guaranteed to be consistent with each other:
    fields, omm_data = load_from_omm(fields, opsmode=opsmode)
    for key in ('line0', 'line1', 'line2'):
        #the TLE lines, if any, are not part of an OMM and would become stale:
        data.pop(key, None)
    data.update(omm_data)
    return fields, data


def load(file_name, file_format=None):
    """
    This function takes a file name that contains OMM data (in JSON, XML, KVN or CSV format),
    and returns a list of OMM objects.

    Parameters:
    ----------------
    file_name (``str``): OMM file name
    file_format (``str``): format of the file, either 'json', 'xml', 'kvn' or 'csv' (if None, it is detected from the file extension and content)

    Returns:
    ----------------
    ``list``: list of `dsgp4.omm.OMM` objects
    """
    with open(file_name) as f:
        text = f.read()
    if file_format is None:
        file_format = FORMAT_BY_EXTENSION.get(os.path.splitext(file_name)[1].lower())
    return [OMM(fields) for fields in loads(text, file_format=file_format)]


class OMM(TLE):
    """
    This class constructs an OMM (i.e. Orbit Mean-Elements Message) object from a string
    containing a single message (in JSON, XML, KVN or CSV format), from a dictionary of OMM
    fields, or from a dictionary of elements (i.e. the same dictionary accepted by
    `dsgp4.tle.TLE`).

    Since an OMM carries the same SGP4 mean elements of a TLE, the resulting object can be used
    everywhere a `dsgp4.tle.TLE` object is expected. Contrarily to a TLE, though, it is not
    constrained by the two fixed-width lines: objects whose catalog number is above 339999
    (i.e. beyond what the Alpha-5 convention can encode) can only be represented as OMMs.

    Parameters:
    ----------------
    data (`str`, `dict`): OMM data

    Returns:
    ----------------
    `dsgp4.omm.OMM` object
    """
    def __init__(self, data):
        if isinstance(data, str):
            records = loads(data)
            if len(records) != 1:
                raise ValueError('Expecting a string with a single OMM, while {} were found: '
                                 'use `dsgp4.omm.load` to read a file with several of them.'.format(len(records)))
            self._fields, self._data = load_from_omm(records[0])
        elif isinstance(data, dict):
            #the keywords of the standard are upper case, while the dictionary of elements
            #uses lower case keys (e.g. 'mean_motion'), so the two are told apart by case:
            if 'MEAN_MOTION' in data or 'EPOCH' in data:
                self._fields, self._data = load_from_omm(data)
            else:
                self._fields, self._data = load_from_data(data)
        else:
            raise RuntimeError('Expecting a string with an OMM message, a dictionary of OMM fields, or a dictionary of elements.')

    def _set_from(self, other):
        """
        This function replaces the content of the object with the one of another OMM object.

        Parameters:
        ----------------
        other (`dsgp4.omm.OMM`): object whose content is copied into `self`
        """
        self._data = other._data
        self._fields = other._fields

    def to_tle(self):
        """
        This function returns the TLE representation of the OMM object. Note that the TLE
        format cannot represent all the objects an OMM can: a `ValueError` is raised when the
        satellite catalog number does not fit the two lines.

        Returns:
            `dsgp4.tle.TLE` object
        """
        return TLE(copy_data(self._data))

    def __getattr__(self, attr):
        #an OMM is not made of two lines: rather than the generic error of the TLE class, the
        #line attributes point at the conversion that does produce them:
        if attr in ('line0', 'line1', 'line2', '_lines'):
            raise AttributeError("an OMM has no TLE lines ('{}'): use `to_tle()` to build the "
                                 "TLE representation, which is only possible when the satellite "
                                 "catalog number fits the two lines.".format(attr))
        return super().__getattr__(attr)

    def __repr__(self):
        return 'OMM(\n{}\n)'.format(dumps(self, file_format='kvn').strip())


def _local_name(tag):
    """
    This function returns the name of an XML tag, stripped of its namespace, if any.

    Parameters:
    ----------------
    tag (``str``): XML tag

    Returns:
    ----------------
    ``str``: name of the tag
    """
    return str(tag).rpartition('}')[2]


def _leaf_fields(element):
    """
    This function returns the non-empty leaves of an XML element, in the form of a dictionary
    that maps the name of each tag to its content.

    Parameters:
    ----------------
    element (``xml.etree.ElementTree.Element``): XML element

    Returns:
    ----------------
    ``dict``: dictionary of OMM fields
    """
    return {_local_name(leaf.tag): leaf.text.strip() for leaf in element.iter()
            if len(leaf) == 0 and leaf.text is not None and leaf.text.strip()}


def _add_xml_fields(element, fields, keys):
    """
    This function adds the requested OMM fields, when available, as children of an XML element.

    Parameters:
    ----------------
    element (``xml.etree.ElementTree.Element``): XML element
    fields (``dict``): dictionary of OMM fields
    keys (``tuple``): OMM fields to be added
    """
    for key in keys:
        if fields.get(key) is not None:
            ElementTree.SubElement(element, key).text = str(fields[key])


def _format_value(value):
    """
    This function returns the string representation of the value of an OMM field. Fourteen
    significant digits are used: this is well beyond the precision of any mean element, and
    avoids the noise that the conversions between radians and degrees would otherwise leave
    in the last digits of the written message.

    Parameters:
    ----------------
    value (``float``, ``torch.tensor``): value of an OMM field

    Returns:
    ----------------
    ``str``: string representation of the value
    """
    return '{:.14g}'.format(float(value))


def _field(fields, key, default=None):
    """
    This function returns the value of an OMM field, or a default value when the field is
    either missing or empty (the CSV format, for instance, leaves the fields that do not apply
    to an object empty).

    Parameters:
    ----------------
    fields (``dict``): dictionary of OMM fields
    key (``str``): OMM field
    default: value returned when the field is missing or empty

    Returns:
    ----------------
    value of the OMM field
    """
    value = fields.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    return value
