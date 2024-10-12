import datetime
import numpy as np
import copy
import torch

from . import util
_, MU_EARTH, _, _, _, _, _, _=util.get_gravity_constants('wgs-84')
MU_EARTH = MU_EARTH*1e9

# This function is from python-sgp4 released under MIT License, (c) 2012–2016 Brandon Rhodes
def compute_checksum(line):
    """
    This function takes a TLE line in the form of a string and computes the checksum.

    Parameters:
    ----------------
    line (``str``): TLE line

    Returns:
    ----------------
    ``int``: checksum
    """
    return sum((int(c) if c.isdigit() else c == '-') for c in line[0:68]) % 10


# This function is from python-sgp4 released under MIT License, (c) 2012–2016 Brandon Rhodes
def read_satellite_catalog_number(string):
    """
    This function takes a string corresponding to part of a TLE line, and returns
    the corresponding satellite catalog number.
    
    Parameters:
    ----------------
    string (``str``): string line
    
    Returns:
    ----------------
    ``int``: satellite catalog number
    """
    if not string[0].isalpha():
        return int(string)
    character = string[0]
    n = ord(character) - ord('A') + 10
    n -= character > 'I'
    n -= character > 'O'
    return n * 10000 + int(string[1:])


# Parts of this function is based on python-sgp4 released under MIT License, (c) 2012–2016 Brandon Rhodes
def load_from_lines(lines, opsmode='i'):
    """
    This function takes a TLE as a list of strings and returns both itself and its dictionary representation.
    
    Parameters:
    ----------------
    lines (``list``): TLE data in the form of a list
    opsmode (``str``): operation mode, either 'i' or 'a'

    Returns:
    ----------------
    ``list``: TLE data in the form of a list
    ``dict``: TLE data in the form of a dictionary
    """
    if isinstance(lines, str):
        lines = util.get_non_empty_lines(lines)
    elif isinstance(lines, list):
        if not isinstance(lines[0], str):
            raise ValueError('Expecting a list of strings')
    else:
        raise ValueError('Expecting either a string or a list of strings')

    if len(lines) == 2:
        line0, line1, line2 = None, lines[0], lines[1]
    elif len(lines) == 3:
        line0, line1, line2 = lines[0], lines[1], lines[2]
    else:
        raise ValueError('Expecting a string with two or three lines')
    #for SGP4:
    xpdotp   =  1440.0 / (2.0 *np.pi);
    # if type(line1)!= str or type(line2)!=str:
        # raise ValueError('Input shall be a list of two string elements (1st and 2nd line).')
    #we initialize the tle dictionary:
    data = {}
    #we remove any trailing character:
    line = line1.rstrip()
    if (len(line) >= 64 and
        line.startswith('1 ') and
        line[8] == ' ' and
        line[23] == '.' and
        line[32] == ' ' and
        line[34] == '.' and
        line[43] == ' ' and
        line[52] == ' ' and
        line[61] == ' ' and
        line[63] == ' '):

        data['satellite_catalog_number'] = read_satellite_catalog_number(line1[2:7])
        data['classification']           = line[7] or 'U'
        data['international_designator'] = line[9:17].rstrip()
        two_digit_year=int(line[18:20])
        if two_digit_year<57:
            year = two_digit_year + 2000#int('19'+line[18:20])
        else:
            year = two_digit_year + 1900
        epochdays = float(line[20:32])
        date_datetime = datetime.datetime(year-1, 12, 31, 0, 0, 0, 0)+datetime.timedelta(days = epochdays)
        date_string = date_datetime.strftime(format = '%Y-%m-%d %H:%M:%S.%f')
        data['epoch_year'] = date_datetime.year
        data['epoch_days'] = epochdays
        data['date_string'] = date_string
        data['date_mjd'] = util.from_datetime_to_mjd(util.from_string_to_datetime(date_string))
        data['mean_motion_first_derivative'] = float(line[33:43])*np.pi/1.86624e9
        data['mean_motion_second_derivative'] = float(line[44]+'.'+line[45:50])*pow(10, int(line[50:52]))*np.pi/5.3747712e13
        data['b_star'] = float(line[53]+'.'+line[54:59])*pow(10, int(line[59:61]))
        data['ephemeris_type'] = int(line[62])
        data['element_number'] = int(line[64:68])
        data['line1'] = line
        #for SGP4:
        data['_epochdays'] = epochdays
        data['_bstar'] = torch.tensor(float(line[53]+'.'+line[54:59])*pow(10, int(line[59:61])))
        data['_ndot'] = torch.tensor(float(line[33:43])/(xpdotp*1440.0))
        data['_nddot']= torch.tensor(float(line[44] + '.' + line[45:50])/(xpdotp*1440.0*1440))

    else:
        raise ValueError('First line not compatible with TLE format.')

    line = line2.rstrip()

    if (len(line) >= 69 and
        line.startswith('2 ') and
        line[7] == ' ' and
        line[11] == '.' and
        line[16] == ' ' and
        line[20] == '.' and
        line[25] == ' ' and
        line[33] == ' ' and
        line[37] == '.' and
        line[42] == ' ' and
        line[46] == '.' and
        line[51] == ' '):

        if data['satellite_catalog_number'] != read_satellite_catalog_number(line[2:7]):
            raise ValueError('Satellite catalog numbers of line1 and line2 do not match.')

        data['inclination'] = np.deg2rad(float(line[8:16]))
        data['raan'] = np.deg2rad(float(line[17:25]))
        data['eccentricity'] = float('0.'+line[26:33].replace(' ', '0'))
        data['argument_of_perigee'] = np.deg2rad(float(line[34:42]))
        data['mean_anomaly'] = np.deg2rad(float(line[43:51]))
        data['mean_motion'] = float(line[52:63])*np.pi/43200.0
        data['revolution_number_at_epoch'] = int(line[63:68])
        data['line2'] = line
        #for SGP4:
        data['_inclo']=torch.tensor(np.deg2rad(float(line[8:16])))
        data['_nodeo']=torch.tensor(np.deg2rad(float(line[17:25])))
        data['_ecco'] =torch.tensor(float('0.'+line[26:33].replace(' ', '0')))
        data['_argpo']=torch.tensor(np.deg2rad(float(line[34:42])))
        data['_mo']=torch.tensor(np.deg2rad(float(line[43:51])))
        data['_no_kozai'] =torch.tensor(float(line[52:63]) / xpdotp);

    else:
        raise ValueError('Second line not compatible with TLE format.')

    mon,day,hr,minute,sec = util.days2mdhms(year, epochdays);
    sec_whole, sec_fraction = divmod(sec, 1.0)
    data['_epochyr'] = torch.tensor(year)
    data['_jdsatepoch'] = torch.tensor(util.jday(year,mon,day,hr,minute,sec)[0]);
    data['_jdsatepochF'] = torch.tensor(util.jday(year,mon,day,hr,minute,sec)[1]);

    #I also add the semi-major axis:
    data['semi_major_axis'] = (MU_EARTH/(data['mean_motion']**2))**(1.0/3.0)

    try:
        data['_epoch'] = datetime.datetime(year, mon, day, hr, minute, int(sec_whole),
                                int(sec_fraction * 1000000.0 // 1.0))
    except ValueError:
        # Sometimes a TLE says something like "2019 + 366.82137887 days"
        # which would be December 32nd which causes a ValueError.
        year, mon, day, hr, minute, sec = util.invjday(data['_jdsatepoch'])
        data['_epoch'] = datetime.datetime(year, mon, day, hr, minute, int(sec_whole),
                                int(sec_fraction * 1000000.0 // 1.0))
    data['_opsmode']=opsmode
    # Process optional line zero
    if line0 is not None:
        data['line0'] = line0
        if line0.startswith('0 '):
            line0 = line0[2:]
        data['name'] = line0.strip()
    return lines, data


# Parts of this function is based on python-sgp4 released under MIT License, (c) 2012–2016 Brandon Rhodes
def load_from_data(data, opsmode='i'):
    """
    This function takes a TLE as a dictionary and returns both itself and its representation as a list of strings.

    Parameters:
    ----------------
    data (`dict`): TLE data in the form of a dictionary
    opsmode (`str`): 'i' for improved, 's' for simplified

    Returns:
    ----------------
    `list`: TLE data in the form of a list
    `dict`: TLE data in the form of a dictionary
    """
    #for SGP4:
    xpdotp   =  1440.0 / (2.0 *np.pi);

    data['raan']=data['raan']%(2*np.pi)
    data['argument_of_perigee']=data['argument_of_perigee']%(2*np.pi)
    data['mean_anomaly']=data['mean_anomaly']%(2*np.pi)
    line1 = ['1 ']
    line1.append(str(data['satellite_catalog_number']).zfill(5)[:5])
    line1.append(str(data['classification'])[0] + ' ')
    line1.append(str(data['international_designator']).ljust(8, ' ')[:8] + ' ')
    line1.append(str(data['epoch_year'])[-2:].zfill(2) + '{:012.8f}'.format(data['epoch_days']) + ' ')
    line1.append('{0: 8.8f}'.format(data['mean_motion_first_derivative'] * (1.86624e9 / np.pi)).replace('0', '', 1) + ' ')
    line1.append('{0: 4.4e}'.format((data['mean_motion_second_derivative'] * (5.3747712e13 / np.pi)) * 10).replace(".", '').replace('e+00', '-0').replace('e-0', '-').replace('e+0', '+') + ' ')
    line1.append('{0: 4.4e}'.format(data['b_star'] * 10).replace('.', '').replace('e+0', '+').replace('e-0', '-') + ' ')
    line1.append('{} '.format(data['ephemeris_type']) + str(data['element_number']).rjust(4, ' '))
    line1 = ''.join(line1)
    line1 += str(compute_checksum(line1))
    two_digit_year=int(line1[18:20])
    if two_digit_year<57:
        year = two_digit_year + 2000#int('19'+line1[18:20])
    else:
        year = two_digit_year + 1900#int('20'+line1[18:20])
    epochdays = float(line1[20:32])
    data['_epochdays'] = epochdays
    date_datetime = datetime.datetime(year-1, 12, 31, 0, 0, 0, 0)+datetime.timedelta(days = epochdays)
    date_string = date_datetime.strftime(format = '%Y-%m-%d %H:%M:%S.%f')
    data['epoch_year'] = date_datetime.year
    data['epoch_days'] = epochdays
    data['date_string'] = date_string
    #for SGP4:
    data['_bstar'] = torch.tensor(float(line1[53]+'.'+line1[54:59])*pow(10, int(line1[59:61])))
    data['_ndot'] = torch.tensor(float(line1[33:43])/(xpdotp*1440.0))
    data['_nddot']= torch.tensor(float(line1[44] + '.' + line1[45:50])/(xpdotp*1440.0*1440))

    line2 = ['2 ']
    line2.append(str(data['satellite_catalog_number']).zfill(5)[:5] + ' ')
    line2.append('{0:8.4f}'.format(data['inclination'] * (180 / np.pi)).rjust(8, ' ') + ' ')
    line2.append('{0:8.4f}'.format(data['raan'] * (180 / np.pi)).rjust(8, ' ') + ' ')
    line2.append(str(round(float(data['eccentricity']) * 1e7)).rjust(7, '0')[:7] + ' ')
    line2.append('{0:8.4f}'.format(data['argument_of_perigee'] * (180 / np.pi)).rjust(8, ' ') + ' ')
    line2.append('{0:8.4f}'.format(data['mean_anomaly'] * (180 / np.pi)).rjust(8, ' ') + ' ')
    line2.append('{0:11.8f}'.format(data['mean_motion'] * 43200.0 / np.pi).rjust(8, ' '))
    line2.append(str(data['revolution_number_at_epoch']).rjust(5))
    line2 = ''.join(line2)
    line2 += str(compute_checksum(line2))
    #for SGP4:
    data['_inclo']=torch.tensor(np.deg2rad(float(line2[8:16])))
    data['_nodeo']=torch.tensor(np.deg2rad(float(line2[17:25])))
    data['_ecco'] =torch.tensor(float('0.'+line2[26:33].replace(' ', '0')))
    data['_argpo']=torch.tensor(np.deg2rad(float(line2[34:42])))
    data['_mo']=torch.tensor(np.deg2rad(float(line2[43:51])))
    data['_no_kozai'] =torch.tensor(float(line2[52:63]) / xpdotp);

    if len(line1) != 69:
        raise RuntimeError('TLE line 1 has unexpected length ({})'.format(len(line1)))
    if len(line2) != 69:
        raise RuntimeError('TLE line 2 has unexpected length ({})'.format(len(line2)))

    data['line0'] = None
    data['line1'] = line1
    data['line2'] = line2
    date_datetime = datetime.datetime(data['epoch_year']-1, 12, 31, 0, 0, 0, 0)+datetime.timedelta(days = data['epoch_days'])
    date_string = date_datetime.strftime(format = '%Y-%m-%d %H:%M:%S.%f')
    data['date_string'] = date_string
    data['date_mjd'] = util.from_datetime_to_mjd(util.from_string_to_datetime(date_string))
    data['semi_major_axis'] = (MU_EARTH/(data['mean_motion']**2))**(1.0/3.0)
    lines = [line1, line2]
    mon,day,hr,minute,sec = util.days2mdhms(year, epochdays);
    sec_whole, sec_fraction = divmod(sec, 1.0)
    data['_epochyr'] = torch.tensor(year)
    data['_jdsatepoch'] = torch.tensor(util.jday(year,mon,day,hr,minute,sec)[0]);
    data['_jdsatepochF'] = torch.tensor(util.jday(year,mon,day,hr,minute,sec)[1]);

    #I also add the semi-major axis:
    data['semi_major_axis'] = (MU_EARTH/(data['mean_motion']**2))**(1.0/3.0)
    try:
        data['_epoch'] = datetime.datetime(year, mon, day, hr, minute, int(sec_whole),
                                int(sec_fraction * 1000000.0 // 1.0))
    except ValueError:
        # Sometimes a TLE says something like "2019 + 366.82137887 days"
        # which would be December 32nd which causes a ValueError.
        year, mon, day, hr, minute, sec = util.invjday(data['_jdsatepoch'])
        data['_epoch'] = datetime.datetime(year, mon, day, hr, minute, int(sec_whole),
                                int(sec_fraction * 1000000.0 // 1.0))
    data['_opsmode']=opsmode
    if 'name' in data:
        data['line0'] = '0 '+data['name']
    return lines, data


def load(file_name):
    """
    This function takes a file name that contains TLE (either with names or without),
    and returns a list of TLE objects.

    Parameters:
    ----------------
    file_name (`str`): TLEs file name

    Returns:
    ----------------
    `list`: list of `dsgp4.tle.TLE` objects
    """
    with open(file_name) as f:
        lines = util.get_non_empty_lines(f.read())

    i = 0
    tles = []
    while True:
        line = lines[i]
        if not (line.startswith('1 ') or line.startswith('2 ')):
            # This is a line0 (extra name line)
            tle = TLE(lines[i:i+3])
            i += 3
        else:
            # This is assumed to be line1 (first line of a TLE)
            tle = TLE(lines[i:i+2])
            i += 2
        tles.append(tle)
        if i == len(lines):
            break
    return tles


class TLE():
    """
    This class constructs a TLE object from either a list of strings (that make up the TLE)
    or from a dictionary.

    Parameters:
    ----------------
    data (`str`, `list`, `dict`): TLE data

    Returns:
    ----------------
    `dsgp4.tle.TLE` object
    """
    def __init__(self, data):
        if isinstance(data, list) or isinstance(data, str):
            self._lines, self._data = load_from_lines(data)
        elif isinstance(data, dict):
            self._lines, self._data = load_from_data(data)
        else:
            raise RuntimeError('Expecting a string of TLE, list of strings with TLE lines, or a dictionary of TLE data.')

    def copy(self):
        """
        This function returns a copy of the TLE object.

        Returns:
            `dsgp4.tle.TLE` object
        """
        d = {k: (v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)) for k, v in self._data.items()}
        return TLE(d)

    def set_time(self, date_mjd):
        """
        This function sets the epoch of the TLE to the given date.


        Parameters:
        ----------------
        date_mjd (`float`): date in MJD
        """
        d = copy.deepcopy(self._data)
        d['epoch_year'] = util.from_mjd_to_datetime(date_mjd).year
        d['epoch_days'] = util.from_mjd_to_epoch_days_after_1_jan(date_mjd)
        mon,day,hr,minute,sec = util.days2mdhms(d['epoch_year'], d['epoch_days']);
        sec_whole, sec_fraction = divmod(sec, 1.0)
        d['_epochyr'] = torch.tensor(d['epoch_year'])
        d['_jdsatepoch'] = torch.tensor(util.jday(d['epoch_year'],mon,day,hr,minute,sec)[0]);
        d['_jdsatepochF'] = torch.tensor(util.jday(d['epoch_year'],mon,day,hr,minute,sec)[1]);
        tle = TLE(d)
        self._data = tle._data
        self._lines = tle._lines

    def update(self, tle_data):
        """
        This function updates the TLE object with the given data.


        Parameters:
        ----------------
        tle_data (`dict`): dictionary of TLE data
        """
        d = copy.deepcopy(self._data)
        for k, v in tle_data.items():
            d[k] = v
        tle = TLE(d)
        self._mo=tle['_mo']
        self._bstar=tle['_bstar']
        self._ndot=tle['_ndot']
        self._nddot=tle['_nddot']
        self._ecco= tle['_ecco']
        self._argpo=tle['_argpo']
        self._inclo=tle['_inclo']
        self._no_kozai=tle['_no_kozai']
        self._nodeo=tle['_nodeo']
        self._data = tle._data
        self._lines = tle._lines

    def perigee_alt(self, R_eq = 6378135.0):
        """
        This function returns the perigee altitude of a given TLE.


        Parameters:
        ----------------
        R_eq (``float``): equatorial radius of the Earth [m]

        Returns:
        ----------------
        ``float``: perigee altitude
        """
        return self.semi_major_axis*(1-self.eccentricity)-R_eq

    def apogee_alt(self, R_eq = 6378135.0):
        """
        This function returns the apogee altitude of a given TLE.


        Parameters:
        ----------------
        R_eq (``float``): equatorial radius of the Earth [m]

        Returns:
        ----------------
        ``float``: apogee altitude
        """
        return self.semi_major_axis*(1+self.eccentricity)-R_eq

    def __repr__(self):
        return 'TLE(\n{}\n)'.format('\n'.join(self._lines))

    def __getitem__(self, index):
        return self._data[index]

    def __getattr__(self, attr):
        if attr == '_data' or attr == '_lines':
            return super().__getattr__(attr)
        elif attr in self._data.keys():
            return self[attr]
        else:
            return super().__getattr__(attr)

