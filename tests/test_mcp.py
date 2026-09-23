"""
Tests of the optional MCP layer (`dsgp4.mcp`). They are skipped when the `mcp`
dependency (the `[mcp]` extra, which requires Python >= 3.10) is not installed.
"""
import asyncio
import base64
import json

import pytest

pytest.importorskip('mcp.server.mcpserver', reason="the MCP layer requires the 'mcp' extra (mcp>=2, Python>=3.10)")

import numpy as np
import torch

import dsgp4
from dsgp4.mcp import create_server
from dsgp4.mcp.server import DOMAINS

TLE_LINES = [
    '1 34427U 93036RU  22068.94647328  .00008100  00000-0  11455-2 0  9999',
    '2 34427  74.0145 306.8269 0033346  13.0723 347.1308 14.76870515693886',
]
TLE = '\n'.join(TLE_LINES)
TLE_2 = ('1 34428U 93036RV  22068.90158861  .00002627  00000-0  63561-3 0  9999\n'
         '2 34428  74.0386 139.1157 0038434 196.7068 279.9147 14.53118052686950')
DEEP_SPACE_TLE = ('1 14128U 83058A   06176.02844893 -.00000158  00000-0  10000-3 0  9627\n'
                  '2 14128  11.4384  35.2134 0011562  26.4582 333.5652  0.98870114 46093')


@pytest.fixture(scope='module')
def server():
    return create_server()


def call(server, name, arguments):
    result = asyncio.run(server.call_tool(name, arguments))
    assert not result.is_error, result.content[0].text
    return json.loads(result.content[0].text)


def test_server_registers_all_domains(server):
    tools = {tool.name for tool in asyncio.run(server.list_tools())}
    expected = {'parse_element_set', 'validate_tle', 'fix_tle_checksum', 'build_tle', 'convert_element_sets',
                'propagate', 'propagate_batch', 'cartesian_to_keplerian', 'convert_time', 'orbital_elements_summary',
                'state_partials_wrt_tle', 'state_partials_wrt_time', 'transform_covariance',
                'update_tle_epoch', 'fit_tle_to_state',
                'mldsgp4_propagate', 'mldsgp4_inspect',
                'plot_orbits', 'plot_element_distributions'}
    assert expected == tools
    prompts = {prompt.name for prompt in asyncio.run(server.list_prompts())}
    assert {'characterize_orbit', 'compare_orbits', 'uncertainty_analysis',
            'tle_determination', 'mldsgp4_training_guide'} <= prompts


def test_domain_selection():
    narrow = create_server(domains=['tle'])
    tools = {tool.name for tool in asyncio.run(narrow.list_tools())}
    assert 'parse_element_set' in tools and 'propagate' not in tools
    with pytest.raises(ValueError):
        create_server(domains=['bogus'])
    assert set(DOMAINS) == {'tle', 'propagation', 'gradients', 'estimation', 'ml', 'plot'}


def test_parse_element_set(server):
    parsed = call(server, 'parse_element_set', {'element_set': TLE})
    assert parsed['format'] == 'TLE'
    assert parsed['satellite_catalog_number'] == 34427
    assert parsed['elements']['inclination_deg'] == pytest.approx(74.0145)
    assert parsed['elements']['mean_motion_revs_per_day'] == pytest.approx(14.76870515693886, abs=1e-8)
    assert parsed['derived']['orbit_class'] == 'LEO'
    assert not parsed['derived']['uses_deep_space_corrections']
    assert parsed['lines'] == TLE_LINES

    deep = call(server, 'parse_element_set', {'element_set': DEEP_SPACE_TLE})
    assert deep['derived']['uses_deep_space_corrections']


def test_validate_and_checksum(server):
    assert call(server, 'validate_tle', {'tle': TLE})['valid']
    corrupted = call(server, 'validate_tle', {'tle': TLE.replace('9999', '9990', 1)})
    assert not corrupted['valid'] and 'checksum' in corrupted['errors'][0]
    fixed = call(server, 'fix_tle_checksum', {'line': TLE_LINES[0][:68]})
    assert fixed['line'] == TLE_LINES[0]


def test_build_tle_roundtrip(server):
    built = call(server, 'build_tle', {
        'satellite_catalog_number': 34427,
        'epoch_utc': '2022-03-09T22:42:55',
        'mean_motion_revs_per_day': 14.76870515,
        'eccentricity': 0.0033346,
        'inclination_deg': 74.0145,
        'raan_deg': 306.8269,
        'argument_of_perigee_deg': 13.0723,
        'mean_anomaly_deg': 347.1308,
        'b_star': 0.0011455,
    })
    assert len(built['lines']) == 2
    reparsed = call(server, 'parse_element_set', {'element_set': '\n'.join(built['lines'])})
    assert reparsed['elements']['inclination_deg'] == pytest.approx(74.0145)
    assert reparsed['elements']['eccentricity'] == pytest.approx(0.0033346)


def test_convert_element_sets(server):
    as_json = call(server, 'convert_element_sets', {'element_sets': TLE, 'output_format': 'json'})
    assert as_json['number_of_objects'] == 1
    fields = json.loads(as_json['content'])[0]
    assert int(fields['NORAD_CAT_ID']) == 34427
    back = call(server, 'convert_element_sets', {'element_sets': as_json['content'], 'output_format': 'tle'})
    reparsed = call(server, 'parse_element_set', {'element_set': back['content']})
    assert reparsed['satellite_catalog_number'] == 34427


def test_propagate_matches_library(server):
    times = [0.0, 30.0, 120.0]
    result = call(server, 'propagate', {'element_set': TLE, 'minutes_since_epoch': times})
    assert result['frame'] == 'TEME' and len(result['states']) == 3 and not result['warnings']
    tle = dsgp4.tle.TLE(TLE_LINES)
    dsgp4.initialize_tle(tle)
    expected = dsgp4.propagate(tle, torch.tensor(times)).detach().numpy()
    for state, reference in zip(result['states'], expected):
        assert np.allclose(state['position_km'], reference[0])
        assert np.allclose(state['velocity_km_s'], reference[1])


def test_propagate_with_dates_and_deep_space(server):
    result = call(server, 'propagate', {'element_set': DEEP_SPACE_TLE,
                                        'dates_utc': ['2006-06-25T02:00:00']})
    state = result['states'][0]
    radius = np.linalg.norm(state['position_km'])
    assert 40000.0 < radius < 45000.0  #geosynchronous altitude
    assert state['date_utc'].startswith('2006-06-25T0')


def test_propagate_batch(server):
    result = call(server, 'propagate_batch', {'element_sets': TLE + '\n' + TLE_2,
                                              'minutes_since_epoch': [10.0]})
    assert [state['satellite_catalog_number'] for state in result['states']] == [34427, 34428]
    single = call(server, 'propagate', {'element_set': TLE, 'minutes_since_epoch': [10.0]})
    assert np.allclose(result['states'][0]['position_km'], single['states'][0]['position_km'])


def test_cartesian_to_keplerian(server):
    state = call(server, 'propagate', {'element_set': TLE, 'minutes_since_epoch': [0.0]})['states'][0]
    elements = call(server, 'cartesian_to_keplerian', {'position_km': state['position_km'],
                                                       'velocity_km_s': state['velocity_km_s']})
    #osculating vs mean elements: only a loose agreement is expected
    assert elements['inclination_deg'] == pytest.approx(74.0, abs=0.5)
    assert elements['orbital_period_minutes'] == pytest.approx(1440.0 / 14.7687, abs=1.0)


def test_convert_time(server):
    result = call(server, 'convert_time', {'date_utc': '2022-03-09T22:42:55', 'element_set': TLE})
    assert result['jd'] == pytest.approx(result['mjd'] + 2400000.5)
    assert abs(result['tsince_minutes']) < 10.0  #close to the TLE epoch
    with pytest.raises(Exception):
        call(server, 'convert_time', {})


def test_state_partials_wrt_tle(server):
    result = call(server, 'state_partials_wrt_tle', {'element_set': TLE, 'minutes_since_epoch': [0.0, 30.0]})
    assert result['columns'][3] == 'eccentricity'
    jacobian = result['entries'][1]['jacobian']
    assert len(jacobian) == 6 and len(jacobian[0]) == 9
    #cross-check one entry against a direct autograd evaluation:
    tle = dsgp4.tle.TLE(TLE_LINES)
    elements = dsgp4.initialize_tle(tle, with_grad=True)
    states = dsgp4.propagate(tle, torch.tensor([0.0, 30.0])).reshape(-1, 2, 3)
    gradient = torch.autograd.grad(states[1, 0, 2], elements)[0]
    assert jacobian[2][3] == pytest.approx(float(gradient[3]))


def test_state_partials_wrt_time(server):
    result = call(server, 'state_partials_wrt_time', {'element_set': TLE, 'minutes_since_epoch': [15.0]})
    entry = result['entries'][0]
    #the derivative of the position wrt time (km/min) matches the velocity (km/s):
    for derivative, velocity in zip(entry['dstate_dt'][:3], entry['velocity_km_s']):
        assert derivative == pytest.approx(velocity * 60.0, rel=1e-2)


def test_transform_covariance(server):
    covariance = (1e-8 * np.eye(6)).tolist()
    result = call(server, 'transform_covariance', {'element_set': TLE, 'covariance': covariance,
                                                   'minutes_since_epoch': [0.0]})
    entry = result['entries'][0]
    matrix = np.array(entry['covariance_teme'])
    assert matrix.shape == (6, 6)
    assert np.allclose(matrix, matrix.T)
    assert all(sigma >= 0.0 for sigma in entry['sigmas'])
    #C = J C0 J^T with C0 = s*I must equal s * J J^T:
    partials = call(server, 'state_partials_wrt_tle', {'element_set': TLE, 'minutes_since_epoch': [0.0]})
    jacobian = np.array(partials['entries'][0]['jacobian'])[:, [3, 4, 5, 6, 7, 8]]
    assert np.allclose(matrix, 1e-8 * jacobian @ jacobian.T)


def test_update_tle_epoch(server):
    result = call(server, 'update_tle_epoch', {'element_set': TLE, 'new_epoch_utc': '2022-03-10T12:00:00'})
    assert result['epoch_utc'].startswith('2022-03-10T1')
    assert result['fit']['position_residual_km'] < 1.0


def test_fit_tle_to_state(server):
    target = call(server, 'propagate', {'element_set': TLE, 'minutes_since_epoch': [720.0]})['states'][0]
    result = call(server, 'fit_tle_to_state', {'template_element_set': TLE,
                                               'position_km': target['position_km'],
                                               'velocity_km_s': target['velocity_km_s'],
                                               'date_utc': target['date_utc']})
    assert result['fit']['position_residual_km'] < 1.0
    assert result['satellite_catalog_number'] == 34427


def test_mldsgp4_propagate_untrained_warns(server):
    result = call(server, 'mldsgp4_propagate', {'element_set': TLE, 'minutes_since_epoch': [0.0, 10.0]})
    assert len(result['states']) == 2
    assert not result['model']['trained']
    assert any('UNTRAINED' in warning for warning in result['warnings'])


def test_mldsgp4_checkpoint_roundtrip(server, tmp_path):
    model = dsgp4.mldsgp4(hidden_size=17)
    path = str(tmp_path / 'mldsgp4.pth')
    torch.save(model.state_dict(), path)
    inspected = call(server, 'mldsgp4_inspect', {'model_path': path})
    assert inspected['hidden_size'] == 17
    assert inspected['number_of_parameters'] == sum(p.numel() for p in model.parameters())
    result = call(server, 'mldsgp4_propagate', {'element_set': TLE, 'minutes_since_epoch': [5.0],
                                                'model_path': path, 'hidden_size': 17})
    assert result['model']['trained'] and not result['warnings']


def _image_bytes(result):
    content = result.content[0]
    assert content.mime_type == 'image/png'
    return base64.b64decode(content.data)


def test_plot_orbits(server, tmp_path):
    path = str(tmp_path / 'orbit.png')
    result = asyncio.run(server.call_tool('plot_orbits', {'element_sets': TLE + '\n' + DEEP_SPACE_TLE,
                                                          'file_path': path}))
    assert not result.is_error
    image = _image_bytes(result)
    assert image[:8] == b'\x89PNG\r\n\x1a\n'
    with open(path, 'rb') as saved:
        assert saved.read()[:8] == b'\x89PNG\r\n\x1a\n'


def test_plot_element_distributions(server):
    result = asyncio.run(server.call_tool('plot_element_distributions',
                                          {'element_sets': TLE + '\n' + TLE_2}))
    assert not result.is_error
    assert _image_bytes(result)[:8] == b'\x89PNG\r\n\x1a\n'


def test_resources(server):
    resources = {str(resource.uri) for resource in asyncio.run(server.list_resources())}
    assert {'dsgp4://reference/overview', 'dsgp4://reference/tle-format', 'dsgp4://reference/omm-format',
            'dsgp4://reference/sgp4-parameters', 'dsgp4://reference/mldsgp4',
            'dsgp4://examples/element-sets'} <= resources
    gravity = list(asyncio.run(server.read_resource('dsgp4://reference/gravity-models/wgs-84')))[0].content
    assert '398600.5' in gravity
    examples = list(asyncio.run(server.read_resource('dsgp4://examples/element-sets')))[0].content
    #every example element set must be parsable (and the TLE checksums valid):
    summary = call(server, 'orbital_elements_summary', {'element_sets': examples})
    assert summary['number_of_objects'] >= 3
    lines = examples.strip().splitlines()
    assert call(server, 'validate_tle', {'tle': '\n'.join(lines[0:3])})['valid']


def test_prompts(server):
    prompt = asyncio.run(server.get_prompt('characterize_orbit', {'element_set': TLE}))
    assert prompt.messages[0].role == 'user'
    assert '34427' in prompt.messages[0].content.text
    guide = asyncio.run(server.get_prompt('mldsgp4_training_guide', {}))
    assert 'state_dict' in guide.messages[0].content.text


def test_tool_errors_carry_messages(server):
    from mcp.server.mcpserver.exceptions import ToolError
    with pytest.raises(ToolError, match='could not be parsed'):
        asyncio.run(server.call_tool('propagate', {'element_set': 'garbage', 'minutes_since_epoch': [0.0]}))
    with pytest.raises(ToolError, match='at most'):
        asyncio.run(server.call_tool('propagate', {'element_set': TLE,
                                                   'minutes_since_epoch': list(range(1500))}))
