import shutil

import pytest
from opyplus import Epm, WeatherData

import sinergym.utils.common as common


@pytest.mark.parametrize(
    'st_year,st_mon,st_day,end_year,end_mon,end_day,expected',
    [
        (2000, 10, 1, 2000, 11, 1, 2764800),
        (2002, 1, 10, 2002, 2, 5, 2332800),
        # st_time=00:00:00 and ed_time=24:00:00
        (2021, 5, 5, 2021, 5, 5, 3600 * 24),
        (2004, 7, 1, 2004, 6, 1, -2505600),  # Negative delta secons test
    ]
)
def test_get_delta_seconds(
        st_year,
        st_mon,
        st_day,
        end_year,
        end_mon,
        end_day,
        expected):
    delta_sec = common.get_delta_seconds(
        st_year, st_mon, st_day, end_year, end_mon, end_day)
    assert isinstance(delta_sec, float)
    assert delta_sec == expected


@pytest.mark.parametrize('sec_elapsed,expected_list',
                         [(2764800,
                           [1991,
                            2,
                            2,
                            0]),
                             (0,
                              [1991,
                               1,
                               1,
                               0]),
                             ((2764800 * 4) + (3600 * 10),
                              [1991,
                                 5,
                                 9,
                                 10]),
                          ])
def test_get_current_time_info(epm, sec_elapsed, expected_list):
    output = common.get_current_time_info(epm, sec_elapsed)
    print(output)
    assert isinstance(output, list)
    assert len(output) == 4
    assert output == expected_list


def test_parse_variables(variable_path):
    # The name of variables we expected
    observation_expected = [
        'Site Outdoor Air Drybulb Temperature (Environment)',
        'Site Outdoor Air Relative Humidity (Environment)',
        'Site Wind Speed (Environment)',
        'Site Wind Direction (Environment)',
        'Site Diffuse Solar Radiation Rate per Area (Environment)',
        'Site Direct Solar Radiation Rate per Area (Environment)',
        'Zone Thermostat Heating Setpoint Temperature (SPACE1-1)',
        'Zone Thermostat Cooling Setpoint Temperature (SPACE1-1)',
        'Zone Air Temperature (SPACE1-1)',
        'Zone Thermal Comfort Mean Radiant Temperature (SPACE1-1 PEOPLE 1)',
        'Zone Air Relative Humidity (SPACE1-1)',
        'Zone Thermal Comfort Clothing Value (SPACE1-1 PEOPLE 1)',
        'Zone Thermal Comfort Fanger Model PPD (SPACE1-1 PEOPLE 1)',
        'Zone People Occupant Count (SPACE1-1)',
        'People Air Temperature (SPACE1-1 PEOPLE 1)',
        'Facility Total HVAC Electricity Demand Rate (Whole Building)']
    action_expected = ['Space1-HtgSetP-RL', 'Space1-ClgSetP-RL']

    variables = common.parse_variables(variable_path)

    assert isinstance(variables, dict)
    assert len(variables) == 2
    for i in range(len(variables['observation'])):
        assert variables['observation'][i] == observation_expected[i]
    for i in range(len(variables['action'])):
        assert variables['action'][i] == action_expected[i]


def test_parse_observation_action_space(space_path):

    output = common.parse_observation_action_space(space_path)

    assert list(output.keys()) == ['observation',
                                   'discrete_action', 'continuous_action']

    assert isinstance(output['observation'], tuple)
    assert output['observation'][0] == float(-5e6)
    assert output['observation'][1] == float(5e6)
    assert output['observation'][2] == (20,)

    assert output['continuous_action'][2] == (2,)

    assert isinstance(output['discrete_action'], dict)
    assert len(output['discrete_action']) == 10
    assert isinstance(output['discrete_action'][0], tuple)
    assert len(output['discrete_action'][0]
               ) == output['continuous_action'][2][0]

    assert isinstance(output['continuous_action'], tuple)
    assert len(output['continuous_action'][0]
               ) == output['continuous_action'][2][0]
    assert len(output['continuous_action'][1]
               ) == output['continuous_action'][2][0]


@ pytest.mark.parametrize(
    'variation',
    [
        (None),
        ((1, 0.0, 0.001)),
        ((5, 0.0, 0.01)),
        ((10, 0.0, 0.1)),
    ]
)
def test_create_variable_weather(variation, weather_data, weather_path):
    output = common.create_variable_weather(
        weather_data, weather_path, ['drybulb'], variation)
    if variation is None:
        assert output is None
    else:
        expected = weather_path.split('.epw')[0] + '_Random_' + str(
            variation[0]) + '_' + str(variation[1]) + '_' + str(variation[2]) + '.epw'
        assert output == expected
