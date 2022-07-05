"""Common utilities."""

import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pydoc import locate
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import pandas as pd
from opyplus import Epm, WeatherData
from opyplus.epm.record import Record


def get_delta_seconds(
        st_year: int,
        st_mon: int,
        st_day: int,
        end_year: int,
        end_mon: int,
        end_day: int) -> float:
    """Returns the delta seconds between st year:st mon:st day:0:0:0 and
    end year:end mon:end day:24:0:0.

    Args:
        st_year (int): Start year.
        st_mon (int): Start month.
        st_day (int): Start day.
        end_year (int): End year.
        end_mon (int): End month.
        end_day (int): End day.

    Returns:
        float: Time difference in seconds.

    """
    startTime = datetime(st_year, st_mon, st_day, 0, 0, 0)
    endTime = datetime(end_year, end_mon, end_day,
                       23, 0, 0) + timedelta(0, 3600)
    delta_sec = (endTime - startTime).total_seconds()
    return delta_sec


def get_current_time_info(
        epm: Epm, sec_elapsed: float) -> List[int]:
    """Returns the current day, month and hour given the seconds elapsed since the simulation started.

    Args:
        epm (opyplus.Epm): EnergyPlus model object.
        sec_elapsed (float): Seconds elapsed since the start of the simulation

    Returns:
        List[int]: A List composed by the current year, day, month and hour in the simulation.

    """
    start_date = datetime(
        year=int(epm.RunPeriod[0]['begin_year']),
        month=int(epm.RunPeriod[0]['begin_month']),
        day=int(epm.RunPeriod[0]['begin_day_of_month'])
    )

    current_date = start_date + timedelta(seconds=sec_elapsed)

    return [
        int(current_date.year),
        int(current_date.month),
        int(current_date.day),
        int(current_date.hour),
    ]


def parse_variables(var_file: str) -> Dict[str, List[str]]:
    """Parse observation and action to dictionary.

    Args:
        var_file (str): Variables file path.

    Returns:
        Dict[str, List[str]]: observation and action keys; a list with the name of the observation <variables> (<zone>) and a list with the name of the action <variables> respectively.
    """

    tree = ET.parse(var_file)
    root = tree.getroot()

    variables = {}
    observation = []
    action = []
    for var in root.findall('variable'):
        if var.attrib['source'] == 'EnergyPlus':
            observation.append(var[0].attrib['type'] +
                               ' (' + var[0].attrib['name'] + ')')
        if var.attrib['source'] == 'Ptolemy':
            action.append(var[0].attrib['schedule'])

    variables['observation'] = observation
    variables['action'] = action

    return variables


def create_variable_weather(
        weather_data: WeatherData,
        original_epw_file: str,
        columns: List[str] = ['drybulb'],
        variation: Optional[Tuple[float, float, float]] = None) -> Optional[str]:
    """Create a new weather file using Ornstein-Uhlenbeck process.

    Args:
        weather_data (opyplus.WeatherData): Opyplus object with the weather for the simulation.
        original_epw_file (str): Path to the original EPW file.
        columns (List[str], optional): List of columns to be affected. Defaults to ['drybulb'].
        variation (Optional[Tuple[float, float, float]], optional): Tuple with the sigma, mean and tau for OU process. Defaults to None.

    Returns:
        Optional[str]: Name of the file created in the same location as the original one.
    """

    if variation is None:
        return None
    else:
        # Get dataframe with weather series
        df = weather_data.get_weather_series()

        sigma = variation[0]  # Standard deviation.
        mu = variation[1]  # Mean.
        tau = variation[2]  # Time constant.

        T = 1.  # Total time.
        # All the columns are going to have the same num of rows since they are
        # in the same dataframe
        n = len(df[columns[0]])
        dt = T / n
        # t = np.linspace(0., T, n)  # Vector of times.

        sigma_bis = sigma * np.sqrt(2. / tau)
        sqrtdt = np.sqrt(dt)

        x = np.zeros(n)

        # Create noise
        for i in range(n - 1):
            x[i + 1] = x[i] + dt * (-(x[i] - mu) / tau) + \
                sigma_bis * sqrtdt * np.random.randn()

        for column in columns:
            # Add noise
            df[column] += x

        # Save new weather data
        weather_data.set_weather_series(df)
        filename = original_epw_file.split('.epw')[0]
        filename += '_Random_%s_%s_%s.epw' % (str(sigma), str(mu), str(tau))
        weather_data.to_epw(filename)
        return filename


def ranges_getter(output_path: str,
                  last_result: Optional[Dict[str, List[float]]] = None
                  ) -> Dict[str, List[float]]:
    """Given a path with simulations outputs, this function is used to extract max and min absolute values of all episodes in each variable. If a dict ranges is given, will be updated.

    Args:
        output_path (str): path with simulations directories (Eplus-env-<env_name>).
        last_result (Optional[Dict[str, List[float]]], optional): Last ranges dict to be updated. This will be created if it is not given.

    Returns:
        Dict[str, List[float]]: list min,max of each variable as a key.

    """

    if last_result is not None:
        result = last_result
    else:
        result = {}

    content = os.listdir(output_path)
    for simulation in content:
        if os.path.isdir(
            output_path +
            '/' +
                simulation) and simulation.startswith('Eplus-env'):
            simulation_content = os.listdir(output_path + '/' + simulation)
            for episode_dir in simulation_content:
                if os.path.isdir(
                    output_path +
                    '/' +
                    simulation +
                    '/' +
                        episode_dir):
                    monitor_path = output_path + '/' + simulation + '/' + episode_dir + '/monitor.csv'
                    print('Reading ' + monitor_path + ' limits.')
                    data = pd.read_csv(monitor_path)

                    if len(result) == 0:
                        for column in data:
                            # variable : [min,max]
                            result[column] = [np.inf, -np.inf]

                    for column in data:
                        if np.min(data[column]) < result[column][0]:
                            result[column][0] = np.min(data[column])
                        if np.max(data[column]) > result[column][1]:
                            result[column][1] = np.max(data[column])
    return result


def get_record_keys(record: Record) -> List[str]:
    """Given an opyplus Epm Record (one element from opyplus.epm object) this function returns list of keys (opyplus hasn't got this functionality explicitly)

     Args:
        record (opyplus.Epm.Record): Element from Epm object.

     Returns:
        List[str]: Key list from record.
    """
    return [field.ref for field in record._table._dev_descriptor._field_descriptors]


def prepare_batch_from_records(records: List[Record]) -> List[Dict[str, Any]]:
    """Prepare a list of dictionaries in order to use Epm.add_batch directly

    Args:
        records List[opyplus.Epm.Record]: List of records which will be converted to dictionary batch.

    Returns:
        List[Dict[str, Any]]: List of dicts where each dictionary is a record element.
    """

    batch = []
    for record in records:
        aux_dict = {}
        for key in get_record_keys(record):
            aux_dict[key] = record[key]
        batch.append(aux_dict)

    return batch


def get_season_comfort_range(year, month, day):
    """Get comfort temperature range depending on season. The comfort ranges are those
    defined by ASHRAE in Standard 55—Thermal Environmental Conditions for Human Occupancy (2004).

    Args:
        year (int): current year
        month (int): current month
        day (int): current day
    """

    summer_start_date = datetime(year, 6, 1)
    summer_final_date = datetime(year, 9, 30)

    range_comfort_summer = (23.0, 26.0)
    range_comfort_winter = (20.0, 23.5)

    current_dt = datetime(year, month, day)

    if current_dt >= summer_start_date and current_dt <= summer_final_date:
        comfort = range_comfort_summer
    else:
        comfort = range_comfort_winter

    return comfort
