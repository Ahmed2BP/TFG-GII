import os

from gym.envs.registration import register

from sinergym.utils.rewards import *

# Set __version__ in module
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()


#========================5ZoneAutoDXVAV========================#
# 0) Demo environment
register(
    id='Eplus-demo-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature (SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        'env_name': 'demo-v1'})

# 1) 5-zone, hot weather, discrete actions
register(
    id='Eplus-5Zone-hot-discrete-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': True,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature (SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0)
        },
        'env_name': '5Zone-hot-discrete-v1'})

# 2) 5-zone, mixed weather, discrete actions
register(
    id='Eplus-5Zone-mixed-discrete-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': True,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature (SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0)
        },
        'env_name': '5Zone-mixed-discrete-v1'})

# 3) 5-zone, cool weather, discrete actions
register(
    id='Eplus-5Zone-cool-discrete-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': True,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature (SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0)
        },
        'env_name': '5Zone-cool-discrete-v1'})

# 4) 5-zone, hot weather, discrete actions and stochastic
register(
    id='Eplus-5Zone-hot-discrete-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': True,
        'weather_variability': (
            1.0,
            0.0,
            0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature (SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0)
        },
        'env_name': '5Zone-hot-discrete-stochastic-v1'})

# 5) 5-zone, mixed weather, discrete actions and stochastic
register(
    id='Eplus-5Zone-mixed-discrete-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': True,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature (SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0)
        },
        'env_name': '5Zone-mixed-discrete-stochastic-v1'})

# 6) 5-zone, cool weather, discrete actions and stochastic
register(
    id='Eplus-5Zone-cool-discrete-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': True,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature (SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0)
        },
        'env_name': '5Zone-cool-discrete-stochastic-v1'})

# 7) 5-zone, hot weather, continuous actions
register(
    id='Eplus-5Zone-hot-continuous-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': False,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature (SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        'env_name': '5Zone-hot-continuous-v1'})

# 8) 5-zone, mixed weather, continuous actions
register(
    id='Eplus-5Zone-mixed-continuous-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': False,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature (SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        'env_name': '5Zone-mixed-continuous-v1'})

# 9) 5-zone, cool weather, continuous actions
register(
    id='Eplus-5Zone-cool-continuous-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': False,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature (SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        'env_name': '5Zone-cool-continuous-v1'})

# 10) 5-zone, hot weather, continuous actions and stochastic
register(
    id='Eplus-5Zone-hot-continuous-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': False,
        'weather_variability': (
            1.0,
            0.0,
            0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature (SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        'env_name': '5Zone-hot-continuous-stochastic-v1'})

# 11) 5-zone, mixed weather, continuous actions and stochastic
register(
    id='Eplus-5Zone-mixed-continuous-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': False,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
                'temperature_variable': 'Zone Air Temperature (SPACE1-1)',
                'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
                'range_comfort_winter': (20.0, 23.5),
                'range_comfort_summer': (23.0, 26.0)
        },
        'env_name': '5Zone-mixed-continuous-stochastic-v1'})

# 12) 5-zone, cool weather, continuous actions and stochastic
register(
    id='Eplus-5Zone-cool-continuous-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'variables_file': 'variablesDXVAV.cfg',
        'spaces_file': '5ZoneAutoDXVAV_spaces.cfg',
        'discrete_actions': False,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
                'temperature_variable': 'Zone Air Temperature (SPACE1-1)',
                'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
                'range_comfort_winter': (20.0, 23.5),
                'range_comfort_summer': (23.0, 26.0)
        },
        'env_name': '5Zone-cool-continuous-stochastic-v1'})

#========================DATACENTER========================#
# 13) DC, hot weather, discrete actions
register(
    id='Eplus-datacenter-hot-discrete-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'variables_file': 'variablesDataCenter.cfg',
        'spaces_file': '2ZoneDataCenterHVAC_wEconomizer_spaces.cfg',
        'discrete_actions': True,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature (West Zone)',
                'Zone Air Temperature (East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-hot-discrete-v1'
    }
)

# 14) DC, hot weather, continuous actions
register(
    id='Eplus-datacenter-hot-continuous-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'variables_file': 'variablesDataCenter.cfg',
        'spaces_file': '2ZoneDataCenterHVAC_wEconomizer_spaces.cfg',
        'discrete_actions': False,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature (West Zone)',
                'Zone Air Temperature (East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-hot-continuous-v1'
    }
)

# 15) DC, hot weather, discrete actions and stochastic
register(
    id='Eplus-datacenter-hot-discrete-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'variables_file': 'variablesDataCenter.cfg',
        'spaces_file': '2ZoneDataCenterHVAC_wEconomizer_spaces.cfg',
        'discrete_actions': True,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature (West Zone)',
                'Zone Air Temperature (East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-hot-discrete-stochastic-v1'
    }
)

# 16) DC, hot weather, continuous actions and stochastic
register(
    id='Eplus-datacenter-hot-continuous-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'variables_file': 'variablesDataCenter.cfg',
        'spaces_file': '2ZoneDataCenterHVAC_wEconomizer_spaces.cfg',
        'discrete_actions': False,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature (West Zone)',
                'Zone Air Temperature (East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-hot-continuous-stochastic-v1'
    }
)

# 17) DC, mixed weather, discrete actions
register(
    id='Eplus-datacenter-mixed-discrete-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'variables_file': 'variablesDataCenter.cfg',
        'spaces_file': '2ZoneDataCenterHVAC_wEconomizer_spaces.cfg',
        'discrete_actions': True,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature (West Zone)',
                'Zone Air Temperature (East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-mixed-discrete-v1'})

# 18) DC, mixed weather, continuous actions
register(
    id='Eplus-datacenter-mixed-continuous-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'variables_file': 'variablesDataCenter.cfg',
        'spaces_file': '2ZoneDataCenterHVAC_wEconomizer_spaces.cfg',
        'discrete_actions': False,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature (West Zone)',
                'Zone Air Temperature (East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-mixed-continuous-v1'})

# 19) DC, mixed weather, discrete actions and stochastic
register(
    id='Eplus-datacenter-mixed-discrete-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'variables_file': 'variablesDataCenter.cfg',
        'spaces_file': '2ZoneDataCenterHVAC_wEconomizer_spaces.cfg',
        'discrete_actions': True,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature (West Zone)',
                'Zone Air Temperature (East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-mixed-discrete-stochastic-v1'})

# 20) DC, mixed weather, continuous actions and stochastic
register(
    id='Eplus-datacenter-mixed-continuous-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'variables_file': 'variablesDataCenter.cfg',
        'spaces_file': '2ZoneDataCenterHVAC_wEconomizer_spaces.cfg',
        'discrete_actions': False,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature (West Zone)',
                'Zone Air Temperature (East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-mixed-continuous-stochastic-v1'})

# 21) DC, cool weather, discrete actions
register(
    id='Eplus-datacenter-cool-discrete-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'variables_file': 'variablesDataCenter.cfg',
        'spaces_file': '2ZoneDataCenterHVAC_wEconomizer_spaces.cfg',
        'discrete_actions': True,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature (West Zone)',
                'Zone Air Temperature (East Zone)'],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (
                18,
                27),
            'range_comfort_summer': (
                18,
                27)},
        'env_name': 'datacenter-cool-discrete-v1'})

# 22) DC, cool weather, continuous actions
register(
    id='Eplus-datacenter-cool-continuous-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'variables_file': 'variablesDataCenter.cfg',
        'spaces_file': '2ZoneDataCenterHVAC_wEconomizer_spaces.cfg',
        'discrete_actions': False,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature (West Zone)',
                'Zone Air Temperature (East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-cool-continuous-v1'})

# 23) DC, cool weather, discrete actions and stochastic
register(
    id='Eplus-datacenter-cool-discrete-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'variables_file': 'variablesDataCenter.cfg',
        'spaces_file': '2ZoneDataCenterHVAC_wEconomizer_spaces.cfg',
        'discrete_actions': True,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature (West Zone)',
                'Zone Air Temperature (East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-cool-discrete-stochastic-v1'})

# 24) DC, cool weather, continuous actions and stochastic
register(
    id='Eplus-datacenter-cool-continuous-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'variables_file': 'variablesDataCenter.cfg',
        'spaces_file': '2ZoneDataCenterHVAC_wEconomizer_spaces.cfg',
        'discrete_actions': False,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature (West Zone)',
                'Zone Air Temperature (East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-cool-continuous-stochastic-v1'})

#========================MULLION========================#
# TODO Change temperature and energy names for reward calculation.
# 25) IW, mixed weather, discrete actions
register(
    id='Eplus-IWMullion-mixed-discrete-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'IW_Mullion.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'variables_file': 'variablesIW.cfg',
        'spaces_file': 'IW_Mullion_spaces.cfg',
        'discrete_actions': True,
        'env_name': 'IWMullion-mixed-discrete-v1'})

# 26) IW, mixed weather, discrete actions and stochastic
register(
    id='Eplus-IWMullion-mixed-discrete-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'IW_Mullion.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'variables_file': 'variablesIW.cfg',
        'spaces_file': 'IW_Mullion_spaces.cfg',
        'discrete_actions': True,
        'weather_variability': (
            1.0,
            0.0,
            0.001),
        'env_name': 'IWMullion-mixed-discrete-stochastic-v1'})

# 27) IW, mixed weather, continuous actions
register(
    id='Eplus-IWMullion-mixed-continuous-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'IW_Mullion.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'variables_file': 'variablesIW.cfg',
        'spaces_file': 'IW_Mullion_spaces.cfg',
        'discrete_actions': False,
        'env_name': 'IWMullion-mixed-continuous-v1'})

# 28) IW, mixed weather, continuous actions and stochastic
register(
    id='Eplus-IWMullion-mixed-continuous-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'IW_Mullion.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'variables_file': 'variablesIW.cfg',
        'spaces_file': 'IW_Mullion_spaces.cfg',
        'discrete_actions': False,
        'weather_variability': (
            1.0,
            0.0,
            0.001),
        'env_name': 'IWMullion-mixed-continuous-stochastic-v1'})

# 29) IW, cool weather, discrete actions
register(
    id='Eplus-IWMullion-cool-discrete-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'IW_Mullion.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'variables_file': 'variablesIW.cfg',
        'spaces_file': 'IW_Mullion_spaces.cfg',
        'discrete_actions': True,
        'env_name': 'IWMullion-cool-discrete-v1'})

# 30) IW, cool weather, discrete actions and stochastic
register(
    id='Eplus-IWMullion-cool-discrete-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'IW_Mullion.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'variables_file': 'variablesIW.cfg',
        'spaces_file': 'IW_Mullion_spaces.cfg',
        'discrete_actions': True,
        'weather_variability': (
            1.0,
            0.0,
            0.001),
        'env_name': 'IWMullion-cool-discrete-stochastic-v1'})

# 31) IW, cool weather, continuous actions
register(
    id='Eplus-IWMullion-cool-continuous-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'IW_Mullion.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'variables_file': 'variablesIW.cfg',
        'spaces_file': 'IW_Mullion_spaces.cfg',
        'discrete_actions': False,
        'env_name': 'IWMullion-cool-continuous-v1'})

# 32) IW, cool weather, continuous actions and stochastic
register(
    id='Eplus-IWMullion-cool-continuous-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'IW_Mullion.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'variables_file': 'variablesIW.cfg',
        'spaces_file': 'IW_Mullion_spaces.cfg',
        'discrete_actions': False,
        'weather_variability': (
            1.0,
            0.0,
            0.001),
        'env_name': 'IWMullion-cool-continuous-stochastic-v1'})
