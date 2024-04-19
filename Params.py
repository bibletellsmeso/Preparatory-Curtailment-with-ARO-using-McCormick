import numpy as np

# 1. microgrid configuration of the NEHLA case study
PERIOD_min = 15 # time resolution of the planner
PERIOD_hour = PERIOD_min / 60  # (hours)

# ------------------------------------------------------------------------------------------------------------------
BATTERY_CAPACITY = 567 # (kWh)
BATTERY_POWER = 250 # (kW)
SOC_INI = 187.5 # (kWh)
SOC_END = SOC_INI # (kWh)
SOC_MAX = 453.6 # (kWh)
SOC_MIN = 113.4 # (kWh)

CHARGE_EFFICIENCY = 0.93 # (%)
DISCHARGE_EFFICIENCY = 0.93 # (%)
CHARGING_POWER = BATTERY_CAPACITY # (kW)
DISCHARGING_POWER = BATTERY_CAPACITY # (kW)

# ------------------------------------------------------------------------------------------------------------------
DG_params = {"DG_min": 100, # (kW)
             "DG_max": 750, # (kW)
             "DG_ramp_up": 800, # (kW)
             "DG_ramp_down": 800,
             "DG_reserve_up": 100,
             "DG_reserve_down": 50,
             "DG_rate": 80}


ESS_params = {"capacity": BATTERY_CAPACITY,  # (kWh)
               "soc_min": SOC_MIN,  # (kWh)
               "soc_max": SOC_MAX,  # (kWh)
               "soc_ini": SOC_INI,  # (kWh)
               "soc_end": SOC_END,  # (kWh)
               "charge_eff": CHARGE_EFFICIENCY,  # (/)
               "discharge_eff": DISCHARGE_EFFICIENCY,  # (/)
               "power_min": 0,  # (kW)
               "power_max": BATTERY_POWER}  # (kW)

DER_params = {"PV_min": 0,
             "PV_max": 600,
             "PV_ramp_up": 520,
             "PV_ramp_down": 520,
             "PV_capacity": 600,
             "WT_min": 0,
             "WT_max": 500,
             "WT_ramp_up": 200,
             "WT_ramp_down": 200,
             "WT_cpapacity": 300}

load_params = {"ramp_up": 280,
               "ramp_down": 280}

cost_params = {"a": 0.001,
               "b": 0.015,
               "c": 0.059,
               "m_pos": 0.005,
               "m_neg": 0.005,
               "m_O&M": 0.0,
               "PV_m_cut_pre": 0.01,
               "WT_m_cut_pre": 0.008,
               "m_pos_re": 0.02,
               "m_neg_re": 0.02,
               "m_O&M_re": 0.01,
               "PV_m_cut_re": 0.08,
               "WT_m_cut_re": 0.07,
               "PV_m_cut_cn": 0.005,
               "WT_m_cut_cn": 0.01}

pwl_params = {"num": 10}

# ------------------------------------------------------------------------------------------------------------------
PARAMETERS = {}
PARAMETERS["period_hours"] = PERIOD_min / 60  # (hours)
PARAMETERS['RG'] = DER_params
PARAMETERS['cost'] = cost_params
PARAMETERS['DE'] = DG_params
PARAMETERS['ES'] = ESS_params
PARAMETERS['load'] = load_params
PARAMETERS['PWL'] = pwl_params
PARAMETERS['u_1'] = np.ones(96)
