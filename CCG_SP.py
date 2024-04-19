import os
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt

from utils import read_file
from SP_primal_LP import *
from Params import PARAMETERS
from root_project import ROOT_DIR
from Data_read import *

class CCG_SP():
    """
    CCGD = Column and Constraint Gneration Dual
    SP = Sub Problem of the CCG dual cutting plane algorithm.
    SP = Max-min problem that is reformulated as a single max problem by taking the dual.
    The resulting maximization problem is bilinear and is linearized using McCormick not big-M method.
    The final reformulated SP is a MILP due to binary variables related to the uncertainty set.
    
    :ivar nb_periods: number of market periods (-)
    :ivar period_hours: period duration (hours)

    :ivar model: a Gurobi model (-)

    """

    def __init__(self, PV_forecast:np.array, WT_forecast:np.array, load_forecast:np.array, PV_pos_100:np.array, PV_neg_100:np.array, PV_pos_50:np.array, PV_neg_50:np.array, PV_pos_20:np.array, PV_neg_20:np.array,
                 WT_pos_100:np.array, WT_neg_100:np.array, WT_pos_50:np.array, WT_neg_50:np.array, WT_pos_20:np.array, WT_neg_20:np.array,
                 load_pos_100:np.array, load_neg_100:np.array, load_pos_50:np.array, load_neg_50:np.array, load_pos_20:np.array, load_neg_20:np.array,
                 DE1_p:np.array, DE1_rp:np.array, DE1_rn:np.array, x_curt_PV:np.array, x_curt_WT:np.array, Pi_PV_t:float=0, Pi_WT_t:float=0, Pi_load:float=0, M:float=None):
        """
        Init the planner.
        """
        self.parameters = PARAMETERS # simulation parameters
        self.period_hours = PARAMETERS['period_hours']  # (hour)
        self.nb_periods = int(24 / self.period_hours)
        self.t_set = range(self.nb_periods)
        self.seg_num = 10
        self.big_M = 1e6
        self.epsilon = 1e-6
        self.PV_forecast = PV_forecast # (kW)
        self.WT_forecast = WT_forecast
        self.load_forecast = load_forecast # (kW)
        self.PV_lb = data.PV_min
        self.PV_ub = data.PV_max
        self.WT_lb = data.WT_min
        self.WT_ub = data.WT_max
        self.load_lb = data.load_min
        self.load_ub = data.load_max
        self.p_1 = DE1_p # (kW) The power of diesel generator
        self.r_pos_1 = DE1_rp # (kw) The reserve rate of diesel generator
        self.r_neg_1 = DE1_rn
        self.x_curt_PV = x_curt_PV # (kW) The curtailment of
        self.x_curt_WT = x_curt_WT
        self.PV_pos_100 = PV_pos_100 # (kW) The maximal deviation betwwen the min and forecast PV uncertainty set bounds
        self.PV_neg_100 = PV_neg_100 # (kW) The maximal deviation between the max and forecast PV uncertainty set bounds
        self.PV_pos_50 = PV_pos_50 
        self.PV_neg_50 = PV_neg_50 
        self.PV_pos_20 = PV_pos_20
        self.PV_neg_20 = PV_neg_20 
        self.WT_pos_100 = WT_pos_100 # (kW) The maximal deviation betwwen the min and forecast WT uncertainty set bounds
        self.WT_neg_100 = WT_neg_100 # (kW) The maximal deviation between the max and forecast WT uncertainty set bounds
        self.WT_pos_50 = WT_pos_50 
        self.WT_neg_50 = WT_neg_50 
        self.WT_pos_20 = WT_pos_20
        self.WT_neg_20 = WT_neg_20 
        self.load_pos_100 = load_pos_100 # (kw) The maximal deviation between the min and forecast load uncertainty set bounds
        self.load_neg_100 = load_neg_100 # (kW) The maximal deviation between the max and forecast load uncertainty set bounds
        self.load_pos_50 = load_pos_50 
        self.load_neg_50 = load_neg_50 
        self.load_pos_20 = load_pos_20 
        self.load_neg_20 = load_neg_20 
        self.Pi_PV_t = Pi_PV_t # uncertainty budget <= self.nb_periods, gamma = 0: no uncertainty
        self.Pi_WT_t = Pi_WT_t
        self.Pi_load = Pi_load
        self.M = M

        # DE1 parameters
        self.u_DE1 = PARAMETERS['u_1'] # on/off
        self.DE1_min = PARAMETERS['DE']['DE1_min'] # (kW)
        self.DE1_max = PARAMETERS['DE']['DE1_max'] # (kW)
        self.DE1_ramp_up = PARAMETERS['DE']['DE1_ramp_up'] # (kW)
        self.DE1_ramp_down = PARAMETERS['DE']['DE1_ramp_down'] # (kW)
        self.DE1_reserve_up = PARAMETERS['DE']['DE1_reserve_up']
        self.DE1_reserve_down = PARAMETERS['DE']['DE1_reserve_down']
        self.DE1_p_rate = PARAMETERS['DE']['DE1_p_rate']

        # ESS parameters
        self.EScapacity = PARAMETERS['ES']['capacity']  # (kWh)
        self.soc_ini = PARAMETERS['ES']['soc_ini']  # (kWh)
        self.soc_end = PARAMETERS['ES']['soc_end']  # (kWh)
        self.soc_min = PARAMETERS['ES']['soc_min']  # (kWh)
        self.soc_max = PARAMETERS['ES']['soc_max']  # (kWh)
        self.charge_eff = PARAMETERS['ES']['charge_eff']  # (/)
        self.discharge_eff = PARAMETERS['ES']['discharge_eff']  # (/)
        self.ES_min = PARAMETERS['ES']['power_min']  # (kW)
        self.ES_max = PARAMETERS['ES']['power_max']  # (kW)

        # PV parameters
        self.PV_min = PARAMETERS['RG']['PV_min']
        self.PV_max = PARAMETERS['RG']['PV_max']
        self.PV_ramp_up = PARAMETERS['RG']['PV_ramp_up']
        self.PV_ramp_down = PARAMETERS['RG']['PV_ramp_down']

        # WT parameters
        self.WT_min = PARAMETERS['RG']['WT_min']
        self.WT_max = PARAMETERS['RG']['WT_max']
        self.WT_ramp_up = PARAMETERS['RG']['WT_ramp_up']
        self.WT_ramp_down = PARAMETERS['RG']['WT_ramp_down']

        # load parameters
        self.load_ramp_up = PARAMETERS['load']['ramp_up']
        self.load_ramp_down = PARAMETERS['load']['ramp_down']

        # Cost parameters
        self.cost_a_DE1 = PARAMETERS['cost']['DE1_a']
        self.cost_b_DE1 = PARAMETERS['cost']['DE1_b']
        self.cost_c_DE1 = PARAMETERS['cost']['DE1_c']
        self.cost_m_pos_DE1 = PARAMETERS['cost']['DE1_m_pos']
        self.cost_m_neg_DE1 = PARAMETERS['cost']['DE1_m_neg']
        self.cost_m_pos_re_DE1 = PARAMETERS['cost']['DE1_m_pos_re']
        self.cost_m_neg_re_DE1 = PARAMETERS['cost']['DE1_m_neg_re']
        self.cost_OM_ES = PARAMETERS['cost']['ES_m_O&M']
        self.cost_OM_ES_re = PARAMETERS['cost']['ES_m_O&M_re']
        self.cost_m_pre_PV = PARAMETERS['cost']['PV_m_cut_pre']
        self.cost_m_re_PV = PARAMETERS['cost']['PV_m_cut_re']
        self.cost_m_cn_PV = PARAMETERS['cost']['PV_m_cut_cn']
        self.cost_m_pre_WT = PARAMETERS['cost']['WT_m_cut_pre']
        self.cost_m_re_WT = PARAMETERS['cost']['WT_m_cut_re']
        self.cost_m_cn_WT = PARAMETERS['cost']['WT_m_cut_cn']

        # Piecewise linearlization parameters
        self.seg_num = PARAMETERS['PWL']['num']

        self.time_building_model = None
        self.time_solving_model = None

        # Create model
        self.model = self.create_model()

        # Sovle model
        self.solver_status = None
   
    def create_model(self):
        """
        Create the optimization problem
        """
        t_build = time.time()

        # -------------------------------------------------------------------------------------------------------------
        # 1. create model
        model = gp.Model("SP_dual_MILP")

        # -------------------------------------------------------------------------------------------------------------
        # 2. Create dual variables -> phi

        # 2.1 Continuous variables
        # primal constraints <= b -> dual variables <= 0, primal constraints = b -> dual varialbes are free, (primal constraints >= b -> dual variables >= 0)
        phi_DE1pos = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_DE1pos")
        phi_DE1neg = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_DE1neg")
        phi_chg = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_chg")
        phi_dis = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_dis")
        phi_ini = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_ini") # free of dual variable 
        phi_S = model.addVars(self.nb_periods - 1, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_S") # num: 95, free dual of ESS dynamics (=)
        phi_end = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_end") # free of dual variable
        phi_Smin = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_Smin")
        phi_Smax = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_Smax")
        phi_PV = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_PV") # free of dual variable
        phi_WT = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_WT") # free of dual variable
        phi_load = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi_load") # free of dual variable
        phi_curt_PV = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_curt_PV") # free of dual variable
        phi_curt_WT = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_curt_WT") # free of dual variable
        phi = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name="phi") # free dual of power balance
        phi_pc = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_pc") # free of dual variable
        phi_wc = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, obj=0, name="phi_wc") # free of dual variable

        # 2.2 Continuous variables related to the uncertainty set
        epsilon_pos_100 = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=0, name="epsilon_pos_100")
        epsilon_neg_100 = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=0, name="epsilon_neg_100")
        epsilon_pos_50 = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=0, name="epsilon_pos_50")
        epsilon_neg_50 = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=0, name="epsilon_neg_50")
        epsilon_pos_20 = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=0, name="epsilon_pos_20")
        epsilon_neg_20 = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=0, name="epsilon_neg_20")
        kapa_pos_100 = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=0, name="kapa_pos_100")
        kapa_neg_100 = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=0, name="kapa_neg_100")
        kapa_pos_50 = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=0, name="kapa_pos_50")
        kapa_neg_50 = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=0, name="kapa_neg_50")
        kapa_pos_20 = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=0, name="kapa_pos_20")
        kapa_neg_20 = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=0, name="kapa_neg_20")
        xi_pos_100 = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=0, name="xi_pos_100")
        xi_neg_100 = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=0, name="xi_neg_100")
        xi_pos_50 = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=0, name="xi_pos_50")
        xi_neg_50 = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=0, name="xi_neg_50")
        xi_pos_20 = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=0, name="xi_pos_20")
        xi_neg_20 = model.addVars(self.nb_periods, lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=0, name="xi_neg_20")
        Pi_PV_100 = model.addVar(vtype=GRB.CONTINUOUS, obj=0, name="Pi_PV_100")
        Pi_PV_50 = model.addVar(vtype=GRB.CONTINUOUS, obj=0, name="Pi_PV_50")
        Pi_PV_20 = model.addVar(vtype=GRB.CONTINUOUS, obj=0, name="Pi_PV_20")
        Pi_WT_100 = model.addVar(vtype=GRB.CONTINUOUS, obj=0, name="Pi_WT_100")
        Pi_WT_50 = model.addVar(vtype=GRB.CONTINUOUS, obj=0, name="Pi_WT_50")
        Pi_WT_20 = model.addVar(vtype=GRB.CONTINUOUS, obj=0, name="Pi_WT_20")
        Pi_load_100 = model.addVar(vtype=GRB.CONTINUOUS, obj=0, name="Pi_load_100")
        Pi_load_50 = model.addVar(vtype=GRB.CONTINUOUS, obj=0, name="Pi_load_50")
        Pi_load_20 = model.addVar(vtype=GRB.CONTINUOUS, obj=0, name="Pi_load_20")

        # 2.3 Continuous varialbes use for the linearization of the bilinear terms
        alpha_pos_100 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='alpha_pos_100')
        alpha_neg_100 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='alpha_neg_100')
        alpha_pos_50 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='alpha_pos_50')
        alpha_neg_50 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='alpha_neg_50')
        alpha_pos_20 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='alpha_pos_20')
        alpha_neg_20 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='alpha_neg_20')
        beta_pos_100 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='beta_pos_100')
        beta_neg_100 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='beta_neg_100')
        beta_pos_50 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='beta_pos_50')
        beta_neg_50 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='beta_neg_50')
        beta_pos_20 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='beta_pos_20')
        beta_neg_20 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='beta_neg_20')
        gamma_pos_100 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='gamma_pos_100')
        gamma_neg_100 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='gamma_neg_100')
        gamma_pos_50 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='gamma_pos_50')
        gamma_neg_50 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='gamma_neg_50')
        gamma_pos_20 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='gamma_pos_20')
        gamma_neg_20 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='gamma_neg_20')
        delta_pos_100 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='delta_pos_100')
        delta_neg_100 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='delta_neg_100')
        delta_pos_50 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='delta_pos_50')
        delta_neg_50 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='delta_neg_50')
        delta_pos_20 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='delta_pos_20')
        delta_neg_20 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='delta_neg_20')
        zeta_pos_100 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='zeta_pos_100')
        zeta_neg_100 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='zeta_neg_100')
        zeta_pos_50 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='zeta_pos_50')
        zeta_neg_50 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='zeta_neg_50')
        zeta_pos_20 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='zeta_pos_20')
        zeta_neg_20 = model.addVars(self.nb_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, obj=0, name='zeta_neg_20')

        x_100 = model.addVars(self.nb_periods, vtype=GRB.BINARY, name="x_pos_100")
        y_100 = model.addVars(self.nb_periods, vtype=GRB.BINARY, name="y_pos_100")
        z_100 = model.addVars(self.nb_periods, vtype=GRB.BINARY, name="z_pos_100")
        x_50 = model.addVars(self.nb_periods, vtype=GRB.BINARY, name="x_pos_50")
        y_50 = model.addVars(self.nb_periods, vtype=GRB.BINARY, name="y_pos_50")
        z_50 = model.addVars(self.nb_periods, vtype=GRB.BINARY, name="z_pos_50")
        x_20 = model.addVars(self.nb_periods, vtype=GRB.BINARY, name="x_pos_20")
        y_20 = model.addVars(self.nb_periods, vtype=GRB.BINARY, name="y_pos_20")
        z_20 = model.addVars(self.nb_periods, vtype=GRB.BINARY, name="z_pos_20")
        # -------------------------------------------------------------------------------------------------------------
        # 3. create objective
        obj_exp = 0
        obj_exp += phi_ini * self.soc_ini + phi_end * self.soc_end
        for i in self.t_set:
            obj_exp += phi_DE1pos[i] * self.r_pos_1[i] + phi_DE1neg[i] * self.r_neg_1[i]
            obj_exp += phi_chg[i] * self.ES_max + phi_dis[i] * self.ES_max
            obj_exp += - phi_Smin[i] * self.soc_min + phi_Smax[i] * self.soc_max
            obj_exp += phi_PV[i] * self.PV_forecast[i] + alpha_pos_100[i] * self.PV_pos_100[i] - alpha_neg_100[i] * self.PV_neg_100[i] + alpha_pos_50[i] * self.PV_pos_50[i] - alpha_neg_50[i] * self.PV_neg_50[i] + alpha_pos_20[i] * self.PV_pos_20[i] - alpha_neg_20[i] * self.PV_neg_20[i]
            obj_exp += phi_WT[i] * self.WT_forecast[i] + beta_pos_100[i] * self.WT_pos_100[i] - beta_neg_100[i] * self.WT_neg_100[i] + beta_pos_50[i] * self.WT_pos_50[i] - beta_neg_50[i] * self.WT_neg_50[i] + beta_pos_20[i] * self.WT_pos_20[i] - beta_neg_20[i] * self.WT_neg_20[i]
            obj_exp += phi_load[i] * self.load_forecast[i] + gamma_pos_100[i] * self.load_pos_100[i] - gamma_neg_100[i] * self.load_neg_100[i] + gamma_pos_50[i] * self.load_pos_50[i] - gamma_neg_50[i] * self.load_neg_50[i] + gamma_pos_20[i] * self.load_pos_20[i] - gamma_neg_20[i] * self.load_neg_20[i]
            obj_exp += phi_curt_PV[i] * self.PV_forecast[i] - phi_curt_PV[i] * self.x_curt_PV[i] + delta_pos_100[i] * self.PV_pos_100[i] - delta_neg_100[i] * self.PV_neg_100[i] + delta_pos_50[i] * self.PV_pos_50[i] - delta_neg_50[i] * self.PV_neg_50[i] + delta_pos_20[i] * self.PV_pos_20[i] - delta_neg_20[i] * self.PV_neg_20[i]
            obj_exp += phi_curt_WT[i] * self.WT_forecast[i] - phi_curt_WT[i] * self.x_curt_WT[i] + zeta_pos_100[i] * self.WT_pos_100[i] - zeta_neg_100[i] * self.WT_neg_100[i] + zeta_pos_50[i] * self.WT_pos_50[i] - zeta_neg_50[i] * self.WT_neg_50[i] + zeta_pos_20[i] * self.WT_pos_20[i] - zeta_neg_20[i] * self.WT_neg_20[i]
            obj_exp += phi_pc[i] * self.x_curt_PV[i] + phi_wc[i] * self.x_curt_WT[i]
        # a constant offset
            obj_exp += PWL_val(self.seg_num, self.DE1_min, self.DE1_max, FC1, self.p_1[i])
            obj_exp += PWL_val(self.seg_num, self.PV_min, data.PV_pred[i], PC_PV, self.x_curt_PV[i])
            obj_exp += PWL_val(self.seg_num, self.WT_min, data.WT_pred[i], PC_WT, self.x_curt_WT[i])
            obj_exp += self.cost_m_pos_DE1 * self.r_pos_1[i] + self.cost_m_neg_DE1 * self.r_neg_1[i]

        model.setObjective(obj_exp, GRB.MAXIMIZE)
        # -------------------------------------------------------------------------------------------------------------
        # 4. Create constraints
        # primal variables >= 0 -> dual constraints <= c, primal variables are free -> dual constraints = c, (primal variables <= 0 -> dual constraints >= c)
        # Constraints related to DE1
        model.addConstrs((phi_DE1pos[i] + phi[i] <= self.cost_m_pos_re_DE1 for i in self.t_set), name='c_DE1_pos')
        model.addConstrs((phi_DE1neg[i] - phi[i] <= self.cost_m_neg_re_DE1 for i in self.t_set), name='c_DE1_neg')
        # Constraints related to the ES charge and discharge
        model.addConstr((phi_chg[0] - phi[0] <= self.cost_OM_ES_re), name='c_ES_chg_first')
        model.addConstrs((phi_chg[i] - phi[i] - phi_S[i - 1] * self.charge_eff * self.period_hours <= self.cost_OM_ES_re for i in range(1, self.nb_periods)), name='c_ES_chg')
        model.addConstr((phi_dis[0] + phi[0] <= self.cost_OM_ES_re), name='c_ES_dis_first')
        model.addConstrs((phi_dis[i] + phi[i] + phi_S[i - 1] / self.discharge_eff * self.period_hours <= self.cost_OM_ES_re for i in range(1, self.nb_periods)), name='c_ES_dis')
        
        # Constraints related to the ES SOC
        model.addConstr((- phi_Smin[0] + phi_Smax[0] + phi_ini - phi_S[0] <= 0), name='c_S_first') # time period 1 for phi_Smin/phi_Smax and time period 2 for phi_S
        model.addConstrs((- phi_Smin[i] + phi_Smax[i] + phi_S[i - 1] - phi_S[i] <= 0 for i in range(1, self.nb_periods - 1)), name='c_S') # time period 3 to nb_periods - 1
        model.addConstr((- phi_Smin[self.nb_periods - 1] + phi_Smax[self.nb_periods - 1] + phi_end + phi_S[self.nb_periods - 2] <= 0), name='c_S_end') # last time period
        
        # Constraints related to PV and load
        model.addConstrs((phi_PV[i] + phi[i] <= 0 for i in self.t_set), name='c_PV')
        model.addConstrs((phi_WT[i] + phi[i] <= 0 for i in self.t_set), name='c_WT')
        model.addConstrs((phi_load[i] - phi[i] <= 0 for i in self.t_set), name='c_load')

        # Constraints related to PV curtailment
        model.addConstrs((phi_curt_PV[i] - phi[i] <= self.cost_m_re_PV for i in self.t_set), name='c_curt_PV')
        model.addConstrs((phi_curt_WT[i] - phi[i] <= self.cost_m_re_WT for i in self.t_set), name='c_curt_WT')

        model.addConstrs((phi_pc[i] + phi[i] <= self.cost_m_cn_PV for i in self.t_set), name='c_pc')
        model.addConstrs((phi_wc[i] + phi[i] <= self.cost_m_cn_WT for i in self.t_set), name='c_wc')
        # -------------------------------------------------------------------------------------------------------------

        # Constraints related to the uncertainty budget
        # model.addConstrs((epsilon_pos_100[i] <= self.big_M * x_100[i] for i in self.t_set), "log_transform_epsilon_pos_100")
        # model.addConstrs((epsilon_neg_100[i] <= self.big_M * x_100[i] for i in self.t_set), "log_transform_epsilon_neg_100")
        # model.addConstrs((epsilon_pos_50[i] <= self.big_M * x_50[i] for i in self.t_set), "log_transform_epsilon_pos_50")
        # model.addConstrs((epsilon_neg_50[i] <= self.big_M * x_50[i] for i in self.t_set), "log_transform_epsilon_neg_50")
        # model.addConstrs((epsilon_pos_20[i] <= self.big_M * x_20[i] for i in self.t_set), "log_transform_epsilon_pos_20")
        # model.addConstrs((epsilon_neg_20[i] <= self.big_M * x_20[i] for i in self.t_set), "log_transform_epsilon_neg_20")
        # model.addConstrs((kapa_pos_100[i] <= self.big_M * y_100[i] for i in self.t_set), "log_transform_kapa_pos_100")
        # model.addConstrs((kapa_neg_100[i] <= self.big_M * y_100[i] for i in self.t_set), "log_transform_kapa_neg_100")
        # model.addConstrs((kapa_pos_50[i] <= self.big_M * y_50[i] for i in self.t_set), "log_transform_kapa_pos_50")
        # model.addConstrs((kapa_neg_50[i] <= self.big_M * y_50[i] for i in self.t_set), "log_transform_kapa_neg_50")
        # model.addConstrs((kapa_pos_20[i] <= self.big_M * y_20[i] for i in self.t_set), "log_transform_kapa_pos_20")
        # model.addConstrs((kapa_neg_20[i] <= self.big_M * y_20[i] for i in self.t_set), "log_transform_kapa_neg_20")
        # model.addConstrs((xi_pos_100[i] <= self.big_M * z_100[i] for i in self.t_set), "log_transform_xi_pos_100")
        # model.addConstrs((xi_neg_100[i] <= self.big_M * z_100[i] for i in self.t_set), "log_transform_xi_neg_100")
        # model.addConstrs((xi_pos_50[i] <= self.big_M * z_50[i] for i in self.t_set), "log_transform_xi_pos_50")
        # model.addConstrs((xi_neg_50[i] <= self.big_M * z_50[i] for i in self.t_set), "log_transform_xi_neg_50")
        # model.addConstrs((xi_pos_20[i] <= self.big_M * z_20[i] for i in self.t_set), "log_transform_xi_pos_20")
        # model.addConstrs((xi_neg_20[i] <= self.big_M * z_20[i] for i in self.t_set), "log_transform_xi_neg_20")


        model.addConstrs((epsilon_pos_100[i] <= self.big_M * x_100[i] for i in self.t_set), "linearization_x_pos_100")
        model.addConstrs((epsilon_neg_100[i] <= self.big_M * (1 - x_100[i]) for i in self.t_set), "linearization_x_neg_100")
        # model.addConstrs((epsilon_pos_100[i] + epsilon_neg_100[i] >= self.epsilon * x_100[i] for i in self.t_set), "linearization_lower")
        model.addConstrs((epsilon_pos_50[i] <= self.big_M * x_50[i] for i in self.t_set), "linearization_x_pos_50")
        model.addConstrs((epsilon_neg_50[i] <= self.big_M * (1 - x_50[i]) for i in self.t_set), "linearization_x_neg_50")
        # model.addConstrs((epsilon_pos_50[i] + epsilon_neg_50[i] >= self.epsilon * x_50[i] for i in self.t_set), "linearization_lower")
        model.addConstrs((epsilon_pos_20[i] <= self.big_M * x_20[i] for i in self.t_set), "linearization_x_pos_20")
        model.addConstrs((epsilon_neg_20[i] <= self.big_M * (1 - x_20[i]) for i in self.t_set), "linearization_x_neg_20")
        # model.addConstrs((epsilon_pos_20[i] + epsilon_neg_20[i] >= self.epsilon * x_20[i] for i in self.t_set), "linearization_lower")
        model.addConstrs((kapa_pos_100[i] <= self.big_M * y_100[i] for i in self.t_set), "linearization_y_pos_100")
        model.addConstrs((kapa_neg_100[i] <= self.big_M * (1 - y_100[i]) for i in self.t_set), "linearization_y_neg_100")
        # model.addConstrs((kapa_pos_100[i] + kapa_neg_100[i] >= self.epsilon * y_100[i] for i in self.t_set), "linearization_lower")
        model.addConstrs((kapa_pos_50[i] <= self.big_M * y_50[i] for i in self.t_set), "linearization_y_pos_50")
        model.addConstrs((kapa_neg_50[i] <= self.big_M * (1 - y_50[i]) for i in self.t_set), "linearization_y_neg_50")
        # model.addConstrs((kapa_pos_50[i] + kapa_neg_50[i] >= self.epsilon * y_50[i] for i in self.t_set), "linearization_lower")
        model.addConstrs((kapa_pos_20[i] <= self.big_M * y_20[i] for i in self.t_set), "linearization_y_pos_20")
        model.addConstrs((kapa_neg_20[i] <= self.big_M * (1 - y_20[i]) for i in self.t_set), "linearization_y_neg_20")
        # model.addConstrs((kapa_pos_20[i] + kapa_neg_20[i] >= self.epsilon * y_20[i] for i in self.t_set), "linearization_lower")
        model.addConstrs((xi_pos_100[i] <= self.big_M * z_100[i] for i in self.t_set), "linearization_z_pos_100")
        model.addConstrs((xi_neg_100[i] <= self.big_M * (1 - z_100[i]) for i in self.t_set), "linearization_z_neg_100")
        # model.addConstrs((xi_pos_100[i] + xi_neg_100[i] >= self.epsilon * z_100[i] for i in self.t_set), "linearization_lower")
        model.addConstrs((xi_pos_50[i] <= self.big_M * z_50[i] for i in self.t_set), "linearization_z_pos_50")
        model.addConstrs((xi_neg_50[i] <= self.big_M * (1 - z_50[i]) for i in self.t_set), "linearization_z_neg_50")
        # model.addConstrs((xi_pos_50[i] + xi_neg_50[i] >= self.epsilon * z_50[i] for i in self.t_set), "linearization_lower")
        model.addConstrs((xi_pos_20[i] <= self.big_M * z_20[i] for i in self.t_set), "linearization_z_pos_20")
        model.addConstrs((xi_neg_20[i] <= self.big_M * (1 - z_20[i]) for i in self.t_set), "linearization_z_neg_20")
        # model.addConstrs((xi_pos_20[i] + xi_neg_20[i] >= self.epsilon * z_20[i] for i in self.t_set), "linearization_lower")


        # model.addConstrs((epsilon_pos_100[i] * epsilon_neg_100[i] == 0 for i in self.t_set), name='c_epsilon_100_simul')
        # model.addConstrs((epsilon_pos_50[i] * epsilon_neg_50[i] == 0 for i in self.t_set), name='c_epsilon_50_simul')
        # model.addConstrs((epsilon_pos_20[i] * epsilon_neg_20[i] == 0 for i in self.t_set), name='c_epsilon_20_simul')
        # model.addConstrs((kapa_pos_100[i] * kapa_neg_100[i] == 0 for i in self.t_set), name='c_kapa_100_simul')
        # model.addConstrs((kapa_pos_50[i] * kapa_neg_50[i] == 0 for i in self.t_set), name='c_kapa_50_simul')
        # model.addConstrs((kapa_pos_20[i] * kapa_neg_20[i] == 0 for i in self.t_set), name='c_kapa_20_simul')
        # model.addConstrs((xi_pos_100[i] * xi_neg_100[i] == 0 for i in self.t_set), name='c_xi_100_simul')
        # model.addConstrs((xi_pos_50[i] * xi_neg_50[i] == 0 for i in self.t_set), name='c_xi_50_simul')
        # model.addConstrs((xi_pos_20[i] * xi_neg_20[i] == 0 for i in self.t_set), name='c_xi_20_simul')
        model.addConstr(gp.quicksum(epsilon_pos_100[i] + epsilon_neg_100[i] for i in self.t_set) <= Pi_PV_100, name='c_Pi_PV_100') # PV uncertainty budget
        model.addConstr(gp.quicksum(epsilon_pos_50[i] + epsilon_neg_50[i] for i in self.t_set) <= Pi_PV_50, name='c_Pi_PV_50')
        model.addConstr(gp.quicksum(epsilon_pos_20[i] + epsilon_neg_20[i] for i in self.t_set) <= Pi_PV_20, name='c_Pi_PV_20')
        model.addConstr(gp.quicksum(kapa_pos_100[i] + kapa_neg_100[i] for i in self.t_set) <= Pi_WT_100, name='c_Pi_WT_100') # WT uncertainty budget
        model.addConstr(gp.quicksum(kapa_pos_50[i] + kapa_neg_50[i] for i in self.t_set) <= Pi_WT_50, name='c_Pi_WT_50')
        model.addConstr(gp.quicksum(kapa_pos_20[i] + kapa_neg_20[i] for i in self.t_set) <= Pi_WT_20, name='c_Pi_WT_20')
        model.addConstr(gp.quicksum(xi_pos_100[i] + xi_neg_100[i] for i in self.t_set) <= Pi_load_100, name='c_Pi_load_100') # load uncertainty budget
        model.addConstr(gp.quicksum(xi_pos_50[i] + xi_neg_50[i] for i in self.t_set) <= Pi_load_50, name='c_Pi_load_50')
        model.addConstr(gp.quicksum(xi_pos_20[i] + xi_neg_20[i] for i in self.t_set) <= Pi_load_20, name='c_Pi_load_20')
        model.addConstrs((epsilon_pos_100[i] + epsilon_pos_50[i] + epsilon_pos_20[i] + epsilon_neg_100[i] + epsilon_neg_50[i] + epsilon_neg_20[i] >= 0 for i in self.t_set), name='c_epsilon_boundary_lb')
        model.addConstrs((epsilon_pos_100[i] + epsilon_pos_50[i] + epsilon_pos_20[i] + epsilon_neg_100[i] + epsilon_neg_50[i] + epsilon_neg_20[i] <= 1 for i in self.t_set), name='c_epsilon_boundary_ub')
        model.addConstrs((kapa_pos_100[i] + kapa_pos_50[i] + kapa_pos_20[i] + kapa_neg_100[i] + kapa_neg_50[i] + kapa_neg_20[i] >= 0 for i in self.t_set), name='c_kapa_boundary_lb')
        model.addConstrs((kapa_pos_100[i] + kapa_pos_50[i] + kapa_pos_20[i] + kapa_neg_100[i] + kapa_neg_50[i] + kapa_neg_20[i] <= 1 for i in self.t_set), name='c_kapa_boundary_ub')
        model.addConstrs((xi_pos_100[i] + xi_pos_50[i] + xi_pos_20[i] + xi_neg_100[i] + xi_neg_50[i] + xi_neg_20[i] >= 0 for i in self.t_set), name='c_xi_boundary_lb')
        model.addConstrs((xi_pos_100[i] + xi_pos_50[i] + xi_pos_20[i] + xi_neg_100[i] + xi_neg_50[i] + xi_neg_20[i] <= 1 for i in self.t_set), name='c_xi_boundary_ub')
        model.addConstr((Pi_PV_100 + Pi_PV_50 * 50 / 100 + Pi_PV_20 * 20 / 100 == self.Pi_PV_t), name='c_Pi_PV')
        model.addConstr((Pi_PV_50 == Pi_PV_100 / 0.15 * 0.35), name='c_Pi_PV_50')
        model.addConstr((Pi_PV_20 == Pi_PV_100 / 0.15 * 0.5), name='c_Pi_PV_20')
        model.addConstr((Pi_WT_100 + Pi_WT_50 * 50 / 100 + Pi_WT_20 * 20 / 100 == self.Pi_WT_t), name='c_Pi_WT')
        model.addConstr((Pi_WT_50 == Pi_WT_100 / 0.15 * 0.35), name='c_Pi_WT_50')
        model.addConstr((Pi_WT_20 == Pi_WT_100 / 0.15 * 0.5), name='c_Pi_WT_20')
        model.addConstr((Pi_load_100 + Pi_load_50 * 50 / 100 + Pi_load_20 * 20 / 100 == self.Pi_load), name='c_Pi_load')
        model.addConstr((Pi_load_50 == Pi_load_100 / 0.15 * 0.35), name='c_Pi_load_50')
        model.addConstr((Pi_load_20 == Pi_load_100 / 0.15 * 0.5), name='c_Pi_load_20')


        # Constraints related to the uncertainty ramp up/down
        model.addConstrs(((self.PV_forecast[i] + self.PV_pos_100[i] * epsilon_pos_100[i] - self.PV_neg_100[i] * epsilon_neg_100[i] + self.PV_pos_50[i] * epsilon_pos_50[i] - self.PV_neg_50[i] * epsilon_neg_50[i] + self.PV_pos_20[i] * epsilon_pos_20[i] - self.PV_neg_20[i] * epsilon_neg_20[i])
                          - (self.PV_forecast[i-1] + self.PV_pos_100[i-1] * epsilon_pos_100[i-1] - self.PV_neg_100[i-1] * epsilon_neg_100[i-1] + self.PV_pos_50[i-1] * epsilon_pos_50[i-1] - self.PV_neg_50[i-1] * epsilon_neg_50[i-1] + self.PV_pos_20[i-1] * epsilon_pos_20[i-1] - self.PV_neg_20[i-1] * epsilon_neg_20[i-1]) <= self.PV_ramp_up * self.period_hours for i in range(1, self.nb_periods)), name='c_PV_ramp_up')
        model.addConstrs(((self.PV_forecast[i-1] + self.PV_pos_100[i-1] * epsilon_pos_100[i-1] - self.PV_neg_100[i-1] * epsilon_neg_100[i-1] + self.PV_pos_50[i-1] * epsilon_pos_50[i-1] - self.PV_neg_50[i-1] * epsilon_neg_50[i-1] + self.PV_pos_20[i-1] * epsilon_pos_20[i-1] - self.PV_neg_20[i-1] * epsilon_neg_20[i-1])
                          - (self.PV_forecast[i] + self.PV_pos_100[i] * epsilon_pos_100[i] - self.PV_neg_100[i] * epsilon_neg_100[i] + self.PV_pos_50[i] * epsilon_pos_50[i] - self.PV_neg_50[i] * epsilon_neg_50[i] + self.PV_pos_20[i] * epsilon_pos_20[i] - self.PV_neg_20[i] * epsilon_neg_20[i]) <= self.PV_ramp_down * self.period_hours for i in range(1, self.nb_periods)), name='c_PV_ramp_down')
        model.addConstrs(((self.WT_forecast[i] + self.WT_pos_100[i] * epsilon_pos_100[i] - self.WT_neg_100[i] * epsilon_neg_100[i] + self.WT_pos_50[i] * epsilon_pos_50[i] - self.WT_neg_50[i] * epsilon_neg_50[i] + self.WT_pos_20[i] * epsilon_pos_20[i] - self.WT_neg_20[i] * epsilon_neg_20[i])
                          - (self.WT_forecast[i-1] + self.WT_pos_100[i-1] * epsilon_pos_100[i-1] - self.WT_neg_100[i-1] * epsilon_neg_100[i-1] + self.WT_pos_50[i-1] * epsilon_pos_50[i-1] - self.WT_neg_50[i-1] * epsilon_neg_50[i-1] + self.WT_pos_20[i-1] * epsilon_pos_20[i-1] - self.WT_neg_20[i-1] * epsilon_neg_20[i-1]) <= self.WT_ramp_up * self.period_hours for i in range(1, self.nb_periods)), name='c_WT_ramp_up')
        model.addConstrs(((self.WT_forecast[i-1] + self.WT_pos_100[i-1] * epsilon_pos_100[i-1] - self.WT_neg_100[i-1] * epsilon_neg_100[i-1] + self.WT_pos_50[i-1] * epsilon_pos_50[i-1] - self.WT_neg_50[i-1] * epsilon_neg_50[i-1] + self.WT_pos_20[i-1] * epsilon_pos_20[i-1] - self.WT_neg_20[i-1] * epsilon_neg_20[i-1])
                          - (self.WT_forecast[i] + self.WT_pos_100[i] * epsilon_pos_100[i] - self.WT_neg_100[i] * epsilon_neg_100[i] + self.WT_pos_50[i] * epsilon_pos_50[i] - self.WT_neg_50[i] * epsilon_neg_50[i] + self.WT_pos_20[i] * epsilon_pos_20[i] - self.WT_neg_20[i] * epsilon_neg_20[i]) <= self.WT_ramp_down * self.period_hours for i in range(1, self.nb_periods)), name='c_WT_ramp_down')
        model.addConstrs(((self.load_forecast[i] + self.load_pos_100[i] * xi_pos_100[i] - self.load_neg_100[i] * xi_neg_100[i] + self.load_pos_50[i] * xi_pos_50[i] - self.load_neg_50[i] * xi_neg_50[i] + self.load_pos_20[i] * xi_pos_20[i] - self.load_neg_20[i] * xi_neg_20[i])
                          - (self.load_forecast[i-1] + self.load_pos_100[i-1] * xi_pos_100[i-1] - self.load_neg_100[i-1] * xi_neg_100[i-1] + self.load_pos_50[i-1] * xi_pos_50[i-1] - self.load_neg_50[i-1] * xi_neg_50[i-1] + self.load_pos_20[i-1] * xi_pos_20[i-1] - self.load_neg_20[i-1] * xi_neg_20[i-1]) <= self.load_ramp_up * self.period_hours for i in range(1, self.nb_periods)), name='c_load_ramp_up')
        model.addConstrs(((self.load_forecast[i-1] + self.load_pos_100[i-1] * xi_pos_100[i-1] - self.load_neg_100[i-1] * xi_neg_100[i-1] + self.load_pos_50[i-1] * xi_pos_50[i-1] - self.load_neg_50[i-1] * xi_neg_50[i-1] + self.load_pos_20[i-1] * xi_pos_20[i-1] - self.load_neg_20[i-1] * xi_neg_20[i-1])
                          - (self.load_forecast[i] + self.load_pos_100[i] * xi_pos_100[i] - self.load_neg_100[i] * xi_neg_100[i] + self.load_pos_50[i] * xi_pos_50[i] - self.load_neg_50[i] * xi_neg_50[i] + self.load_pos_20[i] * xi_pos_20[i] - self.load_neg_20[i] * xi_neg_20[i]) <= self.load_ramp_down * self.period_hours for i in range(1, self.nb_periods)), name='c_load_ramp_down')
        
        # Constraints related to the McCormick method
        # alpha_pos/neg
        model.addConstrs((alpha_pos_100[i] >= - epsilon_pos_100[i] for i in self.t_set), name='c_alpha_pos__100_1')
        model.addConstrs((alpha_pos_100[i] >= phi_PV[i] + epsilon_pos_100[i] - 1 for i in self.t_set), name='c_alpha_pos__100_2')
        model.addConstrs((alpha_pos_100[i] <= epsilon_pos_100[i] for i in self.t_set), name='c_alpha_pos__100_3')
        model.addConstrs((alpha_pos_100[i] <= phi_PV[i] - epsilon_pos_100[i] + 1 for i in self.t_set), name='c_alpha_pos__100_4')
        model.addConstrs((alpha_pos_50[i] >= - epsilon_pos_50[i] for i in self.t_set), name='c_alpha_pos__50_1')
        model.addConstrs((alpha_pos_50[i] >= phi_PV[i] + epsilon_pos_50[i] - 1 for i in self.t_set), name='c_alpha_pos__50_2')
        model.addConstrs((alpha_pos_50[i] <= epsilon_pos_50[i] for i in self.t_set), name='c_alpha_pos__50_3')
        model.addConstrs((alpha_pos_50[i] <= phi_PV[i] - epsilon_pos_50[i] + 1 for i in self.t_set), name='c_alpha_pos__50_4')
        model.addConstrs((alpha_pos_20[i] >= - epsilon_pos_20[i] for i in self.t_set), name='c_alpha_pos__20_1')
        model.addConstrs((alpha_pos_20[i] >= phi_PV[i] + epsilon_pos_20[i] - 1 for i in self.t_set), name='c_alpha_pos__20_2')
        model.addConstrs((alpha_pos_20[i] <= epsilon_pos_20[i] for i in self.t_set), name='c_alpha_pos__20_3')
        model.addConstrs((alpha_pos_20[i] <= phi_PV[i] - epsilon_pos_20[i] + 1 for i in self.t_set), name='c_alpha_pos__20_4')

        model.addConstrs((alpha_neg_100[i] >= - epsilon_neg_100[i] for i in self.t_set), name='c_alpha_neg__100_1')
        model.addConstrs((alpha_neg_100[i] >= phi_PV[i] + epsilon_neg_100[i] - 1 for i in self.t_set), name='c_alpha_neg__100_2')
        model.addConstrs((alpha_neg_100[i] <= epsilon_neg_100[i] for i in self.t_set), name='c_alpha_neg__100_3')
        model.addConstrs((alpha_neg_100[i] <= phi_PV[i] - epsilon_neg_100[i] + 1 for i in self.t_set), name='c_alpha_neg__100_4')
        model.addConstrs((alpha_neg_50[i] >= - epsilon_neg_50[i] for i in self.t_set), name='c_alpha_neg__50_1')
        model.addConstrs((alpha_neg_50[i] >= phi_PV[i] + epsilon_neg_50[i] - 1 for i in self.t_set), name='c_alpha_neg__50_2')
        model.addConstrs((alpha_neg_50[i] <= epsilon_neg_50[i] for i in self.t_set), name='c_alpha_neg__50_3')
        model.addConstrs((alpha_neg_50[i] <= phi_PV[i] - epsilon_neg_50[i] + 1 for i in self.t_set), name='c_alpha_neg__50_4')
        model.addConstrs((alpha_neg_20[i] >= - epsilon_neg_20[i] for i in self.t_set), name='c_alpha_neg__20_1')
        model.addConstrs((alpha_neg_20[i] >= phi_PV[i] + epsilon_neg_20[i] - 1 for i in self.t_set), name='c_alpha_neg__20_2')
        model.addConstrs((alpha_neg_20[i] <= epsilon_neg_20[i] for i in self.t_set), name='c_alpha_neg__20_3')
        model.addConstrs((alpha_neg_20[i] <= phi_PV[i] - epsilon_neg_20[i] + 1 for i in self.t_set), name='c_alpha_neg__20_4')

        # beta_pos/neg
        model.addConstrs((beta_pos_100[i] >= - kapa_pos_100[i] for i in self.t_set), name='c_beta_pos__100_1')
        model.addConstrs((beta_pos_100[i] >= phi_WT[i] + kapa_pos_100[i] - 1 for i in self.t_set), name='c_beta_pos__100_2')
        model.addConstrs((beta_pos_100[i] <= kapa_pos_100[i] for i in self.t_set), name='c_beta_pos__100_3')
        model.addConstrs((beta_pos_100[i] <= phi_WT[i] - kapa_pos_100[i] + 1 for i in self.t_set), name='c_beta_pos__100_4')
        model.addConstrs((beta_pos_50[i] >= - kapa_pos_50[i] for i in self.t_set), name='c_beta_pos__50_1')
        model.addConstrs((beta_pos_50[i] >= phi_WT[i] + kapa_pos_50[i] - 1 for i in self.t_set), name='c_beta_pos__50_2')
        model.addConstrs((beta_pos_50[i] <= kapa_pos_50[i] for i in self.t_set), name='c_beta_pos__50_3')
        model.addConstrs((beta_pos_50[i] <= phi_WT[i] - kapa_pos_50[i] + 1 for i in self.t_set), name='c_beta_pos__50_4')
        model.addConstrs((beta_pos_20[i] >= - kapa_pos_20[i] for i in self.t_set), name='c_beta_pos__20_1')
        model.addConstrs((beta_pos_20[i] >= phi_WT[i] + kapa_pos_20[i] - 1 for i in self.t_set), name='c_beta_pos__20_2')
        model.addConstrs((beta_pos_20[i] <= kapa_pos_20[i] for i in self.t_set), name='c_beta_pos__20_3')
        model.addConstrs((beta_pos_20[i] <= phi_WT[i] - kapa_pos_20[i] + 1 for i in self.t_set), name='c_beta_pos__20_4')
        
        model.addConstrs((beta_neg_100[i] >= - kapa_neg_100[i] for i in self.t_set), name='c_beta_neg__100_1')
        model.addConstrs((beta_neg_100[i] >= phi_WT[i] + kapa_neg_100[i] - 1 for i in self.t_set), name='c_beta_neg__100_2')
        model.addConstrs((beta_neg_100[i] <= kapa_neg_100[i] for i in self.t_set), name='c_beta_neg__100_3')
        model.addConstrs((beta_neg_100[i] <= phi_WT[i] - kapa_neg_100[i] + 1 for i in self.t_set), name='c_beta_neg__100_4')
        model.addConstrs((beta_neg_50[i] >= - kapa_neg_50[i] for i in self.t_set), name='c_beta_neg__50_1')
        model.addConstrs((beta_neg_50[i] >= phi_WT[i] + kapa_neg_50[i] - 1 for i in self.t_set), name='c_beta_neg__50_2')
        model.addConstrs((beta_neg_50[i] <= kapa_neg_50[i] for i in self.t_set), name='c_beta_neg__50_3')
        model.addConstrs((beta_neg_50[i] <= phi_WT[i] - kapa_neg_50[i] + 1 for i in self.t_set), name='c_beta_neg__50_4')
        model.addConstrs((beta_neg_20[i] >= - kapa_neg_20[i] for i in self.t_set), name='c_beta_neg__20_1')
        model.addConstrs((beta_neg_20[i] >= phi_WT[i] + kapa_neg_20[i] - 1 for i in self.t_set), name='c_beta_neg__20_2')
        model.addConstrs((beta_neg_20[i] <= kapa_neg_20[i] for i in self.t_set), name='c_beta_neg__20_3')
        model.addConstrs((beta_neg_20[i] <= phi_WT[i] - kapa_neg_20[i] + 1 for i in self.t_set), name='c_beta_neg__20_4')

        # gamma_pos/neg
        model.addConstrs((gamma_pos_100[i] >= - xi_pos_100[i] for i in self.t_set), name='c_gamma_pos__100_1')
        model.addConstrs((gamma_pos_100[i] >= phi_load[i] + xi_pos_100[i] - 1 for i in self.t_set), name='c_gamma_pos__100_2')
        model.addConstrs((gamma_pos_100[i] <= xi_pos_100[i] for i in self.t_set), name='c_gamma_pos__100_3')
        model.addConstrs((gamma_pos_100[i] <= phi_load[i] - xi_pos_100[i] + 1 for i in self.t_set), name='c_gamma_pos__100_4')
        model.addConstrs((gamma_pos_50[i] >= - xi_pos_50[i] for i in self.t_set), name='c_gamma_pos__50_1')
        model.addConstrs((gamma_pos_50[i] >= phi_load[i] + xi_pos_50[i] - 1 for i in self.t_set), name='c_gamma_pos__50_2')
        model.addConstrs((gamma_pos_50[i] <= xi_pos_50[i] for i in self.t_set), name='c_gamma_pos__50_3')
        model.addConstrs((gamma_pos_50[i] <= phi_load[i] - xi_pos_50[i] + 1 for i in self.t_set), name='c_gamma_pos__50_4')
        model.addConstrs((gamma_pos_20[i] >= - xi_pos_20[i] for i in self.t_set), name='c_gamma_pos__20_1')
        model.addConstrs((gamma_pos_20[i] >= phi_load[i] + xi_pos_20[i] - 1 for i in self.t_set), name='c_gamma_pos__20_2')
        model.addConstrs((gamma_pos_20[i] <= xi_pos_20[i] for i in self.t_set), name='c_gamma_pos__20_3')
        model.addConstrs((gamma_pos_20[i] <= phi_load[i] - xi_pos_20[i] + 1 for i in self.t_set), name='c_gamma_pos__20_4')
        
        model.addConstrs((gamma_neg_100[i] >= - xi_neg_100[i] for i in self.t_set), name='c_gamma_neg__100_1')
        model.addConstrs((gamma_neg_100[i] >= phi_load[i] + xi_neg_100[i] - 1 for i in self.t_set), name='c_gamma_neg__100_2')
        model.addConstrs((gamma_neg_100[i] <= xi_neg_100[i] for i in self.t_set), name='c_gamma_neg__100_3')
        model.addConstrs((gamma_neg_100[i] <= phi_load[i] - xi_neg_100[i] + 1 for i in self.t_set), name='c_gamma_neg__100_4')
        model.addConstrs((gamma_neg_50[i] >= - xi_neg_50[i] for i in self.t_set), name='c_gamma_neg__50_1')
        model.addConstrs((gamma_neg_50[i] >= phi_load[i] + xi_neg_50[i] - 1 for i in self.t_set), name='c_gamma_neg__50_2')
        model.addConstrs((gamma_neg_50[i] <= xi_neg_50[i] for i in self.t_set), name='c_gamma_neg__50_3')
        model.addConstrs((gamma_neg_50[i] <= phi_load[i] - xi_neg_50[i] + 1 for i in self.t_set), name='c_gamma_neg__50_4')
        model.addConstrs((gamma_neg_20[i] >= - xi_neg_20[i] for i in self.t_set), name='c_gamma_neg__20_1')
        model.addConstrs((gamma_neg_20[i] >= phi_load[i] + xi_neg_20[i] - 1 for i in self.t_set), name='c_gamma_neg__20_2')
        model.addConstrs((gamma_neg_20[i] <= xi_neg_20[i] for i in self.t_set), name='c_gamma_neg__20_3')
        model.addConstrs((gamma_neg_20[i] <= phi_load[i] - xi_neg_20[i] + 1 for i in self.t_set), name='c_gamma_neg__20_4')

        # delta
        model.addConstrs((delta_pos_100[i] >= - epsilon_pos_100[i] for i in self.t_set), name='c_delta_pos_100_1')
        model.addConstrs((delta_pos_100[i] >= phi_curt_PV[i] for i in self.t_set), name='c_delta_pos_100_2')
        model.addConstrs((delta_pos_100[i] <= 0 for i in self.t_set), name='c_delta_pos_100_3')
        model.addConstrs((delta_pos_100[i] <= phi_curt_PV[i] - epsilon_pos_100[i] + 1 for i in self.t_set), name='c_delta_pos_100_4')
        model.addConstrs((delta_pos_50[i] >= - epsilon_pos_50[i] for i in self.t_set), name='c_delta_pos_50_1')
        model.addConstrs((delta_pos_50[i] >= phi_curt_PV[i] for i in self.t_set), name='c_delta_pos_50_2')
        model.addConstrs((delta_pos_50[i] <= 0 for i in self.t_set), name='c_delta_pos_50_3')
        model.addConstrs((delta_pos_50[i] <= phi_curt_PV[i] - epsilon_pos_50[i] + 1 for i in self.t_set), name='c_delta_pos_50_4')
        model.addConstrs((delta_pos_20[i] >= - epsilon_pos_20[i] for i in self.t_set), name='c_delta_pos_20_1')
        model.addConstrs((delta_pos_20[i] >= phi_curt_PV[i] for i in self.t_set), name='c_delta_pos_20_2')
        model.addConstrs((delta_pos_20[i] <= 0 for i in self.t_set), name='c_delta_pos_20_3')
        model.addConstrs((delta_pos_20[i] <= phi_curt_PV[i] - epsilon_pos_20[i] + 1 for i in self.t_set), name='c_delta_pos_20_4')

        model.addConstrs((delta_neg_100[i] >= - epsilon_neg_100[i] for i in self.t_set), name='c_delta_neg_100_1')
        model.addConstrs((delta_neg_100[i] >= phi_curt_PV[i] for i in self.t_set), name='c_delta_neg_100_2')
        model.addConstrs((delta_neg_100[i] <= 0 for i in self.t_set), name='c_delta_neg_100_3')
        model.addConstrs((delta_neg_100[i] <= phi_curt_PV[i] - epsilon_neg_100[i] + 1 for i in self.t_set), name='c_delta_neg_100_4')
        model.addConstrs((delta_neg_50[i] >= - epsilon_neg_50[i] for i in self.t_set), name='c_delta_neg_50_1')
        model.addConstrs((delta_neg_50[i] >= phi_curt_PV[i] for i in self.t_set), name='c_delta_neg_50_2')
        model.addConstrs((delta_neg_50[i] <= 0 for i in self.t_set), name='c_delta_neg_50_3')
        model.addConstrs((delta_neg_50[i] <= phi_curt_PV[i] - epsilon_neg_50[i] + 1 for i in self.t_set), name='c_delta_neg_50_4')
        model.addConstrs((delta_neg_20[i] >= - epsilon_neg_20[i] for i in self.t_set), name='c_delta_neg_20_1')
        model.addConstrs((delta_neg_20[i] >= phi_curt_PV[i] for i in self.t_set), name='c_delta_neg_20_2')
        model.addConstrs((delta_neg_20[i] <= 0 for i in self.t_set), name='c_delta_neg_20_3')
        model.addConstrs((delta_neg_20[i] <= phi_curt_PV[i] - epsilon_neg_20[i] + 1 for i in self.t_set), name='c_delta_neg_20_4')

        # zeta
        model.addConstrs((zeta_pos_100[i] >= - kapa_pos_100[i] for i in self.t_set), name='c_zeta_pos_100_1')
        model.addConstrs((zeta_pos_100[i] >= phi_curt_WT[i] for i in self.t_set), name='c_zeta_pos_100_2')
        model.addConstrs((zeta_pos_100[i] <= 0 for i in self.t_set), name='c_zeta_pos_100_3')
        model.addConstrs((zeta_pos_100[i] <= phi_curt_WT[i] - kapa_pos_100[i] + 1 for i in self.t_set), name='c_zeta_pos_100_4')
        model.addConstrs((zeta_pos_50[i] >= - kapa_pos_50[i] for i in self.t_set), name='c_zeta_pos_50_1')
        model.addConstrs((zeta_pos_50[i] >= phi_curt_WT[i] for i in self.t_set), name='c_zeta_pos_50_2')
        model.addConstrs((zeta_pos_50[i] <= 0 for i in self.t_set), name='c_zeta_pos_50_3')
        model.addConstrs((zeta_pos_50[i] <= phi_curt_WT[i] - kapa_pos_50[i] + 1 for i in self.t_set), name='c_zeta_pos_50_4')
        model.addConstrs((zeta_pos_20[i] >= - kapa_pos_20[i] for i in self.t_set), name='c_zeta_pos_20_1')
        model.addConstrs((zeta_pos_20[i] >= phi_curt_WT[i] for i in self.t_set), name='c_zeta_pos_20_2')
        model.addConstrs((zeta_pos_20[i] <= 0 for i in self.t_set), name='c_zeta_pos_20_3')
        model.addConstrs((zeta_pos_20[i] <= phi_curt_WT[i] - kapa_pos_20[i] + 1 for i in self.t_set), name='c_zeta_pos_20_4')

        model.addConstrs((zeta_neg_100[i] >= - kapa_neg_100[i] for i in self.t_set), name='c_zeta_neg_100_1')
        model.addConstrs((zeta_neg_100[i] >= phi_curt_WT[i] for i in self.t_set), name='c_zeta_neg_100_2')
        model.addConstrs((zeta_neg_100[i] <= 0 for i in self.t_set), name='c_zeta_neg_100_3')
        model.addConstrs((zeta_neg_100[i] <= phi_curt_WT[i] - kapa_neg_100[i] + 1 for i in self.t_set), name='c_zeta_neg_100_4')
        model.addConstrs((zeta_neg_50[i] >= - kapa_neg_50[i] for i in self.t_set), name='c_zeta_neg_50_1')
        model.addConstrs((zeta_neg_50[i] >= phi_curt_WT[i] for i in self.t_set), name='c_zeta_neg_50_2')
        model.addConstrs((zeta_neg_50[i] <= 0 for i in self.t_set), name='c_zeta_neg_50_3')
        model.addConstrs((zeta_neg_50[i] <= phi_curt_WT[i] - kapa_neg_50[i] + 1 for i in self.t_set), name='c_zeta_neg_50_4')
        model.addConstrs((zeta_neg_20[i] >= - kapa_neg_20[i] for i in self.t_set), name='c_zeta_neg_20_1')
        model.addConstrs((zeta_neg_20[i] >= phi_curt_WT[i] for i in self.t_set), name='c_zeta_neg_20_2')
        model.addConstrs((zeta_neg_20[i] <= 0 for i in self.t_set), name='c_zeta_neg_20_3')
        model.addConstrs((zeta_neg_20[i] <= phi_curt_WT[i] - kapa_neg_20[i] + 1 for i in self.t_set), name='c_zeta_neg_20_4')

        # -------------------------------------------------------------------------------------------------------------
        # 5. Store variables
        self.allvar = dict()
        self.allvar['phi_DE1pos'] = phi_DE1pos
        self.allvar['phi_DE1neg'] = phi_DE1neg
        self.allvar['phi_chg'] = phi_chg
        self.allvar['phi_dis'] = phi_dis
        self.allvar['phi'] = phi
        self.allvar['phi_ini'] = phi_ini
        self.allvar['phi_S'] = phi_S
        self.allvar['phi_end'] = phi_end
        self.allvar['phi_Smin'] = phi_Smin
        self.allvar['phi_Smax'] = phi_Smax
        self.allvar['phi_PV'] = phi_PV
        self.allvar['phi_WT'] = phi_WT
        self.allvar['phi_load'] = phi_load
        self.allvar['phi_curt_PV'] = phi_curt_PV
        self.allvar['phi_curt_WT'] = phi_curt_WT
        self.allvar['phi_pc'] = phi_pc
        self.allvar['phi_wc'] = phi_wc

        self.allvar['epsilon_pos_100'] = epsilon_pos_100
        self.allvar['epsilon_neg_100'] = epsilon_neg_100
        self.allvar['epsilon_pos_50'] = epsilon_pos_50
        self.allvar['epsilon_neg_50'] = epsilon_neg_50
        self.allvar['epsilon_pos_20'] = epsilon_pos_20
        self.allvar['epsilon_neg_20'] = epsilon_neg_20
        self.allvar['kapa_pos_100'] = kapa_pos_100
        self.allvar['kapa_neg_100'] = kapa_neg_100
        self.allvar['kapa_pos_50'] = kapa_pos_50
        self.allvar['kapa_neg_50'] = kapa_neg_50
        self.allvar['kapa_pos_20'] = kapa_pos_20
        self.allvar['kapa_neg_20'] = kapa_neg_20
        self.allvar['xi_pos_100'] = xi_pos_100
        self.allvar['xi_neg_100'] = xi_neg_100
        self.allvar['xi_pos_50'] = xi_pos_50
        self.allvar['xi_neg_50'] = xi_neg_50
        self.allvar['xi_pos_20'] = xi_pos_20
        self.allvar['xi_neg_20'] = xi_neg_20

        self.allvar['Pi_PV_100'] = Pi_PV_100
        self.allvar['Pi_PV_50'] = Pi_PV_50
        self.allvar['Pi_PV_20'] = Pi_PV_20
        self.allvar['Pi_WT_100'] = Pi_WT_100
        self.allvar['Pi_WT_50'] = Pi_WT_50
        self.allvar['Pi_WT_20'] = Pi_WT_20
        self.allvar['Pi_load_100'] = Pi_load_100
        self.allvar['Pi_load_50'] = Pi_load_50
        self.allvar['Pi_load_20'] = Pi_load_20

        self.allvar['alpha_pos_100'] = alpha_pos_100
        self.allvar['alpha_neg_100'] = alpha_neg_100
        self.allvar['alpha_pos_50'] = alpha_pos_50
        self.allvar['alpha_neg_50'] = alpha_neg_50
        self.allvar['alpha_pos_20'] = alpha_pos_20
        self.allvar['alpha_neg_20'] = alpha_neg_20
        self.allvar['beta_pos_100'] = beta_pos_100
        self.allvar['beta_neg_100'] = beta_neg_100
        self.allvar['beta_pos_50'] = beta_pos_50
        self.allvar['beta_neg_50'] = beta_neg_50
        self.allvar['beta_pos_20'] = beta_pos_20
        self.allvar['beta_neg_20'] = beta_neg_20
        self.allvar['gamma_pos_100'] = gamma_pos_100
        self.allvar['gamma_neg_100'] = gamma_neg_100
        self.allvar['gamma_pos_50'] = gamma_pos_50
        self.allvar['gamma_neg_50'] = gamma_neg_50
        self.allvar['gamma_pos_20'] = gamma_pos_20
        self.allvar['gamma_neg_20'] = gamma_neg_20
        self.allvar['delta_pos_100'] = delta_pos_100
        self.allvar['delta_neg_100'] = delta_neg_100
        self.allvar['delta_pos_50'] = delta_pos_50
        self.allvar['delta_neg_50'] = delta_neg_50
        self.allvar['delta_pos_20'] = delta_pos_20
        self.allvar['delta_neg_20'] = delta_neg_20
        self.allvar['zeta_pos_100'] = zeta_pos_100
        self.allvar['zeta_neg_100'] = zeta_neg_100
        self.allvar['zeta_pos_50'] = zeta_pos_50
        self.allvar['zeta_neg_50'] = zeta_neg_50
        self.allvar['zeta_pos_20'] = zeta_pos_20
        self.allvar['zeta_neg_20'] = zeta_neg_20

        self.time_building_model = time.time() - t_build
        # print("Time spent building the mathematical program %gs" % self.time_building_model)
        
        return model
    
    def solve(self, LogToConsole:bool=False, logfile:str="", Threads:int=0, MIPFocus:int=0, TimeLimit:float=GRB.INFINITY):
        """
        :param LogToConsole: no log in the console if set to False.
        :param logfile: no log in file if set to ""
        :param Threads: Default value = 0 -> use all threads
        :param MIPFocus: If you are more interested in good quality feasible solutions, you can select MIPFocus=1.
                        If you believe the solver is having no trouble finding the optimal solution, and wish to focus more attention on proving optimality, select MIPFocus=2.
                        If the best objective bound is moving very slowly (or not at all), you may want to try MIPFocus=3 to focus on the bound.
        :param TimeLimit: in seconds.
        """

        t_solve = time.time()
        self.model.setParam('LogToConsole', LogToConsole)
        # self.model.setParam('OutputFlag', outputflag) # no log into console and log file if set to True
        # self.model.setParam('MIPGap', 0.01)
        self.model.setParam('TimeLimit', TimeLimit)
        self.model.setParam('MIPFocus', MIPFocus)
        self.model.setParam('LogFile', logfile)
        self.model.setParam('Threads', Threads)

        self.model.optimize()

        if self.model.status == 2 or self.model.status == 9:
            pass
        else:
            self.model.computeIIS()
            self.model.write("infeasible_model.ilp")

        if self.model.status == gp.GRB.Status.UNBOUNDED:
            self.model.computeIIS()
            self.model.write("unbounded_model_ilp")

        self.time_solving_model = time.time() - t_solve

    def store_solution(self):

        m = self.model

        solution = dict()
        solution['status'] = m.status
        solution['obj'] = m.objVal

        # 0 dimensional variables
        for var in ['phi_ini', 'phi_end', 'Pi_PV_100', 'Pi_PV_50', 'Pi_PV_20', 'Pi_WT_100', 'Pi_WT_50', 'Pi_WT_20', 'Pi_load_100', 'Pi_load_50', 'Pi_load_20']:
            solution[var] = self.allvar[var].X

        # 1 dimensional variables
        for var in ['phi_DE1pos', 'phi_DE1neg', 'phi_chg', 'phi_dis', 'phi_Smin', 'phi_Smax',
                    'phi', 'phi_PV', 'phi_WT', 'phi_load', 'phi_curt_PV', 'phi_curt_WT', 'phi_pc', 'phi_wc',
                    'epsilon_pos_100', 'epsilon_neg_100', 'epsilon_pos_50', 'epsilon_neg_50', 'epsilon_pos_20', 'epsilon_neg_20',
                    'kapa_pos_100', 'kapa_neg_100', 'kapa_pos_50', 'kapa_neg_50', 'kapa_pos_20', 'kapa_neg_20',
                    'xi_pos_100', 'xi_neg_100', 'xi_pos_50', 'xi_neg_50', 'xi_pos_20', 'xi_neg_20',
                    'alpha_pos_100', 'alpha_neg_100', 'alpha_pos_50', 'alpha_neg_50', 'alpha_pos_20', 'alpha_neg_20',
                    'beta_pos_100', 'beta_neg_100', 'beta_pos_50', 'beta_neg_50', 'beta_pos_20', 'beta_neg_20',
                    'gamma_pos_100', 'gamma_neg_100', 'gamma_pos_50', 'gamma_neg_50', 'gamma_pos_20', 'gamma_neg_20',
                    'delta_pos_100', 'delta_neg_100', 'delta_pos_50', 'delta_neg_50', 'delta_pos_20', 'delta_neg_20',
                    'zeta_pos_100', 'zeta_neg_100', 'zeta_pos_50', 'zeta_neg_50', 'zeta_pos_20', 'zeta_neg_20']:
            solution[var] = [self.allvar[var][t].X for t in self.t_set]

        for var in ['phi_S']:
            solution[var] = [self.allvar[var][t].X for t in range(self.nb_periods - 1)]

        # 6. Timing indicators
        solution["time_building"] = self.time_building_model
        solution["time_solving"] = self.time_solving_model
        solution["time_total"] = self.time_building_model + self.time_solving_model

        return solution
    
    def export_model(self, filename):
        """
        Export the model into a lp format.
        :param filename: directory and filename of the exported model.
        """

        self.model.write("%s.lp" % filename)

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    dirname = '/Users/Andrew/OneDrive - GIST/Code/Graduation/PC_RGD_CCG_Mc_MIU/'
    day = '2018-07-04'

    PV_forecast = data.PV_pred
    WT_forecast = data.WT_pred
    load_forecast = data.load_egg
    
    DE1_p = read_file(dir='/Users/Andrew/OneDrive - GIST/Code/Graduation/PC_RGD_CCG_Mc_MIU/export_MILP/', name='sol_MILP_DE1_p')
    DE1_rp = read_file(dir='/Users/Andrew/OneDrive - GIST/Code/Graduation/PC_RGD_CCG_Mc_MIU/export_MILP/', name='sol_MILP_DE1_rp')
    DE1_rn = read_file(dir='/Users/Andrew/OneDrive - GIST/Code/Graduation/PC_RGD_CCG_Mc_MIU/export_MILP/', name='sol_MILP_DE1_rn')
    ES_charge = read_file(dir='/Users/Andrew/OneDrive - GIST/Code/Graduation/PC_RGD_CCG_Mc_MIU/export_MILP/', name='sol_MILP_ES_charge')
    ES_discharge = read_file(dir='/Users/Andrew/OneDrive - GIST/Code/Graduation/PC_RGD_CCG_Mc_MIU/export_MILP/', name='sol_MILP_ES_discharge')
    ES_SOC = read_file(dir='/Users/Andrew/OneDrive - GIST/Code/Graduation/PC_RGD_CCG_Mc_MIU/export_MILP/', name='sol_MILP_ES_SOC')
    x_curt_PV = read_file(dir='/Users/Andrew/OneDrive - GIST/Code/Graduation/PC_RGD_CCG_Mc_MIU/export_MILP/', name='sol_MILP_x_curt_PV')
    x_curt_WT = read_file(dir='/Users/Andrew/OneDrive - GIST/Code/Graduation/PC_RGD_CCG_Mc_MIU/export_MILP/', name='sol_MILP_x_curt_WT')

    PV_lb = PV_forecast - data.PV_neg_100
    PV_ub = PV_forecast + data.PV_pos_100
    WT_lb = WT_forecast - data.WT_neg_100
    WT_ub = WT_forecast + data.WT_pos_100
    load_lb = load_forecast - data.load_neg_100
    load_ub = load_forecast + data.load_pos_100
    PV_pos_100 = data.PV_pos_100
    PV_neg_100 = data.PV_neg_100
    PV_pos_50 = data.PV_pos_50
    PV_neg_50 = data.PV_neg_50
    PV_pos_20 = data.PV_pos_20
    PV_neg_20 = data.PV_neg_20
    WT_pos_100 = data.WT_pos_100
    WT_neg_100 = data.WT_neg_100
    WT_pos_50 = data.WT_pos_50
    WT_neg_50 = data.WT_neg_50
    WT_pos_20 = data.WT_pos_20
    WT_neg_20 = data.WT_neg_20
    load_pos_100 = data.load_pos_100
    load_neg_100 = data.load_neg_100
    load_pos_50 = data.load_pos_50
    load_neg_50 = data.load_neg_50
    load_pos_20 = data.load_pos_20
    load_neg_20 = data.load_neg_20

    Pi_PV_t = 96
    Pi_WT_t = 96
    Pi_load = 96
    M = 1e6
    epsilon = 1e-6
    SP_dual = CCG_SP(PV_forecast=PV_forecast, WT_forecast=WT_forecast, load_forecast=load_forecast, PV_pos_100=PV_pos_100, PV_neg_100=PV_neg_100, PV_pos_50=PV_pos_50, PV_neg_50=PV_neg_50, PV_pos_20=PV_pos_20, PV_neg_20=PV_neg_20,
                     WT_pos_100=WT_pos_100, WT_neg_100=WT_neg_100, WT_pos_50=WT_pos_50, WT_neg_50=WT_neg_50, WT_pos_20=WT_pos_20, WT_neg_20=WT_neg_20,
                     load_pos_100=load_pos_100, load_neg_100=load_neg_100, load_pos_50=load_pos_50, load_neg_50=load_neg_50, load_pos_20=load_pos_20, load_neg_20=load_neg_20,
                     DE1_p=DE1_p, DE1_rp=DE1_rp, DE1_rn=DE1_rn, x_curt_PV=x_curt_PV, x_curt_WT=x_curt_WT, Pi_PV_t=Pi_PV_t, Pi_WT_t=Pi_WT_t, Pi_load=Pi_load, M=M)

    SP_dual.export_model(dirname + 'SP_dual_MILP')
    MIPFocus = 0
    TimeLimit = 15
    logname = 'SP_dual_MILP_start_' + 'MIPFocus_' + str(MIPFocus) + '.log'
    SP_dual.solve(LogToConsole=True, logfile=dirname + logname, Threads=1, MIPFocus=MIPFocus, TimeLimit=TimeLimit)
    solution = SP_dual.store_solution()

    print('nominal objective %.2f' % (solution['obj']))
    plt.style.use(['science', 'no-latex'])
    
    plt.figure()
    plt.plot(solution['phi_PV'], label='phi_PV')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(solution['phi_WT'], label='phi_WT')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(solution['phi_load'], label='phi_load')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(solution['epsilon_pos_100'], label='epsilon_pos_100')
    plt.plot(solution['epsilon_neg_100'], label='epsilon_neg_100')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(solution['xi_pos_100'], label='xi_pos_100')
    plt.plot(solution['xi_neg_100'], label='xi_neg_100')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(solution['alpha_pos_100'], label='alpha_pos_100')
    plt.plot(solution['alpha_neg_100'], label='alpha_neg_100')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(solution['beta_pos_100'], label='beta_pos_100')
    plt.plot(solution['beta_neg_100'], label='beta_neg_100')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(solution['gamma_pos_100'], label='gamma_pos_100')
    plt.plot(solution['gamma_neg_100'], label='gamma_neg_100')
    plt.legend()
    plt.show()

    PV_worst_case = [PV_forecast[i] + PV_pos_100[i] * solution['epsilon_pos_100'][i] - PV_neg_100[i] * solution['epsilon_neg_100'][i] + PV_pos_50[i] * solution['epsilon_pos_50'][i] - PV_neg_50[i] * solution['epsilon_neg_50'][i] + PV_pos_20[i] * solution['epsilon_pos_20'][i] - PV_neg_20[i] * solution['epsilon_neg_20'][i] for i in range(96)]
    WT_worst_case = [WT_forecast[i] + WT_pos_100[i] * solution['kapa_pos_100'][i] - WT_neg_100[i] * solution['kapa_neg_100'][i] + WT_pos_50[i] * solution['kapa_pos_50'][i] - WT_neg_50[i] * solution['kapa_neg_50'][i] + WT_pos_20[i] * solution['kapa_pos_20'][i] - WT_neg_20[i] * solution['kapa_neg_20'][i] for i in range(96)]
    load_worst_case = [load_forecast[i] + load_pos_100[i] * solution['xi_pos_100'][i] - load_neg_100[i] * solution['xi_neg_100'][i] + load_pos_50[i] * solution['xi_pos_50'][i] - load_neg_50[i] * solution['xi_neg_50'][i] + load_pos_20[i] * solution['xi_pos_20'][i] - load_neg_20[i] * solution['xi_neg_20'][i] for i in range(96)]

    plt.figure()
    plt.plot(PV_worst_case, marker='.', color='k', label='RG worst case')
    # plt.plot(PV_solution, label = 'Pm')
    plt.plot(PV_forecast, label = 'PV forecast')
    plt.plot(PV_forecast - PV_neg_100, ':', label = 'PV min')
    plt.plot(PV_forecast + PV_pos_100, ':', label = 'PV max')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(WT_worst_case, marker='.', color='k', label='RG worst case')
    # plt.plot(WT_solution, label = 'Pm')
    plt.plot(WT_forecast, label = 'WT forecast')
    plt.plot(WT_forecast - WT_neg_100, ':', label = 'WT min')
    plt.plot(WT_forecast + WT_pos_100, ':', label = 'WT max')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(load_worst_case, marker='.', color='k', label='load_worst case')
    plt.plot(load_forecast, label = 'load forecast')
    plt.plot(load_forecast - load_neg_100, ':', label = 'load min')
    plt.plot(load_forecast + load_pos_100, ':', label = 'load max')
    # plt.ylim(-0.05 * PARAMETERS['RG_capacity'], PARAMETERS['RG_capacity'])
    plt.legend()
    plt.show()

    # Get dispatch variables by solving the primal LP with the worst case PV, load generation trajectory
    SP_primal = SP_primal_LP(PV_forecast=PV_forecast, WT_forecast=WT_forecast, load_forecast=load_forecast, DE1_p=DE1_p, DE1_rp=DE1_rp, DE1_rn=DE1_rn,
                             ES_charge=ES_charge, ES_discharge=ES_discharge, ES_SOC=ES_SOC, x_curt_PV=x_curt_PV, x_curt_WT=x_curt_WT)
    SP_primal.solve()
    SP_primal_sol = SP_primal.store_solution()

    print('RO objective %.2f' % (SP_primal_sol['obj']))

    plt.figure()
    plt.plot(SP_primal_sol['y_S'], label='SOC')
    plt.ylim(0, PARAMETERS['ES']['capacity'])
    plt.legend()
    plt.show()

    PV_worst = np.array(PV_worst_case)
    WT_worst = np.array(WT_worst_case)
    load_worst = np.array(load_worst_case)
    fmt = '%.18e'
    data = np.column_stack((PV_worst, WT_worst, load_worst.flatten()))
    np.savetxt('worst.csv', data, delimiter=',', header='PV_worst, WT_worst, load_worst', comments='', fmt='%.2f')