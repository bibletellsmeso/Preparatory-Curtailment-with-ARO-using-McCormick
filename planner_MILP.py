import math
import os
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from utils import *
from root_project import ROOT_DIR

import matplotlib.pyplot as plt
from Data_read import *
from Params import PARAMETERS

class Planner_MILP():
    """
    MILP capacity firming formulation: binary variables to avoid simultaneous charge and discharge.
    :ivar nb_periods: number of market periods (-)
    :ivar period_hours: period duration (hours)
    :ivar soc_ini: initial state of charge (kWh)
    :ivar soc_end: final state of charge (kWh)
    :ivar RG_forecast: RG point forecasts (kW)
    :ivar load_forecast: load forecast (kW)
    :ivar x: diesel on/off variable (on = 1, off = 0)
          shape = (nb_market periods,)

    :ivar model: a Gurobi model (-)
    """

    def __init__(self, PV_forecast:np.array, WT_forecast:np.array, load_forecast:np.array):
        """
        Init the planner.
        """
        self.parameters = PARAMETERS # simulation parameters
        self.period_hours = PARAMETERS['period_hours'] # 1/4 hour
        self.nb_periods = int(24 / self.period_hours) # 96
        self.t_set = range(self.nb_periods)
        self.PV_forecast = PV_forecast # (kW)
        self.WT_forecast = WT_forecast # (kW)
        self.load_forecast = load_forecast # (kW)
        # Compute the PV, load min and max 
        self.PV_lb = data.PV_min
        self.PV_ub = data.PV_max
        self.WT_lb = data.WT_min
        self.WT_ub = data.WT_max
        self.load_lb = data.load_max
        self.load_ub = data.load_min

        # Parameters required for the MP in the CCG algorithm
        self.u_1 = PARAMETERS['u_1'] # on/off
        self.DE1_min = PARAMETERS['DE']['DE1_min'] # (kW)
        self.DE1_max = PARAMETERS['DE']['DE1_max'] # (kW)
        self.DE1_ramp_up = PARAMETERS['DE']['DE1_ramp_up'] # (kW)
        self.DE1_ramp_down = PARAMETERS['DE']['DE1_ramp_down'] # (kW)
        self.DE1_reserve_up = PARAMETERS['DE']['DE1_reserve_up']
        self.DE1_reserve_down = PARAMETERS['DE']['DE1_reserve_down']
        self.DE1_p_rate = PARAMETERS['DE']['DE1_p_rate']

        # ES parameters
        self.EScapacity = PARAMETERS['ES']['capacity']  # (kWh)
        self.soc_ini = PARAMETERS['ES']['soc_ini']  # (kWh)
        self.soc_end = PARAMETERS['ES']['soc_end']  # (kWh)
        self.soc_min = PARAMETERS['ES']['soc_min']  # (kWh)
        self.soc_max = PARAMETERS['ES']['soc_max']  # (kWh)
        self.charge_eff = PARAMETERS['ES']['charge_eff']  # (/)
        self.discharge_eff = PARAMETERS['ES']['discharge_eff']  # (/)
        self.ES_min = PARAMETERS['ES']['power_min']  # (kW)
        self.ES_max = PARAMETERS['ES']['power_max']  # (kW)

        # RG parameters
        self.PV_min = PARAMETERS['RG']['PV_min']
        self.PV_max = PARAMETERS['RG']['PV_max']
        self.PV_ramp_up = PARAMETERS['RG']['PV_ramp_up']
        self.PV_ramp_down = PARAMETERS['RG']['PV_ramp_down']
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
        self.cost_m_pre_WT = PARAMETERS['cost']['WT_m_cut_pre']
        self.cost_m_re_WT = PARAMETERS['cost']['WT_m_cut_re']

        # Piecewise linearlization parameters
        self.seg_num = PARAMETERS['PWL']['num']

        self.time_building_model = None
        self.time_solving_model = None

        # Create model
        self.model = self.create_model()

        # Solve model
        self.solver_status = None

    def create_model(self):
        """
        Create the optimization problem.
        """
        t_build = time.time()

        # -------------------------------------------------------------------------------------------------------------
        # 1. Create model
        model = gp.Model("planner_MILP_gurobi")

        # -------------------------------------------------------------------------------------------------------------
        # 2. create vairables
        # 2.1 Create First-stage variables -> x
        p_1 = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='p_1') # comtemporary output of DE
        x_chg = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_chg') # charge power
        x_dis = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_dis') # discharge power
        x_curt_PV = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_curt_PV') # pre-dispatch curtailment of RG
        x_curt_WT = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_curt_WT')
        r_pos_1 = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='r_pos_1') # reserve capacity of DE
        r_neg_1 = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='r_neg_1')
        u = model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name='u') # on/off of ES charge
        x_PV = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_PV') # RG output
        x_WT = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_WT') # RG output
        x_load = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_load') # load demand
        x_S = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_SOC') # ES SOC

        u_re = model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name="u_re")
        p_pos_1 = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="p_pos_1")
        p_neg_1 = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="p_neg_1")
        y_chg = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_charge")
        y_dis = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_discharge")
        y_S = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_SOC")
        y_PV = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_PV")
        y_WT = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_WT")
        y_curt_PV = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_curt_PV")
        y_curt_WT = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_curt_WT")
        y_load = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_load")
        y_pc = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_pc")
        y_wc = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_wc")
        # # -------------------------------------------------------------------------------------------------------------
        x_cost_fuel_1 = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x_cost_fuel_1")
        x_cost_fuel_res_1 = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x_cost_fuel_res_1")
        x_cost_OM_ES = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="pre_cost_ES")
        x_cost_curt_PV = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x_cost_curt_PV")
        x_cost_curt_WT = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x_cost_curt_WT")
        y_cost_fuel_1 = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="regulation_cost_DE1")      
        y_cost_OM_ES = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="regulation_cost_ES")
        y_cost_curt_PV = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="PV_curtail_cost")
        y_cost_curt_WT = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="WT_curtail_cost")

        # -------------------------------------------------------------------------------------------------------------
        # 3. Create objective
        obj = gp.quicksum(x_cost_fuel_1[i] + x_cost_fuel_res_1[i] + x_cost_curt_PV[i] + x_cost_curt_WT[i] + x_cost_OM_ES[i] + y_cost_fuel_1[i] + y_cost_OM_ES[i] + y_cost_curt_PV[i] + y_cost_curt_WT[i] for i in self.t_set)
        model.setObjective(obj, GRB.MINIMIZE)


        # -------------------------------------------------------------------------------------------------------------
        # 4. Create constraints
        # 4.1 Fisrt stage constraints

        # 4.2 Second stage constraints
        model.addConstrs((- p_1[i] + r_neg_1[i] <= - self.u_1[i] * self.DE1_min for i in self.t_set), name='c_DE1_generation_min')
        model.addConstrs((p_1[i] + r_pos_1[i] <= self.u_1[i] * self.DE1_max for i in self.t_set), name='c_DE1_generation_max')
        model.addConstrs((p_1[i] - p_1[i-1] + r_pos_1[i] <= self.DE1_ramp_up for i in range(1, self.nb_periods)), name='c_DE1_reserve_min')
        model.addConstrs((p_1[i-1] - p_1[i] + r_neg_1[i] <= self.DE1_ramp_down for i in range(1, self.nb_periods)), name='c_DE1_reserve_max')
        model.addConstrs((- r_pos_1[i] <= - self.DE1_reserve_up for i in self.t_set), name='c_reserve_min_DE1')
        model.addConstrs((- r_neg_1[i] <= - self.DE1_reserve_down for i in self.t_set), name='c_reserve_max_DE1')

        model.addConstrs((x_chg[i] <= u[i] * self.ES_max for i in self.t_set), name='c_chgarge_max') # LP
        model.addConstrs((x_dis[i] <= (1 - u[i]) * self.ES_max for i in self.t_set), name='c_discharge_max') # LP
        model.addConstrs((- x_S[i] <= - self.soc_min for i in self.t_set), name='c_SOC_min')
        model.addConstrs((x_S[i] <= self.soc_max for i in self.t_set), name='c_SOC_max')
        model.addConstr((x_S[0] == self.soc_ini), name='c_SOC_first')
        model.addConstrs((x_S[i] - x_S[i - 1] - ((self.charge_eff * x_chg[i]) - (x_dis[i] / self.discharge_eff)) * self.period_hours == 0 for i in range(1, self.nb_periods)), name='c_SOC_dynamic')
        model.addConstr((x_S[self.nb_periods - 1] == self.soc_ini), name='c_SOC_last')

        model.addConstrs((p_1[i] -x_chg[i] + x_dis[i] + x_PV[i] + x_WT[i] - x_curt_PV[i] - x_curt_WT[i] - x_load[i] == 0 for i in self.t_set), name='c_power_balance')

        model.addConstrs((x_PV[i] == self.PV_forecast[i] for i in self.t_set), name='c_PV_output')
        model.addConstrs((x_WT[i] == self.WT_forecast[i] for i in self.t_set), name='c_WT_output')
        model.addConstrs((x_load[i] == self.load_forecast[i] for i in self.t_set), name='c_x_load_demand')
        model.addConstrs((x_curt_PV[i] <= self.PV_lb[i] for i in self.t_set), name='c_x_curt_PV')
        model.addConstrs((x_curt_WT[i] <= self.WT_lb[i] for i in self.t_set), name='c_x_curt_WT')

        model.addConstrs((x_cost_fuel_res_1[i] == self.cost_m_pos_DE1 * r_pos_1[i] + self.cost_m_neg_DE1 * r_neg_1[i] for i in self.t_set), name='c_cost_fuel_res_1')
        model.addConstrs((x_cost_OM_ES[i] == self.cost_OM_ES * (x_chg[i] + x_dis[i]) for i in self.t_set), name='c_cost_OM_ES')

        for i in self.t_set:
            model.addGenConstrPWL(p_1[i], x_cost_fuel_1[i], PWL(self.seg_num, self.DE1_min, self.DE1_max, FC1)[0],
                                  PWL(self.seg_num, self.DE1_min, self.DE1_max, FC1)[1])
            model.addGenConstrPWL(x_curt_PV[i], x_cost_curt_PV[i], PWL(self.seg_num, self.PV_min, self.PV_lb[i], PC_PV)[0],
                                  PWL(self.seg_num, self.PV_min, self.PV_lb[i], PC_PV)[1])
            model.addGenConstrPWL(x_curt_WT[i], x_cost_curt_WT[i], PWL(self.seg_num, self.WT_min, self.WT_lb[i], PC_WT)[0],
                                  PWL(self.seg_num, self.WT_min, self.WT_lb[i], PC_WT)[1])

        model.addConstrs((p_pos_1[i] <= r_pos_1[i] for i in self.t_set), name='c_reserve_pos_DE1')
        model.addConstrs((p_neg_1[i] <= r_neg_1[i] for i in self.t_set), name='c_reserve_neg_DE1')
        model.addConstrs((y_dis[i] <= u_re[i] * self.ES_max for i in self.t_set), name='c_discharge_re')
        model.addConstrs((y_chg[i] <= (1 - u_re[i]) * self.ES_max for i in self.t_set), name='c_charge_re')
        model.addConstrs((- y_S[i] <= - self.soc_min for i in self.t_set), name='c_min_S')
        model.addConstrs((y_S[i] <= self.soc_max for i in self.t_set), name='c_max_S')
        model.addConstr((y_S[0] == self.soc_ini), name='c_ESS_first_period')
        model.addConstrs((y_S[i] - y_S[i - 1] - self.period_hours * ((self.charge_eff * y_chg[i] - y_dis[i] / self.discharge_eff)) == 0 for i in range(1, self.nb_periods)), name='c_ESS_re-dispatch')
        model.addConstr((y_S[self.nb_periods - 1] == self.soc_end), name='c_ESS_last_period')
        model.addConstrs((y_PV[i] == self.PV_forecast[i] for i in self.t_set), name='c_PV_re-dispatch')
        model.addConstrs((y_WT[i] == self.WT_forecast[i] for i in self.t_set), name='c_WT_re-dispatch')
        model.addConstrs((y_load[i] == self.load_forecast[i] for i in self.t_set), name='c_load_re-dispatch')
        model.addConstrs((y_curt_PV[i] <= self.PV_forecast[i] - x_curt_PV[i] for i in self.t_set), name='c_y_curt_PV')
        model.addConstrs((y_curt_WT[i] <= self.WT_forecast[i] - x_curt_WT[i] for i in self.t_set), name='c_y_curt_WT')
        model.addConstrs((y_pc[i] <= x_curt_PV[i] for i in self.t_set), name='c_y_pc')
        model.addConstrs((y_wc[i] <= x_curt_WT[i] for i in self.t_set), name='c_y_wc')

        model.addConstrs((y_cost_fuel_1[i] == self.cost_m_pos_re_DE1 * p_pos_1[i] + self.cost_m_neg_re_DE1 * p_neg_1[i] for i in self.t_set), name='c_cost_reg_DE1')
        model.addConstrs((y_cost_curt_PV[i] == self.cost_m_re_PV * y_curt_PV[i] for i in self.t_set), name='c_cost_curt_PV')
        model.addConstrs((y_cost_curt_WT[i] == self.cost_m_re_WT * y_curt_WT[i] for i in self.t_set), name='c_cost_curt_WT')
        model.addConstrs((y_cost_OM_ES[i] == self.cost_OM_ES_re * (y_chg[i] + y_dis[i]) for i in self.t_set), name='c_cost_reg_ES')

        # 4.2.2 power balance equation
        model.addConstrs((p_1[i] + p_pos_1[i] - p_neg_1[i] - y_chg[i] + y_dis[i] + y_PV[i] + y_WT[i] - x_curt_PV[i] + y_pc[i] - x_curt_WT[i] - y_curt_PV[i] + y_wc[i] - y_curt_WT[i] - y_load[i] == 0 for i in self.t_set))

        # -------------------------------------------------------------------------------------------------------------
        # 5. Store variables
        self.allvar = dict()
        self.allvar['p_1'] = p_1
        self.allvar['r_pos_1'] = r_pos_1
        self.allvar['r_neg_1'] = r_neg_1
        self.allvar['x_chg'] = x_chg
        self.allvar['x_dis'] = x_dis
        self.allvar['u'] = u
        self.allvar['x_S'] = x_S
        self.allvar['y_chg'] = y_chg
        self.allvar['y_dis'] = y_dis
        self.allvar['y_pc'] = y_pc
        self.allvar['y_wc'] = y_wc
        self.allvar['x_PV'] = x_PV
        self.allvar['x_WT'] = x_WT
        self.allvar['x_load'] = x_load
        self.allvar['x_curt_PV'] = x_curt_PV
        self.allvar['x_curt_WT'] = x_curt_WT
        self.allvar['x_cost_fuel_1'] = x_cost_fuel_1
        self.allvar['x_cost_fule_res_1'] = x_cost_fuel_res_1
        self.allvar['x_cost_OM_ES'] = x_cost_OM_ES
        self.allvar['x_cost_curt_PV'] = x_cost_curt_PV
        self.allvar['x_cost_curt_WT'] = x_cost_curt_WT

        self.time_building_model = time.time() - t_build
        # print("Time spent building the mathematical program: %gs" % self.time_building_model)

        return model

    def solve(self, LogToConsole:bool=False, logfile:str="", Threads:int=0, MIPFocus:int=0, TimeLimit:float=GRB.INFINITY):

        t_solve = time.time()

        self.model.setParam('LogToConsole', LogToConsole) # no log in the console if set to False
        # self.model.setParam('OutputFlag', outputflag) # no log into console and log file if set to True
        # self.model.setParam('MIPGap', 0.01)
        self.model.setParam('TimeLimit', TimeLimit)
        self.model.setParam('MIPFocus', MIPFocus)
        # self.model.setParam('DualReductions', 0) # Model was proven to be either infeasible or unbounded. To obtain a more definitive conclusion, set the DualReductions parameter to 0 and reoptimize.

        # If you are more interested in good quality feasible solutions, you can select MIPFocus=1.
        # If you believe the solver is having no trouble finding the optimal solution, and wish to focus more attention on proving optimality, select MIPFocus=2.
        # If the best objective bound is moving very slowly (or not at all), you may want to try MIPFocus=3 to focus on the bound.

        self.model.setParam('LogFile', logfile) # no log in file if set to ""
        self.model.setParam('Threads', Threads) # Default value = 0 -> use all threads

        self.model.optimize()

        if self.model.status == 2 or self.model.status == 9:
            pass
        else:
            self.model.computeIIS()
            self.model.write("infeasible_model.ilp")

        if self.model.status == gp.GRB.Status.UNBOUNDED:
            self.model.computeIIS()
            self.model.write("unbounded_model_ilp")

        self.solver_status = self.model.status
        self.time_solving_model = time.time() - t_solve

        # self.model.computeIIS()
        # self.model.write("infeasible_model.ilp")

    def store_solution(self):

        m = self.model

        solution = dict()
        solution['status'] = m.status
        if solution['status'] == 2 or solution['status'] == 9:
            solution['obj'] = m.objVal

            # 1 dimensional variables
            for var in ['p_1', 'r_pos_1', 'r_neg_1', 'x_chg', 'x_dis', 'y_chg', 'y_dis', 'x_PV', 'x_WT', 'u', 'x_load',
                        'x_S', 'x_curt_PV', 'x_curt_WT', 'x_cost_fuel_1', 'x_cost_OM_ES', 'x_cost_curt_PV', 'x_cost_curt_WT']:
                solution[var] = [self.allvar[var][t].X for t in self.t_set]
        else:
            print('WARNING planner MILP status %s -> problem not solved, objective is set to nan' %(solution['status']))
            solution['obj'] = math.nan

        # 3. Timing indicators
        solution["time_building"] = self.time_building_model
        solution["time_solving"] = self.time_solving_model
        solution["time_total"] = self.time_building_model + self.time_solving_model

        return solution

    def export_model(self, filename):
        """
        Export the pyomo model into a cpxlp format.
        :param filename: directory and filename of the exported model.
        """

        self.model.write("%s.lp" % filename)
        # self.model.write("%s.mps" % filename)


# Validation set
VS = 'VS1' # 'VS1', 'VS2

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    dirname = '/Users/Andrew/OneDrive - GIST/Code/Graduation/PC_RGD_CCG_Mc_MIU/export_MILP/'

    # load data
    PV_forecast = data.PV_pred
    WT_forecast = data.WT_pred
    load_forecast = data.load_egg

    PV_lb = data.PV_min
    PV_ub = data.PV_max
    WT_lb = data.WT_min
    WT_ub = data.WT_max
    load_lb = data.load_min
    load_ub = data.load_max

    day = '2018-07-04'

    # Plot point forecasts vs observations
    FONTSIZE = 20
    plt.style.use(['science', 'no-latex'])

    plt.figure(figsize=(16,9))
    plt.plot(PV_forecast, label='forecast')
    plt.plot(PV_lb, linestyle='--', color='darkgrey')
    plt.plot(PV_ub, linestyle='--', color='darkgrey')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname + day + '_PV_comparison' + '.pdf')
    plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot(WT_forecast, label='forecast')
    plt.plot(WT_lb, linestyle='--', color='darkgrey')
    plt.plot(WT_ub, linestyle='--', color='darkgrey')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname + day + '_WT_comparison' + '.pdf')
    plt.close('all')

    FONTSIZE = 20
    plt.figure(figsize=(16,9))
    plt.plot(load_forecast, label='forecast')
    plt.plot(load_lb, linestyle='--', color='darkgrey')
    plt.plot(load_ub, linestyle='--', color='darkgrey')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname + day + '_load_comparison' + '.pdf')
    plt.close('all')

    # MILP planner with forecasts
    planner = Planner_MILP(PV_forecast=PV_forecast, WT_forecast=WT_forecast, load_forecast=load_forecast)
    planner.export_model(dirname + 'MILP')
    planner.solve()
    solution = planner.store_solution()

    dump_file(dir=dirname, name='sol_MILP_DE1_p', file=solution['p_1'])
    dump_file(dir=dirname, name='sol_MILP_DE1_rp', file=solution['r_pos_1'])
    dump_file(dir=dirname, name='sol_MILP_DE1_rn', file=solution['r_neg_1'])
    dump_file(dir=dirname, name='sol_MILP_ES_charge', file=solution['x_chg'])
    dump_file(dir=dirname, name='sol_MILP_ES_discharge', file=solution['x_dis'])
    dump_file(dir=dirname, name='sol_MILP_ES_SOC', file=solution['x_S'])
    dump_file(dir=dirname, name='sol_MILP_x_curt_PV', file=solution['x_curt_PV'])
    dump_file(dir=dirname, name='sol_MILP_x_curt_WT', file=solution['x_curt_WT'])

    print('objective point forecasts %.2f' % (solution['obj']))

    plt.figure(figsize=(16,9))
    plt.plot(solution['p_1'], label='DE1 output')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.title('MILP formulation')
    plt.tight_layout()
    plt.savefig(dirname + day + '_DE_units_output_' + '.pdf')
    plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot(solution['x_chg'], label='x_chg')
    plt.plot(solution['x_dis'], label='x_dis')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.title('pre-ESS simultaneous')
    plt.tight_layout()
    plt.savefig(dirname + day + '_pre-ESS_simultaneous_' + '.pdf')
    plt.close('all')    

    plt.figure(figsize=(16,9))
    plt.plot(solution['y_chg'], label='y_chg')
    plt.plot(solution['y_dis'], label='y_dis')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.title('re-ESS simultaneous')
    plt.tight_layout()
    plt.savefig(dirname + day + '_re-ESS_simultaneous_' + '.pdf')
    plt.close('all')    

    plt.figure(figsize=(16,9))
    plt.plot(solution['x_curt_PV'], label='PV curt predict')
    plt.plot(solution['x_curt_WT'], label='WT curt predict')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname + day + '_curt_' + '.pdf')
    plt.close('all')
    
    plt.figure(figsize=(16,9))
    plt.plot(solution['p_1'], linewidth=2, label='DE1 units')
    plt.plot(solution['x_load'], color='darkorange', linewidth=2, label='load')
    plt.plot(solution['x_PV'], color='royalblue', linewidth=2, label='PV generation')
    plt.plot(solution['x_WT'], color='royalblue', linewidth=2, label='WT generation')
    plt.plot(solution['x_curt_PV'], color='dimgrey', linestyle='-.', linewidth=2, label='PV curtailment')
    plt.plot(solution['x_curt_WT'], color='dimgrey', linestyle='-.', linewidth=2, label='WT curtailment')
    plt.plot(([hs - eg for hs, eg in zip(solution['x_PV'], solution['x_curt_PV'])]), linestyle='--', color='darkorchid', linewidth=2, label='PV output')
    plt.plot(([hs - eg for hs, eg in zip(solution['x_WT'], solution['x_curt_WT'])]), linestyle='--', color='darkorchid', linewidth=2, label='WT output')
    plt.legend()
    plt.savefig(dirname + day + 'MILP_forecast' + '.pdf')
    plt.close('all')
