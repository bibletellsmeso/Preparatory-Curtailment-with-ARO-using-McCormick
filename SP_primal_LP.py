import os
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from root_project import ROOT_DIR
import matplotlib.pyplot as plt

from utils import *
from Params import PARAMETERS
from Data_read import *

class SP_primal_LP():
    """
    SP primal of the benders decomposition using gurobi.
    :ivar nb_periods: number of market periods (-)
    :ivar period_hours: period duration (hours)
    :ivar soc_ini: initial state of charge (kWh)
    :ivar soc_end: final state of charge (kWh)
    :ivar PV_forecast: PV forecast (kW)
    :ivar load_forecast: load forecast (kW)
    :ivar x: diesel on/off variable (on = 1, off = 0)
          shape = (nb_market periods,)

    :ivar model: a Gurobi model (-)
    """

    def __init__(self, PV_forecast:np.array, WT_forecast:np.array, load_forecast:np.array, DE1_p:np.array, DE1_rp:np.array, DE1_rn:np.array,
                 ES_charge:np.array, ES_discharge:np.array, ES_SOC:np.array, x_curt_PV:np.array, x_curt_WT:np.array):
        """
        Init the planner.
        """
        self.parameters = PARAMETERS # simulation parameters
        self.period_hours = PARAMETERS['period_hours']  # (hour)
        self.nb_periods = int(24 / self.period_hours)
        self.t_set = range(self.nb_periods)

        self.PV_forecast = PV_forecast # (kW)
        self.WT_forecast = WT_forecast
        self.load_forecast = load_forecast # (kW)
        # Compute the PV, load min and max 
        self.PV_lb = data.PV_min
        self.PV_ub = data.PV_max
        self.WT_lb = data.WT_min
        self.WT_ub = data.WT_max
        self.load_lb = data.load_min
        self.load_ub = data.load_max
        self.p_1 = DE1_p # (kW) The power of diesel generator
        self.r_pos_1 = DE1_rp # (kw) The reserve rate of diesel generator
        self.r_neg_1 = DE1_rn
        self.x_chg = ES_charge # (kW) The power of ESS charge
        self.x_dis = ES_discharge
        self.x_S = ES_SOC
        self.x_curt_PV = x_curt_PV # (kW) The curtailment of PV
        self.x_curt_WT = x_curt_WT # (kW) The curtailment of WT


        # DE1 parameters
        self.u_DE1 = PARAMETERS['u_1'] # on/off
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

        # Solve model
        self.solver_status = None

    def create_model(self):
        """
        Create the optimization problem.
        """
        t_build = time.time()

        # -------------------------------------------------------------------------------------------------------------
        # 1. create model
        model = gp.Model("SP_primal_LP_gurobi")

        # -------------------------------------------------------------------------------------------------------------
        # 2. create Second-stage variables -> y

        p_pos_1 = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="p_pos_1")
        p_neg_1 = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="p_neg_1")
        # Incremental output of ES (kW)
        y_chg = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_chg")
        y_dis = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_dis")
        y_S = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="SoC")
        # Real-time RG
        y_PV = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_PV")
        y_WT = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_WT")
        y_curt_PV = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_curt_PV")
        y_curt_WT = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_curt_WT")
        y_load = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_load")
        y_pc = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_pc")
        y_wc = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_wc")

        # 2.4 Constant offset using piecewise-linearlization
        x_cost_OM_ES = model.addVars(self.nb_periods, vtype=GRB.CONTINUOUS, obj=0, name='x_cost_OM_ES')
        x_cost_fuel_PWL_1 = model.addVars(self.nb_periods, vtype=GRB.CONTINUOUS, obj=0, name='x_cost_fuel_PWL_1')
        x_cost_fuel_res_1 = model.addVars(self.nb_periods, vtype=GRB.CONTINUOUS, obj=0, name='x_cost_fuel_res_1')
        x_cost_curt_PV_PWL = model.addVars(self.nb_periods, vtype=GRB.CONTINUOUS, obj=0, name='x_cost_curt_PV_PWL')
        x_cost_curt_WT_PWL = model.addVars(self.nb_periods, vtype=GRB.CONTINUOUS, obj=0, name='x_cost_curt_WT_PWL')
        y_cost_fuel_1 = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='y_cost_fuel_1_')
        y_cost_OM_ES = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cost_OM_ES")
        y_cost_curt_PV = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cost_curt_PV")
        y_cost_curt_WT = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cost_curt_WT")
        y_cost_cn_PV = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cost_cn_PV")
        y_cost_cn_WT = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cost_cn_WT")

        # -------------------------------------------------------------------------------------------------------------
        # 3. Create objective
        objective = gp.quicksum(x_cost_fuel_PWL_1[i] + x_cost_fuel_res_1[i] + x_cost_curt_PV_PWL[i] + x_cost_curt_WT_PWL[i] + x_cost_OM_ES[i] + y_cost_fuel_1[i] + y_cost_OM_ES[i]
                                + y_cost_curt_PV[i] + y_cost_curt_WT[i] + y_cost_cn_WT[i] + y_cost_cn_WT[i] for i in self.t_set)
        model.setObjective(objective, GRB.MINIMIZE)

        # -------------------------------------------------------------------------------------------------------------
        # 4. Create Second-stage constraints

        model.addConstrs((x_cost_fuel_PWL_1[i] == PWL_val(self.seg_num, self.DE1_min, self.DE1_max, FC1, self.p_1[i]) for i in self.t_set), name='c_cost_fuel_PWL1')
        model.addConstrs((x_cost_fuel_res_1[i] == self.cost_m_pos_DE1 * self.r_pos_1[i] + self.cost_m_neg_DE1 * self.r_neg_1[i] for i in self.t_set), name='c_cost_fuel_res1')
        model.addConstrs((x_cost_OM_ES[i] == self.cost_OM_ES * (self.x_chg[i] + self.x_dis[i]) for i in self.t_set), name='c_cost_re-OM_ES')
        model.addConstrs((x_cost_curt_PV_PWL[i] == PWL_val(self.seg_num, self.PV_min, data.PV_pred[i], PC_PV, self.x_curt_PV[i]) for i in self.t_set), name='c_cost_curt_PV_PWL')
        model.addConstrs((x_cost_curt_WT_PWL[i] == PWL_val(self.seg_num, self.WT_min, data.WT_pred[i], PC_WT, self.x_curt_WT[i]) for i in self.t_set), name='c_cost_curt_WT_PWL')   
        model.addConstrs((y_cost_fuel_1[i] == self.cost_m_pos_re_DE1 * p_pos_1[i] + self.cost_m_neg_re_DE1 * p_neg_1[i] for i in self.t_set), name='c_cost_re-fuel_1')
        model.addConstrs((y_cost_OM_ES[i] == self.cost_OM_ES_re * (y_dis[i] + y_chg[i]) for i in self.t_set), name='c_cost_pre-OM_ES')
        model.addConstrs((y_cost_curt_PV[i] == self.cost_m_re_PV * y_curt_PV[i] for i in self.t_set), name='c_cost_PV_curt_pos')
        model.addConstrs((y_cost_curt_WT[i] == self.cost_m_re_WT * y_curt_WT[i] for i in self.t_set), name='c_cost_WT_curt_pos')
        model.addConstrs((y_cost_cn_PV[i] == self.cost_m_re_PV * y_pc[i] for i in self.t_set), name='c_cost_PV_cn_pos')
        model.addConstrs((y_cost_cn_WT[i] == self.cost_m_re_WT * y_wc[i] for i in self.t_set), name='c_cost_WT_cn_pos')

        model.addConstrs((p_pos_1[i] <= self.r_pos_1[i] for i in self.t_set), name='c_reserve_pos_DE1')
        model.addConstrs((p_neg_1[i] <= self.r_neg_1[i] for i in self.t_set), name='c_reserve_neg_DE1')
        model.addConstrs((y_dis[i] <= self.ES_max for i in self.t_set), name='c_discharge_re')
        model.addConstrs((y_chg[i] <= self.ES_max for i in self.t_set), name='c_charge_re')
        model.addConstrs((- y_S[i] <= - self.soc_min for i in self.t_set), name='c_min_S')
        model.addConstrs((y_S[i] <= self.soc_max for i in self.t_set), name='c_max_S')
        model.addConstr((y_S[0] == self.soc_ini), name='c_ESS_first_period')
        model.addConstrs((y_S[i] - y_S[i - 1] - self.period_hours * (self.charge_eff * y_chg[i] -  y_dis[i] / self.discharge_eff) == 0 for i in range(1, self.nb_periods)), name='c_ESS_re-dispatch')
        model.addConstr((y_S[self.nb_periods - 1] == self.soc_end), name='c_ESS_last_period')
        model.addConstrs((y_PV[i] == self.PV_forecast[i] for i in self.t_set), name='c_y_PV')
        model.addConstrs((y_WT[i] == self.WT_forecast[i] for i in self.t_set), name='c_y_WT')
        model.addConstrs((y_load[i] == self.load_forecast[i] for i in self.t_set), name='c_y_load')
        model.addConstrs((y_curt_PV[i] <= self.PV_forecast[i] - self.x_curt_PV[i] for i in self.t_set), name='c_y_curt_PV')
        model.addConstrs((y_curt_WT[i] <= self.WT_forecast[i] - self.x_curt_WT[i] for i in self.t_set), name='c_y_curt_WT')
        model.addConstrs((y_pc[i] <= self.x_curt_PV[i] for i in self.t_set), name='c_pc')
        model.addConstrs((y_wc[i] <= self.x_curt_WT[i] for i in self.t_set), name='c_wc')
        # 4.2.2 power balance equation
        model.addConstrs((self.p_1[i] + p_pos_1[i] - p_neg_1[i] - y_chg[i] + y_dis[i] + y_PV[i] - self.x_curt_PV[i] - y_curt_PV[i] + y_pc[i] + y_WT[i] - self.x_curt_WT[i] + y_wc[i] - y_curt_WT[i] - y_load[i] == 0 for i in self.t_set), name='c_power_balance_eq')

        self.time_building_model = time.time() - t_build
        # print("Time spent building the mathematical program: %gs" % self.time_building_model)

        return model
    
    def solve(self, outputflag:bool=False):

        t_solve = time.time()
        self.model.setParam('OutputFlag', outputflag)
        self.model.optimize()
        self.time_solving_model = time.time() - t_solve

    def store_solution(self):

        m = self.model

        solution = dict()
        solution['status'] = m.status

        if solution['status'] == 2 or  solution['status'] == 9:
            # solutionStatus = 2: Model was solved to optimality (subject to tolerances), and an optimal solution is available.
            # solutionStatus = 9: Optimization terminated because the time expended exceeded the value specified in the TimeLimit  parameter.

            solution['obj'] = m.objVal

            varname = ['p_pos_1', 'p_neg_1', 'y_chg', 'y_dis', 'y_S', 'y_PV', 'y_WT', 'y_curt_PV', 'y_curt_WT', 'y_load', 'y_wc', 'y_pc',
                       'x_cost_fuel_PWL_1', 'x_cost_fuel_res_1', 'x_cost_OM_ES', 'x_cost_curt_PV_PWL', 'x_cost_curt_WT_PWL',
                       'y_cost_fuel_1', 'y_cost_OM_ES', 'y_cost_curt_PV', 'y_cost_curt_WT', 'y_cost_cn_PV', 'y_cost_cn_WT']
            for key in varname:
                solution[key] = []

            sol = m.getVars()
            solution['all_var'] = sol
            for v in sol:
                for key in varname:
                    if v.VarName.split('[')[0] == key:
                        solution[key].append(v.x)
        else:
            print('WARNING planner SP primal status %s -> problem not solved, objective is set to nan' %(solution['status']))
            self.model.computeIIS()
            self.model.write("infeasible_model.ilp")
            print('WARNING planner MP status %s -> problem not solved, cannot retrieve solution')
            # solutionStatus = 3: Model was proven to be infeasible.
            # solutionStatus = 4: Model was proven to be either infeasible or unbounded.
            solution['obj'] = float('nan')

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

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())
    
    dirname = '/Users/Andrew/OneDrive - GIST/Code/Graduation/PC_RGD_CCG_Mc_MIU/export_MILP/'

    PV_forecast = np.array(pd.read_csv('worst.csv'), dtype=np.float32)[:,0]
    WT_forecast = np.array(pd.read_csv('worst.csv'), dtype=np.float32)[:,1]
    load_forecast = np.array(pd.read_csv('worst.csv'), dtype=np.float32)[:,2]
    # PV_forecast = data.PV_pred
    # WT_forecast = data.WT_pred
    # load_forecast = data.load_egg

    DE1_p = read_file(dir=dirname, name='sol_MILP_DE1_p')
    DE1_rp = read_file(dir=dirname, name='sol_MILP_DE1_rp')
    DE1_rn = read_file(dir=dirname, name='sol_MILP_DE1_rn')
    ES_charge = read_file(dir=dirname, name='sol_MILP_ES_charge')
    ES_discharge = read_file(dir=dirname, name='sol_MILP_ES_discharge')
    ES_SOC = read_file(dir=dirname, name='sol_MILP_ES_SOC')
    x_curt_PV = read_file(dir=dirname, name='sol_MILP_x_curt_PV')
    x_curt_WT = read_file(dir=dirname, name='sol_MILP_x_curt_WT')

    SP_primal = SP_primal_LP(PV_forecast=PV_forecast, WT_forecast=WT_forecast, load_forecast=load_forecast, DE1_p=DE1_p, DE1_rp=DE1_rp, DE1_rn=DE1_rn, 
                             ES_charge=ES_charge, ES_discharge=ES_discharge, ES_SOC=ES_SOC, x_curt_PV=x_curt_PV, x_curt_WT=x_curt_WT)
    SP_primal.export_model(dirname + 'SP_primal_LP')
    SP_primal.solve()
    solution = SP_primal.store_solution()

    print('objective SP primal %.2f' %(solution['obj']))
    
    plt.style.use(['science', 'no-latex'])
    plt.figure()
    plt.plot(solution['y_chg'], label='y chg')
    plt.plot(solution['y_dis'], label='y dis')
    plt.plot(solution['y_S'], label='y S')
    plt.legend()
    plt.show()


    print(solution['y_S'])


    # print(solution['all_var'])


    # Get dual values
    # for c in SP_primal.model.getConstrs():
    #     print('The dual value of %s : %g' % (c.constrName, c.pi))