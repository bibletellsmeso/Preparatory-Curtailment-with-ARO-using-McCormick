import os
import time
import numpy as np

import gurobipy as gp
from gurobipy import GRB

from root_project import ROOT_DIR
from Params import PARAMETERS
from Data_read import *
from utils import *

class CCG_MP():
    """
    CCG = Column-and-Constraint Generation.
    MP = Master Problem of the CCG algorithm.
    The MP is a Linear Programming.

    :ivar nb_periods: number of market periods (-)
    
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

        # Parameters required for the MP in the CCG algorithm
        self.PV_forecast = PV_forecast # (kW)
        self.WT_forecast = WT_forecast # (kW)
        self.load_forecast = load_forecast # (kW)
        # Compute the PV, load min and max 
        self.PV_lb = data.PV_min
        self.PV_ub = data.PV_max
        self.WT_lb = data.WT_min
        self.WT_ub = data.WT_max
        self.load_lb = data.load_min
        self.load_ub = data.load_max

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

        # Solve model
        self.solver_status = None

    def create_model(self):
        """
        Create the optimization problem.
        """
        t_build = time.time()

        # -------------------------------------------------------------------------------------------------------------
        # 1. Create model
        model = gp.Model("MP")

        # -------------------------------------------------------------------------------------------------------------
        # 2.1 Create First-stage variables
        p_1 = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='p_1') # comtemporary output of DE1
        x_chg = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_chg') # charge power
        x_dis = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_dis') # discharge power
        x_curt_PV = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_curt_PV') # pre-dispatch curtailment of RG
        x_curt_WT = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_curt_WT') # pre-dispatch curtailment of RG
        r_pos_1 = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='r_pos_1') # reserve capacity of DE1
        r_neg_1 = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='r_neg_1') # reserve capacity of DE1
        u = model.addVars(self.nb_periods, obj=0, vtype=GRB.BINARY, name='u') # on/off of ES charge
        x_PV = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_PV') # PV output
        x_WT = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_WT') # PV output
        x_load = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_load') # load demand
        x_S = model.addVars(self.nb_periods, obj=0, vtype=GRB.CONTINUOUS, name='x_S') # ES SOC
        x_cost_fuel_1 = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x_cost_fuel_1")
        x_cost_OM_ES = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x_cost_OM_ES")
        x_cost_fuel_res_1 = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x_cost_fuel_res_1")
        x_cost_curt_PV = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x_cost_curt_PV") # PWL  
        x_cost_curt_WT = model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="x_cost_curt_WT") # PWL  

        theta = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=0, name="theta") # objective

        obj = gp.quicksum(x_cost_fuel_1[i] + x_cost_fuel_res_1[i] + x_cost_OM_ES[i] + x_cost_curt_PV[i] + x_cost_curt_WT[i] for i in self.t_set)
        # -------------------------------------------------------------------------------------------------------------
        # 2.2 Create objective
        model.setObjective(obj + theta, GRB.MINIMIZE)

        # -------------------------------------------------------------------------------------------------------------
        # 2.3 Create constraints

        # Limits of generation capacity cst

        model.addConstrs((- p_1[i] + r_neg_1[i] <= - self.u_DE1[i] * self.DE1_min for i in self.t_set), name='c_DE1_generation_min')
        model.addConstrs((p_1[i] + r_pos_1[i] <= self.u_DE1[i] * self.DE1_max for i in self.t_set), name='c_DE1_generation_max')
        model.addConstrs((p_1[i] - p_1[i-1] + r_pos_1[i] <= self.DE1_ramp_up * self.nb_periods for i in range(1, self.nb_periods)), name='c_DE1_reserve_min')
        model.addConstrs((p_1[i-1] - p_1[i] + r_neg_1[i] <= self.DE1_ramp_down * self.nb_periods for i in range(1, self.nb_periods)), name='c_DE1_reserve_max')

        model.addConstrs((- r_pos_1[i] <= - self.DE1_reserve_up for i in self.t_set), name='c_reserve_min_DE1')
        model.addConstrs((- r_neg_1[i] <= - self.DE1_reserve_down for i in self.t_set), name='c_reserve_max_DE1')

        # # Ramping of DE1 generator cst
        # model.addConstrs(((p_1[i] + r_pos_1[i]) - (p_1[i-1] - r_neg_1[i-1]) <= self.u_DE1[i-1] * self.DE1_ramp_up * self.period_hours + (1 - self.u_DE1[i-1]) * self.DE1_max for i in range(1, self.nb_periods)), name='c_DE1_ramping_1')
        # model.addConstrs((- (p_1[i] + r_pos_1[i]) + (p_1[i-1] - r_neg_1[i-1]) <= self.u_DE1[i] * self.DE1_ramp_down * self.period_hours + (1 - self.u_DE1[i]) * self.DE1_max for i in range(1, self.nb_periods)), name='c_DE1_ramping_2')
        # model.addConstrs(((p_1[i] - r_neg_1[i]) - (p_1[i-1] + r_pos_1[i-1]) <= self.u_DE1[i-1] * self.DE1_ramp_up * self.period_hours + (1 - self.u_DE1[i-1]) * self.DE1_max for i in range(1, self.nb_periods)), name='c_DE1_ramping_3')
        # model.addConstrs((- (p_1[i] - r_neg_1[i]) + (p_1[i-1] + r_pos_1[i-1]) <= self.u_DE1[i] * self.DE1_ramp_down * self.period_hours + (1 - self.u_DE1[i]) * self.DE1_max for i in range(1, self.nb_periods)), name='c_DE1_ramping_4')

        # Power balancing condition cst
        model.addConstrs((p_1[i] - x_chg[i] + x_dis[i] + x_PV[i] + x_WT[i] - x_curt_PV[i] -x_curt_WT[i] - x_load[i] == 0 for i in self.t_set), name='c_power_balance')
        # Comtemporary output of ES cst
        model.addConstrs((x_chg[i] <= u[i] * self.ES_max for i in self.t_set), name='c_chgarge_max') # LP
        model.addConstrs((x_dis[i] <= (1 - u[i]) * self.ES_max for i in self.t_set), name='c_discharge_max') # LP
        # SOC of ES cst
        model.addConstrs((- x_S[i] <= - self.soc_min for i in self.t_set), name='c_SOC_min')
        model.addConstrs((x_S[i] <= self.soc_max for i in self.t_set), name='c_SOC_max')
        model.addConstr((x_S[0] == self.soc_ini), name='c_SOC_first')
        model.addConstrs((x_S[i] - x_S[i - 1] - ((self.charge_eff * x_chg[i]) - (x_dis[i] / self.discharge_eff)) * self.period_hours == 0 for i in range(1, self.nb_periods)), name='c_SOC_dynamic')
        model.addConstr((x_S[self.nb_periods - 1] == self.soc_ini), name='c_SOC_last')
        # RG output cst
        model.addConstrs((x_PV[i] == self.PV_forecast[i] for i in self.t_set), name='c_PV_output')
        model.addConstrs((x_WT[i] == self.WT_forecast[i] for i in self.t_set), name='c_WT_output')
        # load demand cst
        model.addConstrs((x_load[i] == self.load_forecast[i] for i in self.t_set), name='c_x_load_demand')
        # Preparatory curtailment cst
        model.addConstrs((x_curt_PV[i] <= self.PV_lb[i] for i in self.t_set), name='c_x_PV_curtailment')
        model.addConstrs((x_curt_WT[i] <= self.WT_lb[i] for i in self.t_set), name='c_x_WT_curtailment')
        
        # pre-dispatch cost cst
        model.addConstrs((x_cost_fuel_res_1[i] == self.cost_m_pos_DE1 * r_pos_1[i] + self.cost_m_neg_DE1 * r_neg_1[i] for i in self.t_set), name='c_cost_fuel_res_DE1')
        model.addConstrs((x_cost_OM_ES[i] == self.cost_OM_ES * (x_chg[i] + x_dis[i]) for i in self.t_set), name='c_cost_OM_ES')
        for i in self.t_set:
            model.addGenConstrPWL(p_1[i], x_cost_fuel_1[i], PWL(self.seg_num, self.DE1_min, self.DE1_max, FC1)[0],
                                  PWL(self.seg_num, self.DE1_min, self.DE1_max, FC1)[1])
            model.addGenConstrPWL(x_curt_PV[i], x_cost_curt_PV[i], PWL(self.seg_num, self.PV_min, self.PV_lb[i], PC_PV)[0],
                                  PWL(self.seg_num, self.PV_min, self.PV_lb[i], PC_PV)[1])
            model.addGenConstrPWL(x_curt_WT[i], x_cost_curt_WT[i], PWL(self.seg_num, self.WT_min, self.WT_lb[i], PC_WT)[0],
                                  PWL(self.seg_num, self.WT_min, self.WT_lb[i], PC_WT)[1])
            
        # -------------------------------------------------------------------------------------------------------------
        # 3. Store variables
        self.allvar = dict()
        self.allvar['p_1'] = p_1
        self.allvar['x_chg'] = x_chg
        self.allvar['x_dis'] = x_dis
        self.allvar['x_curt_PV'] = x_curt_PV
        self.allvar['x_curt_WT'] = x_curt_WT
        self.allvar['r_pos_1'] = r_pos_1
        self.allvar['r_neg_1'] = r_neg_1
        self.allvar['u'] = u
        self.allvar['x_PV'] = x_PV
        self.allvar['x_WT'] = x_WT
        self.allvar['x_load'] = x_load
        self.allvar['x_S'] = x_S
        self.allvar['x_cost_fuel_1'] = x_cost_fuel_1
        self.allvar['x_cost_fuel_res_1'] = x_cost_fuel_res_1
        self.allvar['x_cost_OM_ES'] = x_cost_OM_ES
        self.allvar['x_cost_curt_PV'] = x_cost_curt_PV
        self.allvar['x_cost_curt_WT'] = x_cost_curt_WT
        self.allvar['theta'] = theta

        self.time_building_model = time.time() - t_build
        # print("Time spent building the mathematical program: %gs" % self.time_building_model)

        return model
    
    def update_MP(self, PV_trajectory:np.array, WT_trajectory:np.array, load_trajectory:np.array, iteration:int):
        """
        Add the second-stage variables at CCG iteration i.
        :param MP: MP to update in the CCG algorithm.
        :param PV_trajectory: RG trajectory computed by the SP at iteration i.
        :param iteration: update at iteration i.
        :return: the model is directly updated
        """
        # -------------------------------------------------------------------------------------------------------------
        # 4.1 Second-stage variables
        # Incremental output of DE1 (kW)
        p_pos_1 = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="p_pos_1_" + str(iteration))
        p_neg_1 = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="p_neg_1_" + str(iteration))
        # Incremental output of ES (kW)
        u_re = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.BINARY, name="u_re_" + str(iteration))
        y_chg = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_chg_" + str(iteration))
        y_dis = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_dis_" + str(iteration))
        y_S = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="SoC_" + str(iteration))
        # Real-time RG
        y_PV = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_PV_" + str(iteration))
        y_WT = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_WT_" + str(iteration))
        y_curt_PV = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_curt_PV_" + str(iteration))
        y_curt_WT = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_curt_WT_" + str(iteration))
        y_load = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_load_" + str(iteration))
        y_pc = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cn_PV_" + str(iteration))
        y_wc = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cn_WT_" + str(iteration))

        # -------------------------------------------------------------------------------------------------------------

        # Upward/Downward regulation cost of DE1 generator/ES
        y_cost_fuel_1 = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name='y_cost_fuel_1_' + str(iteration))
        y_cost_OM_ES = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cost_OM_ES_" + str(iteration))
        # RG re-dispatch curtailment cost
        y_cost_curt_PV = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cost_curt_PV_" + str(iteration))
        y_cost_curt_WT = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cost_curt_WT_" + str(iteration))
        y_cost_cn_PV = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cost_cn_PV_" + str(iteration))
        y_cost_cn_WT = self.model.addVars(self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cost_cn_WT_" + str(iteration))

        # -------------------------------------------------------------------------------------------------------------
        # 4.2 Add the constraint related to the objective
        # objective
        objective = gp.quicksum(y_cost_fuel_1[i] + y_cost_OM_ES[i] + y_cost_curt_PV[i] + y_cost_curt_WT[i] for i in self.t_set)
        # theta = MP.model.getVarByname() = only theta variable of the MP model
        self.model.addConstr(self.model.getVarByName('theta') >= objective, name='theta_' + str(iteration))

        # -------------------------------------------------------------------------------------------------------------
        # 4.3 Add the constraint related to the feasbility domain of the secondstage variables -> y
        # 4.3.1 cost cst
        self.model.addConstrs((y_cost_fuel_1[i] == self.cost_m_pos_re_DE1 * p_pos_1[i] + self.cost_m_neg_re_DE1 * p_neg_1[i] for i in self.t_set), name='c_cost_fuel_1' + str(iteration))
        self.model.addConstrs((y_cost_OM_ES[i] == self.cost_OM_ES_re * (y_dis[i] + y_chg[i]) for i in self.t_set), name='c_cost_curt_neg' + str(iteration))
        self.model.addConstrs((y_cost_curt_PV[i] == self.cost_m_re_PV * y_curt_PV[i] for i in self.t_set), name='c_cost_curt_PV' + str(iteration))
        self.model.addConstrs((y_cost_curt_WT[i] == self.cost_m_re_WT * y_curt_WT[i] for i in self.t_set), name='c_cost_curt_WT' + str(iteration))
        self.model.addConstrs((y_cost_cn_PV[i] == self.cost_m_cn_PV * y_pc[i] for i in self.t_set), name='c_cost_cn_PV' + str(iteration))
        self.model.addConstrs((y_cost_cn_WT[i] == self.cost_m_cn_WT * y_wc[i] for i in self.t_set), name='c_cost_cn_WT' + str(iteration))
        # 4.3.2 DE1 reserve power
        # max/min DE1 reserve cst: self.model.getVarByName() -> return variables of the model from name, the x_b variable are index 0 to 95
        self.model.addConstrs((p_pos_1[i] <= self.model.getVars()[i+480] for i in self.t_set), name='c_DE1_reserve_max_' + str(iteration))
        self.model.addConstrs((p_neg_1[i] <= self.model.getVars()[i+576] for i in self.t_set), name='c_DE1_reserve_min_' + str(iteration))
        # 4.3.3 ES reserve power
        self.model.addConstrs((y_chg[i] <= u_re[i] * self.ES_max for i in self.t_set), name='c_ES_chg_re_' + str(iteration))
        self.model.addConstrs((y_dis[i] <= (1 - u_re[i]) * self.ES_max for i in self.t_set), name='c_ES_dis_re_' + str(iteration))
        self.model.addConstr((y_S[0] == self.soc_ini), name='c_y_SOC_first_period_' + str(iteration))
        self.model.addConstrs((y_S[i] - y_S[i - 1] - (self.charge_eff * y_chg[i] - y_dis[i] / self.discharge_eff) * self.period_hours == 0 for i in range(1, self.nb_periods)), name='c_y_S_Incremental_' + str(iteration))
        self.model.addConstr((y_S[self.nb_periods - 1] == self.soc_end), name='c_y_SOC_last_period_' + str(iteration))
        self.model.addConstrs((- y_S[i] <= - self.soc_min for i in self.t_set), name='c_y_SOC_min_' + str(iteration))
        self.model.addConstrs((y_S[i] <= self.soc_max for i in self.t_set), name='c_y_SOC_max_' + str(iteration))
        # 4.3.6 RG generation cst
        self.model.addConstrs((y_PV[i] == PV_trajectory[i] for i in self.t_set), name='c_y_PV_generation_' + str(iteration))
        self.model.addConstrs((y_WT[i] == WT_trajectory[i] for i in self.t_set), name='c_y_WT_generation_' + str(iteration))
        # 4.3.7 load cst
        self.model.addConstrs((y_load[i] == load_trajectory[i] for i in self.t_set), name='c_y_load_' + str(iteration))
        # 4.3.4 real-time curtailment cst
        self.model.addConstrs((y_curt_PV[i] <= PV_trajectory[i] - self.model.getVars()[i+288] for i in self.t_set), name='c_y_PV_curtailment_' + str(iteration))
        self.model.addConstrs((y_curt_WT[i] <= WT_trajectory[i] - self.model.getVars()[i+384] for i in self.t_set), name='c_y_WT_curtailment_' + str(iteration))
        self.model.addConstrs((y_pc[i] <= self.model.getVars()[i+288] for i in self.t_set), name='c_y_pc' + str(iteration))
        self.model.addConstrs((y_wc[i] <= self.model.getVars()[i+384] for i in self.t_set), name='c_y_wc' + str(iteration))
        # 4.3.4 power balance equation
        self.model.addConstrs((self.model.getVars()[i] + p_pos_1[i] - p_neg_1[i] - y_chg[i] + y_dis[i] + y_PV[i] - self.model.getVars()[i+288] - y_curt_PV[i] + y_pc[i] + y_WT[i] - self.model.getVars()[i+384] - y_curt_WT[i] + y_wc[i] - y_load[i] == 0 for i in self.t_set), name='c_real-time_power_balance_' + str(iteration))
        
        # -------------------------------------------------------------------------------------------------------------
        # 5. Store the added variables to the MP in a new dict
        self.allvar['var_' + str(iteration)] = dict()
        self.allvar['var_' + str(iteration)]['y_cost_fuel_1'] = y_cost_fuel_1
        self.allvar['var_' + str(iteration)]['y_cost_OM_ES'] = y_cost_OM_ES
        self.allvar['var_' + str(iteration)]['y_cost_curt_PV'] = y_cost_curt_PV
        self.allvar['var_' + str(iteration)]['y_cost_curt_WT'] = y_cost_curt_WT
        self.allvar['var_' + str(iteration)]['p_pos_1'] = p_pos_1
        self.allvar['var_' + str(iteration)]['p_neg_1'] = p_neg_1
        self.allvar['var_' + str(iteration)]['u_re'] = u_re
        self.allvar['var_' + str(iteration)]['y_chg'] = y_chg
        self.allvar['var_' + str(iteration)]['y_dis'] = y_dis
        self.allvar['var_' + str(iteration)]['y_S'] = y_S
        self.allvar['var_' + str(iteration)]['y_PV'] = y_PV
        self.allvar['var_' + str(iteration)]['y_curt_PV'] = y_curt_PV
        self.allvar['var_' + str(iteration)]['y_pc'] = y_pc
        self.allvar['var_' + str(iteration)]['y_WT'] = y_WT
        self.allvar['var_' + str(iteration)]['y_curt_WT'] = y_curt_WT
        self.allvar['var_' + str(iteration)]['y_load'] = y_load
        self.allvar['var_' + str(iteration)]['y_wc'] = y_wc

        # -------------------------------------------------------------------------------------------------------------
        # 6. Update model to implement the modifications
        self.model.update()

    # True: output a log that is generated during optimization troubleshooting to the console
    def solve(self, LogToConsole:bool=False):
        t_solve = time.time()
        self.model.setParam('LogToConsole', LogToConsole)
        self.model.optimize()
        self.time_solving_model = time.time() - t_solve

    def store_solution(self):
        
        m = self.model

        solution = dict()
        solution['status'] = m.status
        if solution['status'] == 2 or solution['status'] == 9:
            # solutionStatus = 2: Model was solved to optimality (SUbject to tolerances), and an optimal solution is available.
            # solutionStatus = 9: Optimization terminated because the time expended exceeded the value specified in the TimeLimit  parameter.

            # 0 dimensional variables
            solution['theta'] = self.allvar['theta'].X
            # 1D variable
            solution['p_1'] = [self.allvar['p_1'][t].X for t in self.t_set]
            solution['x_curt_PV'] = [self.allvar['x_curt_PV'][t].X for t in self.t_set]
            solution['x_curt_WT'] = [self.allvar['x_curt_WT'][t].X for t in self.t_set]
            solution['x_chg'] = [self.allvar['x_chg'][t].X for t in self.t_set]
            solution['x_dis'] = [self.allvar['x_dis'][t].X for t in self.t_set]
            solution['x_S'] = [self.allvar['x_S'][t].X for t in self.t_set]            
            solution['r_pos_1'] = [self.allvar['r_pos_1'][t].X for t in self.t_set]
            solution['r_neg_1'] = [self.allvar['r_neg_1'][t].X for t in self.t_set]
            solution['u'] = [self.allvar['u'][t].X for t in self.t_set]
            solution['x_PV'] = [self.allvar['x_PV'][t].X for t in self.t_set]
            solution['x_WT'] = [self.allvar['x_WT'][t].X for t in self.t_set]
            solution['x_load'] = [self.allvar['x_load'][t].X for t in self.t_set]
            solution['x_cost_fuel_1'] = [self.allvar['x_cost_fuel_1'][t].X for t in self.t_set]
            solution['x_cost_fuel_res_1'] = [self.allvar['x_cost_fuel_res_1'][t].X for t in self.t_set]
            solution['x_cost_OM_ES'] = [self.allvar['x_cost_OM_ES'][t].X for t in self.t_set]
            solution['x_cost_curt_PV'] = [self.allvar['x_cost_curt_PV'][t].X for t in self.t_set]
            solution['x_cost_curt_WT'] = [self.allvar['x_cost_curt_WT'][t].X for t in self.t_set]
            solution['obj'] = m.objVal
        else:
            print('WARNING MP status %s -> problem not solved, objective is set to nan' %(solution['status']))
            # solutionStatus = 3: Model was proven to be infeasible
            # solutionStatus = 4: Model was proven to be either infeasible or unbounded.
            solution['status'] = float('nan')

        # Timing indicators
        solution['time_building'] = self.time_building_model
        solution['time_solving'] = self.time_solving_model
        solution['time_total'] = self.time_building_model + self.time_solving_model

        return solution
    
    def update_sol(self, MP_sol:dict, i:int):
        """
        Add the solution of the 1 dimensional variables at iteration i.
        :param MP_sol: solution of the MP model at iteration i.
        :param i: index of interation.
        :return: update directly the dict.
        """
        MP_status = MP_sol['status']
        if MP_status == 2 or MP_status == 9:
            MP_sol['var_' + str(i)] = dict()
            # add the solution of the 1 dimensional variables at iteration
            for var in ['y_cost_fuel_1', 'y_cost_OM_ES', 'y_cost_curt_PV', 'y_cost_curt_WT', 'p_pos_1', 'p_neg_1',
                        'u_re', 'y_chg', 'y_dis', 'y_S', 'y_PV', 'y_WT', 'y_curt_PV', 'y_curt_WT', 'y_pc', 'y_wc', 'y_load',]:
                MP_sol['var_' + str(i)][var] = [self.allvar['var_' + str(i)][var][t].X for t in self.t_set]
        else:
            self.model.computeIIS()
            self.model.write("infeasible_model.ilp")
            print('WARNING planner MP status %s -> problem not solved, cannot retrieve solution')

    def export_model(self, filename):
        """
        Export the pyomo model into a cpxlp format.
        :param filename: directory and filename of the exported model.
        """

        self.model.write("%s.lp" % filename)


if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())