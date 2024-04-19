"""
2023.10.31.
GIST Power System Lab.
Hyun-Su Shin.
Column-and-Constraint Generation (CCG) algorithm to solve a two-stage robust optimization problem in the microgrid scheduling.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from CCG_MP import CCG_MP
from CCG_SP import CCG_SP
from SP_primal_LP import SP_primal_LP
from planner_MILP import Planner_MILP
from Data_read import *
from root_project import ROOT_DIR
from Params import PARAMETERS
from utils import dump_file, check_ES

def ccg_algo(dir:str, tol:float, DE1_p:np.array, DE1_rp:np.array, DE1_rn:np.array,
             ES_charge:np.array, ES_discharge:np.array, ES_SOC:np.array, x_curt_PV:np.array, x_curt_WT:np.array,
             Pi_PV_t:int, Pi_WT_t, Pi_load:int, solver_param:dict, day:str, log:bool=False, printconsole:bool=False, M:float=None):
    """
    CCG = Column-and-Constraint Generation
    Column-and-Constraint Generation algorithm.
    Iteration between the MP and SP until convergence criteria is reached.
    :param tol: convergence tolerance.
    :param Pi_RG_t/s, Pi_load: RG/load spatio and temporal budget of uncertainty.
    :param RG/load_max/min: RG/load max/min bound of the uncertainty set (kW).
    :ivar x: pre-dispatch variables.
    :ivar y: re-dispatch variables.
    :param solver_param: Gurobi solver parameters.
    :return: the final preparatory curtailment schedule when the convergence criteria is reached and some data.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # CCG initialization: build the initial MP
    # ------------------------------------------------------------------------------------------------------------------

    # Building the MP
    MP = CCG_MP(PV_forecast=PV_forecast, WT_forecast=WT_forecast, load_forecast=load_forecast)
    MP.model.update()
    print('MP initialized: %d variables %d constraints' % (len(MP.model.getVars()), len(MP.model.getConstrs())))
    MP.export_model(dir + day + '_CCG_MP_initialized')

    # ------------------------------------------------------------------------------------------------------------------
    # CCG loop until convergence criteria is reached
    # ------------------------------------------------------------------------------------------------------------------

    if printconsole:
        print('---------------------------------CCG ITERATION STARTING---------------------------------')

    t_solve = time.time()
    objectives = []
    computation_times = []
    # measure that helps control the trade-off between solution quality and computation time in MILP or MIQP
    mipgap = []
    alpha_pos_100_list = []
    alpha_neg_100_list = []
    alpha_pos_50_list = []
    alpha_neg_50_list = []
    alpha_pos_20_list = []
    alpha_neg_20_list = []
    beta_pos_100_list = []
    beta_neg_100_list = []
    beta_pos_50_list = []
    beta_neg_50_list = []
    beta_pos_20_list = []
    beta_neg_20_list = []
    gamma_pos_100_list = []
    gamma_neg_100_list = []
    gamma_pos_50_list = []
    gamma_neg_50_list = []
    gamma_pos_20_list = []
    gamma_neg_20_list = []
    delta_pos_100_list = []
    delta_neg_100_list = []
    delta_pos_50_list = []
    delta_neg_50_list = []
    delta_pos_20_list = []
    delta_neg_20_list = []
    zeta_pos_100_list = []
    zeta_neg_100_list = []
    zeta_pos_50_list = []
    zeta_neg_50_list = []
    zeta_pos_20_list = []
    zeta_neg_20_list = []
    SP_dual_status = []
    SP_primal_status = []
    tolerance = 1e20

    # with CCG the convergence is stable.
    tolerance_list = [tolerance]
    iteration = 1
    ES_count_list = []
    ES_charge_discharge_list = []
    max_iteration = 4

    while all(i < tol for i in tolerance_list) is not True and iteration < max_iteration:
        logfile = ""
        if log:
            logfile = dir + 'logfile_' + str(iteration) + '.log'
        if printconsole:
            print('i= %s solve SP dual' % (iteration))

        # ------------------------------------------------------------------------------------------------------------------
        # 1. SP part
        # ------------------------------------------------------------------------------------------------------------------

        # 1.1 Solve the SP and get the worst RG and load trajectory to add the new constraints of the MP
        SP_dual = CCG_SP(PV_forecast=PV_forecast, WT_forecast=WT_forecast, load_forecast=load_forecast, PV_pos_100=PV_pos_100, PV_neg_100=PV_neg_100, PV_pos_50=PV_pos_50, PV_neg_50=PV_neg_50, PV_pos_20=PV_pos_20, PV_neg_20=PV_neg_20,
                         WT_pos_100=WT_pos_100, WT_neg_100=WT_neg_100, WT_pos_50=WT_pos_50, WT_neg_50=WT_neg_50, WT_pos_20=WT_pos_20, WT_neg_20=WT_neg_20,
                         load_pos_100=load_pos_100, load_neg_100=load_neg_100, load_pos_50=load_pos_50, load_neg_50=load_neg_50, load_pos_20=load_pos_20, load_neg_20=load_neg_20,
                         DE1_p=DE1_p, DE1_rp=DE1_rp, DE1_rn=DE1_rn, x_curt_PV=x_curt_PV, x_curt_WT=x_curt_WT, Pi_PV_t=Pi_PV_t, Pi_WT_t=Pi_WT_t, Pi_load=Pi_load, M=M)    
        SP_dual.solve(logfile=logfile, Threads=solver_param['Threads'], MIPFocus=solver_param['MIPFocus'], TimeLimit=solver_param['TimeLimit'])
        SP_dual_sol = SP_dual.store_solution()
        SP_dual_status.append(SP_dual_sol['status'])
        # mipgap.append(SP_dual.model.MIPGap)
        alpha_pos_100_list.append(SP_dual_sol['alpha_pos_100'])
        alpha_neg_100_list.append(SP_dual_sol['alpha_neg_100'])
        alpha_pos_50_list.append(SP_dual_sol['alpha_pos_50'])
        alpha_neg_50_list.append(SP_dual_sol['alpha_neg_50'])
        alpha_pos_20_list.append(SP_dual_sol['alpha_pos_20'])
        alpha_neg_20_list.append(SP_dual_sol['alpha_neg_20'])
        beta_pos_100_list.append(SP_dual_sol['beta_pos_100'])
        beta_neg_100_list.append(SP_dual_sol['beta_neg_100'])
        beta_pos_50_list.append(SP_dual_sol['beta_pos_50'])
        beta_neg_50_list.append(SP_dual_sol['beta_neg_50'])
        beta_pos_20_list.append(SP_dual_sol['beta_pos_20'])
        beta_neg_20_list.append(SP_dual_sol['beta_neg_20'])
        gamma_pos_100_list.append(SP_dual_sol['gamma_pos_100'])
        gamma_neg_100_list.append(SP_dual_sol['gamma_neg_100'])
        gamma_pos_50_list.append(SP_dual_sol['gamma_pos_50'])
        gamma_neg_50_list.append(SP_dual_sol['gamma_neg_50'])
        gamma_pos_20_list.append(SP_dual_sol['gamma_pos_20'])
        gamma_neg_20_list.append(SP_dual_sol['gamma_neg_20'])
        delta_pos_100_list.append(SP_dual_sol['delta_pos_100'])
        delta_neg_100_list.append(SP_dual_sol['delta_neg_100'])
        delta_pos_50_list.append(SP_dual_sol['delta_pos_50'])
        delta_neg_50_list.append(SP_dual_sol['delta_neg_50'])
        delta_pos_20_list.append(SP_dual_sol['delta_pos_20'])
        delta_neg_20_list.append(SP_dual_sol['delta_neg_20'])
        zeta_pos_100_list.append(SP_dual_sol['zeta_pos_100'])
        zeta_neg_100_list.append(SP_dual_sol['zeta_neg_100'])
        zeta_pos_50_list.append(SP_dual_sol['zeta_pos_50'])
        zeta_neg_50_list.append(SP_dual_sol['zeta_neg_50'])
        zeta_pos_20_list.append(SP_dual_sol['zeta_pos_20'])
        zeta_neg_20_list.append(SP_dual_sol['zeta_neg_20'])


        # 1.2 Compute the worst RG, load trajectory from the SP dual solution
        PV_worst_case_from_SP = [PV_forecast[i] + PV_pos_100[i] * SP_dual_sol['epsilon_pos_100'][i] - PV_neg_100[i] * SP_dual_sol['epsilon_neg_100'][i] + PV_pos_50[i] * SP_dual_sol['epsilon_pos_50'][i]
                                 - PV_neg_50[i] * SP_dual_sol['epsilon_neg_50'][i] + PV_pos_20[i] * SP_dual_sol['epsilon_pos_20'][i] - PV_neg_20[i] * SP_dual_sol['epsilon_neg_20'][i] for i in range(nb_periods)]
        WT_worst_case_from_SP = [WT_forecast[i] + WT_pos_100[i] * SP_dual_sol['kapa_pos_100'][i] - WT_neg_100[i] * SP_dual_sol['kapa_neg_100'][i] + WT_pos_50[i] * SP_dual_sol['kapa_pos_50'][i]
                                 - WT_neg_50[i] * SP_dual_sol['kapa_neg_50'][i] + WT_pos_20[i] * SP_dual_sol['kapa_pos_20'][i] - WT_neg_20[i] * SP_dual_sol['kapa_neg_20'][i] for i in range(nb_periods)]
        load_worst_case_from_SP = [load_forecast[i] + load_pos_100[i] * SP_dual_sol['xi_pos_100'][i] - load_neg_100[i] * SP_dual_sol['xi_neg_100'][i] + load_pos_50[i] * SP_dual_sol['xi_pos_50'][i]
                                   - load_neg_50[i] * SP_dual_sol['xi_neg_50'][i] + load_pos_20[i] * SP_dual_sol['xi_pos_20'][i] - load_neg_20[i] * SP_dual_sol['xi_neg_20'][i] for i in range(nb_periods)]
        # if printconsole:
        #     print('     i = %s : SP dual status %s solved in %.1f s MIPGap = %.6f' % (iteration, SP_dual_sol['status'], SP_dual_sol['time_total'], SP_dual.model.MIPGap))

        # 1.3 Solve the primal of the SP to check if the objecitves of the primal and dual are equal to each other
        SP_primal = SP_primal_LP(PV_forecast=PV_worst_case_from_SP, WT_forecast=WT_worst_case_from_SP, load_forecast=load_worst_case_from_SP,
                                DE1_p=DE1_p, DE1_rp=DE1_rp, DE1_rn=DE1_rn,
                                ES_charge=ES_charge, ES_discharge=ES_discharge, ES_SOC=ES_SOC, x_curt_PV=x_curt_PV, x_curt_WT=x_curt_WT)
        SP_primal.solve()
        SP_primal_sol = SP_primal.store_solution()
        SP_primal_status.append(SP_primal_sol['status'])

        if printconsole:
            print('     i = %s : SP primal status %s' % (iteration, SP_primal_sol['status']))
            print('     i = %s : SP primal %.1f $ SP dual %.1f $ -> |SP primal - SP dual| = %.2f $' % (iteration, SP_primal_sol['obj'], SP_dual_sol['obj'], abs(SP_primal_sol['obj'] - SP_dual_sol['obj'])))

        # 1.4 SP solved to optimality ? -> Check if there is any simultaneous charge and discharge in the SP primal solution
        if SP_primal_sol['status'] == 2 or SP_primal_sol['status'] == 9: # 2 = optimal, 9 = timelimit has been reached
            nb_count = check_ES(SP_primal_sol = SP_primal_sol)
            if nb_count > 0:
                ES_charge_discharge_list.append([iteration, SP_primal_sol['y_chg'], SP_primal_sol['y_dis']])
            else:
                nb_count = float('nan')
            ES_count_list.append(nb_count)
            if printconsole:
                print('     i = %s : %s simultaneous charge and discharge' % (iteration, nb_count))

        # ------------------------------------------------------------------------------------------------------------------
        # 2. MP part
        # ------------------------------------------------------------------------------------------------------------------

        # Check Sub Problem status -> bounded or unbounded
        if SP_dual_sol['status'] == 2 or SP_dual_sol['status'] == 9:  # 2 = optimal, 9 = timelimit has been reached
            # Add an optimality cut to MP and solve
            MP.update_MP(PV_trajectory=PV_worst_case_from_SP, WT_trajectory=WT_worst_case_from_SP, load_trajectory=load_worst_case_from_SP, iteration=iteration)
            if printconsole:
                print('i = %s : MP with %d variables and %d constraints' % (iteration, len(MP.model.getVars()), len(MP.model.getConstrs())))
            # MP.export_model(dir + 'MP_' + str(iteration))
            if printconsole:
                print('i = %s : solve MP' % (iteration))
            MP.solve()
            MP_sol = MP.store_solution()
            MP.update_sol(MP_sol=MP_sol, i=iteration)
            if MP_sol['status'] == 3 or MP_sol['status'] == 4:
                print('i = %s : WARNING MP status %s -> Create a new MP, increase big-M value and compute a new RG trajectory from SP' % (iteration, MP_sol['status']))

                # MP unbounded of infeasible -> increase big-M's value to get another PV trajectory from the SP
                SP_dual = SP_dual = CCG_SP(PV_forecast=PV_forecast, WT_forecast=WT_forecast, load_forecast=load_forecast, PV_pos_100=PV_pos_100, PV_neg_100=PV_neg_100, PV_pos_50=PV_pos_50, PV_neg_50=PV_neg_50, PV_pos_20=PV_pos_20, PV_neg_20=PV_neg_20,
                                           WT_pos_100=WT_pos_100, WT_neg_100=WT_neg_100, WT_pos_50=WT_pos_50, WT_neg_50=WT_neg_50, WT_pos_20=WT_pos_20, WT_neg_20=WT_neg_20,
                                           load_pos_100=load_pos_100, load_neg_100=load_neg_100, load_pos_50=load_pos_50, load_neg_50=load_neg_50, load_pos_20=load_pos_20, load_neg_20=load_neg_20,
                                           DE1_p=DE1_p, DE1_rp=DE1_rp, DE1_rn=DE1_rn, x_curt_PV=x_curt_PV, x_curt_WT=x_curt_WT, Pi_PV_t=Pi_PV_t, Pi_WT_t=Pi_WT_t, Pi_load=Pi_load, M=M+50)
                SP_dual.solve(logfile=logfile, Threads=solver_param['Threads'], MIPFocus=solver_param['MIPFocus'], TimeLimit=solver_param['TimeLimit'])
                SP_dual_sol = SP_dual.store_solution()

                # Compute a new worst PV trajectory from the SP dual solution
                PV_worst_case_from_SP = [PV_forecast[i] + PV_pos_100[i] * SP_dual_sol['epsilon_pos_100'][i] - PV_neg_100[i] * SP_dual_sol['epsilon_neg_100'][i] + PV_pos_50[i] * SP_dual_sol['epsilon_pos_50'][i]
                                        - PV_neg_50[i] * SP_dual_sol['epsilon_neg_50'][i] + PV_pos_20[i] * SP_dual_sol['epsilon_pos_20'][i] - PV_neg_20[i] * SP_dual_sol['epsilon_neg_20'][i] for i in range(nb_periods)]
                WT_worst_case_from_SP = [WT_forecast[i] + WT_pos_100[i] * SP_dual_sol['kapa_pos_100'][i] - WT_neg_100[i] * SP_dual_sol['kapa_neg_100'][i] + WT_pos_50[i] * SP_dual_sol['kapa_pos_50'][i]
                                        - WT_neg_50[i] * SP_dual_sol['kapa_neg_50'][i] + WT_pos_20[i] * SP_dual_sol['kapa_pos_20'][i] - WT_neg_20[i] * SP_dual_sol['kapa_neg_20'][i] for i in range(nb_periods)]
                load_worst_case_from_SP = [load_forecast[i] + load_pos_100[i] * SP_dual_sol['xi_pos_100'][i] - load_neg_100[i] * SP_dual_sol['xi_neg_100'][i] + load_pos_50[i] * SP_dual_sol['xi_pos_50'][i]
                                        - load_neg_50[i] * SP_dual_sol['xi_neg_50'][i] + load_pos_20[i] * SP_dual_sol['xi_pos_20'][i] - load_neg_20[i] * SP_dual_sol['xi_neg_20'][i] for i in range(nb_periods)]

                # Create a new MP
                MP = CCG_MP()
                MP.model.update()
                MP.update_MP(PV_trajectory=PV_worst_case_from_SP, WT_trajectory=WT_worst_case_from_SP, load_trajectory=load_worst_case_from_SP, iteration=iteration)
                if printconsole:
                    print('i = %s : MP with %d variables and %d constraints' % (iteration, len(MP.model.getVars()), len(MP.model.getConstrs())))
                # MP.export_model(dir + 'MP_' + str(iteration))
                if printconsole:
                    print('i = %s : solve new MP' % (iteration))
                MP.solve()
                MP_sol = MP.store_solution()
                MP.update_sol(MP_sol=MP_sol, i=iteration)

            computation_times.append([SP_dual_sol['time_total'], MP_sol['time_total']])

        else: # 4 = Model was proven to be either infeasible or unbounded.
            print('SP is unbounded: a feasibility cut is required to be added to the Master Problem')

        objectives.append([iteration, MP_sol['obj'], SP_dual_sol['obj'], SP_primal_sol['obj']])

        # ------------------------------------------------------------------------------------------------------------------
        # 3. Update: pre-dispatch variables, lower and upper bounds using the updated MP
        # ------------------------------------------------------------------------------------------------------------------

        # Solve the MILP with the worst case trajectory
        planner = Planner_MILP(PV_forecast=PV_worst_case_from_SP, WT_forecast=WT_worst_case_from_SP, load_forecast=load_worst_case_from_SP)
        planner.solve()
        sol_planner = planner.store_solution()

        # Update x variables
        DE1_p = MP_sol['p_1']
        DE1_rp = MP_sol['r_pos_1']
        DE1_rn = MP_sol['r_neg_1']
        ES_charge = MP_sol['x_chg']
        ES_discharge = MP_sol['x_dis']
        ES_SOC = MP_sol['x_S']
        x_curt_PV = MP_sol['x_curt_PV']
        x_curt_WT = MP_sol['x_curt_WT']
        x_cost_fuel_1 = MP_sol['x_cost_fuel_1']
        x_cost_fuel_res_1 = MP_sol['x_cost_fuel_res_1']
        x_cost_OM_ES = MP_sol['x_cost_OM_ES']
        x_cost_curt_PV = MP_sol['x_cost_curt_PV']
        x_cost_curt_WT = MP_sol['x_cost_curt_WT']

        # Update the lower and upper bounds
        # MP -> give the lower bound
        # SP -> give the upper bound
        tolerance = abs(MP_sol['obj'] - SP_dual_sol['obj'])
        print('i = %s : |MP - SP dual| = %.2f $' % (iteration, tolerance))
        tolerance_list.append(tolerance)
        tolerance_list.pop(0)
        if printconsole:
            print('i = %s : MP %.2f $ SP dual %.2f $ -> |MP - SP dual| = %.2f $' % (iteration, MP_sol['obj'], SP_dual_sol['obj'], tolerance))
            print(tolerance_list)
            print('                                                                                                       ')

        iteration += 1

    y_cost_fuel_1 = MP_sol['var_3']['y_cost_fuel_1']
    y_cost_OM_ES = MP_sol['var_3']['y_cost_OM_ES']
    y_cost_curt_PV = MP_sol['var_3']['y_cost_curt_PV']
    y_cost_curt_WT = MP_sol['var_3']['y_cost_curt_WT']

    # ------------------------------------------------------------------------------------------------------------------
    # CCG loop terminated
    # ------------------------------------------------------------------------------------------------------------------
    if printconsole:
        print('-----------------------------------CCG ITERATION TERMINATED-----------------------------------')
    print('Final iteration  = %s : MP %.2f $ SP dual %.2f $ -> |MP - SP dual| = %.2f $' % (iteration-1, MP_sol['obj'], SP_dual_sol['obj'], tolerance))

    # Export last MP
    MP.export_model(dir + day + '_MP')

    # MP.model.printStats()

    # Dump last engagement plan at iteration
    dump_file(dir=dir, name=day+'_p_1', file=DE1_p)
    dump_file(dir=dir, name=day+'_r_pos_1', file=DE1_rp)
    dump_file(dir=dir, name=day+'_r_neg_1', file=DE1_rn)
    dump_file(dir=dir, name=day+'_x_chg', file=ES_charge)
    dump_file(dir=dir, name=day+'_x_dis', file=ES_discharge)
    dump_file(dir=dir, name=day+'_x_S', file=ES_SOC)
    dump_file(dir=dir, name=day+'_x_curt_PV', file=x_curt_PV)
    dump_file(dir=dir, name=day+'_x_curt_WT', file=x_curt_WT)
    dump_file(dir=dir, name=day+'_x_cost_fuel_1', file=x_cost_fuel_1)
    dump_file(dir=dir, name=day+'_x_cost_fuel_res_1', file=x_cost_fuel_res_1)
    dump_file(dir=dir, name=day+'_x_cost_OM_ES', file=x_cost_OM_ES)
    dump_file(dir=dir, name=day+'_x_cost_curt_PV', file=x_cost_curt_PV)
    dump_file(dir=dir, name=day+'_x_cost_curt_WT', file=x_cost_curt_WT)
    dump_file(dir=dir, name=day+'_y_cost_fuel_1', file=y_cost_fuel_1)
    dump_file(dir=dir, name=day+'_y_cost_OM_ES', file=y_cost_OM_ES)
    dump_file(dir=dir, name=day+'_y_cost_curt_PV', file=y_cost_curt_PV)
    dump_file(dir=dir, name=day+'_y_cost_curt_WT', file=y_cost_curt_WT)

    # print T CPU
    t_total = time.time() - t_solve
    computation_times = np.asarray(computation_times)
    SP_dual_status = np.asarray(SP_dual_status)
    SP_primal_status = np.asarray(SP_primal_status)

    if printconsole:
        print('Total CCG loop t CPU %.1f s' % (t_total))
        print('T CPU (s): Sup Problem max %.1f Master Problem max %.1f' % (computation_times[:, 0].max(), computation_times[:, 1].max()))
        print('nb Sup Problem status 2 %d status 9 %d' % (SP_dual_status[SP_dual_status == 2].shape[0], SP_dual_status[SP_dual_status == 9].shape[0]))

    # Store data
    objectives = np.asarray(objectives)
    df_objectives = pd.DataFrame(index=objectives[:,0], data=objectives[:,1:], columns=['MP', 'SP', 'SP_primal'])

    # Store convergence information
    conv_inf = dict()
    conv_inf['mipgap'] = mipgap
    conv_inf['computation_times'] = computation_times
    conv_inf['SP_status'] = SP_dual_status

    conv_inf['SP_primal_status'] = SP_primal_status
    conv_inf['alpha_pos_100'] = alpha_pos_100_list
    conv_inf['alpha_neg_100'] = alpha_neg_100_list
    conv_inf['alpha_pos_50'] = alpha_pos_50_list
    conv_inf['alpha_neg_50'] = alpha_neg_50_list
    conv_inf['alpha_pos_20'] = alpha_pos_20_list
    conv_inf['alpha_neg_20'] = alpha_neg_20_list
    conv_inf['beta_pos_100'] = beta_pos_100_list
    conv_inf['beta_neg_100'] = beta_neg_100_list
    conv_inf['beta_pos_100'] = beta_pos_100_list
    conv_inf['beta_neg_100'] = beta_neg_100_list
    conv_inf['beta_pos_50'] = beta_pos_50_list
    conv_inf['beta_neg_50'] = beta_neg_50_list
    conv_inf['beta_pos_20'] = beta_pos_20_list
    conv_inf['beta_neg_20'] = beta_neg_20_list
    conv_inf['gamma_pos_100'] = gamma_pos_100_list
    conv_inf['gamma_neg_100'] = gamma_neg_100_list
    conv_inf['gamma_pos_50'] = gamma_pos_50_list
    conv_inf['gamma_neg_50'] = gamma_neg_50_list
    conv_inf['gamma_pos_20'] = gamma_pos_20_list
    conv_inf['gamma_neg_20'] = gamma_neg_20_list
    conv_inf['delta_pos_100'] = delta_pos_100_list
    conv_inf['delta_neg_100'] = delta_neg_100_list
    conv_inf['delta_pos_50'] = delta_pos_50_list
    conv_inf['delta_neg_50'] = delta_neg_50_list
    conv_inf['delta_pos_20'] = delta_pos_20_list
    conv_inf['delta_neg_20'] = delta_neg_20_list
    conv_inf['zeta_pos_100'] = zeta_pos_100_list
    conv_inf['zeta_neg_100'] = zeta_neg_100_list
    conv_inf['zeta_pos_50'] = zeta_pos_50_list
    conv_inf['zeta_neg_50'] = zeta_neg_50_list
    conv_inf['zeta_pos_20'] = zeta_pos_20_list
    conv_inf['zeta_neg_20'] = zeta_neg_20_list
    conv_inf['ES_count'] = ES_count_list
    conv_inf['ES_charge_discharge'] = ES_charge_discharge_list

    return DE1_p, DE1_rp, DE1_rn, ES_charge, ES_discharge, ES_SOC, x_curt_PV, x_curt_WT, df_objectives, conv_inf, \
        x_cost_fuel_1, x_cost_fuel_res_1, x_cost_OM_ES, x_cost_curt_PV, x_cost_curt_WT, \
            y_cost_fuel_1, y_cost_OM_ES, y_cost_curt_PV, y_cost_curt_WT

# ------------------------------------------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------------------------------------------

FONTSIZE = 24

# NB periods
nb_periods = 96

# Solver parameters
solver_param = dict()
solver_param['MIPFocus'] = 3 # Seems to be the best -> focus on the bound
solver_param['TimeLimit'] = 10
solver_param['Threads'] = 1

# Convergence threshold between MP and SP objectives
conv_tol = 5
printconsole = True

# Select the day
day_list = ['2018-07-04']
day = day_list[0]
day = '2018-07-04'

# --------------------------------------
# Static RO parameters: [q_min, gamma]
PI_PV_T = 96 # Budget of uncertainty to specify the number of time periods where PV generation lies within the uncertainty interval: 0: to 95 -> 0 = no uncertainty
PI_WT_T = 96
PI_LOAD = 96
#--------------------------------------
# warm_start
M = 1

# quantile from NE or LSTM
PV_Sandia = True

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    os.chdir(ROOT_DIR)
    print(os.getcwd())

    # Create folder
    dirname = '/Users/Andrew/OneDrive - GIST/Code/Graduation/PC_RGD_CCG_Mc_MIU/export_CCG/'
    if PV_Sandia:
        dirname += 'PV_Sandia/'
        pdfname = str(PV_Sandia) + '_' + str(PI_PV_T) + '_' +str(PI_WT_T) + '_' + str(PI_LOAD)

    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    print('-----------------------------------------------------------------------------------------------------------')
    if PV_Sandia:
        print('CCG: day %s Pi_PV_t %s Pi_WT_t %s Pi_load %s' % (day, PI_PV_T, PI_WT_T, PI_LOAD))
    print('-----------------------------------------------------------------------------------------------------------')

    # RG/Load data
    PV_forecast = data.PV_pred
    WT_forecast = data.WT_pred
    load_forecast = data.load_egg
    PV_lb = data.PV_min # forecast - neg
    PV_ub = data.PV_max # forecast + pos
    WT_lb = data.WT_min
    WT_ub = data.WT_max
    load_lb = data.load_min # forecast - neg
    load_ub = data.load_max # forecast + pos
    PV_pos_100 = data.PV_pos_100 # (kW) The maximal deviation betwwen the min and forecast PV uncertainty set bounds
    PV_neg_100 = data.PV_neg_100 # (kW) The maximal deviation between the max and forecast PV uncertainty set bounds
    PV_pos_50 = data.PV_pos_50 
    PV_neg_50 = data.PV_neg_50 
    PV_pos_20 = data.PV_pos_20 
    PV_neg_20 = data.PV_neg_20 
    WT_pos_100 = data.WT_pos_100 # (kW) The maximal deviation betwwen the min and forecast WT uncertainty set bounds
    WT_neg_100 = data.WT_neg_100 # (kW) The maximal deviation between the max and forecast WT uncertainty set bounds
    WT_pos_50 = data.WT_pos_50 
    WT_neg_50 = data.WT_neg_50 
    WT_pos_20 = data.WT_pos_20 
    WT_neg_20 = data.WT_neg_20 
    load_pos_100 = data.load_pos_100 # (kw) The maximal deviation between the min and forecast load uncertainty set bounds
    load_neg_100 = data.load_neg_100 # (kW) The maximal deviation between the max and forecast load uncertainty set bounds
    load_pos_50 = data.load_pos_50 
    load_neg_50 = data.load_neg_50 
    load_pos_20 = data.load_pos_20 
    load_neg_20 = data.load_neg_20 
    nb_periods = PV_pos_100.shape[0]

    # plot style
    plt.style.use(['science', 'no-latex'])
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False
    x_index = [i for i in range(0, nb_periods)]

    # Store the forecast into a dict
    PV_forecast_dict = dict()
    WT_forecast_dict = dict()
    PV_forecast_dict['forecast'] = PV_forecast
    WT_forecast_dict['forecast'] = WT_forecast

    # Compute the starting point for the first MP = day-ahead planning from the PV using the MILP
    planner = Planner_MILP(PV_forecast=PV_forecast, WT_forecast=WT_forecast, load_forecast=load_forecast)
    planner.solve()
    
    # 처음 풀 때 MP를 풀어보기
    # planner = CCG_MP(RG_forecast=RG_forecast, load_forecast=load_forecast)
    # planner.solve()

    sol_planner_ini = planner.store_solution()
    DE1_p_ini = sol_planner_ini['p_1']
    DE1_rp_ini = sol_planner_ini['r_pos_1']
    DE1_rn_ini = sol_planner_ini['r_neg_1']
    ES_charge_ini = sol_planner_ini['x_chg']
    ES_discharge_ini = sol_planner_ini['x_dis']
    ES_SOC_ini = sol_planner_ini['x_S']
    x_curt_PV_ini = sol_planner_ini['x_curt_PV']
    x_curt_WT_ini = sol_planner_ini['x_curt_WT']
    # ------------------------------------------------------------------------------------------------------------------
    # CCG loop
    # ------------------------------------------------------------------------------------------------------------------
    final_DE1_p, final_DE1_rp, final_DE1_rn, final_ES_charge, final_ES_discharge, final_ES_SOC, \
        final_x_curt_PV, final_x_curt_WT, df_objectives, conv_inf , final_x_cost_fuel_1, \
        final_x_cost_fuel_res_1, final_x_cost_OM_ES, final_x_cost_curt_PV, final_x_cost_curt_WT, \
            final_y_cost_fuel_1, final_y_cost_OM_ES, final_y_cost_curt_PV, final_y_cost_curt_WT \
        = ccg_algo(dir=dirname, tol=conv_tol, DE1_p=DE1_p_ini, DE1_rp=DE1_rp_ini, DE1_rn=DE1_rn_ini,
                   ES_charge=ES_charge_ini, ES_discharge=ES_discharge_ini, ES_SOC=ES_SOC_ini, x_curt_PV=x_curt_PV_ini, x_curt_WT=x_curt_WT_ini,
                   Pi_PV_t=PI_PV_T, Pi_WT_t=PI_WT_T, Pi_load=PI_LOAD, solver_param=solver_param, day=day, printconsole=printconsole, M=M)
    df_objectives.to_csv(dirname + day + 'obj_MP_SP_' + '.csv')

    print('-----------------------------------------------------------------------------------------------------------')
    print('CCG: day %s Pi_PV_t %s Pi_WT_t %s Pi_Load %s' % (day, PI_PV_T, PI_WT_T, PI_LOAD))
    print('-----------------------------------------------------------------------------------------------------------')

    # ------------------------------------------------------------------------------------------------------------------
    # Get the final worst case RG generation trajectory computed by the Sub Problem
    # ------------------------------------------------------------------------------------------------------------------

    # Get the worst case related to the last engagement plan by using the Sub Problem dual formulation
    
    SP_dual = CCG_SP(PV_forecast=PV_forecast, WT_forecast=WT_forecast, load_forecast=load_forecast, PV_pos_100=PV_pos_100, PV_neg_100=PV_neg_100, PV_pos_50=PV_pos_50, PV_neg_50=PV_neg_50, PV_pos_20=PV_pos_20, PV_neg_20=PV_neg_20,
                     WT_pos_100=WT_pos_100, WT_neg_100=WT_neg_100, WT_pos_50=WT_pos_50, WT_neg_50=WT_neg_50, WT_pos_20=WT_pos_20, WT_neg_20=WT_neg_20,
                     load_pos_100=load_pos_100, load_neg_100=load_neg_100, load_pos_50=load_pos_50, load_neg_50=load_neg_50, load_pos_20=load_pos_20, load_neg_20=load_neg_20,
                     DE1_p=final_DE1_p, DE1_rp=final_DE1_rp, DE1_rn=final_DE1_rn, x_curt_PV=final_x_curt_PV, x_curt_WT=final_x_curt_WT, Pi_PV_t=PI_PV_T, Pi_WT_t=PI_WT_T, Pi_load=PI_LOAD, M=M)
    SP_dual.solve(LogToConsole=False, Threads=solver_param['Threads'], MIPFocus=solver_param['MIPFocus'], TimeLimit=10)
    SP_dual_sol = SP_dual.store_solution()
    # Compute the worst RG, load path from the SP dual solution
    PV_worst_case = [PV_forecast[i] + PV_pos_100[i] * SP_dual_sol['epsilon_pos_100'][i] - PV_neg_100[i] * SP_dual_sol['epsilon_neg_100'][i] + PV_pos_50[i] * SP_dual_sol['epsilon_pos_50'][i] - PV_neg_50[i] * SP_dual_sol['epsilon_neg_50'][i] + PV_pos_20[i] * SP_dual_sol['epsilon_pos_20'][i] - PV_neg_20[i] * SP_dual_sol['epsilon_neg_20'][i] for i in range(nb_periods)]
    WT_worst_case = [WT_forecast[i] + WT_pos_100[i] * SP_dual_sol['kapa_pos_100'][i] - WT_neg_100[i] * SP_dual_sol['kapa_neg_100'][i] + WT_pos_50[i] * SP_dual_sol['kapa_pos_50'][i] - WT_neg_50[i] * SP_dual_sol['kapa_neg_50'][i] + WT_pos_20[i] * SP_dual_sol['kapa_pos_20'][i] - WT_neg_20[i] * SP_dual_sol['kapa_neg_20'][i] for i in range(nb_periods)]
    load_worst_case = [load_forecast[i] + load_pos_100[i] * SP_dual_sol['xi_pos_100'][i] - load_neg_100[i] * SP_dual_sol['xi_neg_100'][i] + load_pos_50[i] * SP_dual_sol['xi_pos_50'][i] - load_neg_50[i] * SP_dual_sol['xi_neg_50'][i] + load_pos_20[i] * SP_dual_sol['xi_pos_20'][i] - load_neg_20[i] * SP_dual_sol['xi_neg_20'][i] for i in range(nb_periods)]
    dump_file(dir=dirname, name=day + '_PV_worst_case', file=PV_worst_case)
    dump_file(dir=dirname, name=day + '_WT_worst_case', file=WT_worst_case)
    dump_file(dir=dirname, name=day + '_load_worst_case', file=load_worst_case)
    # Check if the worst RG, load path is on the extreme quantile
    if sum(SP_dual_sol['epsilon_pos_100']) == PI_PV_T:
        print('Worst PV path is the extreme')
    else:
        print('%d PV points on upper boundary, %d points on lower boundary, %d points on 50, %d points on 20' % (sum(SP_dual_sol['epsilon_pos_100']), sum(SP_dual_sol['epsilon_neg_100']), sum(SP_dual_sol['epsilon_pos_50'] + SP_dual_sol['epsilon_neg_50']), sum(SP_dual_sol['epsilon_pos_20'] + SP_dual_sol['epsilon_neg_20'])))
    if sum(SP_dual_sol['kapa_pos_100']) == PI_WT_T:
        print('Worst PV path is the extreme')
    else:
        print('%d PV points on upper boundary, %d points on lower boundary, %d points on 50, %d points on 20' % (sum(SP_dual_sol['kapa_pos_100']), sum(SP_dual_sol['kapa_neg_100']), sum(SP_dual_sol['kapa_pos_50'] + SP_dual_sol['kapa_neg_50']), sum(SP_dual_sol['kapa_pos_20'] + SP_dual_sol['kapa_neg_20'])))
    if sum(SP_dual_sol['xi_pos_100'] + SP_dual_sol['xi_neg_100']) == PI_LOAD:
        print('Worst load path is the extreme')
    else:
        print('%d load points on upper boundary, %d points on lower boundary, %d points on 50, %d points on 20' % (sum(SP_dual_sol['xi_pos_100']), sum(SP_dual_sol['xi_neg_100']), sum(SP_dual_sol['xi_pos_50'] + SP_dual_sol['xi_neg_50']), sum(SP_dual_sol['xi_pos_20'] + SP_dual_sol['xi_neg_20'])))

    # ------------------------------------------------------------------------------------------------------------------
    # Second-stage variables comparison:
    # ------------------------------------------------------------------------------------------------------------------

    # Use the SP primal (SP worst case dispatch max min formulation) to compute the dispatch variables related to the last CCG pre-dispatch computed by the MP
    # Use the worst case dispatch to get the equivalent of the max min formulation
    SP_primal = SP_primal_LP(PV_forecast=PV_worst_case, WT_forecast=WT_worst_case, load_forecast=load_worst_case, DE1_p=final_DE1_p, DE1_rp=final_DE1_rp, DE1_rn=final_DE1_rn,
                             ES_charge=final_ES_charge, ES_discharge=final_ES_discharge, ES_SOC=final_ES_SOC, x_curt_PV=final_x_curt_PV, x_curt_WT=final_x_curt_WT)
    SP_primal.solve()
    SP_primal_sol = SP_primal.store_solution()

    # ------------------------------------------------------------------------------------------------------------------
    # Check if there has been any simultanenaous charge and discharge during all CCG iterations
    # ------------------------------------------------------------------------------------------------------------------

    # 1. Check if there is any simultaneous charge and discharge at the last CCG iteration
    nb_count = check_ES(SP_primal_sol=SP_primal_sol)
    print('CCG last iteration %d simultaneous charge and discharge' % (nb_count))

    # 2. Check if there is any simultaneous charge and discharge over all CCG iteration
    # check if there is nan value (meaning during an iteration the SP primal has not been solved because infeasible, etc)
    ES_count = conv_inf['ES_count']
    if sum(np.isnan(ES_count)) > 0:
        print('WARNING %s nan values' %(sum(np.isnan(conv_inf['ES_count']))))
    # “python list replace nan with 0” Code
    ES_count = [0 if x != x else x for x in ES_count]

    print('%d total simultaneous charge and discharge over all CCG iterations' % (sum(ES_count)))
    if sum(conv_inf['ES_count']) > 0:
        plt.figure(figsize=(16,9))
        plt.plot(conv_inf['ES_count'], 'k', linewidth=2, label='ES_count')
        plt.ylim(0, max(conv_inf['ES_count']))
        plt.xlabel('iteration $j$', fontsize=FONTSIZE)
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.tight_layout()
        plt.legend()
        plt.savefig(dirname + day + '_ES_count_' + pdfname + '.pdf')
        plt.close('all')

        # Plot at each iteration where there has been a simultaneous charge and discharge
        for l in conv_inf['ES_charge_discharge']:
            plt.figure(figsize = (8,6))
            plt.plot(l[1], linewidth=2, label='charge')
            plt.plot(l[2], linewidth=2, label='discharge')
            plt.ylim(0, PARAMETERS['ES']['capacity'])
            plt.ylabel('kW', fontsize=FONTSIZE)
            plt.xticks(fontsize=FONTSIZE)
            plt.yticks(fontsize=FONTSIZE)
            plt.legend(fontsize=FONTSIZE)
            plt.title('simultaneous charge discharge at iteration %s' %(l[0]))
            plt.tight_layout()
            plt.close('all')

    # ------------------------------------------------------------------------------------------------------------------
    # Check CCG convergence by computing the planning for the PV worst trajectory from CCG last iteration
    # ------------------------------------------------------------------------------------------------------------------
    planner = Planner_MILP(PV_forecast=PV_worst_case, WT_forecast=WT_worst_case, load_forecast=load_worst_case)
    planner.solve()
    sol_planner = planner.store_solution()

    # ------------------------------------------------------------------------------------------------------------------
    # First-stage variables comparison: x and objectives
    # ------------------------------------------------------------------------------------------------------------------
    # Convergence plot
    error_MP_SP = np.abs(df_objectives['MP'].values - df_objectives['SP'].values)
    error_SP = np.abs(df_objectives['SP'].values - df_objectives['SP_primal'].values)

    plt.figure(figsize = (16,9))
    plt.plot(error_MP_SP, marker=10, markersize=10, linewidth=2, label='|MP - SP dual| $')
    plt.plot(error_SP, marker=11, markersize=10, linewidth=2, label='|SP primal - SP dual| $')
    plt.plot(100 * np.asarray(conv_inf['mipgap']), label='SP Dual mipgap %')
    plt.xlabel('Iteration $j$', fontsize=FONTSIZE)
    plt.ylabel('Gap', fontsize=FONTSIZE)
    # plt.ylim(-1, 10)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.tight_layout()
    plt.savefig(dirname + 'error_conv_' + pdfname + '.pdf')
    plt.savefig(dirname + '_error_conv.png', dpi=300)
    plt.close('all')

    print('')
    print('-----------------------CHECK COLUMN AND CONSTRAINT GENERATION CONVERGENCE-----------------------')
    print('Final iteration %s MP %s |MP - SP dual| %.2f $' % (len(df_objectives),
    df_objectives['MP'].values[-1]/4, abs(df_objectives['MP'].values[-1]/4 - df_objectives['SP'].values[-1]/4)))
    print('SP primal %.2f $ SP dual %.2f $ -> |SP primal - SP dual| = %.2f' % (
    SP_primal_sol['obj']/4, SP_dual_sol['obj']/4, abs(SP_primal_sol['obj']/4 - SP_dual_sol['obj']/4)))
    err_planner_CCG = abs(df_objectives['MP'].values[-1] - df_objectives['SP'].values[-1])
    # print('MILP planner %.2f $ MP CCG %.2f $ -> |MILP planner - MP CCG| = %.2f' % (
    # sol_planner['obj'], df_objectives['MP'].values[-1], err_planner_CCG))

    if err_planner_CCG > conv_tol:
        print('-----------------------WARNING COLUMN AND CONSTRAINT GENERATION IS NOT CONVERGED-----------------------')
        print('abs error %.4f $' % (err_planner_CCG))
    else:
        print('-----------------------COLUMN AND CONSTRAINT GENERATION IS CONVERGED-----------------------')
        print('CCG is converged with |MILP planner - MP CCG| = %.4f $' % (err_planner_CCG))

    # print('cost x fuel1 MP:', sum(final_x_cost_fuel_1), 'vs SP:', sum(SP_primal_sol['x_cost_fuel_PWL_1']))
    # print('cost x fuel1 res MP:', sum(final_x_cost_fuel_res_1), 'vs SP:', sum(SP_primal_sol['x_cost_fuel_res_1']))
    # print('cost x OM ES MP:', sum(final_x_cost_OM_ES), 'vs SP:', sum(SP_primal_sol['x_cost_OM_ES']))
    # print('cost x curt PV MP:', sum(final_x_cost_curt_PV), 'vs SP:', sum(SP_primal_sol['x_cost_curt_PV_PWL']))
    # print('cost x curt WT MP:', sum(final_x_cost_curt_WT), 'vs SP:', sum(SP_primal_sol['x_cost_curt_WT_PWL']))
    # print('cost y fuel MP:', sum(final_y_cost_fuel_1), 'vs SP:', sum(SP_primal_sol['y_cost_fuel_1']))
    # print('cost y OM ES MP:', sum(final_y_cost_OM_ES), 'vs SP:', sum(SP_primal_sol['y_cost_OM_ES']))
    # print('cost y PV curt MP:', sum(final_y_cost_curt_PV), 'vs SP:', sum(SP_primal_sol['y_cost_curt_PV']))
    # print('cost y WT curt MP:', sum(final_y_cost_curt_WT), 'vs SP:', sum(SP_primal_sol['y_cost_curt_WT']))
    # print('cost MP:', df_objectives['MP'].values[-1], 'vs SP:', SP_primal_sol['obj'])

    plt.figure(figsize=(16,9))
    plt.plot(WT_forecast, color='royalblue', marker="s", markersize=6, zorder=2, linewidth=4, label='$ w^{*}_{t} $')
    plt.plot(PV_forecast, color='green', marker="D", markersize=6, zorder=3, linewidth=4, label='$ v^{*}_{t} $')
    plt.plot(load_forecast, color='darkgoldenrod', marker='o', markersize=6, zorder=1, linewidth=4,  label='$ l^{*}_{t} $')
    plt.xlabel('Time (h)', fontsize=FONTSIZE)
    plt.ylabel('Power (kW)', fontsize=FONTSIZE)
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'], fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.tight_layout()
    plt.savefig(dirname + day + '_Forecast_' + pdfname + '.pdf')
    plt.savefig(dirname + '_Forecast', dpi=300)
    plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot(PV_worst_case, color='crimson', marker="o", markersize=8, linewidth=4, zorder=3, label='$ \hat{v}_t$')
    plt.plot(PV_forecast, 'steelblue', linestyle='solid', marker="s", markersize=8, linewidth=4, label='$ v^{*}_{t} $', zorder=1)
    plt.plot(PV_ub, 'dimgrey', linestyle=(0, (5, 5)), linewidth=2, label="$ v^{*}_{t} + v^{100+}%+} $", zorder=1)
    plt.plot(data.PV_pred + data.PV_pos_50, color='dimgrey', linestyle=(0, (3, 10, 1, 10)), linewidth=2, label="$ v^{*}_{t} + v_{t}^{50+}%+} $")
    plt.plot(data.PV_pred + data.PV_pos_20, color='dimgrey', linestyle=(0, (3, 5, 1, 5)), linewidth=2, label="$ v^{*}_{t} + v_{t}^{20+}%+} $")
    plt.plot(PV_lb, 'dimgrey', linestyle=(0, (5, 10)), linewidth=2, label="$ v^{*}_{t} - v^{100-}%-} $", zorder=1)
    plt.plot(data.PV_pred - data.PV_neg_50, color='dimgrey', linestyle=(3, (3, 5, 1, 5, 1, 5)), linewidth=2, label="$ v^{*}_{t} - v_{t}^{50-}%-} $")
    plt.plot(data.PV_pred - data.PV_neg_20, color='dimgrey', linestyle=(0, (3, 10, 1, 10, 1, 10)), linewidth=2, label="$ v^{*}_{t} - v_{t}^{20-}%-} $")    
    plt.xlabel('Time (h)', fontsize=FONTSIZE)
    plt.ylabel('Power (kW)', fontsize=FONTSIZE)
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'], fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.tight_layout()
    plt.savefig(dirname + day + '_PV_trajectory_' + pdfname + '.pdf')
    plt.savefig(dirname + '_PV_trajectory', dpi=300)
    plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot(WT_worst_case, color='crimson', linestyle='solid', marker="o", markersize=8, linewidth=4, label='$ \hat{w}_t$')
    plt.plot(WT_forecast, 'steelblue', linestyle='solid', marker="s", markersize=8, linewidth=4, label='$ w^{*}_{t} $', zorder=1)
    plt.plot(WT_ub, 'dimgrey', linestyle=(0, (5, 5)), linewidth=2, label="$ w^{*}_{t} + w^{100+}%+} $", zorder=1)
    plt.plot(data.WT_pred + data.WT_pos_50, color='dimgrey', linestyle=(0, (3, 10, 1, 10)), linewidth=2, label="$ w^{*}_{t} + w_{t}^{50+}%+} $")
    plt.plot(data.WT_pred + data.WT_pos_20, color='dimgrey', linestyle=(0, (3, 5, 1, 5)), linewidth=2, label="$ w^{*}_{t} + w_{t}^{20+}%+} $")
    plt.plot(WT_lb, 'dimgrey', linestyle=(0, (5, 10)), linewidth=2, label="$ w^{*}_{t} - w^{100-}%-} $", zorder=1)
    plt.plot(data.WT_pred - data.WT_neg_50, color='dimgrey', linestyle=(3, (3, 5, 1, 5, 1, 5)), linewidth=2, label="$ w^{*}_{t} - w_{t}^{50-}%-} $")
    plt.plot(data.WT_pred - data.WT_neg_20, color='dimgrey', linestyle=(0, (3, 10, 1, 10, 1, 10)), linewidth=2, label="$ w^{*}_{t} - w_{t}^{20-}%-} $")    	
    plt.xlabel('Time (h)', fontsize=FONTSIZE)
    plt.ylabel('Power (kW)', fontsize=FONTSIZE)
    plt.ylim(101,299)
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'], fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.tight_layout()
    plt.savefig(dirname + day + '_WT_trajectory_' + pdfname + '.pdf')
    plt.savefig(dirname + '_WT_trajectory', dpi=300)
    plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot(load_worst_case, color='crimson', linestyle='solid', marker="o", markersize=8, linewidth=4, label='$ \hat{l}_t$')
    plt.plot(load_forecast, 'orange', linestyle='solid', marker="d", markersize=8, linewidth=4, label='$ l^{*}_{t} $', zorder=1)
    plt.plot(load_ub, 'dimgrey', linestyle=(0, (5, 5)), linewidth=2, label="$ l^{*}_{t} + l^{100+}%+} $", zorder=1)
    plt.plot(data.load_egg + data.load_pos_50, color='dimgrey', linestyle=(0, (3, 10, 1, 10)), linewidth=2, label="$ l^{*}_{t} + l_{t}^{50+}%+} $")
    plt.plot(data.load_egg + data.load_pos_20, color='dimgrey', linestyle=(0, (3, 5, 1, 5)), linewidth=2, label="$ l^{*}_{t} + l_{t}^{20+}%+} $")
    plt.plot(load_lb, 'dimgrey', linestyle=(0, (5, 10)), linewidth=2, label="$ l^{*}_{t} - l^{100-}%-} $", zorder=1)
    plt.plot(data.load_egg - data.load_neg_50, color='dimgrey', linestyle=(3, (3, 5, 1, 5, 1, 5)), linewidth=2, label="$ l^{*}_{t} - l_{t}^{50-}%-} $")
    plt.plot(data.load_egg - data.load_neg_20, color='dimgrey', linestyle=(0, (3, 10, 1, 10, 1, 10)), linewidth=2, label="$ l^{*}_{t} - l_{t}^{20-}%-} $")    	
    plt.xlabel('Time (h)', fontsize=FONTSIZE)
    plt.ylabel('Power (kW)', fontsize=FONTSIZE)
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'], fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.tight_layout()
    plt.savefig(dirname + day + '_load_trajectory_' + pdfname + '.pdf')
    plt.savefig(dirname + '_load_trajectory', dpi=300)
    plt.close('all')

    a = SP_primal_sol['y_chg']
    b = SP_primal_sol['y_dis']
    c = np.zeros(95)
    c[0] = 187.5
    c[94] = 187.5
    for i in range(1,95):
        c[i] = c[i-1] + (a[i] * 0.93 - b[i] / 0.93)/4
    plt.figure(figsize=(16,9))
    plt.plot()
    plt.plot(c, linewidth=2, label='SOC')
    plt.xlabel('Time (h)', fontsize=FONTSIZE)
    plt.ylabel('SOC (kWh)', fontsize=FONTSIZE)
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'], fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)
    plt.tight_layout()
    plt.savefig(dirname + day + '_SOC_' + pdfname + '.pdf')
    plt.savefig(dirname + '_SOC', dpi=300)
    plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot(final_DE1_p, color='firebrick', marker='o', markersize=6, zorder=1, linewidth=3, label='DE output')
    # plt.plot(([hs + eg for hs, eg in zip(final_DE1_p, final_DE1_rp)]), marker='^', markersize=1, zorder=3, linewidth=2, alpha=0.5, label='DE reserve up')
    # plt.plot(([hs - eg for hs, eg in zip(final_DE1_p, final_DE1_rn)]), marker='v', markersize=1, zorder=3, linewidth=2, alpha=0.5, label='DE reserve down')
    plt.plot(WT_forecast, color='steelblue', alpha=0.8, linestyle="-.", markersize=6, zorder=3, linewidth=3, label='WT predicted')
    plt.plot(PV_forecast, color='forestgreen', alpha=0.8, linestyle="--", markersize=6, zorder=3, linewidth=3, label='PV predicted')
    plt.plot(([hs - eg for hs, eg in zip(WT_forecast, final_x_curt_WT)]), color='royalblue', marker="s", markersize=6, zorder=3, linewidth=3, label='WT output')
    plt.plot(([hs - eg for hs, eg in zip(PV_forecast, final_x_curt_PV)]), color='green', marker="D", markersize=6, zorder=3, linewidth=3, label='PV output')
    plt.plot(([hs - eg for hs, eg in zip(final_ES_discharge, final_ES_charge)]), color='gold', markersize=6, zorder=1, linewidth=3,  label='ESS')
    plt.plot(load_forecast, color='darkgoldenrod', marker='^', markersize=6, zorder=1, linewidth=3, label= 'Load demand')
    plt.ylim(-120, 599)
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'], fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel('Time (h)', fontsize=FONTSIZE)
    plt.ylabel('Power (kW)', fontsize=FONTSIZE)
    plt.legend(ncol=1, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8, fontsize=FONTSIZE)
    plt.savefig(dirname + day + '_Pre-dispatch_' + pdfname + '.pdf', dpi=300)
    plt.savefig(dirname + day + '_Pre-dispatch.png', dpi=300)
    plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot(final_DE1_p, color='firebrick', marker='o', markersize=6, zorder=1, linewidth=3, label='DE output')
    plt.plot(WT_worst_case, color='steelblue', alpha=0.8, linestyle="-.", markersize=6, zorder=3, linewidth=3, label='WT generation')
    plt.plot(PV_worst_case, color='forestgreen', alpha=0.8, linestyle="--", markersize=6, zorder=3, linewidth=3, label='PV generation')
    plt.plot(([hs - shs - eg + egg for hs, shs, eg, egg in zip(SP_primal_sol['y_WT'], final_x_curt_WT, SP_primal_sol['y_curt_WT'], SP_primal_sol['y_wc'])]), color='royalblue', marker="s", markersize=6, zorder=2, linewidth=3, label='WT output')
    plt.plot(([hs - shs - eg + egg for hs, shs, eg, egg in zip(SP_primal_sol['y_PV'], final_x_curt_PV, SP_primal_sol['y_curt_PV'], SP_primal_sol['y_pc'])]), color='green', marker="D", markersize=6, zorder=2, linewidth=3, label='PV output')
    plt.plot(([hs - eg for hs, eg in zip(SP_primal_sol['y_dis'], SP_primal_sol['y_chg'])]), color='gold', markersize=6, zorder=1, linewidth=3, label='ESS')
    plt.plot(load_worst_case, color='darkgoldenrod', marker='^', markersize=6, zorder=1, linewidth=3, label= 'Load demand')
    plt.ylim(-120, 599)
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'], fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel('Time (h)', fontsize=FONTSIZE)
    plt.ylabel('Power (kW)', fontsize=FONTSIZE)
    plt.legend(ncol=1, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8, fontsize=FONTSIZE)
    plt.savefig(dirname + day + '_Result_' + pdfname + '.pdf', dpi=300)
    plt.savefig(dirname + day + '_Result.png', dpi=300)
    plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot()
    plt.plot(SP_primal_sol['y_chg'], linewidth=2, label='real-time charge')
    plt.plot(SP_primal_sol['y_dis'], linewidth=2, label='real-time discharge')
    # plt.ylim(0, PARAMETERS['ES']['capacity'])
    plt.ylabel('kW', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname + day + 'realtime_charge_discharge_' + pdfname + '.pdf')
    plt.close('all')

    plt.figure(figsize=(16,9))
    plt.plot()
    plt.plot(SP_primal_sol['y_pc'], linewidth=2, label='PV curtail cancel')
    plt.plot(SP_primal_sol['y_wc'], linewidth=2, label='WT curtail cancel')
    plt.ylabel('kW', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname + day + 'realtime_curtail_cancle_' + pdfname + '.pdf')
    plt.close('all')

    # a = np.array(final_cost_curt)
    # b = np.array(SP_primal_sol['x_cost_curt_PWL'])
    # fmt = '%.18e'
    # data = np.column_stack((a, b.flatten()))
    # np.savetxt('pre-curtailment.csv', data, delimiter=',', header='MP, SP', comments='', fmt='%.2f')
    PV_worst = np.array(SP_primal_sol['y_PV'])
    WT_worst = np.array(SP_primal_sol['y_WT'])
    load_worst = np.array(SP_primal_sol['y_load'])
    fmt = '%.18e'
    data = np.column_stack((PV_worst, WT_worst, load_worst.flatten()))
    np.savetxt('worst.csv', data, delimiter=',', header='PV_worst, WT_worst, load_worst', comments='', fmt='%.2f')

    phi_WT_value = np.array(SP_dual_sol['phi_WT'])
    phi_PV_value = np.array(SP_dual_sol['phi_PV'])
    phi_load_value = np.array(SP_dual_sol['phi_load'])
    phi_curt_WT_value = np.array(SP_dual_sol['phi_curt_WT'])
    phi_curt_PV_value = np.array(SP_dual_sol['phi_curt_PV'])
    phi_data = np.column_stack((phi_PV_value, phi_WT_value, phi_load_value, phi_curt_PV_value, phi_curt_WT_value.flatten()))
    np.savetxt('phi.csv', phi_data, delimiter=',', header='phi_WT,phi_PV,phi_load,phi_curt_WT,phi_curt_PV', comments='', fmt='%.2f')

    # Load data from CSV files
    phi_bM = pd.read_csv('phi_bM.csv', header=0)  # Assuming header is in the first row
    phi_Mc = pd.read_csv('phi.csv', header=0)

    # Define LaTeX-style labels
    labels = {
        'phi_WT': r'$\phi^{\mathrm{WT}}_t$',
        'phi_PV': r'$\phi^{\mathrm{PV}}_t$',
        'phi_load': r'$\phi^{\mathrm{load}}_t$',
        'phi_curt_WT': r'$\phi^{\mathrm{WT-}}_t$',
        'phi_curt_PV': r'$\phi^{\mathrm{PV-}}_t$'
    }

    # Plot the data
    fig, axs = plt.subplots(len(labels), 1, figsize=(16, len(labels) * 3), sharex=True)

    for i, (col, label) in enumerate(labels.items()):
        axs[i].plot(phi_bM[col], label=f'{label} (big-M)', linestyle='-', linewidth=4)
        axs[i].plot(phi_Mc[col], label=f'{label} (McCormick)', linestyle='--', linewidth=4)
        axs[i].set_ylabel(f'{label} values', fontsize=FONTSIZE)
        axs[i].legend(ncol=1, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8, fontsize=FONTSIZE)
        axs[i].tick_params(axis='y', labelsize=FONTSIZE)  # Set y-axis tick font size

    plt.xlabel('Time (h)', fontsize=FONTSIZE)
    plt.xticks([0, 16, 32, 48, 64, 80, 96],['0','4','8','12','16','20','24'], fontsize=FONTSIZE)
    plt.tight_layout()
    plt.savefig(dirname + day + '_Phi_' + pdfname + '.pdf', dpi=300)
    plt.savefig(dirname + day + '_Phi.png', dpi=300)
    plt.close('all')