"""
Utils file containing several functions to be used.
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
import sympy as sp

# from sklearn.preprocessing import StandardScaler
from Params import PARAMETERS

def check_ES(SP_primal_sol: dict):
    """
    Check if there is any simultanenaous charge and discharge of the ES.
    :param SP_primal_sol: solution of the SP primal, dict with at least the keys arguments y_chg and y_chg.
    :return: number of simultanenaous charge and discharge.
    """
    df_check = pd.DataFrame(SP_primal_sol['y_chg'], columns=['y_chg'])
    df_check['y_dis'] = SP_primal_sol['y_dis']

    nb_count = 0
    for i in df_check.index:
        if (df_check.loc[i]['y_chg'] > 0) and (df_check.loc[i]['y_dis'] > 0):
            nb_count += 1
    return nb_count

def build_point_forecast(dir: str= '/Users/Andrew/OneDrive - GIST/Code/Graduation/'):
    """
    Load RG dad point forecasts of VS1 and VS2.
    :return: RG_solution, RG_dad_VS1, and RG_dad_VS2
    """

    k1 = 11 # 0 or 11
    k2 = 80 # 95 or 80
    pv_dad_VS = pd.read

def dump_file(dir: str, name: str, file):
    """
    Dump a file into a picke.
    """
    file_name = open(dir + name + '.pickle', 'wb')
    pickle.dump(file, file_name)
    file_name.close()

def read_file(dir: str, name: str):
    """
    Read a file dumped into a pickle.
    """
    file_name = open(dir + name + '.pickle', 'rb')
    file = pickle.load(file_name)
    file_name.close()

    return file

cost_a_1 = PARAMETERS['cost']['DE1_a']
cost_b_1 = PARAMETERS['cost']['DE1_b']
cost_c_1 = PARAMETERS['cost']['DE1_c']
cost_m_pre_PV = PARAMETERS['cost']['PV_m_cut_pre']
cost_m_re_PV = PARAMETERS['cost']['PV_m_cut_re']
cost_m_pre_WT = PARAMETERS['cost']['WT_m_cut_pre']
cost_m_re_WT = PARAMETERS['cost']['WT_m_cut_re']

def FC1(p):
    return(cost_a_1 * p * p + cost_b_1 * p + cost_c_1)

def PC_PV(g):
    return cost_m_pre_PV * g * g

def PC_WT(g):
    return cost_m_pre_WT * g * g

def RC_PV(g):
    return cost_m_re_PV * g * g

def RC_WT(g):
    return cost_m_re_WT * g * g

# def DRC(g):
#     return - 1 / (4 * cost_m_re_PV) * g * g

def PWL(PWL_num, lb, ub, egg):
    x = []
    y = []
    for i in range(PWL_num + 1):
        x.append(lb + (ub - lb) * i / PWL_num)
        y.append(egg(x[i]))
    return x, y

def PWL_val(PWL_num, lb, ub, egg, x):
    interval = (ub - lb) / PWL_num
    y = 0
    for i in range(PWL_num):
        if PWL(PWL_num, lb, ub, egg)[0][i] <= x < PWL(PWL_num, lb, ub, egg)[0][i + 1]:
            y = (PWL(PWL_num, lb, ub, egg)[1][i + 1] - PWL(PWL_num, lb, ub, egg)[1][i]) / interval * (x - PWL(PWL_num, lb, ub, egg)[0][i]) + PWL(PWL_num, lb, ub, egg)[1][i]
        else:
            pass
    return y


# def piecewise_linearization(a, b, c, min_x, max_x, num, x):
#     intervals = (max_x - min_x) / num
#     for i in range(num + 1):
#         x_point[i] = min_x + intervals * i
#         y_point[i] = a * x_point[i] * x_point[i] + b * x_point[i] + c
    
#     for i in range(10):
#         if x_point[i] <= x < x_point[i+1]:
#             y = (y_point[i+1] - y_point[i]) / (x_point[i+1] - x_point[i]) * (x - x_point[i])
