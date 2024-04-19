import os
import pandas as pd
import numpy as np
import random
from PV_model import PV_forecast
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
np.set_printoptions(precision=6, suppress=True)


# os.listdir: 전체 파일목록 가져오기, endswith(): 특정 확장자 파일인지 확인
file_list = [i for i in os.listdir('./') if i.endswith('.xls')]

for i in file_list:
	SMP = pd.read_excel('{}'.format(i), sheet_name='Sheet1')
    # 행 날리기
	SMP = SMP.drop(index=[0, 1, 26, 27, 28], axis=0)
    # label 열 날리기
	SMP = SMP.drop(labels=['육지 SMP 목록'], axis=1)
    # 7.11(월) SMP 사용 위해 나머지 날리기
	SMP = SMP.drop(labels=['Unnamed: {}'.format(i) for i in range(1, 7)], axis=1)
    # 열을 행 모습으로 Transpose
	SMP = np.array(SMP).flatten()
    # np.tile(A, reps=(a,b)): A를 행으로 a번 열로 b번 반복 쌓기
	SMP = np.tile(SMP, reps=1)
    # 24시간을 15분 단위로 쪼개서 쌓기 
	SMP = np.repeat(SMP, repeats=4, axis=0)/4

# kpx 데이터 불러오기, 클래스 내의 지역 변수들
class data_read:
	def __init__(self):
		self.Sim_time = 96
		self.N_PWL = 10
		self.RTE = 0.93

		self.kpx_PV_data = pd.read_csv('KPX_PV.csv', sep=',', names=['Source', 'Location', 'Date', 'Hour', 'Power'], dtype={'Date': str, 'Hour': str, 'Power': str}, encoding='CP949')[1:] # dtype = DataFrame
		self.kpx_PV_data = pd.DataFrame(self.kpx_PV_data, columns=['Hour', 'Power']).to_numpy(dtype=np.float32)

		self.kpx_WT_data = pd.read_csv('KPX_WT.csv', sep=',', names=['Date', 'Hour', 'Location', 'Power'], encoding='CP949')[1:]
		self.kpx_WT_data = pd.DataFrame(self.kpx_WT_data, columns=['Hour', 'Power']).to_numpy(dtype=np.float32)

		self.kpx_load = pd.read_csv('KPX_Load.csv', sep=',', names=['Date', 'Load_1', 'Load_2', 'Load_3', 'Load_4', 'Load_5', 'Load_6', 'Load_7', 'Load_8', 'Load_9', 'Load_10', 'Load_11',
							      'Load_12', 'Load_13', 'Load_14', 'Load_15', 'Load_16', 'Load_17', 'Load_18', 'Load_19', 'Load_20', 'Load_21', 'Load_22', 'Load_23', 'Load_0'], encoding='CP949')[1:]
		self.kpx_load = self.kpx_load.drop(['Date'], axis=1).to_numpy(dtype=np.float32) / 1000

        # 각 시간대별로 데이터 모음(전처리)
		self.kpx_PV = []
		self.kpx_WT = []
		for i in range(24):
			self.kpx_PV.append(self.kpx_PV_data[self.kpx_PV_data[:, 0] == i, -1])
			self.kpx_WT.append(self.kpx_WT_data[self.kpx_WT_data[:, 0] == i, -1])

		self.PV_var = np.array([])
		self.WT_var = np.array([])
		self.load_var = np.array([])
		self.WT_std = np.array([])
		for i in range(24):
			self.PV_var = np.append(self.PV_var, np.nanvar(self.kpx_PV[i]/np.max(self.kpx_PV)))
			self.WT_var = np.append(self.WT_var, np.nanvar(self.kpx_WT[i]/np.max(self.kpx_WT)))
			self.load_var = np.append(self.load_var, np.nanvar(self.kpx_load[:, i - 1] / np.max(self.kpx_load)))
			self.WT_std = np.append(self.WT_std, np.nanstd(self.kpx_WT[i] / np.max(self.kpx_WT)))

		self.PV_pred = np.array(pd.read_csv('PV_for_scheduling.txt', names=['PV']), dtype=np.float32)[:,0]
		self.PV_pred[self.PV_pred < 0] = 0

		self.WT_pred = np.array(pd.read_csv('WT_for_scheduling.txt', names=['WT']), dtype=np.float32)[:,0]
		self.WT_pred[self.WT_pred < 0] = 0

		self.load_pred = np.array(pd.read_csv('Load_for_scheduling.txt', names=['Load']), dtype=np.float32)[:,0]
		self.load_pred[self.load_pred < 0] = 0

		df = pd.read_csv('egg_load.csv', sep=',', names=['Date', 'Total', 'Real'], dtype={'Date': str, 'Total': str, 'Real': str}, encoding='CP949')
		self.load_egg = df.loc[range(0, 286, 3)].reset_index(drop=True)['Total'].astype(float)
		self.load_egg = self.load_egg / np.mean(self.load_egg)
		self.load_egg = self.load_egg * np.mean(self.load_pred) * 1.3
		
		# self.data = np.array(pd.concat([pd.read_csv('PV_prediction.txt', names=['PV']), pd.read_csv('Load_for_scheduling.txt', names=['Load'])], axis=1), dtype=np.float32)
		# self.PV_Gen, self.load = (max(self.data[:, 0]), max(self.data[:, 1]))
        # 신뢰구간 95% , 표본 1개, 24h을 96time slot으로 쪼개기

		self.PV_p_100 = self.PV_pred * 1.96 * np.sqrt(np.repeat(self.PV_var, repeats=4, axis=0))/np.sqrt(1)
		self.PV_n_100 = self.PV_pred * 1.96 * np.sqrt(np.repeat(self.PV_var, repeats=4, axis=0))/np.sqrt(1)
		self.PV_p_50 = self.PV_p_100 * 0.5
		self.PV_n_50 = self.PV_n_100 * 0.5
		self.PV_p_20 = self.PV_p_100 * 0.2
		self.PV_n_20 = self.PV_n_100 * 0.2

		self.WT_p_100 = self.WT_pred * 1.96 * np.sqrt(np.repeat(self.WT_var, repeats=4, axis=0))/np.sqrt(1)
		self.WT_n_100 = self.WT_pred * 1.96 * np.sqrt(np.repeat(self.WT_var, repeats=4, axis=0))/np.sqrt(1)
		self.WT_p_50 = self.WT_p_100 * 0.5
		self.WT_n_50 = self.WT_n_100 * 0.5
		self.WT_p_20 = self.WT_p_100 * 0.2
		self.WT_n_20 = self.WT_n_100 * 0.2		

		self.load_p_100 = self.load_egg * 1.96 * np.sqrt(np.repeat(self.load_var, repeats=4, axis=0))/np.sqrt(1)
		self.load_n_100 = self.load_egg * 1.96 * np.sqrt(np.repeat(self.load_var, repeats=4, axis=0))/np.sqrt(1)
		self.load_p_50 = self.load_p_100 * 0.5
		self.load_n_50 = self.load_n_100 * 0.5
		self.load_p_20 = self.load_p_100 * 0.2
		self.load_n_20 = self.load_n_100 * 0.2

		self.PV_UB = self.PV_pred + self.PV_p_100
		self.PV_LB = self.PV_pred - self.PV_n_100
		self.WT_UB = self.WT_pred + self.WT_p_100
		self.WT_LB = self.WT_pred - self.WT_n_100
		self.load_UB = self.load_egg + self.load_p_100
		self.load_LB = self.load_egg - self.load_n_100

		# Initialize the PV_oracle array
		self.PV_oracle = np.zeros(self.Sim_time)
		self.load_oracle = np.zeros(self.Sim_time)

		# Set the initial value for PV_oracle as the first predicted value
		self.PV_oracle[0] = self.PV_pred[0] 
		self.load_oracle[0] = self.load_egg[0]

		def get_random_value(lower_bound, upper_bound):
			return random.uniform(lower_bound, upper_bound)
		
		self.trend_factor = []
		for i in range(self.Sim_time):
			self.trend = i / (self.Sim_time - 1)
			self.trend_factor.append(self.trend)

		# Calculate the error based on the time index and trend factor
		self.PV_error = self.PV_p_100 * self.trend_factor
		self.load_error = self.load_p_100 * self.trend_factor

		# Calculate the upper and lower limits for the current time index
		self.PV_upper_limit = self.PV_pred + self.PV_error
		self.PV_lower_limit = self.PV_pred - self.PV_error
		self.load_upper_limit = self.load_egg + self.load_error
		self.load_lower_limit = self.load_egg - self.load_error

		# Calculate the value for PV_oracle based on the previous time period's predicted value
		for i in range(1, self.Sim_time):
			if self.PV_oracle[i-1] < self.PV_pred[i]:
				self.PV_oracle[i] = get_random_value(lower_bound=self.PV_oracle[i-1], upper_bound=self.PV_upper_limit[i])
			else:
				self.PV_oracle[i] = get_random_value(lower_bound=self.PV_lower_limit[i], upper_bound=self.PV_oracle[i-1])
			
			if self.PV_oracle[i] >= self.PV_UB[i]:
				self.PV_oracle[i] = self.PV_UB[i]
			elif self.PV_oracle[i] <= self.PV_LB[i]:
				self.PV_oracle[i] = self.PV_LB[i]

		# Calculate the value for load_oracle based on the previous time period's predicted value
			if self.load_oracle[i-1] < self.load_egg[i]:
				self.load_oracle[i] = get_random_value(lower_bound=self.load_oracle[i-1], upper_bound=self.load_upper_limit[i])
			else:
				self.load_oracle[i] = get_random_value(lower_bound=self.load_lower_limit[i], upper_bound=self.load_oracle[i-1])

			if self.load_oracle[i] >= self.load_UB[i]:
				self.load_oracle[i] = self.load_UB[i]
			elif self.load_oracle[i] <= self.load_LB[i]:
				self.load_oracle[i] = self.load_LB[i]


		normal = norm.rvs(size=len(self.kpx_WT[11]), loc = self.WT_pred[47], scale = np.sqrt(self.WT_pos_100[47]))
		plt.figure(figsize= (16,9))
		sns.kdeplot(normal, color='brown', linewidth=4, label = 'WT PDF')
		lower_bound = self.WT_pred[47] - self.PV_n_50[47]
		upper_bound = self.WT_pred[47] + self.PV_n_50[47]
		kde_values = sns.kdeplot(normal, color='brown', linewidth=4).get_lines()[0].get_ydata()

# Shade the area between lower_bound and upper_bound
		plt.fill_betweenx(y=kde_values, x1=lower_bound, x2=upper_bound, color='g', alpha=0.5)
		plt.ylabel('f(x)', fontsize=24)
		plt.xlabel('Power (kW)', fontsize=24)
		plt.xticks(fontsize=24)
		plt.yticks(fontsize=24)
		plt.legend(ncol=1, loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.8, fontsize=24)
		plt.savefig('/Users/Andrew/OneDrive - GIST/Code/Graduation/PC_RGD_CCG_Mc_MIU/' + 'PDF.png', dpi=300)
		plt.close('all')
		print(upper_bound)

data = data_read()

