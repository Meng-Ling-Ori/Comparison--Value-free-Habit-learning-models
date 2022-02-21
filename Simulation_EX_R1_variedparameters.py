# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 21:24:31 2022

@author: orian
"""

import numpy as np
import scipy as sc
import analysis_plot as anls
import run_simulations as rs
import misc_mbp as mim
import misc_baye as mib

#%%
"""
simulate the Bayesian model
"""

# =============================================================================
# define parameters
# =============================================================================
T = 2 #number of time steps in each trial
nb = 2
no = nb+1 #number of observations
ns = nb+1 #number of states
na = nb #number of actions
nr = nb+1
nc = nb
alpha_policies = np.array([1, 100]) # habitual tendency h
alpha_context = [99]
alpha_rewards = [90]

u = 0.99
preference = np.zeros(nr)
for i in range(1,nr):
    preference[i] = u/(nr-1)
preference[0] = (1.-u)
preference_deval=[]

repetitions = 50
avg = True

n_test = 100 # extinction phase trials number
trials = [200] # sum of all the trials

alpha_policies_series = np.linspace(1/alpha_policies[0], 1/alpha_policies[1], 50)
alpha_policies_series = 1/alpha_policies_series

alpha_list = [alpha_policies_series, alpha_context, alpha_rewards]

# =============================================================================
# simulate
# =============================================================================
worlds_reversal_1_test_baye = rs.simulation_reversal_baye(repetitions, preference, 
                            preference_deval, avg, T, ns, na, nr, 
                             nc, alpha_list, n_test, trials)

# =============================================================================
# plot 
# =============================================================================
 # plot habit strenth
data_a_test_baye = anls.plot_habit_strength(worlds_reversal_1_test_baye, repetitions=50, 
                                            habit_learning_strength=alpha_policies_series, series=True,  baye=True)

#%%
"""
simulate the MB/VF model
"""

# =============================================================================
# define parameters
# =============================================================================
alpha_Hs = [0.009, 0.0001]#[0.013, 0.001]#[0.03, 0.008]#[0.0055, 0.00005]#[0.001]
alpha_R = 0.09#0.05#0.25#0.35 #0.01
w_0 = 1#0.8 #1
w_g = 5 #10
w_h = 5 #10
theta_h = 5#2.5 #5
theta_g = 5#2.5 #5
alpha_w = 1 #1

na=2 #number of actions
nm=2 #number of rewards/reinforcers
alpha_reward = [0.9] #probability of rewards

trials_phase1s = [100] #training trials
trials_phase2 = 100 #extinction trials

u = 0.99
preference = np.zeros(nr)
for i in range(1,nr):
    preference[i] = u/(nr-1)
preference[0] = (1.-u)
U = mim.softmax_reversal(preference)
U_deval = []

repetitions = 50

alpha_Hs_series =  np.linspace(alpha_Hs[0], alpha_Hs[1], 50)
paras = alpha_Hs_series, alpha_R, w_0, w_g, w_h, theta_g, theta_h, alpha_w

# =============================================================================
# simulate
# =============================================================================
worlds_reversal_1_test_mbp = rs.simulation_mbp(na, nm, paras, alpha_reward, 
                                          repetitions,trials_phase1s, trials_phase2, U, U_deval)
# =============================================================================
# plot 
# =============================================================================
 # plot habit strenth
data_a_test_mbp = anls.plot_habit_strength(worlds_reversal_1_test_mbp, repetitions, 
                                           habit_learning_strength=alpha_Hs_series, series=True)  





"""
data analysis
"""
print('\nData Analysis\n')
# =============================================================================
# Linear regression h or alpha_H and habit strength
# =============================================================================

# baye
print('Linear regression (h and habit strength) baye:')
data1 = np.repeat(1/alpha_policies_series,50)
data2 = data_a_test_baye[0].T.flatten()
data1, data2 = mim.remove_nan(data1, data2)
reg_baye = mim.lin_regress(data1, data2) 

# mbp
print('\nLinear regression (alpha_H and habit strength) mbp:')
data1 = np.repeat(alpha_Hs_series,50)
data2 = data_a_test_mbp[0].T.flatten()
data1, data2 = mim.remove_nan(data1, data2)
reg_mbp = mim.lin_regress(data1, data2)
#%%

# =============================================================================
# Test a significant difference between two slope values
# =============================================================================

b1 = reg_baye[0]
b2 = reg_mbp[0]

se1 = reg_baye[-1]
se2 = reg_mbp[-1]

z_score = (b1-b2) / (((se1*(b1**2) + se2*(b2**2)))**0.5)

print('z_score: ' +str(z_score))

#%%
"""
save data
"""

file_name = ['worlds_EXR1_baye_paratest.pkl', 'worlds_EXR1_mbp_paratest.pkl']
data = [worlds_reversal_1_test_baye, worlds_reversal_1_test_mbp]

file_path_0 = 'E:/CAN/thesis/Habit learning models/data/'
for i in range(len(data)):
    file_path = file_path_0 + file_name[i]
    mim.save_load_data(data[i], file_path, mode = 'wb')



#%%
"""
load data
"""
import misc_mbp as mim

file_name = ['worlds_EXR1_baye_paratest.pkl', 'worlds_EXR1_mbp_paratest.pkl']

file_path_0 = 'E:/CAN/thesis/Habit learning models/data/'
worlds = []
for i in range(len(file_name)):
    file_path = file_path_0 + file_name[i]
    worlds.append(mim.save_load_data(file_path=file_path, mode = 'rb'))

worlds_reversal_1_test_baye, worlds_reversal_1_test_mbp= worlds
