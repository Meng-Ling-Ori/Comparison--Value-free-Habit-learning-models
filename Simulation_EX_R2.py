# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 21:40:44 2022

@author: orian
"""

import numpy as np
import scipy as sc
import analysis_plot as anls
import run_simulations as rs
import misc_mbp as mim
import misc_baye as mib

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
repetitions = 50
avg = True

alpha_policies = [8, 100] 
alpha_context = [99]
alpha_rewards = [50] #90
alpha_list = [alpha_policies, alpha_context, alpha_rewards]
utility= np.array([0.0, 1.0, 1.0]) #no-rewards/rewards(pellets) after actionA/rewards(pellets) after actionB/ rewards(pellets+leisure)/rewards(leisure)
#utility=mib.softmax(5*utility)
preference=mib.softmax(utility)
preference_deval = []

trials = [2000] #training trials
n_test = 1000 #extinction trials
alpha_list = [alpha_policies, alpha_context, alpha_rewards]
#%%
# =============================================================================
# simulate
# =============================================================================
worlds_reversal_2_baye =  rs.simulation_reversal_baye(repetitions, preference, preference_deval, avg, T, ns, na, nr, 
                             nc, alpha_list, n_test, trials, EX1_2 = True)
#%%
# =============================================================================
# plot 
# =============================================================================

 # plot chosen probability 
anls.plot_chosen_probability(worlds_reversal_2_baye[:50], repetitions=50, 
                             habit_learning_strength=alpha_policies[0], weak=False, baye=True, EX1_2 = True)
anls.plot_chosen_probability(worlds_reversal_2_baye[50:], repetitions=50, 
                             habit_learning_strength=alpha_policies[1], weak=True, baye=True, EX1_2 = True)

 # plot habit strenth
data_a_2_baye = anls.plot_habit_strength(worlds_reversal_2_baye, repetitions=50, 
                                         habit_learning_strength=alpha_policies, EX1_2=True, baye=True)
# =============================================================================
# data analysis 
# =============================================================================
 # t-text strong vs weak
data1, data2 = mim.remove_nan(data_a_2_baye[0][:,0],data_a_2_baye[0][:,1])
mim.sample_two_t_test(data1, data2)

#%%
"""
simulate the MB/VF model
"""

# =============================================================================
# define parameters
# =============================================================================
na=2 #number of actions
nm=2 #number of rewards/reinforcers
repetitions = 50

alpha_Hs = [0.001, 0.0005]
alpha_R = 0.01
alpha_stoch = []
w_0 = 1
w_g = 5
w_h = 5
theta_h = 5
theta_g = 5
alpha_w = 1
paras = alpha_Hs, alpha_R, w_0, w_g, w_h, theta_g, theta_h, alpha_w

alpha_reward = [0.5] #probability of rewards

trials_phase1s = [1000] #training trials
trials_phase2 = 1000 #5000 #extinction trials

U = np.array([0.0, 1.0, 1.0])
U_deval = []
repetitions = 50
#%%
# =============================================================================
# simulate
# =============================================================================
worlds_reversal_2_mbp = rs.simulation_mbp(na, nm, paras, alpha_reward, 
                                          repetitions,
                                          trials_phase1s, trials_phase2, U, U_deval,
                                          EX1_2 = True)
#%%
# =============================================================================
# plot 
# =============================================================================

 # replication plot dynamic changes of key variables
anls.plot_variables_dynamic_mbp(worlds_reversal_2_mbp[:50], repetitions) 

 # plot chosen probability 
anls.plot_chosen_probability(worlds_reversal_2_mbp[:50], repetitions=50, 
                             habit_learning_strength=alpha_Hs[0], weak=False, baye=False, EX1_2 = True)
anls.plot_chosen_probability(worlds_reversal_2_mbp[50:], repetitions=50, 
                             habit_learning_strength=alpha_Hs[1], weak=True, baye=False, EX1_2 = True)

 # plot habit strenth
data_a_2_mbp = anls.plot_habit_strength(worlds_reversal_2_mbp, 
                                        repetitions=50, habit_learning_strength=alpha_Hs, EX1_2=True) 

# =============================================================================
# data analysis 
# =============================================================================
 # t-text strong vs weak
data1, data2 = mim.remove_nan(data_a_2_mbp[0][:,0],data_a_2_mbp[0][:,1])
mim.sample_two_t_test(data1, data2)

#%%
"""
save data
"""

file_name = ['worlds_EXR2_baye.pkl', 'worlds_EXR2_mbp.pkl']
data = [worlds_reversal_2_baye, worlds_reversal_2_mbp]

file_path_0 = 'E:/CAN/thesis/Habit learning models/data/'
for i in range(len(data)):
    file_path = file_path_0 + file_name[i]
    mim.save_load_data(data[i], file_path, mode = 'wb')



#%%
"""
load data
"""
import misc_mbp as mim

file_name = ['worlds_EXR2_baye.pkl', 'worlds_EXR2_mbp.pkl']

file_path_0 = 'E:/CAN/thesis/Habit learning models/data/'
worlds = []
for i in range(len(file_name)):
    file_path = file_path_0 + file_name[i]
    worlds.append(mim.save_load_data(file_path=file_path, mode = 'rb'))

worlds_reversal_2_baye,  worlds_reversal_2_mbp= worlds