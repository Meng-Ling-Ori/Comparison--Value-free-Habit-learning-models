# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 21:44:18 2022

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
alpha_list = [alpha_policies, alpha_context, alpha_rewards]

u = 0.99
preference = np.zeros(nr)
for i in range(1,nr):
    preference[i] = u/(nr-1)
preference[0] = (1.-u)




ut = preference[1:].sum()
preference_deval=np.zeros(nr)
preference_deval[2:] = ut / (nr-2)
preference_deval[:2] = (1-ut) / 2

repetitions = 50
avg = True

n_test = 100 # extinction phase trials number
trials = [200] # sum of all the trials

#%%
# =============================================================================
# simulate
# =============================================================================
worlds_deval_baye = rs.simulation_reversal_baye(repetitions,preference, 
                            preference_deval, avg, T, ns, na, nr, 
                             nc, alpha_list, n_test, trials, 
                             deval_1 = True)

# =============================================================================
# plot 
# =============================================================================
 # plot habit strenth
data_a_deval_baye = anls.plot_habit_strength(worlds_deval_baye, repetitions=50, 
                                             habit_learning_strength=alpha_policies, baye=True)



#%%
"""
simulate the MB/VF model
"""

# =============================================================================
# define parameters
# =============================================================================
alpha_Hs = [0.009, 0.0001]#[0.02, 0.008]#[0.0055, 0.00005]#[0.001]
alpha_R = 0.09#0.25#0.35 #0.01
w_0 = 1 #1
w_g = 5 #10
w_h = 5 #10
theta_h = 5 #2.5 #5
theta_g = 5 #2.5 #5
alpha_w = 1 #1
paras = alpha_Hs, alpha_R, w_0, w_g, w_h, theta_g, theta_h, alpha_w

na=2 #number of actions
nm=2 #number of rewards/reinforcers
alpha_reward = [0.9] #probability of rewards

trials_phase1s = [100] #training trials
trials_phase2 = 100 #extinction trials

    
repetitions = 50
u = 0.99
preference = np.zeros(nr)
for i in range(1,nr):
    preference[i] = u/(nr-1)
preference[0] = (1.-u)
U = mim.softmax_reversal(preference)

ut = preference[1:].sum()
preference_deval=np.zeros(nr)
preference_deval[2:] = ut / (nr-2)
preference_deval[:2] = (1-ut) / 2

U_deval = mim.softmax_reversal(preference_deval)
#%%
U_deval = U.copy()
U_deval[1] = U[0]

# =============================================================================
# simulate
# =============================================================================
worlds_deval_mbp = rs.simulation_mbp(na, nm, paras, alpha_reward, repetitions,
                                     trials_phase1s, trials_phase2, U, U_deval,
                                     deval_1=True)

# =============================================================================
# plot 
# =============================================================================
 # plot habit strenth
data_a_deval_mbp = anls.plot_habit_strength(worlds_deval_mbp, repetitions=50, 
                                            habit_learning_strength=alpha_Hs) 

anls.plot_chosen_probability_retrieval(worlds_deval_mbp, repetitions=50, 
                                       habit_learning_strength=alpha_Hs)


#%%
"""
data analysis
"""
print('\nData Analysis\n')
# =============================================================================
# t-test strong vs weak
# =============================================================================

# baye
print('t-test habit strength (strong vs weak) baye:')
data1, data2 = mim.remove_nan(data_a_deval_baye[0][:,0],data_a_deval_baye[0][:,1])
mim.sample_two_t_test(data1, data2)

# mbp
print('t-test habit strength (strong vs weak) mbp:')
data1, data2 = mim.remove_nan(data_a_deval_mbp[0][:,0],data_a_deval_mbp[0][:,1])
mim.sample_two_t_test(data1, data2)

#%%
"""
save data
"""

file_name = ['worlds_EXD1_baye.pkl', 'worlds_EXD1_mbp.pkl']
data = [worlds_deval_baye, worlds_deval_mbp]

file_path_0 = 'E:/CAN/thesis/Habit learning models/data/'
for i in range(len(data)):
    file_path = file_path_0 + file_name[i]
    mim.save_load_data(data[i], file_path, mode = 'wb')



#%%
"""
load data
"""
import misc_mbp as mim

file_name = ['worlds_EXD1_baye.pkl', 'worlds_EXD1_mbp.pkl']

file_path_0 = 'E:/CAN/thesis/Habit learning models/data/'
worlds = []
for i in range(len(file_name)):
    file_path = file_path_0 + file_name[i]
    worlds.append(mim.save_load_data(file_path=file_path, mode = 'rb'))

worlds_deval_baye, worlds_deval_mbp= worlds

