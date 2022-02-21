# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 22:10:37 2022

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
preference_deval=[]

repetitions = 50
avg = True

n_test = 100 # extinction phase trials number
trials = [200] # sum of all the trials
# =============================================================================
# simulate
# =============================================================================
worlds_retrieval_baye = rs.simulation_reversal_baye(repetitions, preference, 
                            preference_deval, avg, T, ns, na, nr, 
                             nc, alpha_list, n_test, trials, retrieval= True)
# =============================================================================
# plot 
# =============================================================================
 # plot chosen probability 
anls.plot_chosen_probability(worlds_retrieval_baye[:50], repetitions=50, 
                             habit_learning_strength=alpha_policies[0], weak=False, baye=True)
anls.plot_chosen_probability(worlds_retrieval_baye[50:], repetitions=50, 
                             habit_learning_strength=alpha_policies[1], weak=True, baye=True)


 # plot convergence time
 
data_t_retrieval_baye = anls.plot_convergence_time(worlds_retrieval_baye, repetitions=50, habit_learning_strength=alpha_policies, baye=True)



 # plot probabilities

data_p_retrieval_baye = anls.plot_chosen_probability_retrieval(worlds_retrieval_baye, repetitions=50, 
                                                               habit_learning_strength=alpha_policies, baye=True, re=True)


#%%
"""
simulate the MB/VF model
"""

# =============================================================================
# define parameters
# =============================================================================
alpha_Hs = [0.009, 0.0001]#[0.0055, 0.00005]#[0.001]
alpha_R = 0.09#0.25#0.35 #0.01
w_0 = 1#0.8
w_g = 5 #10
w_h = 5 #10
theta_h = 5#2.5 #5
theta_g = 5#2.5 #5
alpha_w = 1 #1
paras = alpha_Hs, alpha_R, w_0, w_g, w_h, theta_g, theta_h, alpha_w

na=2 #number of actions
nm=2 #number of rewards/reinforcers

trials_phase1s = [100] #training trials
trials_phase2 = 100 #extinction trials

repetitions = 50

u = 0.99
preference = np.zeros(nr)
for i in range(1,nr):
    preference[i] = u/(nr-1)
preference[0] = (1.-u)
U = mim.softmax_reversal(preference)
U_deval = []

alpha_reward = [0.9] #probability of rewards
# =============================================================================
# simulate
# =============================================================================
worlds_retrieval_mbp = rs.simulation_mbp(na, nm, paras, alpha_reward, 
                                          repetitions,
                                          trials_phase1s, trials_phase2, U, U_deval, 
                                          retrieval=True)
# =============================================================================
# plot 
# =============================================================================
 # plot chosen probability 
anls.plot_chosen_probability(worlds_retrieval_mbp[:50], repetitions=50, 
                             habit_learning_strength=alpha_Hs[0], weak=False, baye=False)
anls.plot_chosen_probability(worlds_retrieval_mbp[50:], repetitions=50, 
                             habit_learning_strength=alpha_Hs[1], weak=True, baye=False)


 # plot convergence time
data_t_retrieval_mbp = anls.plot_convergence_time(worlds_retrieval_mbp, repetitions=50, habit_learning_strength=alpha_Hs)



 # plot probabilities
data_p_retrieval_mbp = anls.plot_chosen_probability_retrieval(worlds_retrieval_mbp, repetitions=50,
                                                              habit_learning_strength=alpha_Hs, re=True)

#%%
"""
data analysis
"""
print('\nData Analysis\n')
# =============================================================================
# t-test convergence time strong vs weak
# =============================================================================
# baye
print('t-test convergence time (strong vs weak) baye:')
print('\n naive:')
data1, data2 = mim.remove_nan(data_t_retrieval_baye[0,0,:],data_t_retrieval_baye[1,0,:])
mim.sample_two_t_test(data1, data2) 
print('\n experienced:')
data1, data2 = mim.remove_nan(data_t_retrieval_baye[0,1,:],data_t_retrieval_baye[1,1,:])
mim.sample_two_t_test(data1, data2) 

# mbp
print('t-test convergence time (strong vs weak) mbp:')
print('\n naive:')
data1, data2 = mim.remove_nan(data_t_retrieval_mbp[0,0,:],data_t_retrieval_mbp[1,0,:])
mim.sample_two_t_test(data1, data2) 
print('\n experienced:')
data1, data2 = mim.remove_nan(data_t_retrieval_mbp[0,1,:],data_t_retrieval_mbp[1,1,:])
mim.sample_two_t_test(data1, data2) 

# =============================================================================
# t-test convergence time naive vs exped
# =============================================================================
# baye
print('t-test convergence time (naive vs experienced) baye:')
print('\n strong:')
data1, data2 = mim.remove_nan(data_t_retrieval_baye[0,0,:],data_t_retrieval_baye[0,1,:])
mim.sample_two_t_test(data1, data2) 
print('\n weak:')
data1, data2 = mim.remove_nan(data_t_retrieval_baye[1,0,:],data_t_retrieval_baye[1,1,:])
mim.sample_two_t_test(data1, data2) 

# mbp
print('t-test convergence time (naive vs experienced) mbp:')
print('\n strong:')
data1, data2 = mim.remove_nan(data_t_retrieval_mbp[0,0,:],data_t_retrieval_mbp[0,1,:])
mim.sample_two_t_test(data1, data2) 
print('\n weak:')
data1, data2 = mim.remove_nan(data_t_retrieval_mbp[1,0,:],data_t_retrieval_mbp[1,1,:])
mim.sample_two_t_test(data1, data2)



#%%
"""
save data
"""

file_name = ['worlds_EXRe_baye.pkl', 'worlds_EXRe_mbp.pkl']
data = [worlds_retrieval_baye, worlds_retrieval_mbp]

file_path_0 = 'E:/CAN/thesis/Habit learning models/data/'
for i in range(len(data)):
    file_path = file_path_0 + file_name[i]
    mim.save_load_data(data[i], file_path, mode = 'wb')



#%%
"""
load data
"""
import misc_mbp as mim

file_name = ['worlds_EXRe_baye.pkl', 'worlds_EXRe_mbp.pkl']

file_path_0 = 'E:/CAN/thesis/Habit learning models/data/'
worlds = []
for i in range(len(file_name)):
    file_path = file_path_0 + file_name[i]
    worlds.append(mim.save_load_data(file_path=file_path, mode = 'rb'))

worlds_retrieval_baye, worlds_retrieval_mbp= worlds