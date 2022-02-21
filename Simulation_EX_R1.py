# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 21:17:27 2022

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
#%%
# =============================================================================
# simulate
# =============================================================================

worlds_reversal_1_baye = rs.simulation_reversal_baye(repetitions, preference, 
                            preference_deval, avg, T, ns, na, nr, 
                             nc, alpha_list, n_test, trials)

#%%
# =============================================================================
# plot 
# =============================================================================

 # replication plot dynamic changes of key variables
anls.plot_variables_dynamic_baye(worlds_reversal_1_baye, repetitions, alpha_policies)
 # replication plot lineplot - habit strength
anls.plot_habit_strength_replication(worlds_reversal_1_baye, repetitions, alpha_policies, 
                        EX_D1=False) 

 # plot chosen probability 
anls.plot_chosen_probability(worlds_reversal_1_baye[:50], repetitions=50, 
                             habit_learning_strength=alpha_policies[0],weak=False, baye=True)
anls.plot_chosen_probability(worlds_reversal_1_baye[50:], repetitions=50, 
                             habit_learning_strength=alpha_policies[1],weak=True, baye=True)


anls.plot_chosen_probability_retrieval(worlds_reversal_1_baye, repetitions=50, 
                                       habit_learning_strength=alpha_policies, baye=True)

 # plot habit strenth
data_a_baye = anls.plot_habit_strength(worlds_reversal_1_baye, repetitions=50, habit_learning_strength=alpha_policies, baye=True)

#%%
"""
simulate the MB/VF model
"""

# =============================================================================
# define parameters
# =============================================================================
alpha_Hs = [0.009, 0.0001]#[0.012, 0.0001]#[0.013, 0.001]#[0.03, 0.008]#[0.0055, 0.00005]#[0.001]
alpha_R = 0.09#0.05#0.25#0.35 #0.01
w_0 = 1 #1
w_g = 5 #10
w_h = 5 #10
theta_h = 5 #5
theta_g = 5 #5
alpha_w = 1 #1



paras = alpha_Hs, alpha_R, w_0, w_g, w_h, theta_g, theta_h, alpha_w

na=2 #number of actions
nm=2 #number of rewards/reinforcers
alpha_reward = [0.9] #probability of rewards
alpha_stoch = [0.9, 0.8, 0.7, 0.6] #probability of rewards

trials_phase1s = [100] #training trials
trials_phase2 = 100 #extinction trials

u = 0.99
preference = np.zeros(nr)
for i in range(1,nr):
    preference[i] = u/(nr-1)
preference[0] = (1.-u)

U = mim.softmax_reversal(preference)
#U = np.array([0.09802733042535536, 4.0, 4.0])/4 #utility of rewards/reinforcers
U_deval = []
repetitions = 50

# =============================================================================
# simulate
# =============================================================================
worlds_reversal_1_mbp = rs.simulation_mbp(na, nm, paras, alpha_reward, 
                                          repetitions,
                                          trials_phase1s, trials_phase2, U, U_deval)

# =============================================================================
# plot 
# =============================================================================

 # plot chosen probability 
anls.plot_chosen_probability(worlds_reversal_1_mbp[:50], repetitions=50, 
                             habit_learning_strength=alpha_Hs[0], weak=False, baye=False)
anls.plot_chosen_probability(worlds_reversal_1_mbp[50:], repetitions=50, 
                             habit_learning_strength=alpha_Hs[1], weak=True, baye=False)

anls.plot_chosen_probability_retrieval(worlds_reversal_1_mbp, repetitions=50,
                                       habit_learning_strength=alpha_Hs)
 # plot habit strenth
data_a_mbp = anls.plot_habit_strength(worlds_reversal_1_mbp, repetitions=50, habit_learning_strength=alpha_Hs) 

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
data1, data2 = mim.remove_nan(data_a_baye[0][:,0],data_a_baye[0][:,1])
mim.sample_two_t_test(data1, data2)

# mbp
print('t-test habit strength (strong vs weak) mbp:')
data1, data2 = mim.remove_nan(data_a_mbp[0][:,0],data_a_mbp[0][:,1])
mim.sample_two_t_test(data1, data2)



# =============================================================================
# t-test on stable belief between strong and weak habit learners
# =============================================================================

  # baye
print('t-test stable belief (strong vs weak) baye:')
for n in range(len(trials_phase1s)):
    data1, data2 = np.zeros(repetitions), np.zeros(repetitions)
    for i in range(repetitions):
        results_a_param = data_a_baye[1][:,i]
        results_a_param_type = data_a_baye[2][:,i]
        data1[i] = mim.stable_level(results_a_param[0], results_a_param_type[0])
        data2[i] = mim.stable_level(results_a_param[1], results_a_param_type[1])
    data1, data2 = mim.remove_nan(data1, data2)
    mim.sample_two_t_test(data1, data2)
    
  # mbp
print('t-test stable belief (strong vs weak) mbp:')
for n in range(len(trials_phase1s)):
    data1, data2 = np.zeros(repetitions), np.zeros(repetitions)
    for i in range(repetitions):
        results_a_param = data_a_mbp[1][:,i]
        results_a_param_type = data_a_mbp[2][:,i]
        data1[i] = mim.stable_level(results_a_param[0], results_a_param_type[0])
        data2[i] = mim.stable_level(results_a_param[1], results_a_param_type[1])
    data1, data2 = mim.remove_nan(data1, data2)
    mim.sample_two_t_test(data1, data2)
    
#%%
"""
save data
"""

file_name = ['worlds_EXR1_baye.pkl', 'worlds_EXR1_mbp.pkl']
file_path_0 = 'E:/CAN/thesis/Habit learning models/data/'
data = [worlds_reversal_1_baye, worlds_reversal_1_mbp]
for i in range(len(data)):
    file_path = file_path_0 + file_name[i]
    mim.save_load_data(data[i], file_path, mode = 'wb')



#%%
"""
load data
"""
import misc_mbp as mim

file_name = ['worlds_EXR1_baye.pkl', 'worlds_EXR1_mbp.pkl']
file_path_0 = 'E:/CAN/thesis/Habit learning models/data/'
worlds = []
for i in range(len(file_name)):
    file_path = file_path_0 + file_name[i]
    worlds.append(mim.save_load_data(file_path=file_path, mode = 'rb'))
worlds_reversal_1_baye,  worlds_reversal_1_mbp= worlds


#%%
"""
to find similar strong and weak agents for two models.
"""

# =============================================================================
# def test_datas(data1, data2, d):
#     if (data1 > data2 - d).all() and (data1 < data2 + d).all():
#         return True
#     else:
#         return False
# 
# data_a_median_baye = np.percentile(data_a_baye[1], 50, axis={1}) # strong-weak, 4 parameters for fitted functions
# habit_strength_median_baye = data_a_median_baye[:,2] - 100
# data_a_median_mbp = np.percentile(data_a_mbp[1], 50, axis={1}) # strong-weak, 4 parameters for fitted functions
# habit_strength_median_mbp = data_a_median_mbp[:,2] - 100
# 
# #%%
# 
# alpha_Hs = np.arange(0.0001, 0.05, 0.0005)
# alpha_R = np.arange(0.01, 0.35, 0.01)
# w_0 = [1]
# w_g_h = [5] 
# theta_g_h = [5]#np.arange(1, 5.5, 0.5)
# alpha_w = [1] #1
# repetitions = 10
# 
# #%%
# 
# potential_strong_paras = []
# strong_paras = []
# potential_weak_paras = []
# weak_paras = []
# 
# for i in range(len(alpha_Hs)):
#     print(str(i) + '/' + str(len(alpha_Hs)))
#     for j in range(len(alpha_R)):
#         for k in range(len(w_0)):
#             for l in range(len(w_g_h)):
#                 for m in range(len(theta_g_h)):
#                     paras = [[alpha_Hs[i]], alpha_R[j], w_0[k], w_g_h[l],
#                              w_g_h[l], theta_g_h[m], theta_g_h[m], alpha_w[0]]
#                     worlds = rs.simulation_mbp(na, nm, paras, alpha_reward, 
#                                       repetitions,trials_phase1s, trials_phase2, 
#                                         U, U_deval)
#                     
#                     results_a = np.zeros((repetitions, 4))
#                     a_type = np.zeros((repetitions), dtype=int)                        
#                     results_a, a_type = anls.infer_time_mbp(worlds, repetitions)
#                     habit_strength = results_a[:,2]-100
#                     habit_strength = habit_strength.T 
#                     habit_strength_median = np.percentile(habit_strength, 50, axis={0})
#                     parameters_median = np.percentile(results_a, 50, axis={0})
#                     
#                     if test_datas(habit_strength_median, habit_strength_median_baye[0], 1):
#                         #here habit strength = habit strength of strong Bayesian agent
#                         potential_strong_paras.append(paras)
#                         if test_datas(parameters_median, data_a_median_baye[0], d=data_a_median_baye[0]/5):
#                             print (paras)
#                             print ('this is strong.................')
#                             strong_paras.append(paras)
#                     elif test_datas(habit_strength_median, habit_strength_median_baye[1], 1):
#                         #here habit strength = habit strength of strong Bayesian agent
#                         potential_weak_paras.append(paras)
#                         if test_datas(parameters_median, data_a_median_baye[1], d=data_a_median_baye[1]/5):
#                             print (paras)
#                             print ('this is weak.................')
#                             weak_paras.append(paras)
# 
# print(strong_paras)
# print(weak_paras)
# =============================================================================
