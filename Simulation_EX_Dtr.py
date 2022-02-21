# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 21:54:22 2022

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
n_test =  100
trials =  [100+n_test, 200+n_test, 300+n_test, 400+n_test, 500+n_test, 1000+n_test, \
               5000+n_test, 10000+n_test]

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
#%%
# =============================================================================
# simulate
# =============================================================================
worlds_training_baye = rs.simulation_reversal_baye(repetitions, preference, preference_deval, 
                                                   avg, T, ns, na, nr, 
                                                   nc, alpha_list, n_test, trials)
#%%
# =============================================================================
# plot 
# =============================================================================

trials_phase1s = np.array(trials) - n_test
 # plot habit strenth
data_a_training_baye = anls.plot_habit_strength_training(worlds_training_baye, 
                                repetitions, alpha_policies, 
                                 trials_phase1s, EX4_1=True, baye=True)


#%%
"""
simulate the MB/VF model
"""

# =============================================================================
# define parameters
# =============================================================================
na=2 #number of actions
nm=3 #number of rewards/reinforcers

alpha_Hs = [0.009, 0.0001]#[0.013, 0.001]#[0.0055, 0.00005]#[0.001]
alpha_R = 0.09#0.05#0.25#0.35 #0.01
w_0 = 1 #1
w_g = 5 #10
w_h = 5 #10
theta_h = 5#2.5 #5
theta_g = 5#2.5 #5
alpha_w = 1 #1
paras = alpha_Hs, alpha_R, w_0, w_g, w_h, theta_g, theta_h, alpha_w


alpha_reward = [0.9]

trials_phase1s=[100,200,300,400,500,1000,5000,10000]
trials_phase2 = 100

repetitions = 50

T = 2 #number of time steps in each trial
nb = 2
no = nb+1 #number of observations
ns = nb+1 #number of states
na = nb #number of actions
nr = nb+1
nc = nb
nm=nr-1
U = np.array([-5.20455776256869, 1, 1])
U_deval=[]

u = 0.99
preference = np.zeros(nr)
for i in range(1,nr):
    preference[i] = u/(nr-1)
preference[0] = (1.-u)

U = mim.softmax_reversal(preference)

alpha_Hs_test = [0.005, 0.0001]
alpha_R_test = 0.1#0.35 #0.01
theta_h_test = 5 #5
theta_g_test = 5 #5
paras_test = alpha_Hs_test, alpha_R_test, w_0, w_g, w_h, theta_g_test, theta_h_test, alpha_w
U_test = np.array([-5.20455776256869, 1, 1])
# =============================================================================
# simulate
# =============================================================================
worlds_training_mbp = rs.simulation_mbp(na, nm, paras, alpha_reward, 
                                          repetitions,
                                          trials_phase1s, trials_phase2, U, U_deval)
worlds_training_mbp_test = rs.simulation_mbp(na, nm, paras_test, alpha_reward, 
                                          repetitions,
                                          trials_phase1s, trials_phase2, U_test, U_deval)
#%%
# =============================================================================
# plot 
# =============================================================================
 # plot habit strenth

data_a_training_mbp = anls.plot_habit_strength_training(worlds_training_mbp, 
                                repetitions, alpha_Hs, 
                                 trials_phase1s, EX4_1=True, baye=False)
data_a_training_mbp_test = anls.plot_habit_strength_training(worlds_training_mbp_test, 
                                repetitions, alpha_Hs_test, 
                                 trials_phase1s, EX4_1=True, baye=False)

#%%
"""
data analysis
"""
print('\nData Analysis\n')

# =============================================================================
# t-test habit strength (strong vs weak)
# =============================================================================
# baye
print('t-test habit strength (strong vs weak) baye:')
for n in range(len(trials_phase1s)):
    print('\n training duration = ' + str(trials_phase1s[n]) + ' :')
    data1, data2 = mim.remove_nan(data_a_training_baye[0][0,:,n],data_a_training_baye[0][1,:,n])
    mim.sample_two_t_test(data1, data2)


# mbp
print('t-test habit strength (strong vs weak) mbp:')
for n in range(len(trials_phase1s)):
    print('\n training duration = ' + str(trials_phase1s[n]) + ' :')
    data1, data2 = mim.remove_nan(data_a_training_mbp[0][0,:,n],data_a_training_mbp[0][1,:,n])
    mim.sample_two_t_test(data1, data2)

#%%
# =============================================================================
# t-test on stable belief between strong and weak habit learners
# =============================================================================
trials_phase1s=[100,200,300,400,500,1000,5000,10000]
  # baye
print('t-test stable belief (strong vs weak) baye:')
for n in range(len(trials_phase1s)):
    print('\n training duration:' + str(trials_phase1s[n]))
    data1, data2 = np.zeros(repetitions), np.zeros(repetitions)
    for i in range(repetitions):
        results_a_param = data_a_training_baye[1][:,n,i]
        results_a_param_type = data_a_training_baye[2][:,n,i]
        data1[i] = mim.stable_level(results_a_param[0], results_a_param_type[0])
        data2[i] = mim.stable_level(results_a_param[1], results_a_param_type[1])
        data1[i], data2[i] = mim.remove_nan(results_a_param[0,3], results_a_param[1,3])
    data1, data2 = mim.remove_nan(data1, data2)
    mim.sample_two_t_test(data1, data2)
    
  # mbp
print('t-test stable belief (strong vs weak) mbp:')
for n in range(len(trials_phase1s)):
    print('\n training duration:' + str(trials_phase1s[n]))
    data1, data2 = np.zeros(repetitions), np.zeros(repetitions)
    for i in range(repetitions):
        results_a_param = data_a_training_mbp[1][:,n,i]
        results_a_param_type = data_a_training_mbp[2][:,n,i]
        data1[i] = mim.stable_level(results_a_param[0], results_a_param_type[0])
        data2[i] = mim.stable_level(results_a_param[1], results_a_param_type[1])
        data1[i], data2[i] = mim.remove_nan(results_a_param[0,3], results_a_param[1,3])
    data1, data2 = mim.remove_nan(data1, data2)
    mim.sample_two_t_test(data1, data2)

  
#%%
""" 
plot probability
"""
# =============================================================================
# trials_phase1s=[100,200,300,400,500,1000,5000,10000]
# trials_phase2 = 100
# n = 5
# m = 2*n
# data_p_training_baye = anls.plot_chosen_probability_training(worlds_training_baye[:m*repetitions], 
#                      repetitions, trials_phase1s[:n], trials_phase2, 'extinction phase', baye=True)
# data_p_training_mbp = anls.plot_chosen_probability_training(worlds_training_mbp[:m*repetitions], 
#                      repetitions, trials_phase1s[:n], trials_phase2, 'extinction phase', baye=False)
# =============================================================================

#%%
"""
save data
"""

file_name = ['worlds_EXDtr_baye.pkl', 'worlds_EXDtr_mbp.pkl', 'worlds_EXDtr_mbp_test.pkl']
data = [worlds_training_baye, worlds_training_mbp, worlds_training_mbp_test]

file_path_0 = 'E:/CAN/thesis/Habit learning models/data/'
for i in range(len(data)):
    file_path = file_path_0 + file_name[i]
    mim.save_load_data(data[i], file_path, mode = 'wb')



#%%
"""
load data
"""
import misc_mbp as mim

file_name = ['worlds_EXDtr_baye.pkl', 'worlds_EXDtr_mbp.pkl', 'worlds_EXDtr_mbp_test.pkl']

file_path_0 = 'E:/CAN/thesis/Habit learning models/data/'
worlds = []
for i in range(len(file_name)):
    file_path = file_path_0 + file_name[i]
    worlds.append(mim.save_load_data(file_path=file_path, mode = 'rb'))

worlds_training_baye, worlds_training_mbp, worlds_training_mbp_test = worlds


#%%
"""
with devaluation 
"""
#%%
"""
simulate the Bayesian model
"""

# =============================================================================
# define parameters
# =============================================================================
n_test =  100
trials =  [100+n_test, 200+n_test, 300+n_test, 400+n_test, 500+n_test, 1000+n_test, \
               5000+n_test, 10000+n_test]

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

# =============================================================================
# simulate
# =============================================================================
worlds_training_baye_testdeval = rs.simulation_reversal_baye(repetitions, preference, preference_deval, 
                                                   avg, T, ns, na, nr, 
                                                   nc, alpha_list, n_test, trials, 
                                                   deval_1=True)
#%%
# =============================================================================
# plot 
# =============================================================================

trials_phase1s = np.array(trials) - n_test
 # plot habit strenth
data_a_training_baye_testdeval = anls.plot_habit_strength_training(worlds_training_baye_testdeval, 
                                repetitions, alpha_policies, 
                                 trials_phase1s, EX4_1=True, baye=True)
 #plot choice probability at the end of phases
data_p_training_baye_testdeval = anls.plot_chosen_probability_phaseend_mean(worlds_training_baye_testdeval, 
                                     alpha_Hs, repetitions, 
                                     devaluation=True, baye=True)

#%%
"""
simulate the MB/VF model
"""

# =============================================================================
# define parameters
# =============================================================================
na=2 #number of actions
nm=3 #number of rewards/reinforcers

alpha_Hs = [0.009, 0.0001]#[0.013, 0.001]#[0.0055, 0.00005]#[0.001]
alpha_R = 0.09#0.05#0.25#0.35 #0.01
w_0 = 1 #1
w_g = 5 #10
w_h = 5 #10
theta_h = 5#2.5 #5
theta_g = 5#2.5 #5
alpha_w = 1 #1
paras = alpha_Hs, alpha_R, w_0, w_g, w_h, theta_g, theta_h, alpha_w


alpha_reward = [0.9]

trials_phase1s=[100,200,300,400,500,1000,5000,10000]
trials_phase2 = 100

repetitions = 50

T = 2 #number of time steps in each trial
nb = 2
no = nb+1 #number of observations
ns = nb+1 #number of states
na = nb #number of actions
nr = nb+1
nc = nb
nm=nr-1

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


# =============================================================================
# simulate
# =============================================================================
worlds_training_mbp_testdeval = rs.simulation_mbp(na, nm, paras, alpha_reward, 
                                          repetitions,
                                          trials_phase1s, trials_phase2, U, U_deval,
                                          deval_1=True)

#%%
# =============================================================================
# plot  worlds_training_testdeval_
# =============================================================================
 # plot habit strenth

data_a_training_mbp_testdeval = anls.plot_habit_strength_training(worlds_training_mbp_testdeval, 
                                repetitions, alpha_Hs, 
                                 trials_phase1s, EX4_1=True, baye=False)
 #plot choice probability at the end of phases
data_p_training_mbp_testdeval = anls.plot_chosen_probability_phaseend_mean(worlds_training_mbp_testdeval, 
                                     alpha_Hs, repetitions, 
                                     devaluation=True, baye=False)
#%%
"""
Data Analysis
"""

# =============================================================================
# t-test probability: end of training phase vs  end of devaluation phase
# =============================================================================

# baye
print('t-test choice probability (end of training vs deval) baye:')
 #strong
print('\n strong') 
for n in range(len(trials_phase1s)):
    print('\n training duration = ' + str(trials_phase1s[n]) + ' :')
    data1, data2 = mim.remove_nan(data_p_training_baye_testdeval[0][0,n,:],data_p_training_baye_testdeval[1][0,n,:])
    mim.sample_two_t_test(data1, data2)

 #weak
print('\n weak') 
for n in range(len(trials_phase1s)):
    print('\n training duration = ' + str(trials_phase1s[n]) + ' :')
    data1, data2 = mim.remove_nan(data_p_training_baye_testdeval[0][1,n,:],data_p_training_baye_testdeval[1][1,n,:])
    mim.sample_two_t_test(data1, data2)    

# mbp
print('t-test choice probability (end of training vs deval) mbp:')
 #strong
print('\n strong') 
for n in range(len(trials_phase1s)):
    print('\n training duration = ' + str(trials_phase1s[n]) + ' :')
    data1, data2 = mim.remove_nan(data_p_training_mbp_testdeval[0][0,n,:],data_p_training_mbp_testdeval[1][0,n,:])
    mim.sample_two_t_test(data1, data2)
 #weak
print('\n weak') 
for n in range(len(trials_phase1s)):
    print('\n training duration = ' + str(trials_phase1s[n]) + ' :')
    data1, data2 = mim.remove_nan(data_p_training_mbp_testdeval[0][1,n,:],data_p_training_mbp_testdeval[1][1,n,:])
    mim.sample_two_t_test(data1, data2)    
