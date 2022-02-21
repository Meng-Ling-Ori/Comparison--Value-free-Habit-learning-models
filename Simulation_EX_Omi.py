# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 21:48:27 2022

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
nb = 4
no = nb+1 #number of observations
ns = 3 #number of states
na = 2 #number of actions
nr = nb+1
nc = 2

repetitions = 50
avg = True

alpha_policies = [10/10, 100]#[10/6]#[1, 100]
alpha_context = [99]
alpha_rewards = [50] #90
alpha_list = [alpha_policies, alpha_context, alpha_rewards]

u= np.array([0.0, 1.0, 1.0, 1.1, 0.1]) #no-rewards/rewards(pellets) after actionA/rewards(pellets) after actionB/ rewards(pellets+leisure)/rewards(leisure)
#utility=mib.softmax(u*4)
preference=mib.softmax(u)
#utility = np.zeros(nr)
#utility[:3] = u[:3]
#utility[3] = u[2]+u[3]
#utility[4] = u[3]
preference_deval=[]

n_test = 500
trials = np.arange(100,2001,100) + n_test
#%%
# =============================================================================
# simulate
# =============================================================================
worlds_training_omission_baye = rs.simulation_reversal_baye(repetitions, preference, 
                                                            preference_deval, 
                                                   avg, T, ns, na, nr, 
                                                   nc, alpha_list, n_test, trials, 
                                                   omission = True)
#%%
# =============================================================================
# plot 
# =============================================================================
trials_phase1s = np.array(trials) - n_test
 # plot habit strenth
data_a_training_omission_baye = anls.plot_habit_strength_training(worlds_training_omission_baye, 
                                repetitions, alpha_policies, 
                                 trials_phase1s, EX4_1=False, baye=True)
 #  plot choice probability (the trial at the end of training phase and omission phase) as a function of training durations 
data_p_training_omission_baye = anls.plot_chosen_probability_phaseend_mean(worlds_training_omission_baye, 
                                      alpha_policies, repetitions, 
                                      devaluation=False, baye=True)


#%%
"""
simulate the MB/VF model
"""

# =============================================================================
# define parameters
# =============================================================================
na=2 #number of actions
nm=3 #number of rewards/reinforcers


alpha_Hs = [0.001, 0.0001]
alpha_R = 0.01
alpha_stoch = []
w_0 = 1
w_g = 5
w_h = 5
theta_h = 5
theta_g = 5
alpha_w = 1
paras = alpha_Hs, alpha_R, w_0, w_g, w_h, theta_g, theta_h, alpha_w

repetitions=50

alpha_reward = [0.5] #probability of rewards

trials_phase1s = np.arange(100,2001,100) #training trials
trials_phase2 = 500 #extinction trials

U = np.array([0.0, 1.0, 1.0, 0.1])
U_deval=[]
#%%
# =============================================================================
# simulate
# =============================================================================
worlds_training_omission_mbp = rs.simulation_mbp(na, nm, paras, alpha_reward, 
                                                  repetitions,
                                                  trials_phase1s, trials_phase2, U, U_deval,
                                                  omission=True)
#%%
# =============================================================================
# plot 
# =============================================================================
 # plot habit strenth
data_a_training_omission_mbp = anls.plot_habit_strength_training(worlds_training_omission_mbp, 
                                repetitions, alpha_Hs, 
                                 trials_phase1s, EX4_1=False, baye=False)
 #  plot choice probability (the trial at the end of training phase and omission phase) as a function of training durations
data_p_training_omission_mbp = anls.plot_chosen_probability_phaseend_mean(worlds_training_omission_mbp, 
                                      alpha_Hs, repetitions, 
                                      devaluation=False, baye=False)
#%%
   # replication
worlds = []
for i in range(len(trials_phase1s)):
    for j in range(i*2*50, ((i*2)+1)*50):
        worlds.append(worlds_training_omission_mbp[j])
anls.plot_chosen_probability_phaseend_replication(worlds, repetitions)
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
for n in range(len(trials_phase1s)):
    print('\n training duration = ' + str(trials_phase1s[n]) + ' :')
    data1, data2 = mim.remove_nan(data_a_training_omission_baye[0][0,:,n],data_a_training_omission_baye[0][1,:,n])
    mim.sample_two_t_test(data1, data2)

# mbp
print('t-test habit strength (strong vs weak) mbp:')
for n in range(len(trials_phase1s)):
    print('\n training duration = ' + str(trials_phase1s[n]) + ' :')
    data1, data2 = mim.remove_nan(data_a_training_omission_mbp[0][0,:,n],data_a_training_omission_mbp[0][1,:,n])
    mim.sample_two_t_test(data1, data2)
    
    
# =============================================================================
# Linear regression training duration and habit strength
# =============================================================================
# baye
print('Linear regression (d_train and H) baye:')
     #strong habit learner
print('\n strong')
data1 = np.repeat(trials_phase1s,repetitions)
data2 = data_a_training_omission_baye[0][0].T.flatten()
data1, data2 = mim.remove_nan(data1, data2)
mim.lin_regress(data1, data2) 

     #weak habit learner
print('\n weak')     
data1 = np.repeat(trials_phase1s,repetitions)
data2 = data_a_training_omission_baye[0][1].T.flatten()
data1, data2 = mim.remove_nan(data1, data2)
mim.lin_regress(data1, data2) 

# mbp
print('Linear regression (d_train and H) mbp:')
     #strong habit learner 
print('\n strong')
data1 = np.repeat(trials_phase1s,repetitions)
data2 = data_a_training_omission_mbp[0][0].T.flatten()
data1, data2 = mim.remove_nan(data1, data2)
mim.lin_regress(data1, data2) 
     #weak habit learner
print('\n weak')
data1 = np.repeat(trials_phase1s,repetitions)
data2 = data_a_training_omission_mbp[0][1].T.flatten()
data1, data2 = mim.remove_nan(data1, data2)
mim.lin_regress(data1, data2) 

# =============================================================================
# test pearson correlation between trainingduration and probability
# =============================================================================

# =============================================================================
# # baye
# print('\npearson correlation (d_train and press rate) baye:')
# print('\n d_train and press rate at end of training phase:')
# data0 = np.repeat(trials_phase1s,repetitions)
# data2 = data_p_training_omission_baye[0][0].flatten()
# data3 = data_p_training_omission_baye[0][1].flatten()
# data1, data2 = mim.remove_nan(data0, data2)
# correlation, p_value = sc.stats.pearsonr(data1, data2)
# print('\n strong:')
# print(' Pearson’s correlation coefficient: ' + '%.4f'%correlation 
#       + ' \n p_value: ' + '%.4f'%p_value)
# data1, data3 = mim.remove_nan(data0, data3)
# correlation, p_value = sc.stats.pearsonr(data1, data3)
# print('\n weak:')
# print(' Pearson’s correlation coefficient: ' + '%.4f'%correlation 
#       + ' \n p_value: ' + '%.4f'%p_value)
# print('\n d_train and press rate at end of omission phase:')
# data0 = np.repeat(trials_phase1s,repetitions)
# data2 = data_p_training_omission_baye[1][0].flatten()
# data3 = data_p_training_omission_baye[1][1].flatten()
# data1, data2 = mim.remove_nan(data0, data2)
# correlation, p_value = sc.stats.pearsonr(data1, data2)
# print('\n strong:')
# print(' Pearson’s correlation coefficient: ' + '%.4f'%correlation 
#       + ' \n p_value: ' + '%.4f'%p_value)
# data1, data3 = mim.remove_nan(data0, data3)
# correlation, p_value = sc.stats.pearsonr(data1, data3)
# print('\n weak:')
# print(' Pearson’s correlation coefficient: ' + '%.4f'%correlation 
#       + ' \n p_value: ' + '%.4f'%p_value)
# 
# # mbp
# print('\npearson correlation (d_train and press rate) mbp:')
# print('\n d_train and press rate at end of training phase:')
# data0 = np.repeat(trials_phase1s,repetitions)
# data2 = data_p_training_omission_mbp[0][0].flatten()
# data3 = data_p_training_omission_mbp[0][1].flatten()
# data1, data2 = mim.remove_nan(data0, data2)
# correlation, p_value = sc.stats.pearsonr(data1, data2)
# print('\n strong:')
# print(' Pearson’s correlation coefficient: ' + '%.4f'%correlation 
#       + ' \n p_value: ' + '%.4f'%p_value)
# data1, data3 = mim.remove_nan(data0, data3)
# correlation, p_value = sc.stats.pearsonr(data1, data3)
# print('\n weak:')
# print(' Pearson’s correlation coefficient: ' + '%.4f'%correlation 
#       + ' \n p_value: ' + '%.4f'%p_value)
# print('\n d_train and press rate at end of omission phase:')
# data0 = np.repeat(trials_phase1s,repetitions)
# data2 = data_p_training_omission_mbp[1][0].flatten()
# data3 = data_p_training_omission_mbp[1][1].flatten()
# data1, data2 = mim.remove_nan(data0, data2)
# correlation, p_value = sc.stats.pearsonr(data1, data2)
# print('\n strong:')
# print(' Pearson’s correlation coefficient: ' + '%.4f'%correlation 
#       + ' \n p_value: ' + '%.4f'%p_value)
# data1, data3 = mim.remove_nan(data0, data3)
# correlation, p_value = sc.stats.pearsonr(data1, data3)
# print('\n weak:')
# print(' Pearson’s correlation coefficient: ' + '%.4f'%correlation 
#       + ' \n p_value: ' + '%.4f'%p_value)
# =============================================================================
#%%
# =============================================================================
# Linear regression training duration and difference in probability 
# =============================================================================
# baye
print('Linear regression (d_train and difference in probability) baye:')
     #strong habit learner
print('\n strong')
data1 = np.repeat(trials_phase1s,repetitions)
data2 = data_p_training_omission_baye[0][0].flatten() - data_p_training_omission_baye[1][0].flatten()
data1, data2 = mim.remove_nan(data1, data2)
mim.lin_regress(data1, data2) 

     #weak habit learner
print('\n weak')     
data1 = np.repeat(trials_phase1s,repetitions)
data2 = data_p_training_omission_baye[0][1].flatten() - data_p_training_omission_baye[1][1].flatten()
data1, data2 = mim.remove_nan(data1, data2)
mim.lin_regress(data1, data2) 

# mbp
print('Linear regression (d_train and difference in probability) mbp:')
     #strong habit learner 
print('\n strong')
data1 = np.repeat(trials_phase1s,repetitions)
data2 = data_p_training_omission_mbp[0][0].flatten() - data_p_training_omission_mbp[1][0].flatten()
data1, data2 = mim.remove_nan(data1, data2)
mim.lin_regress(data1, data2) 
     #weak habit learner
print('\n weak')
data1 = np.repeat(trials_phase1s,repetitions)
data2 = data_p_training_omission_mbp[0][1].flatten() - data_p_training_omission_mbp[1][1].flatten()
data1, data2 = mim.remove_nan(data1, data2)
mim.lin_regress(data1, data2)

#%%
# =============================================================================
# test pearson correlation between training duration and difference in probability 
# =============================================================================
# baye
print('\npearson correlation (d_train and  difference in press rate) baye:')

data0 = np.repeat(trials_phase1s,repetitions)
data2 = data_p_training_omission_baye[0][0].flatten() - data_p_training_omission_baye[1][0].flatten()
data3 = data_p_training_omission_baye[0][1].flatten() - data_p_training_omission_baye[1][1].flatten()
data1, data2 = mim.remove_nan(data0, data2)
correlation, p_value = sc.stats.pearsonr(data1, data2)
print('\n strong:')
print(' Pearson’s correlation coefficient: ' + '%.4f'%correlation 
      + ' \n p_value: ' + '%.4f'%p_value)
data1, data3 = mim.remove_nan(data0, data3)
correlation, p_value = sc.stats.pearsonr(data1, data3)
print('\n weak:')
print(' Pearson’s correlation coefficient: ' + '%.4f'%correlation 
      + ' \n p_value: ' + '%.4f'%p_value)

# mbp
print('\npearson correlation (d_train and  difference in press rate) mbp:')

data0 = np.repeat(trials_phase1s,repetitions)
data2 = data_p_training_omission_mbp[0][0].flatten() - data_p_training_omission_mbp[1][0].flatten()
data3 = data_p_training_omission_mbp[0][1].flatten() - data_p_training_omission_mbp[1][1].flatten()
data1, data2 = mim.remove_nan(data0, data2)
correlation, p_value = sc.stats.pearsonr(data1, data2)
print('\n strong:')
print(' Pearson’s correlation coefficient: ' + '%.4f'%correlation 
      + ' \n p_value: ' + '%.4f'%p_value)
data1, data3 = mim.remove_nan(data0, data3)
correlation, p_value = sc.stats.pearsonr(data1, data3)
print('\n weak:')
print(' Pearson’s correlation coefficient: ' + '%.4f'%correlation 
      + ' \n p_value: ' + '%.4f'%p_value)


#%%
"""
save data
"""

file_name = ['worlds_EXOmi_baye.pkl', 'worlds_EXOmi_mbp.pkl']
data = [worlds_training_omission_baye, worlds_training_omission_mbp]

file_path_0 = 'E:/CAN/thesis/Habit learning models/data/'
for i in range(len(data)):
    file_path = file_path_0 + file_name[i]
    mim.save_load_data(data[i], file_path, mode = 'wb')



#%%
"""
load data
"""
import misc_mbp as mim

file_name = ['worlds_EXOmi_baye.pkl', 'worlds_EXOmi_mbp.pkl']

file_path_0 = 'E:/CAN/thesis/Habit learning models/data/'
worlds = []
for i in range(len(file_name)):
    file_path = file_path_0 + file_name[i]
    worlds.append(mim.save_load_data(file_path=file_path, mode = 'rb'))

worlds_training_omission_baye, worlds_training_omission_mbp= worlds