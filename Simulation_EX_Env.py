# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 22:04:10 2022

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
alpha_rewards = [90, 80, 70, 60]
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
worlds_stoch_baye = rs.simulation_reversal_baye(repetitions, preference, 
                            preference_deval, avg, T, ns, na, nr, 
                             nc, alpha_list, n_test, trials)
#%%
# =============================================================================
# plot 
# =============================================================================
 # plot habit strenth
data_a_stoch_baye = anls.plot_habit_strength_stoch(worlds_stoch_baye, repetitions=50, 
                                             habit_learning_strength=alpha_policies, 
                                             alpha_rewards=alpha_rewards, baye=True)
 
#%%
"""
simulate the MB/VF model
"""

# =============================================================================
# define parameters
# =============================================================================
alpha_Hs = [0.009, 0.0001]#[0.03, 0.008]#[0.0055, 0.00005]#[0.001]
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

alpha_reward = [0.9, 0.8, 0.7, 0.6] #probability of rewards

alpha_Hs_test = [0.013, 0.001] #[0.005, 0.005]
alpha_R_test = 0.05 #0.1 #0.35 #0.01
theta_h_test = 5 #5
theta_g_test = 5 #5
paras_test = alpha_Hs_test, alpha_R_test, w_0, w_g, w_h, theta_g_test, theta_h_test, alpha_w
U_test = np.array([-5.20455776256869, 1, 1])

#%%
# =============================================================================
# simulate
# =============================================================================

worlds_stoch_mbp = rs.simulation_mbp(na, nm, paras, alpha_reward, 
                                      repetitions,trials_phase1s, trials_phase2, 
                                        U, U_deval)

worlds_stoch_mbp_test = rs.simulation_mbp(na, nm, paras_test, alpha_reward, 
                                      repetitions,trials_phase1s, trials_phase2, 
                                        U, U_deval)
#%%
# =============================================================================
# plot 
# =============================================================================
 # plot habit strenth
data_a_stoch_mbp = anls.plot_habit_strength_stoch(worlds_stoch_mbp, repetitions=50, 
                                            habit_learning_strength=alpha_Hs,
                                            alpha_rewards=alpha_reward) 
data_a_stoch_mbp_test = anls.plot_habit_strength_stoch(worlds_stoch_mbp_test, repetitions=50, 
                                            habit_learning_strength=alpha_Hs_test,
                                            alpha_rewards=alpha_reward) 

#%%
"""
data analysis
"""
print('\nData Analysis\n')
# =============================================================================
# linear regression (1-v and habit strength)
# =============================================================================

# baye
print('\nlinear regression (1-v and habit strength) baye:')
     #strong habit learner
print('\n strong:')     
data1 = np.repeat(1-np.array(alpha_rewards),repetitions)
data2 = data_a_stoch_baye[0][0,:,:].flatten()
data1, data2 = mim.remove_nan(data1, data2)
mim.lin_regress(data1, data2) 

     #weak habit learner
print('\n weak:')
data1 = np.repeat(1-np.array(alpha_rewards),repetitions)
data2 = data_a_stoch_baye[0][1,:,:].flatten()
data1, data2 = mim.remove_nan(data1, data2)
mim.lin_regress(data1, data2) 

# mbp
print('\nlinear regression (1-v and habit strength) mbp:') 
     #strong habit learner
print('\n strong:')
data1 = np.repeat(1-np.array(alpha_reward),repetitions)
data2 = data_a_stoch_mbp[0][0,:,:].flatten()
data1, data2 = mim.remove_nan(data1, data2)
mim.lin_regress(data1, data2) 

     #weak habit learner
print('\n weak:')
data1 = np.repeat(1-np.array(alpha_reward),repetitions)
data2 = data_a_stoch_mbp[0][1,:,:].flatten()
data1, data2 = mim.remove_nan(data1, data2)
mim.lin_regress(data1, data2) 

# =============================================================================
# two sample t-test on the habit strength weak vs strong.
# =============================================================================

# baye
print('\nt-test habit strength (strong vs weak) baye:')
         # 1-v=0.1
print('\n 1-v=0.1')
data1, data2 = mim.remove_nan(data_a_stoch_baye[0][0,0,:],data_a_stoch_baye[0][1,0,:])
mim.sample_two_t_test(data1, data2)
         # 1-v=0.2
print('\n 1-v=0.2')         
data1, data2 = mim.remove_nan(data_a_stoch_baye[0][0,1,:],data_a_stoch_baye[0][1,1,:])
mim.sample_two_t_test(data1, data2)
         # 1-v=0.3
print('\n 1-v=0.3')
data1, data2 = mim.remove_nan(data_a_stoch_baye[0][0,2,:],data_a_stoch_baye[0][1,2,:])
mim.sample_two_t_test(data1, data2)
         # 1-v=0.4
print('\n 1-v=0.4')
data1, data2 = mim.remove_nan(data_a_stoch_baye[0][0,3,:],data_a_stoch_baye[0][1,3,:])
mim.sample_two_t_test(data1, data2)

#mbp
print('\nt-test habit strength (strong vs weak) mbp:')
         # 1-v=0.1
print('\n 1-v=0.1')
data1, data2 = mim.remove_nan(data_a_stoch_mbp[0][0,0,:],data_a_stoch_mbp[0][1,0,:])
mim.sample_two_t_test(data1, data2)
         # 1-v=0.2
print('\n 1-v=0.2') 
data1, data2 = mim.remove_nan(data_a_stoch_mbp[0][0,1,:],data_a_stoch_mbp[0][1,1,:])
mim.sample_two_t_test(data1, data2)
         # 1-v=0.3
print('\n 1-v=0.3') 
data1, data2 = mim.remove_nan(data_a_stoch_mbp[0][0,2,:],data_a_stoch_mbp[0][1,2,:])
mim.sample_two_t_test(data1, data2)
         # 1-v=0.4
print('\n 1-v=0.4') 
data1, data2 = mim.remove_nan(data_a_stoch_mbp[0][0,3,:],data_a_stoch_mbp[0][1,3,:])
mim.sample_two_t_test(data1, data2)

#%%
# =============================================================================
# test pearson correlation between 1-v and habit strength
# =============================================================================

# baye
print('\npearson correlation (1-v and habit strength) baye:')
data0 = np.repeat(np.array([0.1, 0.2, 0.3, 0.4]),50)
data2 = data_a_stoch_baye[0][0].flatten()
data3 = data_a_stoch_baye[0][1].flatten()
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
print('\npearson correlation (1-v and habit strength) mbp:')
data0 = np.repeat(np.array([0.1, 0.2, 0.3, 0.4]),50)
data2 = data_a_stoch_mbp[0][0].flatten()
data3 = data_a_stoch_mbp[0][1].flatten()
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

# =============================================================================
# linear regression (1-v and stable belief level)
# =============================================================================

# baye
print('\nlinear regression (1-v and habit strength) baye:')
     #strong habit learner
print('\n strong:')     
data1 = np.repeat(1-np.array(alpha_rewards),repetitions)
data2 = data_a_stoch_baye[1][0,:,:,3].flatten()
data2[np.where(data2==0.0)] = np.nan
data1, data2 = mim.remove_nan(data1, data2)
mim.lin_regress(data1, data2) 

     #weak habit learner
print('\n weak:')
data1 = np.repeat(1-np.array(alpha_rewards),repetitions)
data2 = data_a_stoch_baye[1][1,:,:,3].flatten()
data2[np.where(data2==0.0)] = np.nan
data1, data2 = mim.remove_nan(data1, data2)
mim.lin_regress(data1, data2) 

# mbp
print('\nlinear regression (1-v and habit strength) mbp:') 
     #strong habit learner
print('\n strong:')
data1 = np.repeat(1-np.array(alpha_reward),repetitions)
data2 = data_a_stoch_mbp[1][0,:,:,3].flatten()
data2[np.where(data2==0.0)] = np.nan
data1, data2 = mim.remove_nan(data1, data2)
mim.lin_regress(data1, data2) 

     #weak habit learner
print('\n weak:')
data1 = np.repeat(1-np.array(alpha_reward),repetitions)
data2 = data_a_stoch_mbp[1][1,:,:,3].flatten()
data2[np.where(data2==0.0)] = np.nan
data1, data2 = mim.remove_nan(data1, data2)
mim.lin_regress(data1, data2) 
#%%
# =============================================================================
# test pearson correlation between 1-v and stable belief level
# =============================================================================

# baye
print('\npearson correlation (1-v and stable be.) baye:')
data0 = np.repeat(np.array([0.1, 0.2, 0.3, 0.4]),50)
data2 = data_a_stoch_baye[1][0,:,:,3].flatten()
data3 = data_a_stoch_baye[1][1,:,:,3].flatten()
data2[np.where(data2==0.0)] = np.nan
data3[np.where(data3==0.0)] = np.nan
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
print('\npearson correlation (1-v and stable be.) mbp:')
data0 = np.repeat(np.array([0.1, 0.2, 0.3, 0.4]),50)
data2 = data_a_stoch_mbp[1][0,:,:,3].flatten()
data3 = data_a_stoch_mbp[1][1,:,:,3].flatten()
data2[np.where(data2==0.0)] = np.nan
data3[np.where(data3==0.0)] = np.nan
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
# =============================================================================
# t-test on stable belief between strong and weak habit learners
# =============================================================================
  # baye
print('t-test stable belief (strong vs weak) baye:')
r_p = [0.1, 0.2, 0.3, 0.4]
for n in range(len(r_p)):
    print('\n enmvironmental stoch. (1-v):' + str(r_p[n]))
    data1, data2 = np.zeros(repetitions), np.zeros(repetitions)
    for i in range(repetitions):
        results_a_param = data_a_stoch_baye[1][:,n,i]
#        results_a_param_type = data_a_stoch_baye[2][:,n,i]
#        data1[i] = mim.stable_level(results_a_param[0], results_a_param_type[0])
#       data2[i] = mim.stable_level(results_a_param[1], results_a_param_type[1])
        data1[i], data2[i] = results_a_param[0,3], results_a_param[1,3]
    data1, data2 = mim.remove_nan(data1, data2)
    mim.sample_two_t_test(data1, data2)
    
  # mbp
print('t-test stable belief (strong vs weak) mbp:')
for n in range(len(r_p)):
    print('\n enmvironmental stoch. (1-v):' + str(r_p[n]))
    data1, data2 = np.zeros(repetitions), np.zeros(repetitions)
    for i in range(repetitions):
        results_a_param = data_a_stoch_mbp[1][:,n,i]
#        results_a_param_type = data_a_training_mbp[2][:,n,i]
#        data1[i] = mim.stable_level(results_a_param[0], results_a_param_type[0])
#        data2[i] = mim.stable_level(results_a_param[1], results_a_param_type[1])
        data1[i], data2[i] = results_a_param[0,3], results_a_param[1,3]
    data1, data2 = mim.remove_nan(data1, data2)
    mim.sample_two_t_test(data1, data2)
#%%
"""
plot chosen probabilities
"""
data_probability_baye = anls.plot_chosen_probability_stoch(worlds_stoch_baye, 
                            repetitions, habit_learning_strength=alpha_policies, baye=True)
data_probability_mbp = anls.plot_chosen_probability_stoch(worlds_stoch_mbp, 
                            repetitions, habit_learning_strength=alpha_Hs, baye=False)
data_probability_mbp_test = anls.plot_chosen_probability_stoch(worlds_stoch_mbp_test, 
                            repetitions, habit_learning_strength=alpha_Hs, baye=False)



#%%
"""
Test a significant difference between two slope values
"""
# =============================================================================
# reg_baye = np.zeros((2,5))
# data0 = np.repeat(np.array([0.1, 0.2, 0.3, 0.4]),50)
# data2 = data_a_stoch_baye[0].flatten()
# data3 = data_a_stoch_baye[1].flatten()
# data1, data2 = mim.remove_nan(data0, data2)
# reg_baye[0] = mim.lin_regress(data1, data2)
# data1, data3 = mim.remove_nan(data0, data3)
# reg_baye[1] = mim.lin_regress(data1, data3)
# 
# reg_mbp = np.zeros((2,5))
# data0 = np.repeat(np.array([0.1, 0.2, 0.3, 0.4]),50)
# data2 = data_a_stoch_mbp[0].flatten()
# data3 = data_a_stoch_mbp[1].flatten()
# data1, data2 = mim.remove_nan(data0, data2)
# reg_mbp[0] = mim.lin_regress(data1, data2)
# data1, data3 = mim.remove_nan(data0, data3)
# reg_mbp[1] = mim.lin_regress(data1, data3)
# 
# b1 = reg_baye[0,0] #slope
# b2 = reg_mbp[0,0]
# 
# se1 = reg_baye[0,-1] #std
# se2 = reg_mbp[0,-1]
# 
# z_score = (b1-b2) / (((se1*(b1**2) + se2*(b2**2)))**0.5)
# print('zscore:strong habit learners: ' + str(z_score))
# 
# b1 = reg_baye[1,0]
# b2 = reg_mbp[1,0]
# 
# se1 = reg_baye[1,-1]
# se2 = reg_mbp[1,-1]
# 
# z_score = (b1-b2) / (((se1*(b1**2) + se2*(b2**2)))**0.5)
# 
# print('zscore:weak habit learners: ' + str(z_score))
# print(b1,b2,se1,se2)
# 
# # =============================================================================
# # hypothesis: b1 = b2
# # |Z|	 P value	 sig.
# # >2.58	 <0.01	     very sig. different
# # >1.96	 <0.05	     sig. different
# # <1.96	 >0.05	     not sig.different
# # =============================================================================
# 
# 
# #%%
# 
# print(((se1*(b1**2) + se2*(b2**2)))**0.5)
# print(se2*(b2**2))
# =============================================================================
#%%
"""
save data
"""

file_name = ['worlds_EXStoch_baye.pkl', 'worlds_EXStoch_mbp.pkl']
data = [worlds_stoch_baye, worlds_stoch_mbp]

file_path_0 = 'E:/CAN/thesis/Habit learning models/data/'
for i in range(len(data)):
    file_path = file_path_0 + file_name[i]
    mim.save_load_data(data[i], file_path, mode = 'wb')



#%%
"""
load data
"""
import misc_mbp as mim

file_name = ['worlds_EXStoch_baye.pkl', 'worlds_EXStoch_mbp.pkl']

file_path_0 = 'E:/CAN/thesis/Habit learning models/data/'
worlds = []
for i in range(len(file_name)):
    file_path = file_path_0 + file_name[i]
    worlds.append(mim.save_load_data(file_path=file_path, mode = 'rb'))

worlds_stoch_baye, worlds_stoch_mbp= worlds

#%%
"""
to find similar strong and weak agents for two models.
"""

# =============================================================================
# def test_datas(data1, data2):
#     if data1 > data2 - 1 and data1 < data2 + 1:
#         return True
#     else:
#         return False
# 
# data_a_median_baye = np.percentile(data_a_stoch_baye, 50, axis={2})
# 
# alpha_Hs = np.arange(0.001, 0.02, 0.002)
# alpha_R = np.arange(0.1, 0.3, 0.05)
# w_0 = [0.8]
# w_g_h = [5] #10
# theta_g_h = np.arange(2.5, 5, 0.5)
# alpha_w = [1] #1
# repetitions = 10
# 
# 
# for i in range(len(alpha_Hs)):
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
#                     results_a = np.zeros((4, repetitions, 4))
#                     a_type = np.zeros((4, repetitions), dtype=int)
#                     for p in range(4):                        
#                         results_a[p], a_type[p] = anls.infer_time_mbp(worlds[p*repetitions:(p+1)*repetitions], repetitions)
#                     data_a_0 = results_a[:,:,2]-100
#                     data = data_a_0.T 
#                     data_median = np.percentile(data, 50, axis={0})
#                     
#                     if test_datas(data_median[0], data_a_median_baye[0,0]):
#                         print (paras)
#                         print ('satisfy 0 : this is strong.................')
#                         if test_datas(data_median[1], data_a_median_baye[0,1]):
#                             print (paras)
#                             print ('satisfy 1 : this is strong..............')
#                             if test_datas(data_median[2], data_a_median_baye[0,2]):
#                                 print (paras)
#                                 print ('satisfy 3 : this is strong.............')
#                                 if test_datas(data_median[3], data_a_median_baye[0,3]):
#                                     print (paras)
#                                     print ('this is strong...................')
#                     elif test_datas(data_median[0], data_a_median_baye[1,0]):
#                         print (paras)
#                         print ('satisfy 0 : this is weak.....................')
#                         if test_datas(data_median[1], data_a_median_baye[1,1]):
#                             print (paras)
#                             print ('satisfy 1 : this is weak................')
#                             if test_datas(data_median[2], data_a_median_baye[1,2]):
#                                 print (paras)
#                                 print ('satisfy 2 : this is weak..............')
#                                 if test_datas(data_median[3], data_a_median_baye[1,3]):
#                                     print (paras)
#                                     print ('this is weak.....................')
# =============================================================================
                                    
                        
                            
                        

                    
                    