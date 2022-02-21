# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:47:57 2021

@author: orian
"""
import numpy as np
import scipy as sc
import os
import pickle

def logistic_regression_model(x,*args):
    # parameters: vector variable, 0-9: beta_a1, 10-19: beta_r, 20-29:beta_x1, 30-39: beta_a2, 40-49: beta_x2, 50:beta_0
    a_list, r_list, axr_list = x    
    parameters = args
    f = 0.
    lag_trials = 10
    for l in range(lag_trials):
        f += parameters[l] * a_list[l] \
            + parameters[l+lag_trials] * r_list[l] \
                + parameters[l+2*lag_trials] * axr_list[l]
#    f = sum(parameters[:10] * a_list) \
#        + sum(parameters[10:20] * r_list)\
#            + sum(parameters[20:] * axr_list)
#    for tau in range(lag_trials):        
#        f += parameters[tau]*a_list[lag_trials-tau-1] \
#               + parameters[lag_trials + tau] * r_list[lag_trials-tau-1] \
#                   +parameters[2*lag_trials + tau] * axr_list[lag_trials-tau-1]  
    return f
    
def logit(x):
    with np.errstate(divide = 'ignore'):
        return np.nan_to_num(np.log(x/(1-x)))
    
def gaussian_function(x,mean,sigma):
    return np.exp( - ((x - mean)**2)/ (2*sigma**2))    

def sample_two_t_test(data1,data2, median_value=False):
    if not median_value:        
        W, levene_P = sc.stats.levene(data1, data2, center='mean')
    else:
        W, levene_P = sc.stats.levene(data1, data2, center='median')
    #print("Levene's W statistic for the test of homogeneity of variance is " + str(W))
    #print("Levene's p value for the test of homogeneity of variance is " + str(levene_P))
    if levene_P > 0.05:
        equal_var = True
    else:
        equal_var = False
    print( ' '+ str(equal_var))
    if not median_value: 
        t, p_two = sc.stats.ttest_ind(data1, data2, equal_var=equal_var, 
                                      nan_policy='omit')
        mean1 = data1.mean()    
        mean2 = data2.mean()
        print(' mean: ' + str(f'{mean1:.2f}')+ ' vs '+ str(f'{mean2:.2f}'))
        print(' t = ' + str(f'{t:.4f}'))
        print(' P value = ' + str(f'{p_two:.4f}  \n'))         
    else: 
        t, p_two = sc.stats.ttest_ind(data1, data2, equal_var=equal_var, 
                                      nan_policy='omit')        
        print('\n t=' + str(f'{t:.20f}'))
        print('\n P value=' + str(f'{p_two:.20f}'))
    
        median1 = data1.median()
        median2 = data2.median()
        print('\n median :' + str(f'{median1:.20f}')+ ','+ str(f'{median2:.20f}'))  
    
def lin_regress(data1,data2, median_value=False):
    if not median_value:
        reg = sc.stats.linregress(data1, data2)
        slope, intercept, r_value, p_value, std_err = reg
    else:
        print('error')
    print(' slope = ' + str(f'{slope:.4f}'))# 输出斜率
    print(' intercept = ' + str(f'{intercept:.4f}')) # 输出截距
    print(' r = ' + str(f'{r_value:.4f}')) # 输出 r
    print(' standard error = ' + str(f'{std_err:.4f}'))
    print(' p value = '+str(f'{p_value:.4f}'))  
    return reg
    
    

def remove_nan(data1,data2):
    if np.isnan(data1).any() or np.isnan(data2).any():
        if np.isnan(data1).any():
            list_nan = ~np.isnan(data1)
            #print(list_nan)
            data1 = data1[list_nan]
            data2 = data2[list_nan]
        if np.isnan(data2).any():
            list_nan = ~np.isnan(data2)
            #print(list_nan)
            data1 = data1[list_nan]
            data2 = data2[list_nan]
    return data1, data2

def Action_to_Rewards(trials_phase1, trials_phase2, na, nm, alpha):
    trials = trials_phase1 + trials_phase2
    AM = np.zeros((trials, na, nm)) #Probability of leading to different reinforcers from different actions
    AM[:trials_phase1,0,:] = [1-alpha, alpha]
    AM[:trials_phase1,1,:] = [alpha, 1-alpha]
    AM[trials_phase1:,1,:] = [1-alpha, alpha]
    AM[trials_phase1:,0,:] = [alpha, 1-alpha]
    return AM

def Action_to_Rewards_re(trials_phase1, trials_phase2, na, nm, alpha):
    trials = trials_phase1 + trials_phase2 + trials_phase1
    AM = np.zeros((trials, na, nm)) #Probability of leading to different reinforcers from different actions
    AM[:trials_phase1,0,:] = [1-alpha, alpha]
    AM[:trials_phase1,1,:] = [alpha, 1-alpha]
    AM[trials_phase1:trials_phase1+trials_phase2,1,:] = [1-alpha, alpha]
    AM[trials_phase1:trials_phase1+trials_phase2,0,:] = [alpha, 1-alpha]
    AM[trials_phase1+trials_phase2:,0,:] = [1-alpha, alpha]
    AM[trials_phase1+trials_phase2:,1,:] = [alpha, 1-alpha]
    return AM

def sigmoid(x, a=1., b=1., c=0., d=0.):
    f = a/(1. + np.exp(-b*(x-c))) + d
    return f

def exponential(x, b=1., c=0., d=0.):
    f = np.exp(b*(x-c)) + d
    return f

def confidance_interval_95(data):
    #create 95% confidence interval for population mean weight
    a,b = sc.stats.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=sc.stats.sem(data))
    new_data = data[np.where(data>=a)[0]]
    new_data = new_data[np.where(new_data<=b)[0]]
    return new_data

def confidance_interval_95_mean(data):
    #create 95% confidence interval for population mean weight
    a,b = sc.stats.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=sc.stats.sem(data))
    new_data = data[np.where(data>=a)[0]]
    new_data = new_data[np.where(new_data<=b)[0]]
    mean_data = np.mean(new_data) 
    return mean_data  

def stable_level(results_param, results_param_type):
    fit_functions = [sigmoid, exponential]
    l = fit_functions[results_param_type](0, *results_param[results_param_type:]) 
    return l    

def save_load_data(html=None, file_path=None, mode=None, encoding='utf-8'):
    """
    save or load data: 
        html: str-data
        file_path: str-file path
        mode: str-save or load [w,r]
        encoding: str-[utf-8, gbk]
    """
    file_path_dir = os.path.dirname(file_path)
    if not os.path.exists(file_path_dir):
        os.makedirs(file_path_dir)
    try:
        with open(file_path, mode) as f:
            if mode == 'wb':
                pickle.dump(html,f)
                print('Saved successfully')
            elif mode == 'rb':
                data = pickle.load(f)
                print('Loaded successfully')
                return(data)
                
    except Exception as e:
        if mode == 'w':
            print('Failed to save:{}'.format(e))
        else:
            print('Failed to load:{}'.format(e))
    f.close()              

def softmax_reversal(preference, u_max=1.):
    p_max = np.max(preference)
    e_x_sum = u_max/p_max
    utility = np.log(preference * e_x_sum) + u_max
    return utility                                                                                                                                        