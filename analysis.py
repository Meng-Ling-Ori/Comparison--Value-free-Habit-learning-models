# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 02:35:31 2021

@author: orian
"""

import matplotlib.pylab as plt
import numpy as np
import scipy.special as scs
import pandas as pd
import scipy as sc
import seaborn as sns

from misc_baye import sigmoid, exponential

fit_functions = [sigmoid, exponential]

def plot_variables_dynamic_baye(worlds, repetitions, alpha_policies):
    """
    plot the dynamic changes of variables over trials
    """
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4, ncols=2, figsize= (10,10))
    alpha_policies = 1/ np.array(alpha_policies)
    strong = '$h=$'+ str(alpha_policies[0])
    weak = '$h=$'+ str(alpha_policies[1])
    
    #rewards_schedule
    trials = worlds[0].trials
    rewards_prob = [worlds[0].environment.Rho[:,1,1], worlds[0].environment.Rho[:,2,2]]   
    
    #posterior_over_contexts
    post_c_strong = posterior_over_contexts(worlds[:repetitions], repetitions)
    post_c_weak = posterior_over_contexts(worlds[repetitions:], repetitions)

    ax1[0].plot(post_c_strong[:,0], '.', color='#FF81C0', label = 'context', alpha=1)
    ax1[0].plot(post_c_strong[:,1], '-', color='#FF81C0', alpha=1)
    ax1[1].plot(post_c_weak[:,0], '.', color='#FF81C0', alpha=1)
    ax1[1].plot(post_c_weak[:,1], '-', color='#FF81C0', alpha=1)    
    ax1[0].set_ylabel('$Q_c$:posterior over contexts')
    ax1[0].set_title('strong habit learner ' + repr(strong), fontsize=14, fontweight='bold')
    ax1[1].set_title('weak habit learner '  + repr(weak), fontsize=14, fontweight='bold')
    
    #likelihood over actions(goal-direct)
    likelihood_strong = likelihood_over_actions(worlds[:repetitions], repetitions)
    likelihood_weak = likelihood_over_actions(worlds[repetitions:], repetitions)
    ax2[0].plot(likelihood_strong[:,1], '.', color='#F97306', label = 'action', alpha=1)
    ax2[1].plot(likelihood_weak[:,1], '.', color='#F97306',  alpha=1)
    ax2[0].set_ylabel('$L_a$:likelihood over actions')
    
    #prior over actions(habit)
    prior_a_strong = prior_over_actions(worlds[:repetitions], repetitions)
    prior_a_weak = prior_over_actions(worlds[repetitions:], repetitions)
    
    ax3[0].plot(prior_a_strong[:,1], '.', color='#F97306', alpha=1)
    ax3[1].plot(prior_a_weak[:,1], '.', color='#F97306', alpha=1)
    ax3[0].set_ylabel('$P_a$:prior over actions')
    
    #posterior over actions
    post_a_strong = posterior_over_actions(worlds[:repetitions], repetitions)
    post_a_weak = posterior_over_actions(worlds[repetitions:], repetitions)
   
    ax4[0].plot(post_a_strong[:,1], '.', color='#F97306', alpha=1)
    ax4[1].plot(post_a_weak[:,1], '.', color='#F97306', alpha=1)
    ax4[0].plot(post_a_strong[:,1], '-', color='#F97306', alpha=1)
    ax4[1].plot(post_a_weak[:,1], '-', color='#F97306', alpha=1)    
    ax4[0].set_ylabel('$Q_a$:posterior over actions')
    ax4[0].set_xlabel("trial number")
    ax4[1].set_xlabel("trial number")

    ax0 = [ax1[0],ax2[0],ax3[0],ax4[0]]
    ax1 = [ax1[1],ax2[1],ax3[1],ax4[1]]    
    for i in range(len(ax0)):
        ax1[i].yaxis.tick_right()
        ax1[i].yaxis.set_label_position("right")
        
        ax0[i].set_ylim([-0.01,1.01])
        ax1[i].set_ylim([-0.01,1.01])
        ax0[i].set_yticks([0,0.5,1.0])
        if i==0:            
            ax1[i].set_yticks([0,1])
            ax1[i].set_yticklabels(['c1','c2'])
            #ax1[i].set_ylabel('context')
        else:
            ax1[i].set_yticks([0,1])
            ax1[i].set_yticklabels(['a1','a2'])
            #ax1[i].set_ylabel('action')            
        ax0[i].set_xlim([-10,trials+10])  
        ax1[i].set_xlim([-10,trials+10])
        if i <3:
            ax0[i].set_xticks([])
            ax1[i].set_xticks([])
        ax0[i].plot(rewards_prob[0], color = 'xkcd:light blue', label = 'lever 1',alpha=0.5)
        ax0[i].plot(rewards_prob[1], color = 'blue', label = 'lever 2',alpha=0.5)
        ax1[i].plot(rewards_prob[0], color = 'xkcd:light blue', alpha=0.5)
        ax1[i].plot(rewards_prob[1], color = 'blue', alpha=0.5)  
        
    
    import matplotlib.lines as mlines
    
    lever1 = mlines.Line2D([], [], color='xkcd:light blue', alpha=0.5,
                          label='lever 1')
    lever2 = mlines.Line2D([], [], color='blue', alpha=0.5,
                          label='lever 2')
    action = mlines.Line2D([], [], color='#F97306', alpha=1,
                          label='action')
    context = mlines.Line2D([], [], color='#FF81C0', alpha=1,
                          label='context')
    
    ax0[0].legend([lever1, lever2, action, context], 
                  ['lever 1', 'lever 2', 'action', 'context'], 
                  loc = 'center left', bbox_to_anchor=(0.15, 0.5))

    plt.tight_layout()
    plt.show()    
    
def plot_variables_dynamic_mbp(worlds, repetitions):
    """
    plot the dynamic changes of variables over trials
    """
    na = worlds[0].na
    trials = worlds[0].trials

        
    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5, figsize= (5,8))
#    plt.suptitle('$w_g$:'+str(worlds[0].w_g)+' '+'$w_h$:'+str(worlds[0].w_h)+' '+ \
#                     '$w_0$:'+str(worlds[0].w_0)+' '+r'$\alpha_H$'+str(worlds[0].alpha_H)+' '+ \
#                         r'$\alpha_R$:'+str(worlds[0].alpha_R))
        
    # for probability of reinforcement
    ax1.plot(worlds[0].env.AM[:,0,1], color='#0343DF', label='Action A', linewidth=2.5)
    ax1.plot(worlds[0].env.AM[:,1,1], color='#008000', label='Action B', linewidth=2.5)
    ax1.set_title("probability of reinforcement")
    ax1.text(30,0.7, 'Action A', color = '#0343DF', fontweight = 'demibold')
    ax1.text(130,0.7, 'Action B', color = '#008000', fontweight = 'demibold')
    ax1.set_ylim([-0.01,1.01])
    ax1.set_yticks([0,0.5,1])
    
    # for Goal-Dierected Values(Q)
    Q = np.zeros((trials,na))
    for i in range(repetitions):
        Q += worlds[i].Q
    Q /= repetitions
    
    ax2.plot(Q[:,0], color='#0343DF', label='Action A', linewidth=2.5)
    ax2.plot(Q[:,1], color='#008000', label='Action B', linewidth=2.5)
    ax2.set_title("Goal-Dierected Values(Q)")
    ax2.set_ylim([-0.01,4.01])
    ax2.set_yticks([0,2,4])
    
    # for Habit Strengths(H)
    H = np.zeros((trials,na))
    for i in range(repetitions):
        H += worlds[i].H
    H /= repetitions

    ax3.plot(H[:,0], color='#0343DF', label='Action A', linewidth=2.5)
    ax3.plot(H[:,1], color='#008000', label='Action B', linewidth=2.5)
    ax3.set_title("Habit Strengths(H)")
    ax3.set_ylim([-0.01,0.61])
    ax3.set_yticks([0,0.3,0.6])    

    # for Weight of Goal-Directed Control(W)
    W = np.zeros(trials)
    for i in range(repetitions):
        W += worlds[i].weights
    W /= repetitions
    W = 1 - W
    
    ax4.plot(W, color='black', linewidth=2.5)
    ax4.plot(np.ones(trials)*0.5, color='#808080', linestyle='--',linewidth=1.5)
    ax4.set_title("Weight of Goal-Directed Control(W)")
    ax4.set_ylim([-0.01,1.01])
    ax4.set_yticks([0,0.5,1])    

    # for Choice Probability
    P = np.zeros((trials,na))
    for i in range(repetitions):
        P += worlds[i].P
    P /= repetitions  

    ax5.plot(P[:,0], color='#0343DF', label='Action A', linewidth=2.5)
    ax5.plot(P[:,1], color='#008000', label='Action B', linewidth=2.5)
    ax5.set_title("Choice Probability")
    ax5.set_xlabel('trial number')
    ax5.set_ylim([-0.01,1.01])
    ax5.set_yticks([0,0.5,1])    

    ax = [ax1,ax2,ax3,ax4,ax5]
    for i in range(len(ax)):
        ax[i].set_xlim([-1,trials])

    plt.tight_layout()
    plt.show()    






def posterior_over_contexts(worlds, repetitions):
    if repetitions != 1:
        post_c = np.zeros((worlds[0].agent.posterior_context.shape[0], 2))
        for i in range (0, repetitions):
            post_c[:,0] += np.array(worlds[i].agent.posterior_context[:,0,1])
            post_c[:,1] += np.array(worlds[i].agent.posterior_context[:,1,1])
        post_c /= repetitions
    else:
        post_c = np.zeros((worlds.agent.posterior_context.shape[0], 2))
        post_c[:,0] = np.array(worlds.agent.posterior_context[:,0,1])
        post_c[:,1] = np.array(worlds.agent.posterior_context[:,1,1])

    return post_c

def likelihood_over_actions(worlds, repetitions):
    likelihoods = np.zeros((worlds[0].agent.likelihood.shape[0], 2))
    for i in range (0, repetitions ):
        likelihood = []
        likelihood = (worlds[i].agent.likelihood[:,0,:,:]* worlds[i].agent.posterior_context[:,0,np.newaxis,:]).sum(axis=-1)
        likelihood /= likelihood.sum(axis=1)[:,np.newaxis]
        likelihoods += likelihood
    likelihoods /= repetitions
    return likelihoods
    
def prior_over_actions(worlds, repetitions):
    prior = np.zeros((worlds[0].agent.posterior_dirichlet_pol.shape[0], 2))

    for i in range (0, repetitions):
        prior_pol = np.exp(scs.digamma(worlds[i].agent.posterior_dirichlet_pol) - scs.digamma(worlds[i].agent.posterior_dirichlet_pol.sum(axis=1))[:,np.newaxis,:])
        prior_pol /= prior_pol.sum(axis=1)[:,np.newaxis,:]
        marginal_prior = (prior_pol[:,:,:] * worlds[i].agent.posterior_context[:,0,np.newaxis,:]).sum(axis=-1)
        prior += marginal_prior
    prior /= repetitions
    return prior
   
def posterior_over_actions(worlds, repetitions):
    if repetitions != 1:
        post_a = np.zeros((worlds[0].agent.posterior_policies.shape[0], 2))
        for i in range (0, repetitions):
            post_pol = (worlds[i].agent.posterior_policies[:,0,:,:]* worlds[i].agent.posterior_context[:,0,np.newaxis,:]).sum(axis=-1)
            post_a += post_pol
        post_a /= repetitions
    else:
        post_a = np.zeros((worlds.agent.posterior_policies.shape[0], 2))
        post_a = (worlds.agent.posterior_policies[:,0,:,:]* worlds.agent.posterior_context[:,0,np.newaxis,:]).sum(axis=-1)

    return post_a
    
def infer_time_baye(worlds, repetitions, trials):
    results_c_param = np.zeros((repetitions,4)) #(repetitions x 4)
    results_a_param = np.zeros((repetitions,4)) #(repetitions x 4)
    results_a_param_type = np.zeros((repetitions), dtype=int)
    results_c_param_type = np.zeros((repetitions), dtype=int)
    t_trials = trials - 100
    times = np.arange(0.+1,trials+1,1.) #(200 x 1)
    
    for i in range(repetitions):
        
        posterior_contexts = posterior_over_contexts(worlds[i], repetitions=1)
        posterior_actions = posterior_over_actions(worlds[i], repetitions=1)

        try:
            results_a_param[i], pcov = sc.optimize.curve_fit(sigmoid, times[10:], posterior_actions[10:,1], p0=[1.,1.,t_trials,0.])
            results_a_param_type[i] = 0
        except RuntimeError:
            try:
                results_a_param[i], pcov = sc.optimize.curve_fit(exponential, times[10:], posterior_actions[10:,1], p0=[1.,t_trials,0.])
                results_a_param[i, 0] = 1
                results_a_param_type[i] = 1
            except:
                results_a_param[i] = np.nan
                results_a_param_type[i] = 0

        try:
            results_c_param[i], pcov = sc.optimize.curve_fit(sigmoid, times, posterior_contexts[:,1], p0=[1.,1.,t_trials,0.])
            results_c_param_type[i] = 0
        except RuntimeError:
            try:
                results_c_param[i], pcov = sc.optimize.curve_fit(exponential, times, posterior_contexts[:,1], p0=[1.,t_trials,0.])
                results_c_param[i, 0] = 1
                results_c_param_type[i] = 1
            except:
                results_c_param[i] = np.nan
                results_c_param_type[i] = 0

    #拟合函数sigmoid，自变量：times，因变量：posterior_context，p0：给函数的参数确定一个初始值来减少计算机的计算量

        if results_a_param[i,0] < 0.1 or results_a_param[i,1] < 0.0 or results_a_param[i,2] < 15 or results_a_param[i,2] > trials:
            results_a_param[i] = [0,0,trials+1,0]

        if results_c_param[i,0] < 0.1 or results_c_param[i,1] < 0.0 or results_c_param[i,2] < 15 or results_c_param[i,2] > trials:
            results_c_param[i] = [0,0,trials+1,0]
    
    return results_c_param, results_a_param, results_c_param_type, results_a_param_type[i]    