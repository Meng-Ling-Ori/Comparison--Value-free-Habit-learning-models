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
import matplotlib.lines as mlines

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

def plot_chosen_probability(worlds, repetitions, habit_learning_strength, weak=False, baye=False, EX1_2=False):
    
    trials=worlds[0].trials
    post_a = np.zeros((trials, 2))
# =============================================================================
#     post_a = np.zeros((repetitions, trials, 2))
# =============================================================================
    
    if baye: 
        clr = '#F97306'
        habit_learning_strength = 1/habit_learning_strength
        label = 'h=' + str(habit_learning_strength)
    else:
        clr = '#008000'
        label = r'$\alpha_H$=' + str(habit_learning_strength)
    
# =============================================================================
#     if baye:
#         for i in range(0,repetitions):
#             post_a[i] = (worlds[i].agent.posterior_policies[:,0,:,:]* worlds[i].agent.posterior_context[:,0,np.newaxis,:]).sum(axis=-1)
#     else:
#         for i in range(0,repetitions):
#             post_a[i] = worlds[i].P
# =============================================================================
    if baye:
        for i in range (0, repetitions ):
            post_pol = (worlds[i].agent.posterior_policies[:,0,:,:]* worlds[i].agent.posterior_context[:,0,np.newaxis,:]).sum(axis=-1)
            post_a += post_pol
    else:
        for i in range (0, repetitions ):
            chosen_p = worlds[i].P
            post_a += chosen_p
    post_a /= repetitions                
    
    
    fig, ax = plt.subplots(nrows=1, figsize= (5,2))
    if not weak:
        ax.plot(post_a[:,1], color=clr, linewidth=2.5)
    else:
        ax.plot(post_a[:,1], color=clr, linewidth=2.5, linestyle='--')
    ax.plot([trials/2,trials/2], [0,1], color='dimgrey', linestyle='--', linewidth=1.5)
    ax.text(trials/8,1.1, 'context 1', color = 'black', fontweight = 'regular')
    ax.text(trials/8*6,1.1, 'context 2', color = 'black', fontweight = 'regular')
    ax.axvspan(trials/2,trials,facecolor='lightgrey', alpha=0.7)
    ax.set_xlabel("trial number", fontsize='large')
    if not EX1_2:        
        ax.set_ylabel('Lever 2 press probability', va='center', 
                 rotation='vertical', fontsize='large')  
    else:
        ax.set_ylabel('Withhold press probability', va='center', 
                 rotation='vertical', fontsize='large')  
    ax.set_ylim([-0.01,1.01])
    ax.set_yticks([0,0.5,1.0])
    ax.set_xlim([-10,trials+10]) 
    
    ax.plot(np.ones(trials)*0.5, color="black", linestyle='--',linewidth=1.5)
    
    label_0 = mlines.Line2D([], [], color=clr, alpha=0.5,
                          label=label)
    ax.legend([label_0], 
                  [label], 
                  loc = 'upper left', bbox_to_anchor=(0.0, 1), fontsize='small')    

    plt.tight_layout()
    plt.show()

def plot_chosen_probability_stoch(worlds, repetitions, habit_learning_strength, baye=False, EX1_2=False):
    
    trials=worlds[0].trials
    post_a = np.zeros((trials, 2))
    
    if baye: 
        clr = '#F97306'
        habit_learning_strength = 1/habit_learning_strength
        label_strong = 'h=' + str(habit_learning_strength[0])
        label_weak = 'h=' + str(habit_learning_strength[1])
    else:
        clr = '#008000'
        label_strong = r'$\alpha_H$=' + str(habit_learning_strength[0])
        label_weak = r'$\alpha_H$=' + str(habit_learning_strength[1])
    
    stochasticities = [0.1, 0.2, 0.3, 0.4]
    n_env = len(stochasticities)
    n_agt = 2
    post_a = np.zeros((n_env,n_agt,trials,2))
    for i in range(n_env):
        for j in range(n_agt):                
            for k in range (0, repetitions):
                n = j*repetitions*n_env + i*repetitions + k
                if baye:
                    post_pol = (worlds[n].agent.posterior_policies[:,0,:,:]* 
                                worlds[n].agent.posterior_context[:,0,np.newaxis,:]).sum(axis=-1)
                    post_a[i,j] += post_pol
                else:
                    chosen_p = worlds[n].P
                    post_a[i,j] += chosen_p  
            post_a[i, j] /= repetitions                
    
    
    fig, ax = plt.subplots(nrows=n_env, figsize= (5,8))
    
    for i in range(n_env):
        ax[i].plot(post_a[i,0,:,1], color=clr, linewidth=2.5)
        ax[i].plot(post_a[i,1,:,1], color=clr, linewidth=2.5, linestyle='--')
        ax[i].set_ylabel('1-v='+str(stochasticities[i]))
        ax[i].set_ylim([-0.01,1.01])
        ax[i].set_yticks([0,0.5,1.0])
        ax[i].set_xlim([-10,trials+10]) 
        ax[i].plot(np.ones(trials)*0.5, color='#808080', linestyle='--',linewidth=1.5)
        
    fig.text(0.5, 0, 'trial number', ha='center', fontsize='x-large')
    if not EX1_2:        
        fig.text(0, 0.5, 'Lever 2 press probability', va='center', 
                 rotation='vertical', fontsize='x-large')  
    else:
        fig.text(0, 0.5, 'Withhold press probability', va='center', 
                 rotation='vertical', fontsize='x-large')        

    import matplotlib.lines as mlines
    
    strong_label = mlines.Line2D([], [], color=clr, alpha=0.5,
                          label=label_strong)
    weak_label = mlines.Line2D([], [], color=clr, alpha=0.5, linestyle='--',
                          label=label_weak)

    
    ax[0].legend([strong_label, weak_label], 
                  [label_strong, label_weak], 
                  loc = 'upper left', bbox_to_anchor=(0.0, 1))

    plt.tight_layout()
    plt.show() 
    return post_a

def plot_chosen_probability_training_fake(worlds, repetitions, training_durations, test_duration, baye=False):
    
    trials_list = np.array(training_durations) + test_duration
    n_duration = len(trials_list)
    n_agt = 2
    
    post_a = np.zeros((n_duration, n_agt, trials_list[-1], 2))
    
    if baye: 
        clr = '#F97306'
    else:
        clr = '#008000'
    

    for i in range(n_duration):
        for j in range(n_agt):                
            for k in range (repetitions):
                n = i*repetitions*n_agt + j*repetitions + k
                print(i,j,k,n)
                t_start = trials_list[-1] - worlds[n].trials
                post_a[i,j,:t_start] = - 10000
                if baye:
                    chosen_p = (worlds[n].agent.posterior_policies[:,0,:,:]* 
                                worlds[n].agent.posterior_context[:,0,np.newaxis,:]).sum(axis=-1)
                else:
                    chosen_p = worlds[n].P
                post_a[i,j,t_start:] += chosen_p     
            post_a[i, j] /= repetitions                
    
    
    fig, ax = plt.subplots(nrows=n_duration, figsize= (8,8))
    
    for i in range(n_duration):
        ax[i].plot(post_a[i,0,:,1], color=clr, linewidth=2.5)
        ax[i].plot(post_a[i,1,:,1], color=clr, linewidth=2.5, linestyle='--')
        ax[i].set_ylabel('training duration:'+str(training_durations[i]))
        ax[i].set_ylim([-0.01,1.01])
        ax[i].set_yticks([0,0.5,1.0])
        ax[i].set_xlim([-10,trials_list[-1]+10])
        n = i*repetitions*n_agt
        ax[i].set_xticks([trials_list[-1] - worlds[n].trials, 
                          trials_list[-1] - test_duration, 
                          trials_list[-1]])
        ax[i].set_xticklabels(['0', str(training_durations[i]), str(trials_list[i])])
        x = np.arange(trials_list[-1] - worlds[n].trials, trials_list[-1])
        y = np.zeros(worlds[n].trials) + 0.5
        data_sup = np.zeros((trials_list[-1])) + 0.5
        ax[i].plot(x, y,
                   color='#808080', linestyle='--',linewidth=1.5)
        
    fig.text(0.5, 0, 'trial number', ha='center')
    fig.text(0, 0.5, 'Action B chosen probability', va='center', rotation='vertical')  

    import matplotlib.lines as mlines
    
    strong_label = mlines.Line2D([], [], color=clr, alpha=0.5,
                          label='strong habit learner')
    weak_label = mlines.Line2D([], [], color=clr, alpha=0.5, linestyle='--',
                          label='weak habit learner')

    
    ax[0].legend([strong_label, weak_label], 
                  ['strong habit learner', 'weak habit learner'], 
                  loc = 'upper left', bbox_to_anchor=(0.0, 1))

    plt.tight_layout()
    plt.show() 
    return post_a

def plot_chosen_probability_training(worlds, repetitions, training_durations, test_duration, phasename, baye=False):
    
    trials_list = np.array(training_durations) + test_duration
    n_duration = len(trials_list)
    n_agt = 2
    
    post_a = np.zeros((n_duration, n_agt, trials_list[-1], 2))
    
    if baye: 
        clr = '#F97306'
    else:
        clr = '#008000'
    

    for i in range(n_duration):
        for j in range(n_agt):                
            for k in range (repetitions):
                n = i*repetitions*n_agt + j*repetitions + k
                #print(i,j,k,n)
                t_start = trials_list[-1] - worlds[n].trials
                post_a[i,j,:t_start] = - 10000
                if baye:
                    chosen_p = (worlds[n].agent.posterior_policies[:,0,:,:]* 
                                worlds[n].agent.posterior_context[:,0,np.newaxis,:]).sum(axis=-1)
                else:
                    chosen_p = worlds[n].P
                post_a[i,j,t_start:] += chosen_p     
            post_a[i, j, :] /= repetitions            
    
    fig = plt.figure(figsize= (10,2*n_duration))
    
    for i in range(n_duration):
        #print(i)
        ax = plt.subplot2grid((n_duration, int(trials_list[-1]/100)), 
                              (i, int((trials_list[-1]-trials_list[i])/100)), 
                              colspan=int(trials_list[-1]/100))
        n = i*repetitions*n_agt
        t_start = trials_list[-1] - worlds[n].trials
        ax.plot(post_a[i,0,t_start:,1], color=clr, linewidth=2.5)
        ax.plot(post_a[i,1,t_start:,1], color=clr, linewidth=2.5, linestyle='--')
        ax.text(0.0, 0.8, 'training duration:'+str(training_durations[i]),
                bbox=dict(facecolor='white', edgecolor='black',linewidth=1.5))
        ax.set_ylim([-0.01,1.01])                                                                                                                                                                               
        ax.set_yticks([0,0.5,1.0])
        ax.set_xlim([-10,trials_list[i]+10])                                                                        
        ax.set_xticks([0, training_durations[i], trials_list[i]])
#        ax.set_xticklabels([0, str(training_durations[i])+' (context swiches)', trials_list[i]])
        ax.plot(np.zeros(trials_list[-1])+0.5,
                   color='#808080', linestyle='--',linewidth=1.5)
        if i == n_duration-1:
            ax.set_xlabel('trial number')
                                                                                                                                                                     
    #fig.text(0.5, 0, 'trial number', ha='center')
    fig.text(0, 0.5, 'Action B chosen probability', va='center', rotation='vertical')  

    import matplotlib.lines as mlines
    
    strong_label = mlines.Line2D([], [], color=clr, alpha=0.5,
                          label='strong habit learner')
    weak_label = mlines.Line2D([], [], color=clr, alpha=0.5, linestyle='--',
                          label='weak habit learner')

    
    ax.legend([strong_label, weak_label], 
                  ['strong habit learner', 'weak habit learner'], 
                  loc = 'upper left', bbox_to_anchor=(0.2, 1))
    
    ax.text(training_durations[-1]/2, -0.45, 'training phase', backgroundcolor="white",
            ha='center', va='top', weight='bold', color='black') #x, y, text
    ax.annotate('', xy = (training_durations[-1]/2, -0.2),
            xytext = (0, -0.2),
            arrowprops=dict(arrowstyle='->', facecolor='black'))  
    ax.text((trials_list[-1]-training_durations[-1])/2+training_durations[-1], -0.45, phasename, backgroundcolor="white",
            ha='center', va='top', weight='bold', color='black') #x, y, text

    fig = plt.figure(figsize= (10,2*n_duration))
    
    for i in range(n_duration):
        #print(i)
        ax = plt.subplot2grid((n_duration, int(trials_list[-1]/100)), 
                              (i, int((trials_list[-1]-trials_list[i])/100)), 
                              colspan=int(trials_list[-1]/100))
        n = i*repetitions*n_agt
        t_start = trials_list[-1] - worlds[n].trials
        ax.plot(post_a[i,0,t_start:,0], color=clr, linewidth=2.5)
        ax.plot(post_a[i,1,t_start:,0], color=clr, linewidth=2.5, linestyle='--')
        ax.text(0.0, 0.2, 'training duration:'+str(training_durations[i]),
                bbox=dict(facecolor='white', edgecolor='black',linewidth=1.5))
        ax.set_ylim([-0.01,1.01])                                                                                                                                                                               
        ax.set_yticks([0,0.5,1.0])
        ax.set_xlim([-10,trials_list[i]+10])                                                                        
        ax.set_xticks([0, training_durations[i], trials_list[i]])
#        ax.set_xticklabels([0, str(training_durations[i])+' (context swiches)', trials_list[i]])
        ax.plot(np.zeros(trials_list[-1])+0.5,
                   color='#808080', linestyle='--',linewidth=1.5)
        if i == n_duration-1:
            ax.set_xlabel('trial number')
                                                                                                                                                                     
    #fig.text(0.5, 0, 'trial number', ha='center')
    fig.text(0, 0.5, 'Action A chosen probability', va='center', rotation='vertical')  

    import matplotlib.lines as mlines
    
    strong_label = mlines.Line2D([], [], color=clr, alpha=0.5,
                          label='strong habit learner')
    weak_label = mlines.Line2D([], [], color=clr, alpha=0.5, linestyle='--',
                          label='weak habit learner')

    
    ax.legend([strong_label, weak_label], 
                  ['strong habit learner', 'weak habit learner'], 
                  loc = 'lower left', bbox_to_anchor=(0.2, 0))
    
    ax.text(training_durations[-1]/2, -0.45, 'training phase', backgroundcolor="white",
            ha='center', va='top', weight='bold', color='black') #x, y, text
    ax.annotate('', xy = (training_durations[-1]/2, -0.2),
            xytext = (0, -0.2),
            arrowprops=dict(arrowstyle='->', facecolor='black'))  
    ax.text((trials_list[-1]-training_durations[-1])/2+training_durations[-1], -0.45, phasename, backgroundcolor="white",
            ha='center', va='top', weight='bold', color='black') #x, y, text

    
    plt.tight_layout()
    plt.show()
    
    return post_a

def plot_habit_strength_replication(worlds, repetitions, alpha_policies, 
                        EX_D1=False):
    
    ls = 1/np.array(alpha_policies) 
    t_trials = worlds[0].trials_training
    strong = str(ls[0])
    weak = str(ls[1])
    
    

    results_a = np.zeros((len(ls), repetitions ,4))
    a_type = np.zeros((len(ls), repetitions), dtype=int)
    results_c = np.zeros((len(ls), repetitions ,4))
    c_type = np.zeros((len(ls), repetitions), dtype=int)
    for i in range(len(ls)):
        results_c[i], results_a[i], c_type[i], a_type[i] = infer_time_baye(
                                worlds[i*repetitions:(i+1)*repetitions], 
                                repetitions)
       

    plot_c_0 = results_c[:,:,2]-100
    plot_a_0 = results_a[:,:,2]-100
    
    plot_a = plot_a_0.T 
    plot_c = plot_c_0.T 
    
    num_tend = plot_a.shape[1]
    num_runs = plot_a.shape[0]
    plot_c_data = plot_c.T.reshape(plot_c.size)
    plot_a_data = plot_a.T.reshape(plot_a.size)
    
    
    labels = np.tile(ls, (num_runs, 1)).reshape(-1, order='f')

    data_a = pd.DataFrame({'chosen': plot_a_data, 'tendencies': labels})
    data_c = pd.DataFrame({'chosen': plot_c_data, 'tendencies': labels})
    


    plt.figure()
    plt.title("action and context infer times")    
    sns.lineplot(x='tendencies', y='chosen', data=data_a, ci = 95, 
                     color='#F97306', estimator=np.nanmedian, label="action", linewidth=3)
    sns.lineplot(x='tendencies', y='chosen', data=data_c, ci = 95, 
                 color='#FF81C0', estimator=np.nanmedian, label="context", linewidth=3)

        
    ax = plt.gca()
    ax.set_ylabel("habit strength/trial after reverse", fontsize='x-large')
    ax.set_xlabel("habitual tendency $h$", fontsize='x-large')
    ax.set_xticks([0.01,1.0])
    ax.set_xticklabels([weak,strong])
    ax.xaxis.set_ticks_position('bottom')         
    
    if not EX_D1:
        ax.set_ylim([0,20])
        ax.set_yticks([0,4,8,12,16,20])        
    else:
        ax.set_ylim([0,5])
        ax.set_yticks([0,1,2,3,4,5])

            
    plt.tight_layout()
    plt.show()

def plot_habit_strength(worlds, repetitions, habit_learning_strength, 
                        EX1_2=False,
                        series=False, baye=False, EX3=False):
    
    
    if baye:
        clr = '#F97306'
        ls = 1/np.array(habit_learning_strength) 
        t_trials = worlds[0].trials_training
    else:
        clr = '#008000'
        ls = habit_learning_strength  
        t_trials = worlds[0].trials_phase1
    strong = str(ls[0])
    weak = str(ls[1]) 
    
    
    
    results_a = np.zeros((len(ls), repetitions ,4))
    a_type = np.zeros((len(ls), repetitions), dtype=int)
    if baye:
        results_c = np.zeros((len(ls), repetitions ,4))
        c_type = np.zeros((len(ls), repetitions), dtype=int)
        for i in range(len(ls)):
            results_c[i], results_a[i], c_type[i], a_type[i] = infer_time_baye(
                                worlds[i*repetitions:(i+1)*repetitions], 
                                repetitions)
    else:
        for i in range(len(ls)):
            results_a[i], a_type[i] = infer_time_mbp(
                                worlds[i*repetitions:(i+1)*repetitions], 
                                repetitions)
        
#    results_c[0] = results_c_param_s #i h浓度,k 奖励先验浓度，p_values， l 没啥用
#    results_c[1] = results_c_param_w

#    plot_c_0 = results_c[:,:,2]-100

    plot_a_0 = results_a[:,:,2]-t_trials
    plot_a = plot_a_0.T 
    
    num_tend = plot_a.shape[1]
    num_runs = plot_a.shape[0]
    plot_a_data = plot_a_0.flatten()
    
    plot_b = np.zeros((2,num_runs))
    plot_b[0] = plot_a_0[0]
    plot_b[1] = plot_a_0[-1]
    plot_b_data = plot_b.flatten()
    
    labels = np.tile(ls, (num_runs, 1)).reshape(-1, order='f')
    labels_b = np.tile([ls[0],ls[-1]], (num_runs, 1)).reshape(-1, order='f')
    #labels = labels[: :-1] 
    data_a = pd.DataFrame({'chosen': plot_a_data, 'tendencies': labels})
    data_b = pd.DataFrame({'chosen_b': plot_b_data, 'tendencies_b': labels_b})
    
    import misc_mbp as mim
    if series:
        x = ls
        y = np.median(plot_a_0,1)

        x, y = mim.remove_nan(x, y)
        res = sc.stats.linregress(x, y)
        #print(res.slope, res.rvalue, res.pvalue)


    plt.figure()
    #plt.title("action and context infer times")
    if not series:
        plt.boxplot([plot_a_data[repetitions:], plot_a_data[:repetitions]], 
                    labels=[weak,strong], boxprops={'color':clr, 'linewidth':'3'},
                    flierprops={'markerfacecolor':clr},medianprops={'color':clr})
    
    else:
        sns.lineplot(x='tendencies', y='chosen', data=data_a, ci = 95, 
                     color=clr, estimator=np.nanmedian, linewidth=3)

        
    ax = plt.gca()
    ax.set_ylabel(r"habit strength $H$", fontsize='x-large')
    if series:
        ax.set_xticks([min(ls), np.median(ls), max(ls)])
        plt.plot(ls, res.intercept + res.slope*ls, color='#808080', 
                 linestyle='--', linewidth=1.5)
        #sns.lineplot(x='tendencies_b', y='chosen_b', data=data_b, ci = None, color='#808080', estimator=np.nanmedian, linestyle='--', linewidth=1.5)
        l = (max(ls) - min(ls))/100
        ax.set_xlim([min(ls)-l,max(ls)+l])
        ax.set_ylabel(r"habit strength $H$", fontsize='x-large')
        ax.set_ylim([0,20])
        ax.set_yticks(np.array([0,4,8,12,16,20]))
    if EX1_2:
        ax.set_ylim(np.array([0,20])*10)
        ax.set_yticks(np.array([0,4,8,12,16,20])*10)  
    elif EX3:
        ax.set_ylim(np.array([0,400]))
        ax.set_yticks(np.array([0,80,160,240,320,400]))         
    else: 
        if not series:
            ax.set_ylim([0,40])
            ax.set_yticks(np.array([0,8,16,24,32,40]))
    if baye:
        plt.xlabel("habitual tendency $h$", fontsize='x-large')
    else:
        plt.xlabel(r"step parameter in habitual control $\alpha_H$", fontsize='x-large')
            


    plt.tight_layout()
    plt.show()
    return plot_a, results_a, a_type

def plot_habit_strength_training(worlds, repetitions, habit_learning_strength, 
                                 training_durations=[0], EX4_1=False, baye=False):
    
    
    if baye:
        clr = '#F97306'
        ls = 1/np.array(habit_learning_strength)
        strong = 'h='+str(ls[0])
        weak = 'h='+str(ls[1]) 
    else:
        clr = '#008000'
        ls = habit_learning_strength 
        strong = r'$\alpha_H$='+str(ls[0])
        weak = r'$\alpha_H$='+str(ls[1])         

    n = len(training_durations)
        
    results_a = np.zeros((len(ls), n, repetitions ,4))
    a_type = np.zeros((len(ls), n, repetitions), dtype=int)
    if baye:
        results_c = np.zeros((len(ls), n, repetitions ,4))
        c_type = np.zeros((len(ls), n, repetitions), dtype=int)
        for i in range(len(ls)):
            for j in range(n):
                results_c[i,j], results_a[i,j], c_type[i,j], a_type[i,j] = \
                                infer_time_baye(
                                worlds[(j*len(ls)+i)*repetitions:(j*len(ls)+i+1)*repetitions], 
                                repetitions)
    else:
        for i in range(len(ls)):
            for j in range(n):
                results_a[i,j], a_type[i,j] = infer_time_mbp(
                                worlds[(j*len(ls)+i)*repetitions:(j*len(ls)+i+1)*repetitions], 
                                repetitions)
        
#    results_c[0] = results_c_param_s #i h浓度,k 奖励先验浓度，p_values， l 没啥用
#    results_c[1] = results_c_param_w

#    plot_c_0 = results_c[:,:,2]-100
    t_trials = np.zeros((len(ls), n, repetitions))
    for i in range(n):
        if baye:
            t_trials[:,i,:] = worlds[i*len(ls)*repetitions].trials_training
        else:
            t_trials[:,i,:] = worlds[i*len(ls)*repetitions].trials_phase1
        
    plot_a_0 = results_a[:,:,:,2] - t_trials
    plot_a = np.zeros((len(ls), repetitions, n))
    for i in range(len(ls)):
        plot_a[i] = plot_a_0[i,:,:].T
    
    num_durations = plot_a.shape[1]
    num_runs = repetitions
    plot_a1_data = plot_a_0[0,:,:].flatten()
    plot_a2_data = plot_a_0[1,:,:].flatten()
    
    labels = np.tile(training_durations, (num_runs, 1)).reshape(-1, order='f')
    #labels = labels[: :-1] 
    data_a1 = pd.DataFrame({'chosen': plot_a1_data, 'durations': labels})
    data_a2 = pd.DataFrame({'chosen': plot_a2_data, 'durations': labels})

    plt.figure()
    #plt.title("action and context infer times")
    sns.lineplot(x='durations', y='chosen', data=data_a1, ci = 95, color=clr, 
                 label = strong, estimator=np.nanmedian, linewidth=3)
    sns.lineplot(x='durations', y='chosen', data=data_a2, ci = 95, color=clr, 
                 label = weak, estimator=np.nanmedian, linewidth=3, linestyle='--')

    
    if EX4_1:
        plt.xlabel("training duration (in log scale)", fontsize='x-large')    
        plt.ylim(0,100)
        plt.xscale('log')
        ax = plt.gca()
        ax.set_xticks([100,1000,10000])
        ax.set_xticklabels(["$100$","$1000$","$10000$"])
    else:
        plt.xlabel("training duration", fontsize='x-large')    
        plt.ylim(0,500)
        ax = plt.gca()
        ax.set_xticks([100, 500, 1000, 1500, 2000])            
    plt.ylabel(r"habit strength $H$", fontsize='x-large')
 
    
    plt.tight_layout()
    plt.legend(loc='upper left')
    plt.show()
    return plot_a, results_a, a_type     
    
def plot_habit_strength_stoch(worlds, repetitions, habit_learning_strength, 
                              alpha_rewards,baye=False):
    
    
    if baye:
        clr = '#F97306'
        ls = 1/np.array(habit_learning_strength)
        strong = 'h='+str(ls[0])
        weak = 'h='+str(ls[1]) 
    else:
        clr = '#008000'
        ls = habit_learning_strength 
        strong = r'$\alpha_H$='+str(ls[0])
        weak = r'$\alpha_H$='+str(ls[1])         

    n = len(alpha_rewards)
       
    results_a = np.zeros((len(ls), n, repetitions ,4))
    a_type = np.zeros((len(ls), n, repetitions), dtype=int)
    if baye:
        results_c = np.zeros((len(ls), n, repetitions ,4))
        c_type = np.zeros((len(ls), n, repetitions), dtype=int)
        for i in range(len(ls)):
            for j in range(n):
                results_c[i,j], results_a[i,j], c_type[i,j], a_type[i,j] = \
                                infer_time_baye(
                                worlds[(i*n+j)*repetitions:(i*n+j+1)*repetitions], 
                                repetitions)
    else:
        for i in range(len(ls)):
            for j in range(n):
                results_a[i,j], a_type[i,j] = infer_time_mbp(
                                worlds[(i*n+j)*repetitions:(i*n+j+1)*repetitions], 
                                repetitions)
        
#    results_c[0] = results_c_param_s #i h浓度,k 奖励先验浓度，p_values， l 没啥用
#    results_c[1] = results_c_param_w

#    plot_c_0 = results_c[:,:,2]-100
    t_trials = np.zeros((len(ls), n, repetitions))
    for i in range(n):
        if baye:
            t_trials[:,i,:] = worlds[i*len(ls)*repetitions].trials_training
        else:
            t_trials[:,i,:] = worlds[i*len(ls)*repetitions].trials_phase1
        
    plot_a_0 = results_a[:,:,:,2] - t_trials
    plot_a = np.zeros((len(ls), repetitions, n))
    for i in range(len(ls)):
        plot_a[i] = plot_a_0[i,:,:].T
    
    num_rewards = n
    num_runs = repetitions
    plot_a1_data = plot_a_0[0,:,:].flatten()
    plot_a2_data = plot_a_0[1,:,:].flatten()
    #plot_a1_data = plot_a1_data.flatten()

    labels = np.tile(np.arange(num_rewards), (num_runs, 1)).reshape(-1, order='f')
    #labels = labels[: :-1] 
    data_a1 = pd.DataFrame({'chosen': plot_a1_data, 'rewards': labels})
    data_a2 = pd.DataFrame({'chosen': plot_a2_data, 'rewards': labels})

    plt.figure()
    #plt.title("action and context infer times")
    sns.lineplot(x='rewards', y='chosen', data=data_a1, ci = 95, color=clr, 
                 label = strong, estimator=np.nanmedian, linewidth=3)
    sns.lineplot(x='rewards', y='chosen', data=data_a2, ci = 95, color=clr, 
                 label = weak, estimator=np.nanmedian, linewidth=3, linestyle='--')

    

    plt.xlabel("Environmental stochasticity $1-v$", fontsize='large')    
    plt.ylim(0,100)
    ax = plt.gca()
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(["$0.1$","$0.2$","$0.3$","$0.4$"])        
    plt.ylabel(r"habit strength $H$", fontsize='x-large')
    
    plt.legend(loc='upper left', fontsize='large')
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    #plt.title("action and context infer times")
    sns.lineplot(x='rewards', y='chosen', data=data_a1, ci = 95, color=clr, 
                 label = strong, estimator='mean', err_style="bars",  linewidth=3)
    sns.lineplot(x='rewards', y='chosen', data=data_a2, ci = 95, color=clr, 
                 label = weak, estimator='mean', err_style="bars",  linewidth=3, linestyle='--')

    

    plt.xlabel("Environmental stochasticity $1-v$", fontsize='large')    
    plt.ylim(0,100)
    ax = plt.gca()
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(["$0.1$","$0.2$","$0.3$","$0.4$"])        
    plt.ylabel("habit strength", fontsize='large')
    
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    return plot_a_0, results_a

def plot_chosen_probability_phaseend(worlds, habit_learning_strength, repetitions, 
                                     devaluation=False, baye=False):

    if baye:
        clr = '#F97306'
        ls = 1/np.array(habit_learning_strength)
#        strong = 'h='+str(ls[0])
#        weak = 'h='+str(ls[1]) 
    else:
        clr = '#008000'
        ls = habit_learning_strength 
#        strong = r'\alpha_H='+str(ls[0])
#        weak = r'\alpha_H='+str(ls[1])     

    

    trials_training = []
    for i in range(int(len(worlds)/(repetitions*len(ls)))):
        if baye:
            trials_training.append(worlds[i*repetitions*len(ls)].trials_training)
        else:
            trials_training.append(worlds[i*repetitions*len(ls)].trials_phase1)
    n = len(trials_training)

    
#    trials_training = np.arange(100,1001,100)
#    n = len(trials_training)
     
    p_actions = np.zeros((len(ls), n, repetitions))
    p2_actions = np.zeros((len(ls), n, repetitions))
    for i in range(len(ls)):
        for j in range(n):
            for k in range(repetitions):
                if baye:
                    chosen_p = posterior_over_actions(worlds[(j*len(ls)+i)*repetitions+k], 
                                                  repetitions=1)
                else:
                    chosen_p = worlds[(j*len(ls)+i)*repetitions+k].P
                p_actions[i,j,k] = chosen_p[int(trials_training[j]-1),0]
                p2_actions[i,j,k] = chosen_p[-1,0]
            
    labels = np.tile(np.arange(n), (repetitions, 1)).reshape(-1, order='f')
    
    plot_p_a1_data = p_actions[0,:,:].flatten()
    plot_p_a2_data = p_actions[1,:,:].flatten()
    plot_p2_a1_data = p2_actions[0,:,:].flatten()
    plot_p2_a2_data = p2_actions[1,:,:].flatten()
    
    data_p_a1 = pd.DataFrame({'chosen': plot_p_a1_data, 'training durations': labels})
    data_p_a2 = pd.DataFrame({'chosen': plot_p_a2_data, 'training durations': labels})
    data_p2_a1 = pd.DataFrame({'chosen': plot_p2_a1_data, 'training durations': labels})
    data_p2_a2 = pd.DataFrame({'chosen': plot_p2_a2_data, 'training durations': labels})

    plt.figure()
    if devaluation:
        label2 = 'after devaluation manipulate'
    else:
        label2 = 'after omission manipulate'
    #plt.title("action and context infer times")
    sns.lineplot(x='training durations', y='chosen', data=data_p_a1, ci = 95, 
                 color='black', label = 'end of training phase', estimator=np.nanmedian, linewidth=3)
    sns.lineplot(x='training durations', y='chosen', data=data_p2_a1, ci = 95, 
                 color=clr, label = label2, estimator=np.nanmedian, linewidth=3)
    

    plt.xlabel("training duration")
    plt.ylabel('action A chosen probability')

    plt.xlim([-1,n-1])
    plt.xticks([0-1,5-1,10-1,15-1,20-1], [0,500,1000,1500,2000])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.show() 

    plt.figure()
    #plt.title("action and context infer times")
    sns.lineplot(x='training durations', y='chosen', data=data_p_a2, ci = 95, 
                color='black', label = 'end of training phase', estimator=np.nanmedian, linewidth=3, linestyle = '--')
    sns.lineplot(x='training durations', y='chosen', data=data_p2_a2, ci = 95, 
                 color=clr, label = label2, estimator=np.nanmedian, linewidth=3, linestyle = '--')
    

    plt.xlabel("training duration")
    plt.ylabel('action A chosen probability')

    plt.xlim([-1,n-1])
    plt.xticks([0-1,5-1,10-1,15-1,20-1], [0,500,1000,1500,2000])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.show() 

    plt.figure()
    #plt.title("action and context infer times")
#    sns.lineplot(x='training durations', y='chosen', data=data_p_a2, ci = 95, 
#                color='black', label = 'end of training phase', estimator=np.nanmedian, linewidth=3, linestyle = '--')
#    sns.lineplot(x='training durations', y='chosen', data=data_p2_a2, ci = 95, 
#                 color=clr, label = label2, err_style="bars", estimator=np.nanmedian, linewidth=3, linestyle = '--')
#    sns.lineplot(x='training durations', y='chosen', data=data_p2_a2, ci = 95, 
#                 color=clr, label = label2, err_style="bars", estimator=np.nanmedian, linewidth=3)    
    

    plt.xlabel("training duration")
    plt.ylabel('action A chosen probability')

    plt.xlim([-1,n-1])
    plt.xticks([0-1,5-1,10-1,15-1,20-1], [0,500,1000,1500,2000])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.show()     
    
    return p_actions, p2_actions


def plot_chosen_probability_phaseend_mean(worlds, habit_learning_strength, repetitions, 
                                     devaluation=False, baye=False):

    if baye:
        clr = '#F97306'
        ls = 1/np.array(habit_learning_strength)
        strong = 'h='+str(round(ls[0],4))
        weak = 'h='+str(round(ls[1],4)) 
    else:
        clr = '#008000'
        ls = habit_learning_strength 
        strong = r'$\alpha_H$='+str(ls[0])
        weak = r'$\alpha_H$='+str(ls[1])     

    
    trials_training = []
    for i in range(int(len(worlds)/(repetitions*len(ls)))):
        if baye:
            trials_training.append(worlds[i*repetitions*len(ls)].trials_training)
        else:
            trials_training.append(worlds[i*repetitions*len(ls)].trials_phase1)
    n = len(trials_training)

    p_actions = np.zeros((len(ls), n, repetitions))
    p2_actions = np.zeros((len(ls), n, repetitions))
    for i in range(len(ls)):
        for j in range(n):
            for k in range(repetitions):
                if baye:
                    chosen_p = posterior_over_actions(worlds[(j*len(ls)+i)*repetitions+k], 
                                                  repetitions=1)
                else:
                    chosen_p = worlds[(j*len(ls)+i)*repetitions+k].P
                p_actions[i,j,k] = chosen_p[int(trials_training[j]-1),0]
                p2_actions[i,j,k] = chosen_p[-1,0]
            
    labels = np.tile(np.arange(n), (repetitions, 1)).reshape(-1, order='f')
    
    plot_p_a1_data = p_actions[0,:,:].flatten()
    plot_p_a2_data = p_actions[1,:,:].flatten()
    plot_p2_a1_data = p2_actions[0,:,:].flatten()
    plot_p2_a2_data = p2_actions[1,:,:].flatten()
    
    data_p_a1 = pd.DataFrame({'chosen': plot_p_a1_data, 'training durations': labels})
    data_p_a2 = pd.DataFrame({'chosen': plot_p_a2_data, 'training durations': labels})
    data_p2_a1 = pd.DataFrame({'chosen': plot_p2_a1_data, 'training durations': labels})
    data_p2_a2 = pd.DataFrame({'chosen': plot_p2_a2_data, 'training durations': labels})

    plt.figure()
    if devaluation:
        label2 = 'end of devaluation phase'
    else:
        label2 = 'end of omission phase'
    #plt.title("action and context infer times")
    sns.lineplot(x='training durations', y='chosen', data=data_p_a1, ci = 95, 
                 color='black', label = 'end of training phase', estimator='mean', 
                 err_style="bars", linewidth=3)
    sns.lineplot(x='training durations', y='chosen', data=data_p2_a1, ci = 95, 
                 color=clr, label = label2, estimator='mean', 
                 err_style="bars", linewidth=3)
    sns.lineplot(x='training durations', y='chosen', data=data_p_a2, ci = 95, 
                color='black', label = 'end of training phase', estimator='mean', 
                err_style="bars",linewidth=3, linestyle = '--')
    sns.lineplot(x='training durations', y='chosen', data=data_p2_a2, ci = 95, 
                 color=clr, label = label2, estimator='mean', 
                 err_style="bars", linewidth=3, linestyle = '--')     
    
    
    ax = plt.gca()
    ax.set_xlabel("training duration", fontsize='large')
    ax.set_ylabel('press probability', fontsize='large')
    label1 = 'end of training phase'
    label_1_strong = mlines.Line2D([], [], color='black', alpha=0.5,
                          label=label1+' '+strong)
    label_1_weak = mlines.Line2D([], [], color='black', alpha=0.5,
                          label=label1+' '+weak, linestyle = '--')  
    label_2_strong = mlines.Line2D([], [], color=clr, alpha=0.5,
                          label=label2+' '+strong)
    label_2_weak = mlines.Line2D([], [], color=clr, alpha=0.5,
                          label=label2+' '+weak, linestyle = '--')
    if devaluation:
        ax.legend([label_1_strong, label_1_weak, label_2_strong, label_2_weak], 
                  [label1+' '+strong, label1+' '+weak, label2+' '+strong, label2+' '+weak], 
                  loc = 'lower right', bbox_to_anchor=(1, 0.05), fontsize='small')    
    else:
        if baye:            
            ax.legend([label_1_strong, label_1_weak, label_2_strong, label_2_weak], 
                  [label1+' '+strong, label1+' '+weak, label2+' '+strong, label2+' '+weak], 
                  loc = 'lower right', bbox_to_anchor=(1, 0.0001), fontsize='small')  
        else:
            ax.legend([label_1_strong, label_1_weak, label_2_strong, label_2_weak], 
                  [label1+' '+strong, label1+' '+weak, label2+' '+strong, label2+' '+weak], 
                  loc = 'center right', bbox_to_anchor=(1, 0.4), fontsize='small')  

    plt.xlim([-1,n-1])
    plt.xticks([0-1,5-1,10-1,15-1,20-1], [0,500,1000,1500,2000])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.show() 
    
    return p_actions, p2_actions

def plot_chosen_probability_phaseend_replication(worlds, repetitions, devaluation=False):
    trials_training = []
    for i in range(int(len(worlds)/(repetitions))):
        trials_training.append(worlds[i*repetitions].trials_phase1)
    n = len(trials_training)

    p_actions = np.zeros((n, repetitions))
    p2_actions = np.zeros((n, repetitions))

    for j in range(n):
        for k in range(repetitions):
            chosen_p = worlds[j*repetitions+k].P
            p_actions[j,k] = chosen_p[int(trials_training[j]-1),0]
            p2_actions[j,k] = chosen_p[-1,0]
            
    labels = np.tile(np.arange(n), (repetitions, 1)).reshape(-1, order='f')
    
    plot_p_a1_data = p_actions[:,:].flatten()
    plot_p_a2_data = p2_actions[:,:].flatten()

    
    data_p_a1 = pd.DataFrame({'chosen': plot_p_a1_data, 'training durations': labels})
    data_p_a2 = pd.DataFrame({'chosen': plot_p_a2_data, 'training durations': labels})

    plt.figure()
    if devaluation:
        label2 = 'end of devaluation phase'
    else:
        label2 = 'end of omission phase'
    #plt.title("action and context infer times")
    sns.lineplot(x='training durations', y='chosen', data=data_p_a1, ci = 95, 
                 color='black', label = 'end of training phase', estimator='mean', 
                 err_style="bars", linewidth=3)
    sns.lineplot(x='training durations', y='chosen', data=data_p_a2, ci = 95, 
                color='#008000', label = 'end of training phase', estimator='mean', 
                err_style="bars",linewidth=3, linestyle = '--')    
    
    
    ax = plt.gca()
    ax.set_xlabel("training duration", fontsize='large')
    ax.set_ylabel('press probability', fontsize='large')
    label1 = 'end of training phase'
    label_1 = mlines.Line2D([], [], color='black', alpha=0.5,
                          label=label1)
    label_2 = mlines.Line2D([], [], color='#008000', alpha=0.5,
                          label=label2)
    if devaluation:
        ax.legend([label_1, label_2], 
                  [label1, label2], 
                  loc = 'lower right', bbox_to_anchor=(1, 0.05), fontsize='small')    
    else:

        ax.legend([label_1, label_2], 
                  [label1, label2], 
                  loc = 'center right', bbox_to_anchor=(1, 0.4), fontsize='small')  

    plt.xlim([-1,n-1])
    plt.xticks([0-1,5-1,10-1,15-1,20-1], [0,500,1000,1500,2000])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.show()      
    
    
def plot_chosen_probability_retrieval(worlds, repetitions, habit_learning_strength, 
                                      baye=False, EX1_2=False, re=False):
    
    trials=worlds[0].trials
    post_a = np.zeros((trials, 2))
    
    if baye: 
        clr = '#F97306'
        habit_learning_strength = 1/np.array(habit_learning_strength)
        label_strong = 'h='+str(habit_learning_strength[0])
        label_weak = 'h='+str(habit_learning_strength[1])
    else:
        clr = '#008000'
        label_strong = r'$\alpha_H$='+str(habit_learning_strength[0])
        label_weak = r'$\alpha_H$='+str(habit_learning_strength[1])        
    

    n_agt = 2
    post_a = np.zeros((n_agt,trials,2))

    for j in range(n_agt):                
        for k in range (0, repetitions):
            n = j*repetitions + k
            if baye:
                post_pol = (worlds[n].agent.posterior_policies[:,0,:,:]* 
                                worlds[n].agent.posterior_context[:,0,np.newaxis,:]).sum(axis=-1)
                post_a[j] += post_pol
            else:
                chosen_p = worlds[n].P
                post_a[j] += chosen_p  
        post_a[j] /= repetitions                
    
    
    fig, ax = plt.subplots(nrows=1, figsize= (5,2))
    

    ax.plot(post_a[0,:,1], color=clr, linewidth=2.5)
    ax.plot(post_a[1,:,1], color=clr, linewidth=2.5, linestyle='--')
    ax.set_ylim([-0.01,1.01])
    ax.set_yticks([0,0.5,1.0])
    ax.set_xlim([-10,trials+10]) 
    #ax.plot(np.ones(trials)*0.5, color='#808080', linestyle='--',linewidth=1.5)
        
    ax.set_xlabel('trial number', ha='center', fontsize='large')
    if not EX1_2:        
        ax.set_ylabel('Lever 2 press probability', va='center', 
                 rotation='vertical', fontsize='large')  
    else:
        ax.set_ylabel('Withhold press probability', va='center', 
                 rotation='vertical', fontsize='large')   

    ax.plot(np.ones(trials)*0.5, color="black", linestyle='--',linewidth=1.5)
    if not re:
        ax.plot([trials/2,trials/2], [0,1], color='dimgrey', linestyle='--', linewidth=1.5)
        ax.text(trials/8,1.1, 'context 1', color = 'black', fontweight = 'regular')
        ax.text(trials/8*6,1.1, 'context 2', color = 'black', fontweight = 'regular')
        ax.axvspan(trials/2,trials,facecolor='lightgrey', alpha=0.7)
    else:
        ax.plot([100,100], [0,1], color='dimgrey', linestyle='--', linewidth=1.5)
        ax.plot([200,200], [0,1], color='dimgrey', linestyle='--', linewidth=1.5)
        ax.text(25,1.1, 'context 1', color = 'black', fontweight = 'regular')
        ax.text(125,1.1, 'context 2', color = 'black', fontweight = 'regular')
        ax.text(225,1.1, 'context 1', color = 'black', fontweight = 'regular')
        ax.axvspan(100,200,facecolor='lightgrey', alpha=0.7)        


    strong_label = mlines.Line2D([], [], color=clr, alpha=0.5,
                          label=label_strong)
    weak_label = mlines.Line2D([], [], color=clr, alpha=0.5, linestyle='--',
                          label=label_weak)    
# =============================================================================
#     if baye:
#         strong_label = mlines.Line2D([], [], color=clr, alpha=0.5,
#                           label=label_strong)
#         weak_label = mlines.Line2D([], [], color=clr, alpha=0.5, linestyle='--',
#                           label=label_weak)
#     else:
#         strong_label = mlines.Line2D([], [], color=clr, alpha=0.5,
#                           label='strong habit learner '+r'\alpha_H='+str(habit_learning_strength[0]))
#         weak_label = mlines.Line2D([], [], color=clr, alpha=0.5, linestyle='--',
#                           label='weak habit learner '+r'\alpha_H='+str(habit_learning_strength[0]))       
# =============================================================================

    
    ax.legend([strong_label, weak_label], 
                  [label_strong, label_weak], 
                  loc = 'upper left', bbox_to_anchor=(0.0, 1), fontsize='small')

    plt.tight_layout()
    plt.show() 
    return post_a

def plot_convergence_time(worlds, repetitions, habit_learning_strength, baye=False):
    if baye:
        clr = '#F97306'
        clr2 = '#FFA500'
        ls = 1/np.array(habit_learning_strength)
        n_strong = 'h=' + str(ls[0]) + '\n(naive)'
        n_weak = 'h=' + str(ls[1]) + '\n(naive)'
        e_strong = 'h=' + str(ls[0]) + '\n(experienced)'
        e_weak = 'h=' + str(ls[1])+ '\n(experienced)'
    else:
        clr = '#008000'
        clr2 = '#15B01A'
        ls = habit_learning_strength 
        n_strong = r'$\alpha_H$=' + str(ls[0]) + '\n(naive)'
        n_weak = r'$\alpha_H$=' + str(ls[1])  + '\n(naive)'
        e_strong = r'$\alpha_H$=' + str(ls[0]) + '\n(experienced)'
        e_weak = r'$\alpha_H$=' + str(ls[1])  + '\n(experienced)'
    
    n_agents = 2
    converge_time = np.zeros((len(ls), n_agents, repetitions))
    for i in range(len(ls)):
        for j in range(repetitions):
            converge_time[i,:,j] = convergence_time(worlds[i*repetitions+j], baye=baye)
    
    plot_naive_data = converge_time[:,0,:].flatten()
    plot_exped_data = converge_time[:,1,:].flatten()
    
    labels = np.tile([1,0], (repetitions, 1)).reshape(-1, order='f')
     
    data_naive = pd.DataFrame({'time': plot_naive_data, 'tendencies': labels})
    data_exped = pd.DataFrame({'time': plot_exped_data, 'tendencies': labels})

    plt.figure()
    fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(5,4))
    ax1.set_title('naive agents')
    ax2.set_title('experienced agents')
    ax1.set_yticks([0,10,20,30,40,50])
    ax2.set_yticks([0,10,20,30,40,50])
    ax1.set_ylim(-0.1,50)
    ax2.set_ylim(-0.1,50)  
    
    sns.lineplot(x='tendencies', y='time', data=data_naive, ci = 95, ax=ax1, color=clr, estimator=np.nanmedian, linewidth=3)
    sns.lineplot(x='tendencies', y='time', data=data_exped, ci = 95, ax=ax2, color=clr, estimator=np.nanmedian, linewidth=3)

    ax1.set_xticks([0,1])
    ax1.set_xlim([-0.01,1.00])
#    ax1.set_xticklabels([weak, strong])
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set(xlabel=None)

    ax2.set_xticks([0,1])
    ax2.set_xlim([-0.01,1.00])
#    ax2.set_xticklabels([weak, strong])
    ax2.xaxis.set_ticks_position('bottom')

    plt.tight_layout()
    plt.show()
    
    plt.figure()
    ax = plt.gca()
    ax.set_title('naive agents')
    ax.set_yticks([0,10,20,30,40,50])
    ax.set_ylim(-0.1,50)
 
    sns.lineplot(x='tendencies', y='time', data=data_naive, ci = 95, color=clr, estimator=np.nanmedian, linewidth=3)
    
    if baye:
        plt.xlabel(r"habitual tendency $h$")
    else:
        plt.xlabel(r"step parameter in habitual control $\alpha_H$")
    ax.set_xticks([0,1])
    ax.set_xlim([-0.01,1.00])
#    ax.set_xticklabels([weak, strong])
    ax.xaxis.set_ticks_position('bottom')

    plt.tight_layout()
    plt.show()   
 
    plt.figure()
    ax = plt.gca()
    ax.set_title('experienced agents')
    ax.set_yticks([0,10,20,30,40,50])
    ax.set_ylim(-0.1,50)  
    
    sns.lineplot(x='tendencies', y='time', data=data_exped, ci = 95, color=clr, estimator=np.nanmedian, linewidth=3)

    if baye:
        plt.xlabel(r"habitual tendency $h$")
    else:
        plt.xlabel(r"step parameter in habitual control $\alpha_H$")
    ax.set_xticks([0,1])
    ax.set_xlim([-0.01,1.00])
#    ax.set_xticklabels([weak, strong])
    ax.xaxis.set_ticks_position('bottom')

    plt.tight_layout()
    plt.show() 
    
    plt.figure()
    
    plt.boxplot([plot_naive_data[:repetitions], plot_naive_data[repetitions:], plot_exped_data[:repetitions], plot_exped_data[repetitions:]], 
                    labels=[n_weak,n_strong, e_weak, e_strong], 
                    boxprops={'color':clr, 'linewidth':'3'},
                    flierprops={'markerfacecolor':clr},
                    medianprops={'color':clr})
    ax = plt.gca()
    ax.set_yticks([0,20,40,60,80,100])
    ax.set_ylabel('Convergence time')
    
    return converge_time

def plot_posterior_context_training(worlds, repetitions, training_durations, test_duration, phasename, baye=False):
    """
this function is only for the Bayesian model
    """
    
    trials_list = np.array(training_durations) + test_duration
    n_duration = len(trials_list)
    n_agt = 2
    
    post_c = np.zeros((n_duration, n_agt, trials_list[-1], 2))
    
 
    clr = '#FF81C0'
    
    for i in range(n_duration):
        for j in range(n_agt):                
            for k in range (repetitions):
                n = i*repetitions*n_agt + j*repetitions + k
                #print(i,j,k,n)
                t_start = trials_list[-1] - worlds[n].trials
                post_c[i,j,:t_start] = - 10000

                post_c[i,j,t_start:,0] += np.array(worlds[n].agent.posterior_context[:,0,0])
                post_c[i,j,t_start:,1] += np.array(worlds[n].agent.posterior_context[:,1,0])
                
            post_c[i, j, :] /= repetitions            
    
    fig = plt.figure(figsize= (10,2*n_duration))
    
    for i in range(n_duration):
        #print(i)
        ax = plt.subplot2grid((n_duration, int(trials_list[-1]/100)), 
                              (i, int((trials_list[-1]-trials_list[i])/100)), 
                              colspan=int(trials_list[-1]/100))
        n = i*repetitions*n_agt
        t_start = trials_list[-1] - worlds[n].trials
        ax.plot(post_c[i,0,t_start:,0], color=clr, linewidth=2.5)
        ax.plot(post_c[i,1,t_start:,0], color=clr, linewidth=2.5, linestyle='--')
        ax.text(0.0, 0.2, 'training duration:'+str(training_durations[i]),
                bbox=dict(facecolor='white', edgecolor='black',linewidth=1.5))
        ax.set_ylim([-0.01,1.01])                                                                                                                                                                               
        ax.set_yticks([0,0.5,1.0])
        ax.set_xlim([-10,trials_list[i]+10])                                                                        
        ax.set_xticks([0, training_durations[i], trials_list[i]])
#        ax.set_xticklabels([0, str(training_durations[i])+' (context swiches)', trials_list[i]])
        ax.plot(np.zeros(trials_list[-1])+0.5,
                   color='#808080', linestyle='--',linewidth=1.5)
        if i == n_duration-1:
            ax.set_xlabel('trial number')
                                                                                                                                                                     
    #fig.text(0.5, 0, 'trial number', ha='center')
    fig.text(0, 0.5, 'Posterior over context 1', va='center', rotation='vertical')  

    import matplotlib.lines as mlines
    
    strong_label = mlines.Line2D([], [], color=clr, alpha=0.5,
                          label='strong habit learner')
    weak_label = mlines.Line2D([], [], color=clr, alpha=0.5, linestyle='--',
                          label='weak habit learner')

    
    ax.legend([strong_label, weak_label], 
                  ['strong habit learner', 'weak habit learner'], 
                  loc = 'lower left', bbox_to_anchor=(0.2, 0))
    
    ax.text(training_durations[-1]/2, -0.45, 'training phase', backgroundcolor="white",
            ha='center', va='top', weight='bold', color='black') #x, y, text
    ax.annotate('', xy = (training_durations[-1]/2, -0.2),
            xytext = (0, -0.2),
            arrowprops=dict(arrowstyle='->', facecolor='black'))  
    ax.text((trials_list[-1]-training_durations[-1])/2+training_durations[-1], -0.45, phasename, backgroundcolor="white",
            ha='center', va='top', weight='bold', color='black') #x, y, text

def convergence_time(worlds, baye=False):
    
    results_a_param = np.zeros((4))
    results_a_param_type = np.zeros((1), dtype=int) 
    if baye:
        p_action = posterior_over_actions(worlds, repetitions=1)
    else:
        p_action = worlds.P
    
    trials = 200
    t_trials = trials - 100
    times = np.arange(0.+1,trials+1,1.)
    
    try:
        results_a_param, pcov = sc.optimize.curve_fit(sigmoid, times[10:], p_action[10:trials,1], p0=[1.,1.,t_trials,0.])
        results_a_param_type = 0
    except RuntimeError:
        try:
            results_a_param[1:], pcov = sc.optimize.curve_fit(exponential, times[10:], p_action[10:trials,1], p0=[1.,t_trials,0.])
            results_a_param_type = 1
        except RuntimeError:
            results_a_param = np.nan
            results_a_param_type = 0
    if results_a_param[0] < 0.1 or results_a_param[1] < 0.0 or results_a_param[2] > trials:
        results_a_param = [0, 0, trials+1, 0]
        
    threshold_a = fit_functions[results_a_param_type](0, *results_a_param[results_a_param_type:])
    if threshold_a < 0.5:
        if threshold_a < 0.001:
            threshold_a = 0.001
        time_1 = np.where(p_action[:trials-t_trials,1] <= threshold_a)[0]
        time_2 = np.where(p_action[trials:,1] <= threshold_a)[0]
        if len(time_1)>0:
            time_1 = time_1[0]
        else:
            time_1 = 101
        if len(time_2)>0:
            time_2 = time_2[0]
        else:
            time_2 = 101
    else:
        time_1 = np.nan
        time_2 = np.nan
    #print(time_1, time_2)     
        
    return time_1, time_2          
     
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
    
def infer_time_baye(worlds, repetitions):

    results_c_param = np.zeros((repetitions,4)) #(repetitions x 4)
    results_a_param = np.zeros((repetitions,4)) #(repetitions x 4)
    results_a_param_type = np.zeros((repetitions), dtype=int)
    results_c_param_type = np.zeros((repetitions), dtype=int)

    
    for i in range(repetitions):
        trials = worlds[i].trials
        t_trials = worlds[i].trials_training
        times = np.arange(0.+1,trials+1,1.) #(200 x 1)
        
        posterior_contexts = posterior_over_contexts(worlds[i], repetitions=1)
        posterior_actions = posterior_over_actions(worlds[i], repetitions=1)

        try:
            results_a_param[i], pcov = sc.optimize.curve_fit(sigmoid, times[10:], posterior_actions[10:,1], p0=[1.,1.,t_trials,0.])
            results_a_param_type[i] = 0
        except RuntimeError:
            try:
                results_a_param[i,1:], pcov = sc.optimize.curve_fit(exponential, times[10:], posterior_actions[10:,1], p0=[1.,t_trials,0.])
                results_a_param[i,0] = 1
                results_a_param_type[i] = 1
            except:
                results_a_param[i] = np.nan
                results_a_param_type[i] = 0

        try:
            results_c_param[i], pcov = sc.optimize.curve_fit(sigmoid, times, posterior_contexts[:,1], p0=[1.,1.,t_trials,0.])
            results_c_param_type[i] = 0
        except RuntimeError:
            try:
                results_c_param[i,1:], pcov = sc.optimize.curve_fit(exponential, times, posterior_contexts[:,1], p0=[1.,t_trials,0.])
                results_c_param[i,0] = 1
                results_c_param_type[i] = 1
            except:
                results_c_param[i] = np.nan
                results_c_param_type[i] = 0

    #拟合函数sigmoid，自变量：times，因变量：posterior_context，p0：给函数的参数确定一个初始值来减少计算机的计算量

        if results_a_param[i,0] < 0.1 or results_a_param[i,1] < 0.0 or results_a_param[i,2] < 15 or results_a_param[i,2] > trials:
            results_a_param[i] = [0,0,trials+1,0]

        if results_c_param[i,0] < 0.1 or results_c_param[i,1] < 0.0 or results_c_param[i,2] < 15 or results_c_param[i,2] > trials:
            results_c_param[i] = [0,0,trials+1,0]
    
    return results_c_param, results_a_param, results_c_param_type, results_a_param_type   

def infer_time_mbp(worlds, repetitions):

    results_a_param = np.zeros((repetitions,4)) #(repetitions x 4)
    results_a_param_type = np.zeros((repetitions), dtype=int)
    
    for i in range(repetitions):
        trials = worlds[i].trials
        t_trials = worlds[i].trials_phase1
        times = np.arange(0.+1,trials+1,1.) #(200 x 1)
        
        p_actions = worlds[i].P

        try:
            results_a_param[i], pcov = sc.optimize.curve_fit(sigmoid, times[10:], p_actions[10:,1], p0=[1.,1.,t_trials,0.])
            results_a_param_type[i] = 0
        except RuntimeError:
            try:
                results_a_param[i,1:], pcov = sc.optimize.curve_fit(exponential, times[10:], p_actions[10:,1], p0=[1.,t_trials,0.])
                results_a_param[i, 0] = 1
                results_a_param_type[i] = 1
            except:
                results_a_param[i] = np.nan
                results_a_param_type[i] = 0

    #拟合函数sigmoid，自变量：times，因变量：posterior_context，p0：给函数的参数确定一个初始值来减少计算机的计算量

        if results_a_param[i,0] < 0.1 or results_a_param[i,1] < 0.0 or results_a_param[i,2] < 15 or results_a_param[i,2] > trials:
            results_a_param[i] = [0,0,trials+1,0]

    
    return results_a_param, results_a_param_type 


def plot_chosen_probability_phaseend_average(worlds, habit_learning_strength, repetitions, 
                                     devaluation=False, baye=False):

    if baye:
        clr = '#F97306'
        ls = 1/np.array(habit_learning_strength)
        strong = 'h='+str(ls[0])
        weak = 'h='+str(ls[1]) 
    else:
        clr = '#008000'
        ls = habit_learning_strength 
        strong = r'\alpha_H='+str(ls[0])
        weak = r'\alpha_H='+str(ls[1])     

    
    trials_training = []
    for i in range(int(len(worlds)/(repetitions*len(ls)))):
        if baye:
            trials_training.append(worlds[i*repetitions*len(ls)].trials_training)
        else:
            trials_training.append(worlds[i*repetitions*len(ls)].trials_phase1)
    n = len(trials_training)

    p_actions = np.zeros((len(ls), n, repetitions))
    p2_actions = np.zeros((len(ls), n, repetitions))
    for i in range(len(ls)):
        for j in range(n):
            for k in range(repetitions):
                if baye:
                    chosen_p = posterior_over_actions(worlds[(j*len(ls)+i)*repetitions+k], 
                                                  repetitions=1)
                else:
                    chosen_p = worlds[(j*len(ls)+i)*repetitions+k].P
                p_actions[i,j,k] = chosen_p[int(trials_training[j]-1),0]
                p2_actions[i,j,k] = chosen_p[-1,0]
            
    

    plt.figure()
    if devaluation:
        label2 = 'after devaluation manipulate'
    else:
        label2 = 'after omission manipulate'

    plt.plot(np.arange(n), np.mean(p2_actions[0,:,:], axis=(1))) 
    plt.plot(np.arange(n), np.mean(p_actions[0,:,:], axis=(1))) 
#    ax = plt.gca()
    plt.xlabel("training duration")
    plt.ylabel('action A chosen probability')

    plt.xlim([-1,n-1])
    plt.xticks([0-1,5-1,10-1,15-1,20-1], [0,500,1000,1500,2000])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.show() 

    plt.figure()
    plt.plot(np.arange(n), np.mean(p2_actions[1,:,:], axis=(1))) 
    plt.plot(np.arange(n), np.mean(p_actions[1,:,:], axis=(1))) 
    
#    ax = plt.gca()
    plt.xlabel("training duration")
    plt.ylabel('action A chosen probability')

    plt.xlim([-1,n-1])
    plt.xticks([0-1,5-1,10-1,15-1,20-1], [0,500,1000,1500,2000])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.show()     
    
    return p_actions, p2_actions