# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 01:25:07 2021

@author: orian
"""

import numpy as np
import itertools
import misc_baye as mib
import misc_mbp as mim
import world_baye
import environment_baye as env_b
import environment_mbp as env_m
import agent_baye as agt_b
import agent_mbp as agt_m
import perception_baye as prc_b
import action_selection_baye as asl_b

np.set_printoptions(threshold = 100000, precision = 5)


"""
run functions
"""
def run_agent_baye(par_list, trials, T, ns, na, nr, nc, deval=False, retrieval=False, retrieval_test=False, worlds_old=None, omission=False, devaluation=False,trials_training=100, u_deval=[]):

    #set parameters:
    #learn_pol: initial concentration paramter for policy prior
    #trans_prob: reward probability
    #avg: True for average action selection, False for maximum selection
    #Rho: Environment's reward generation probabilities as a function of time
    #utility: goal prior, preference p(o)
    learn_pol, trans_prob, avg, Rho, utility = par_list


    """
    create matrices
    """


    #generating probability of observations in each state
    A = np.eye(ns)


    #state transition generative probability (matrix)
    B = np.zeros((ns, ns, na))

    for i in range(0,na):
        B[i+1,:,i] += 1

    # agent's beliefs about reward generation

        # concentration parameters
    C_alphas = np.ones((nr, ns, nc))
        # initialize state in front of levers so that agent knows it yields no reward
    C_alphas[0,0,:] = 100
    for i in range(1,nr):
        C_alphas[i,0,:] = 1

        # agent's initial estimate of reward generation probability
    C_agent = np.zeros((nr, ns, nc))
    for c in range(nc):
        C_agent[:,:,c] = np.array([(C_alphas[:,i,c])/(C_alphas[:,i,c]).sum() for i in range(ns)]).T
    


    # context transition matrix

    p = trans_prob
    q = 1.-p
    transition_matrix_context = np.zeros((nc, nc))
    transition_matrix_context += q/(nc-1)
    for i in range(nc):
        transition_matrix_context[i,i] = p

    """
    create environment (grid world)
    """

    environment = env_b.MultiArmedBandid(A, B, Rho, trials = trials, T = T)


    """
    create policies
    """

    pol = np.array(list(itertools.product(list(range(na)), repeat=T-1)))

    npi = pol.shape[0]

    # concentration parameters

    alphas = np.zeros((npi, nc)) + learn_pol


    prior_pi = alphas / alphas.sum(axis=0)


    """
    set state prior (where agent thinks it starts)
    """

    state_prior = np.zeros((ns))

    state_prior[0] = 1.

    """
    set action selection method
    """

    if avg:

        ac_sel = asl_b.AveragedSelector(trials = trials, T = T,
                                      number_of_actions = na)
    else:

        ac_sel = asl_b.MaxSelector(trials = trials, T = T,
                                      number_of_actions = na)

    """
    set context prior
    """

    prior_context = np.zeros((nc)) + 0.1/(nc-1)
    prior_context[0] = 0.9

    """
    set up agent
    """

    # perception
    bayes_prc = prc_b.HierarchicalPerception(A, B, C_agent, transition_matrix_context, \
                                           state_prior, utility, prior_pi, alphas, \
                                               C_alphas, T=T, \
                                                   omission=omission, devaluation=devaluation)

    # agent
    bayes_pln = agt_b.BayesianPlanner(bayes_prc, ac_sel, pol,
                      trials = trials, T = T,
                      prior_states = state_prior,
                      prior_policies = prior_pi,
                      number_of_states = ns,
                      prior_context = prior_context,
                      learn_habit = True,
                      #save_everything = True,
                      number_of_policies = npi,
                      number_of_rewards = nr)


    """
    create world
    """

    w = world_baye.World(environment, bayes_pln, trials = trials, 
                         trials_training=trials_training, T = T)

    """
    simulate experiment
    """
    if deval or devaluation:
        w.simulate_experiment(range(trials_training))
        # reset utility to implement devaluation
        utility = u_deval

        bayes_prc.prior_rewards = utility

        w.simulate_experiment(range(trials_training, trials))               
    elif retrieval:
        w.simulate_experiment(range(trials-trials_training))

        prior_context = np.zeros((nc)) + 0.5
        w.agent.posterior_context[trials-trials_training-1,-1] = prior_context
        # w.simulate_experiment([trials//2])
        
        # prior_context = np.zeros((nc)) + 0.1/(nc-1)
        # prior_context[0] = 0.9
        w.simulate_experiment(range(trials-trials_training, trials))        
    else:     
         w.simulate_experiment(range(trials))
        

    return w



"""
run simulations (Experiments)
"""

def simulation_reversal_baye(repetitions, utility, u_deval, avg, T, ns, na, nr, 
                             nc, alpha_list, n_test, trials, 
                             deval_1 = False, omission = False, 
                             deval_2 = False, EX1_2 = False, retrieval=False):

    alpha_policies, alpha_context, alpha_rewards = alpha_list

    worlds = []
    for trials_n in trials:
        trials_training = trials_n - n_test
        Rho = np.zeros((trials_n, nr, ns))
        print('training duration:', trials_training)
        for tendency in alpha_policies: #1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]:
            for trans in alpha_context:#[100,99,98,97,96,95,94]:
                for prob in alpha_rewards:
                    print(tendency, trans, prob)
                    
                    if deval_2:
                        #Rho[:] = mib.generate_bandit_timeseries_habit_rw(trials_training, nr, ns, n_test,p=prob/100.)
                        Rho[:] = mib.generate_bandit_timeseries_habit_rw_devaluation(trials_training, nr, ns, n_test,p=prob/100.)
                    elif omission:
                        Rho[:] = mib.generate_bandit_timeseries_habit_rw_omission(trials_training, nr, ns, n_test,p=prob/100.)
                    elif EX1_2:
                        Rho[:] = mib.generate_bandit_timeseries_habit_rw(trials_training, nr, ns, n_test,p=prob/100.)
                    elif retrieval:
                        trials_n = 300
                        trials_training = 100
                        Rho = mib.generate_bandit_timeseries_habit_retrieval(trials_training, nr, ns, n_test,p=prob/100.)
                    else:
                        Rho[:] = mib.generate_bandit_timeseries_habit(trials_training, nr, ns, n_test,p=prob/100.)
                    learn_pol = tendency
                    parameters = [learn_pol, trans/100., avg, Rho, utility]

                    for i in range(repetitions):
                        worlds.append(run_agent_baye(parameters, trials_n, T, ns, na, 
                                                     nr, nc, deval=deval_1, retrieval=retrieval, 
                                                     retrieval_test=False, 
                                                     worlds_old=None, omission=omission, devaluation=deval_2,
                                                     trials_training=trials_training,
                                                     u_deval=u_deval))

    return worlds


def simulation_retrieval_baye(worlds_0, repetitions, utility, avg, T, ns, na, nr, 
                              nc, alpha_list):

    alpha_policies, alpha_context, alpha_rewards = alpha_list

    n_test = 1000
    trials_training = 1000
    trials =  trials_training+n_test#number of trials
    trials = 2*trials
    #trials_training = trials - n_test

    Rho = np.zeros((trials, nr, ns))
    worlds = worlds_0.copy()

    for tendency in alpha_policies: #1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]:
        for trans in alpha_context:#[100,99,98,97,96,95,94]:
            for prob in alpha_rewards:
                print(tendency, trans, prob)

                Rho[:trials//2] = mib.generate_bandit_timeseries_habit_rw(trials_training, nr, ns, n_test,p=prob/100.)
                Rho[trials//2:] = mib.generate_bandit_timeseries_habit_rw(trials_training, nr, ns, n_test,p=prob/100.)

                learn_pol = tendency
                parameters = [learn_pol, trans/100., avg, Rho, utility]
                retrieval_test=True

                for i in range(repetitions):
                    worlds.append(run_agent_baye(parameters, trials, T, ns, na, 
                                                 nr, nc, retrieval_test=retrieval_test))

    return worlds

def simulation_mbp(na, nm, paras, alpha_list, repetitions, 
                   trials_phase1s, trials_phase2, U, U_deval, 
                   deval_1=False, deval_2=False, omission=False,
                   retrieval=False, EX1_2=False):

    alpha_Hs, alpha_R, w_0, w_g, w_h, theta_g, theta_h, alpha_w = paras
    devaluation = deval_1 or deval_2
    worlds = []
    for trials_phase1 in trials_phase1s:
        if len(trials_phase1s) > 1:
            print('training duration:', trials_phase1)
        for par_H in alpha_Hs: 
            for alpha in alpha_list:
# =============================================================================
#                 print(par_H, alpha)
# =============================================================================
                for i in range(repetitions):
            
                    if not retrieval:
                        trials = trials_phase1 + trials_phase2
                        if EX1_2:
                            AM = np.zeros((trials, na, nm)) #Probability of leading to different reinforcers from different actions
                            AM[:,:,0] = 1
                            AM[:trials_phase1,0,:] = [0.5, 0.5]
                            AM[trials_phase1:,1,:] = [0.5, 0.5]
                        elif omission:
                            AM = np.zeros((trials,na,nm)) #Probability of leading to different reinforcers from different actions
                            AM[:,:,0] = 1 
                            AM[:trials_phase1,0,:] = [0.5, 0.5, 0.0] # training trials: press:0.5 no rewards/ 0.5 pellet
                            AM[:trials_phase1,1,:] = [0.0, 0.0, 1.0] # training trials: hold-press: 100% leisure
                            AM[trials_phase1:,1,:] = [0.5, 0.5, 0.0] # omission trials:  hold-press:0.5 no rewards/ 0.5 pellet (during omission, hold-press results in 100% leisure, which is implemented in the run_agent)                        
                        elif deval_2:
                            AM = np.zeros((trials,na,nm)) #Probability of leading to different reinforcers from different actions
                            AM[:,:,0] = 1 
                            AM[:trials_phase1,0,:] = [0.5, 0.5, 0.0] # training trials: press:0.5 no rewards/ 0.5 pellet
                            AM[:trials_phase1,1,:] = [0.0, 0.0, 1.0] # training trials: hold-press: 100% leisure
                            AM[trials_phase1:,0,:] = [0.5, 0.5, 0.0] # devaluation trials   
                            AM[trials_phase1:,1,:] = [0.0, 0.0, 1.0] 
                        else:
                            AM = mim.Action_to_Rewards(trials_phase1, trials_phase2, na, nm, alpha)
                        
                        environment = env_m.environment_two_alternative(trials,AM,nm=nm) 
                        runs = agt_m.sim_two_alternative(trials=trials,environment=environment,\
                                                 na=na, nm=nm, alpha_H=par_H,\
                                                     alpha_R=alpha_R, alpha_w=alpha_w,\
                                                         w_g=w_g, w_h=w_h, w_0=w_0,\
                                                                theta_g=theta_g, theta_h=theta_h)    
                        for t in range(trials):
                            runs.run_agent(U, U_deval, t, reversal = True, trials_phase1=trials_phase1, \
                                   trials_phase2=trials_phase2, omission=omission, devaluation=devaluation)   
          
                    else:
                        trials = trials_phase1 + trials_phase2 + trials_phase1
                        AM = mim.Action_to_Rewards_re(trials_phase1, trials_phase2, na, nm, alpha)
                        environment = env_m.environment_two_alternative(trials,AM,nm=nm) 
                        runs = agt_m.sim_two_alternative(trials=trials,environment=environment,\
                                                 na=na, nm=nm, alpha_H=par_H,\
                                                     alpha_R=alpha_R, alpha_w=alpha_w,\
                                                         w_g=w_g, w_h=w_h, w_0=w_0,\
                                                                theta_g=theta_g, theta_h=theta_h)    
                        for t in range(trials):
                            runs.run_agent(U, U_deval, t, reversal = True, trials_phase1=trials_phase1, \
                                   trials_phase2=trials_phase2, omission=omission, devaluation=devaluation) 
                    worlds.append(runs)
    return worlds