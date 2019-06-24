import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

"""
Code for two agents tabular Q-Learning in a environment in which
two firms have to simultaneously set their price with the goal of
maximizing their profits.

To correctly import the environment, ensure that the two files are
located in the same directory and that it is also set as the working directory.

Main features of the algorithm:
- Epsilon-greedy policy (with eps decay) -> random exploration
- Q-learning (with lr decay)
- Optimistic initialization -> non-random exploration
- Stop after 10^5 iterations without changes in strategy ...
- ... or after 3*10^6 iterations in any case
"""

# import environment
path='/home/lorenzo/Desktop'
os.chdir(path)
from MA_Firms_Pricing import MultiAgentFirmsPricing

# Environment configs
env_config = {"num_agents": 2,
              "max_steps":  3*10**6,
             }

# Dictionary from states to states_ids. E.g. (13,7) -> 202
# !!! keys are tuples:(x,y) rather than [x,y] !!!
        
actions_space = list(range(15))
possible_states = [p for p in itertools.product(actions_space, repeat=2)]
index_states = list(range(225))

s_dict = dict(zip(possible_states, index_states))

### TABULAR Q-LEARNING ###

def eps_greedy_tabular_q_learning(
        env=MultiAgentFirmsPricing(),
        num_episodes=5,
        conv_steps = 10**5
        ):
    
    # initialize variables and parameters that remain constant across episodes
    lr_max = 0.2
    lr_decay = 0.000001
    eps_max = 0.5
    eps_decay = 0.01
    gamma = 0.95
    rew_list = list()    
    
    
    # 1) loop over episodes
    for i in range(num_episodes):
        print("Starting episode: {} of {}".format(i, num_episodes))
        
        # initialize q-tables
        q0 = np.full((225,15), 10.0)
        q1 = np.full((225,15), 10.0)
        
        # reset environment
        s = tuple(env.reset()['agent_0'])
        done = False
        
        # initialize training metrics
        training_progress_ag0 = list()
        training_progress_ag1 = list()
        ep_reward_0 = 0
        ep_reward_1 = 0
        
        # initialize parameters
        eps = eps_max
        lr = lr_max
        
        # initialize convergence counters
        new_strategy_0 = np.zeros(15)
        new_strategy_1 = np.zeros(15)
        conv_count_0 = 0
        conv_count_1 = 0
        
        # 2) loop within episodes
        while not done:
            
            # action agent 0
            if np.random.random() < eps or np.sum(q0[s_dict[s],:])==0:
                act0 = np.random.randint(0,15)
            else:
                act0 = np.argmax(q0[s_dict[s],:])
                
            # action agent 1
            if np.random.random() < eps or np.sum(q1[s_dict[s],:])==0:
                act1 = np.random.randint(0,15)
            else:
                act1 = np.argmax(q1[s_dict[s],:])
                
            # Combine actions into a dictionary
            action = {'agent_0':act0, 'agent_1':act1}
            
            # step env and divide stuff among agents
            new_s, rewards, dones, _ = env.step(action)
            new_s = tuple(new_s['agent_0'])
            rew0, rew1 = rewards.values()
            don0, don1, done = dones.values()
                       
            # update agent_0 q-table
            q0[s_dict[s],act0] +=  lr * (rew0 +
                    gamma * np.max(q0[s_dict[new_s],:]) - q0[s_dict[s],act0])
            
            # update agent_1 q-table
            q1[s_dict[s],act1] +=  lr * (rew1 +
                    gamma * np.max(q1[s_dict[new_s],:]) - q1[s_dict[s],act1])
            
            # update state for next iteration
            s = new_s
            
            # decay lr and eps
            eps *= np.exp(-eps_decay)
            lr *= np.exp(-lr_decay)
            
            
            # store and print training metrics
            training_progress_ag0.append(rew0)
            training_progress_ag1.append(rew1)
            
            if env.local_steps % 10000==0:
                # print step, scores, ...
                print("step: {}, score: {}, e: {:.2}, lr: {:.2}"
                 .format(env.local_steps, rewards, eps, lr))
                # print convergence counters
                print("count_0: {}, count_1: {}, max_count: {}"
                 .format(conv_count_0, conv_count_1, conv_steps))

            ep_reward_0 += rew0
            ep_reward_1 += rew1
            
            # break loop after convergence
            old_strategy_0 = new_strategy_0
            old_strategy_1 = new_strategy_1
            
            new_strategy_0 = np.argmax(q0, axis=1)
            new_strategy_1 = np.argmax(q1, axis=1)
            
            if np.array_equal(old_strategy_0, new_strategy_0):
                conv_count_0 += 1
            else:
                conv_count_0 = 0
                
            if np.array_equal(old_strategy_1, new_strategy_1):
                conv_count_1 += 1
            else:
                conv_count_1 = 0
            
            if conv_count_0 >= conv_steps and conv_count_1 >= conv_steps:
                print("Policies convergence reached. Breaking episode loop.")
                break
            
        # average reward per episode for agent_0     
        rew_list.append(ep_reward_0/env.local_steps)
        
        
    return q0, q1, training_progress_ag0, rew_list

# execute training
q0, q1, training_progress_ag0, rew_list = eps_greedy_tabular_q_learning(MultiAgentFirmsPricing(env_config=env_config), 10)
