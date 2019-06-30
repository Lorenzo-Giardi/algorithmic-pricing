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

Main function:

eps_greedy_tabular_q_learning(env, num_episodes, conv_steps)

- Trains 2 tabular Q-learning agents
- Arguments
      env: environment (provide env_config dictionary!)
      num_episodes: number of episodes
      conv_steps: number of steps after policy convergence
-Outputs
      q0: q-table for agent_0
      q1: q-table for agent_1
      training_progress_ag0: array of rewards for agent_0 (only last episode)
      rew_list: array of average reward per episode for agent_0
      
TO DO:
- Improve reporting of training metrics
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
        num_episodes=1,
        conv_steps = 10**5,
        OPT_INIT = False
        ):
    
    # initialize variables and parameters that remain constant across episodes
    lr_max = 0.15
    lr_min = 0.05
    lr_decay = 0.0000005
    eps_max = 0.2
    eps_decay = 0.00001
    eps_min = 0.0001
    gamma = 0.95  
    
    # 1) loop over episodes
    for i in range(num_episodes):
        
        print("Starting episode: {} of {}".format(i, num_episodes))
        
        # initialize q-tables
        if OPT_INIT:
            q0 = np.full((225,15), 10.0)
            q1 = np.full((225,15), 10.0)
        else:
            q0 = 10*np.random.rand(225, 15)
            q1 = 10*np.random.rand(225, 15)
        # reset environment
        s = tuple(env.reset()['agent_0'])
        done = False
        # initialize training metrics
        training_progress_ag0 = list()
        training_progress_ag1 = list()
        # initialize parameters
        eps = eps_max
        lr = lr_max
        # initialize convergence counter
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
            
            s = new_s
            
            # decay lr and eps
            if eps > eps_min:
                eps *= np.exp(-eps_decay)
            if lr > lr_min:
                lr *= np.exp(-lr_decay)
            
            
            # store and print training metrics
            training_progress_ag0.append(rew0)
            training_progress_ag1.append(rew1)
            
            if env.local_steps % 50000==0:
                # print step, scores, ...
                print("step: {}, score: {}, e: {:.2}, lr: {:.2}"
                 .format(env.local_steps, rewards, eps, lr))
                # print counters
                print("count_0: {}, count_1: {}, max_count: {}"
                 .format(conv_count_0, conv_count_1, conv_steps))

            
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
                delta0 = (rew0 - 0.22589)/(0.337472 - 0.22589)
                delta1 = (rew1 - 0.22589)/(0.337472 - 0.22589)
                final_deltas = np.array([delta0, delta1])
                print("Policies convergence reached. Breaking loop. \n \
                      Final deltas: {}".format(final_deltas))
                break
        
    return q0, q1, training_progress_ag0, new_strategy_0, new_strategy_1, s


env_config = {"num_agents": 2,
              "max_steps":  3*10**6,
             }


# EVALUATION OF TRAINED POLICIES AND IRFs

def play_optimal_strategy(env=MultiAgentFirmsPricing(), STEPS=1000):
    
    s = tuple(env.reset()['agent_0'])
    s = final_state
    done = False
    
    for _ in range(STEPS):
        act0 = sigma0[s_dict[s]]
        act1 = sigma1[s_dict[s]]
        
        # Combine actions into a dictionary
        action = {'agent_0':act0, 'agent_1':act1}
        
        # step env and divide stuff among agents
        new_s, rewards, dones, _ = env.step(action)
        new_s = tuple(new_s['agent_0'])
        rew0, rew1 = rewards.values()
        don0, don1, done = dones.values()
        
        s = new_s
        
        if env.local_steps/STEPS > 0.95:
            print("step: {}, score: {}, acts: {}"
                 .format(env.local_steps, rewards.values(), action.values()))
    
    return s, rew0, rew1
   


def defection(env=MultiAgentFirmsPricing(), STEPS=50):
        
    s_list = [state]
    rew0_list = [last_rew0]
    rew1_list = [last_rew1]
    
    act0 = 2
    act1 = sigma1[s_dict[state]]
    action = {'agent_0':act0, 'agent_1':act1}
        
    # step env and divide stuff among agents
    new_s, rewards, dones, _ = env.step(action)
    new_s = tuple(new_s['agent_0'])
    rew0, rew1 = rewards.values()
    don0, don1, done = dones.values()
    
    s = new_s
    
    print("T=0, score: {}, actions: {}"
                 .format(rewards.values(), action.values()))
    
    s_list.append(s)
    rew0_list.append(rew0)
    rew1_list.append(rew1)
    
    for i in range(STEPS):
        act0 = sigma0[s_dict[s]]
        act1 = sigma1[s_dict[s]]
        
        # Combine actions into a dictionary
        action = {'agent_0':act0, 'agent_1':act1}
        
        # step env and divide stuff among agents
        new_s, rewards, dones, _ = env.step(action)
        new_s = tuple(new_s['agent_0'])
        rew0, rew1 = rewards.values()
        don0, don1, done = dones.values()
        
        s = new_s
        
        s_list.append(s)
        rew0_list.append(rew0)
        rew1_list.append(rew1)
        
        print("T={}, score: {}, actions: {}"
                 .format(i+1, rewards.values(), action.values()))
    
    return s_list, rew0_list, rew1_list

# EXECUTE TRAINING CODE
q0, q1, training_progress_ag0, sigma0, sigma1, final_state  = eps_greedy_tabular_q_learning(
        MultiAgentFirmsPricing(env_config=env_config),
        num_episodes=1
        )

plt.plot(training_progress_ag0)

# EXECUTE IRFs CODE
state, last_rew0, last_rew1 = play_optimal_strategy()
s_list, rew0_list, rew1_list = defection()

# plot absolute profits
plt.plot(rew0_list[0:25])
plt.plot(rew1_list[0:25])
plt.show()

# plot deltas
rew0_arr = np.array(rew0_list)
rew1_arr = np.array(rew1_list)
d0_arr = (rew0_arr - 0.22589)/(0.337472 - 0.22589)
d1_arr = (rew1_arr - 0.22589)/(0.337472 - 0.22589)

plt.plot(d0_arr[0:25])
plt.plot(d1_arr[0:25])
plt.show()
