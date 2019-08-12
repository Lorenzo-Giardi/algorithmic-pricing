"""
Code for training two agents using tabular Q-Learning in a environment in which
two firms have to simultaneously set their price with the goal of
maximizing their individual profits.
To correctly import the environment, ensure that the two files are
located in the same directory and that it is also set as the working directory.
Main features of the algorithm:
- Epsilon-greedy policy (with eps decay) -> random exploration
- Q-learning (with lr decay)
- Optimistic initialization -> non-random exploration
- Stop after 10^5 iterations without changes in strategy ...
- ... or after 3*10^6 iterations in any case
PARALLEL EXECUTION OF EPISODES
Different episodes are scheduled to run on different ray actors (~ CPU cores)
-> The function decorator @ray.remote signals that the function can be
    executed in parallel within a ray cluster
-> The function.remote() command returns an Object_ID and creates a ray taks
    that is scheduled to run on a ray actor
-> The ray.get(Object_IDs) command returns the actual Object, i.e. the result
    that has been returned by the function
"""

import os
import itertools
import numpy as np
import ray
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# import environment
path='/home/lorenzo/Desktop/FirmsPricing_DiscrObs'
os.chdir(path)
from MA_Firms_Pricing import MultiAgentFirmsPricing

  
# TABULAR Q LEARNING

NUM_CPUS = 6
NUM_BATCHES = 50 
NUM_EPISODES = NUM_CPUS * NUM_BATCHES
AGGREGATE_STATS_EVERY = 1000
PRINT_EVERY = 100_000

sns.set_style("ticks") 
ray.init(num_cpus=NUM_CPUS)

ENV_CONFIG={
           "num_agents":2,
           "max_steps":2*10**6,
           "p_min":1.4315251,
           "p_max":1.9509807,
           "p_num":15,}

ENV = MultiAgentFirmsPricing(env_config=ENV_CONFIG)

# Dictionary from states to states_ids. E.g. (13,7) -> 202
# !!! keys are tuples!!!
        
actions_space = list(range(15))
possible_states = [p for p in itertools.product(actions_space, repeat=2)]
index_states = list(range(225))

s_dict = dict(zip(possible_states, index_states))

@ray.remote
def tabQ_training(
        INIT = 0,
        EPS_MAX = 0.2,
        EPS_MIN = 0.001,
        EPS_DECAY = 0.00005,
        LR_MAX = 0.15,
        LR_MIN = 0.05,
        LR_DECAY = 0.00000001,
        GAMMA = 0.95,
        CONV_STEPS = 10**5,
        ENV_CONFIG = ENV_CONFIG,
        ):
    # initialize env
    env=MultiAgentFirmsPricing(env_config=ENV_CONFIG)
    # initialize q-tables
    if INIT==0:
        q0 = np.full((225,15), 10.0)
        q1 = np.full((225,15), 10.0)
    elif INIT==1:
        q0 = np.full((225,15), 0.0)
        q1 = np.full((225,15), 0.0)
    else:
        q0 = 3*np.exp(np.random.rand(225,15))
        q1 = 3*np.exp(np.random.rand(225,15))
        
    # reset environment
    s = tuple(env.reset()['agent_0'])
    done = False
    # initialize training metrics
    rew_list_0 = []
    rew_list_1 = []
    avg_rew_0_list = []
    avg_rew_1_list = []
    # initialize parameters
    eps = EPS_MAX
    lr = LR_MAX
    # initialize convergence counter
    new_strategy_0 = np.zeros(15)
    new_strategy_1 = np.zeros(15)
    conv_count_0 = 0
    conv_count_1 = 0

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
        new_s, rewards, dones, deltas = env.step(action)
        new_s = tuple(new_s['agent_0'])
        rew0, rew1 = rewards.values()
        don0, don1, done = dones.values()
                   
        # update agent_0 q-table
        q0[s_dict[s],act0] +=  lr * (rew0 +
                GAMMA * np.max(q0[s_dict[new_s],:]) - q0[s_dict[s],act0])
        
        # update agent_1 q-table
        q1[s_dict[s],act1] +=  lr * (rew1 +
                GAMMA * np.max(q1[s_dict[new_s],:]) - q1[s_dict[s],act1])
        
        # update state
        s = new_s
        
        # decay lr and eps
        if eps > EPS_MIN:
                eps *= np.exp(-EPS_DECAY)
        if lr > LR_MIN:
                lr *= np.exp(-LR_DECAY)
        
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
        
        if conv_count_0 >= CONV_STEPS and conv_count_1 >= CONV_STEPS:
            print("Policies convergence reached. Breaking loop.")
            break
    
    
        # Append episode reward to a list and log stats (every given number of episodes)
        rew_list_0.append(rew0)
        rew_list_1.append(rew1)
        if env.local_steps % AGGREGATE_STATS_EVERY == 0:
            avg_rew_0 = sum(rew_list_0[-AGGREGATE_STATS_EVERY:])/len(rew_list_0[-AGGREGATE_STATS_EVERY:])
            avg_rew_1 = sum(rew_list_0[-AGGREGATE_STATS_EVERY:])/len(rew_list_0[-AGGREGATE_STATS_EVERY:])
            
            avg_rew_0_list.append(avg_rew_0)
            avg_rew_1_list.append(avg_rew_1)
        
        # Print training metrics
        if env.local_steps % PRINT_EVERY ==0:
            # print step, scores, ...
            print("step: {}; avg_scores: {:.3}, {:.3}; e: {:.2}, lr: {:.2}"
             .format(env.local_steps, avg_rew_0, avg_rew_1, eps, lr))
            # print counters
            print("count_0: {}, count_1: {}, max_count: {}"
             .format(conv_count_0, conv_count_1, CONV_STEPS))
            

    final_deltas = np.array(list(deltas.values()))
    final_avg_q = np.mean([q0, q1], axis=0)
    
    #return results of interest
    return final_deltas, new_strategy_0, new_strategy_1, s,\
            env.local_steps, final_avg_q, avg_rew_0_list, avg_rew_1_list


# EXECUTE TRAINING
results = list()
for i in range(NUM_BATCHES):
    print("Starting training batch: {} of {}".format(i+1, NUM_BATCHES))
    
    results_ids = []
    for _ in range(NUM_CPUS):
        results_ids.append(tabQ_training.remote(
                INIT = 2,
                EPS_MAX = 1,
                EPS_MIN = 0.00001,
                EPS_DECAY = 10**(-5),
                LR_MAX = 0.15,
                LR_MIN = 0.05,
                LR_DECAY = 0,
                GAMMA = 0.95,
                CONV_STEPS = 10**5,
                ENV_CONFIG = ENV_CONFIG,
                ))
    batch_results = ray.get(results_ids)
    results.append(batch_results)
    print('Finished training batch number {} of {}'.format(i+1, NUM_BATCHES))

ray.shutdown()

# results is a sort of 3D dataset
# split results into multiple variables indexed by episode number
train_deltas = []
strat0 = []
strat1 = []
final_state = []
steps_at_conv = []
final_avg_q = []
avg_rew_0 = []
avg_rew_1 = []
for batch in range(NUM_BATCHES):
    for cpu in range(NUM_CPUS):
        train_deltas.append(results[batch][cpu][0])
        strat0.append(results[batch][cpu][1])
        strat1.append(results[batch][cpu][2])
        final_state.append(results[batch][cpu][3])
        steps_at_conv.append(results[batch][cpu][4])
        final_avg_q.append(results[batch][cpu][5])
        avg_rew_0.append(results[batch][cpu][6])
        avg_rew_1.append(results[batch][cpu][7])
final_avg_q = np.mean(final_avg_q, axis=0)

### CONVERT TO PANDAS DATAFRAME
# Strategy table (episodes x states_ids)
strat0_df = pd.DataFrame(strat0)
strat1_df = pd.DataFrame(strat1)
# Final states table (episodes x agents_ids)
final_state_df = pd.DataFrame(final_state)
final_state_df.columns = ['Agent_0', 'Agent_1']
# Number of steps table (episodes x 1)
steps_at_conv_df = pd.DataFrame({'steps_at_conv': steps_at_conv})
# Final Q-table (episodes x states_ids)
final_avg_q_df = pd.DataFrame(final_avg_q)
# Strategy table (episodes x states_ids)
avg_rew_0_df = pd.DataFrame(avg_rew_0)
avg_rew_1_df = pd.DataFrame(avg_rew_1)

"""
# save to disk
strat0_df.to_csv('strategy_0.csv')
strat1_df.to_csv('strategy_1.csv')
final_state_df.to_csv('final_state_df.csv')
steps_at_conv_df.to_csv('steps_at_conv_df.csv')
final_avg_q_df.to_csv('final_avg_q_df.csv')
avg_rew_0_df.to_csv('avg_rew_0_df.csv')
avg_rew_1_df.to_csv('avg_rew_1_df.csv')
"""
# steps at convergence
steps_at_conv_df.describe()
sns.boxplot(steps_at_conv_df, color="#069af3")
plt.xlabel('Timesteps')
plt.savefig('ts_boxplot.png', dpi=600)
plt.show()
            
#heatmap plot
index = []
for i in range(15):
    for j in range(15):
        if i==j:
            index.append(s_dict[(i,j)])
labels = []
for i in index:
    labels.append(list(s_dict.keys())[i])
    
sns.heatmap(final_avg_q[index,], yticklabels=labels, cmap="Blues")
plt.xlabel('Actions')
plt.ylabel('States')
plt.savefig('fig1.png', dpi=600)
plt.show()

# plot some training metrics
i = np.random.randint(NUM_EPISODES)
avg_delta_0 = (np.array(avg_rew_0[i]) - 0.22589)/(0.337472 - 0.22589)
avg_delta_1 = (np.array(avg_rew_1[i]) - 0.22589)/(0.337472 - 0.22589)

print(f"Showing training progress for Episode {i} of {NUM_EPISODES}")
plt.plot(avg_rew_0[i], label='agent_0', c="#069af3", lw=0.7)
plt.xlabel('Timesteps')
plt.ylabel('Average reward')
plt.savefig(f'fig2-{i}.png', dpi=600)
plt.show()

plt.plot(avg_delta_0, label='agent_0', c="#069af3", lw=0.7)
plt.xlabel('Timesteps')
plt.ylabel('Average profit gain')
plt.savefig(f'fig3-{i}.png', dpi=600)
plt.show()


# EVALUATION OF TRAINED POLICIES AND IRFs

def play_optimal_strategy(final_state,
                          sigma0,
                          sigma1,
                          STEPS=1000,
                          env=MultiAgentFirmsPricing(env_config=ENV_CONFIG),
                          verbose=True):
    
    s = tuple(env.reset()['agent_0'])
    s = final_state
    done = False
    delta0=[]
    delta1=[]
    
    for _ in range(STEPS):
        act0 = sigma0[s_dict[s]]
        act1 = sigma1[s_dict[s]]
        
        # Combine actions into a dictionary
        action = {'agent_0':act0, 'agent_1':act1}
        
        # step env and divide stuff among agents
        new_s, rewards, dones, infos = env.step(action)
        new_s = tuple(new_s['agent_0'])
        rew0, rew1 = rewards.values()
        don0, don1, done = dones.values()
        
        delta0.append(infos['agent_0']['delta'])
        delta1.append(infos['agent_1']['delta'])
        
        s = new_s
        
        if verbose and env.local_steps/STEPS > 0.95:
            print("step: {}, score: {}, acts: {}"
                 .format(env.local_steps, rewards.values(), action.values()))
    
    return s, rew0, rew1, np.mean(delta0), np.mean(delta1)
   

def defection(state,
              sigma0,
              sigma1,
              env=MultiAgentFirmsPricing(env_config=ENV_CONFIG),
              STEPS=23,
              verbose=True):
        
    s_list = [state]
    rew0_list = [last_rew0]
    rew1_list = [last_rew1]
    
    act0 = 2 # play nash price... not necessarily best response!!!
    act1 = sigma1[s_dict[state]]
    action = {'agent_0':act0, 'agent_1':act1}
        
    # step env and divide stuff among agents
    new_s, rewards, dones, infos = env.step(action)
    new_s = tuple(new_s['agent_0'])
    rew0, rew1 = rewards.values()
    don0, don1, done = dones.values()
    
    s = new_s
    
    if verbose:
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
        new_s, rewards, dones, infos = env.step(action)
        new_s = tuple(new_s['agent_0'])
        rew0, rew1 = rewards.values()
        don0, don1, done = dones.values()
        
        s = new_s
        
        s_list.append(s)
        rew0_list.append(rew0)
        rew1_list.append(rew1)
        
        if verbose:
            print("T={}, score: {}, actions: {}"
                 .format(i+1, rewards.values(), action.values()))
    
    return s_list, rew0_list, rew1_list

    
# EXECUTE IRFs CODE
state0 = np.zeros((NUM_EPISODES, 25))
state1 = np.zeros((NUM_EPISODES, 25))
reward0 = np.zeros((NUM_EPISODES, 25))
reward1 = np.zeros((NUM_EPISODES, 25))
delta0_arr = np.zeros(NUM_EPISODES)
delta1_arr = np.zeros(NUM_EPISODES)

for i in range(NUM_EPISODES):
    state, last_rew0, last_rew1, delta0, delta1 = play_optimal_strategy(final_state=final_state[i],
                                                        sigma0=strat0[i],
                                                        sigma1=strat1[i],
                                                        verbose=False)
    s_list, rew0_list, rew1_list = defection(state=state,
                                                        sigma0=strat0[i],
                                                        sigma1=strat1[i],
                                                        verbose=False)
    s_arr = np.array(s_list)
    s0 = s_arr[:,0]
    s1 = s_arr[:,1]
    
    state0[i] = np.array(s0)
    state1[i] = np.array(s1)
    reward0[i] = np.array(rew0_list)
    reward1[i] = np.array(rew1_list)
    delta0_arr[i] = delta0
    delta1_arr[i] = delta1
    
mean_state0 = state0.mean(axis=0, dtype=np.int32)
mean_state1 = state1.mean(axis=0, dtype=np.int32) 
mean_reward0 = reward0.mean(axis=0)
mean_reward1 = reward1.mean(axis=0)


# plot absolute profits
plt.plot(mean_reward0, label='agent_0', c="#247afd", marker='D')
plt.plot(mean_reward1, label='agent_1', c="#fd8d49", marker='o')
plt.axhline(0.337472, linestyle='dashed', c="#929591")
plt.axhline(0.22589, linestyle='dashed', c="#929591")
plt.legend()
plt.savefig('profits-irf.png', dpi=600)
plt.show()

# plot deltas
d0_arr = (mean_reward0 - 0.22589)/(0.337472 - 0.22589)
d1_arr = (mean_reward1 - 0.22589)/(0.337472 - 0.22589)

plt.plot(d0_arr, label='agent_0', c="#247afd", marker='D')
plt.plot(d1_arr, label='agent_1', c="#fd8d49", marker='o')
plt.axhline(1, linestyle='dashed', c="#929591")
plt.axhline(0, linestyle='dashed', c="#929591")
plt.legend()
plt.savefig('deltas-irf.png', dpi=600)
plt.show()

# plot prices
actions_to_prices_dict = ENV.actions_to_prices_dict


p0_list = list()
p1_list = list()

for element in mean_state0:
    p0_list.append(actions_to_prices_dict[element])
for element in mean_state1:
    p1_list.append(actions_to_prices_dict[element])
    

plt.plot(p0_list, label='agent_0', c="#247afd", marker='D')
plt.plot(p1_list, label='agent_1', c="#fd8d49", marker='o')
plt.axhline(1.48, linestyle='dashed', c="#929591")
plt.axhline(1.93, linestyle='dashed', c="#929591")
plt.legend()
plt.savefig('prices-irf.png', dpi=600)
plt.show()


# bivariate density plot
df2 = pd.DataFrame({'Agent_0':delta0_arr, 'Agent_1':delta1_arr})

sns.kdeplot(df2['Agent_0'], df2['Agent_1'], shade=True, cbar=True, cmap='Blues')
plt.savefig('Density-plot.png', dpi=600)
plt.show()
