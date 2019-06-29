# imports
import os
import itertools
import numpy as np
import ray
import matplotlib.pyplot as plt

# import environment
path='/home/lorenzo/Desktop/FirmsPricing_DiscrObs'
os.chdir(path)
from MA_Firms_Pricing import MultiAgentFirmsPricing
  
# some parameters
NUM_CPUS = 6 # number of ray actors for parallel execution
NUM_EPISODES = 1  # multiplied by NUM_CPUs! 
ray.init(num_cpus=NUM_CPUS)

# Create a dictionary for mapping states to states_ids. 
# E.g. state = (13,7) --> state_id = s_dict[(13,7)] = 202
# !!! notice that dictionary keys are tuples !!!
        
actions_space = list(range(15))
possible_states = [p for p in itertools.product(actions_space, repeat=2)]
index_states = list(range(225))

s_dict = dict(zip(possible_states, index_states))

### TRAINING FUNCTION ###

@ray.remote
def tabQ_training(
        OPT_INIT = True,
        EPS_MAX = 0.2,
        EPS_DECAY = 0.001,
        LR_MAX = 0.2,
        LR_DECAY = 0.00000001,
        GAMMA = 0.3,
        CONV_STEPS = 10**5,
        ENV_CONFIG = {"num_agents": 2, "max_steps":  2*10**6}
        ):
    # initialize env
    env=MultiAgentFirmsPricing(env_config=ENV_CONFIG)
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
        new_s, rewards, dones, _ = env.step(action)
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
        eps *= np.exp(-EPS_DECAY)
        lr *= np.exp(-LR_DECAY)
             
        # store and print training metrics
        training_progress_ag0.append(rew0)
        training_progress_ag1.append(rew1)
        
        if env.local_steps % 100000==0:
            # print step, scores, ...
            print("step: {}, score: {}, e: {:.2}, lr: {:.2}"
             .format(env.local_steps, rewards, eps, lr))
            # print counters
            print("count_0: {}, count_1: {}, max_count: {}"
             .format(conv_count_0, conv_count_1, CONV_STEPS))
        
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
            delta0 = (rew0 - 0.22589)/(0.337472 - 0.22589)
            delta1 = (rew1 - 0.22589)/(0.337472 - 0.22589)
            final_deltas = np.array([delta0, delta1])
            break
    
    #return results of interest
    return final_deltas

### PARALLEL EXECUTION OF EPISODES ###

results = list()
for i in range(NUM_EPISODES):
    print("Starting episode: {} of {}".format(i, NUM_EPISODES))
    
    results_ids = []
    for _ in range(NUM_CPUS):
        results_ids.append(tabQ_training.remote(
                OPT_INIT = True,
                EPS_MAX = 0.2,
                EPS_DECAY = 0.001,
                LR_MAX = 0.2,
                LR_DECAY = 0.00000001,
                GAMMA = 0.3,
                CONV_STEPS = 10**5,
                ENV_CONFIG = {"num_agents": 2, "max_steps":  2*10**6}
                ))
    
    batch_results = ray.get(results_ids)
    results.append(batch_results)

ray.shutdown()

#flatten list of results
results = [item for sublist in results for item in sublist]
