import random
import gym
import ray
import numpy as np
import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print

# from envs.MA_Firms_Pricing import MultiAgentFirmsPricing

# initialize the environment with the given configs
env_config = {"num_agents": 2,"max_steps":10**9}
env=MultiAgentFirmsPricing(env_config=env_config)

# Function for mapping agents to policies
def policy_mapping_fn(agent_id):
    if agent_id=="agent_0":
        return "policy_0"
    elif agent_id=="agent_1":
        return "policy_1"
    elif agent_id=="agent_2":
        return "policy_2"
    else:
        return "policy_3"


# RLLIB DQN TRAINER

ray.shutdown()
ray.init()

trainer = dqn.DQNAgent(env=MultiAgentFirmsPricing, config={
        "env_config": env_config,
        "horizon": 1000,
        "soft_horizon": True, # compute rewards without resetting the env
        "double_q": True, # use Double-DQN (DDQN)
        "dueling": True, # use Dueling-DQN
        "hiddens": [], # fully connected layer for postprocessing
        "learning_starts":100000,
        "lr":0.1,
        "exploration_final_eps":0.001,
        "buffer_size": 10**5, # replay buffer size
        "target_network_update_freq": 5000,
        "timesteps_per_iteration":10000,
        "sample_batch_size":16, # batches of this size are collected untile train_batch_size is met
        "train_batch_size":128, # batch size used for training the neural network
        "num_envs_per_worker": 16, # number of envs to evaluate vectorwise per worker
        "num_cpus_per_worker": 4,
        "num_cpus_for_driver": 2,
        # Additional options for multiagent. Map agents in the environment to policies,
        # i.e., 'agents' in the training algorithm.
        "multiagent": {
                "policy_graphs": {
                        "policy_0": (None, env.observation_space, env.action_space, {}),
                        "policy_1": (None, env.observation_space, env.action_space, {}),
                        #"policy_2": (None, env.observation_space, env.action_space, {}),
                        #"policy_3": (None, env.observation_space, env.action_space, {}),
                },
                "policy_mapping_fn": policy_mapping_fn
        },
        # additional options for the model (i.e. the neural network estimating Qvalues)
        "model": {
                "fcnet_hiddens":[64],
                },
        
})

# Train the agents for 1000*(10**6) iterations    
for i in range(10**6):
    result = trainer.train()
    print(pretty_print(result))
       
"""   
APEX-DQN TRAINER

Distributed Prioritized Experience Replay (APE-X) variation of DQN uses a single GPU learner
and many CPU workers for experience collection, which can scale to hundreds of CPU workers 
due to the distributed prioritizaion of experience prior to storage in replay buffers.

!!! Huge RAM consumption with multiple workers
    (the code below uses ~8GB of RAM with only two workers)
"""

ray.shutdown()
ray.init()


trainer = dqn.ApexAgent(env=MultiAgentFirmsPricing, config={
        "env_config": env_config,
        "horizon": 1000,
        "soft_horizon": True,
        "double_q": True,
        "dueling": True,
        "hiddens": [],
        "learning_starts":50000,
        "lr":0.1,
        "exploration_final_eps":0.001,
        "buffer_size": 10**5,
        "timesteps_per_iteration":25000,
        "target_network_update_freq": 50000,
        "sample_batch_size":100,
        "train_batch_size":200,
        "num_workers": 2,
        "num_envs_per_worker": 8,
        "num_cpus_per_worker": 2,
        "num_cpus_for_driver": 2,
        "multiagent": {
                "policy_graphs": {
                        "policy_0": (None, env.observation_space, env.action_space, {}),
                        "policy_1": (None, env.observation_space, env.action_space, {}),
                        #"policy_2": (None, env.observation_space, env.action_space, {}),
                        #"policy_3": (None, env.observation_space, env.action_space, {}),
                },
                "policy_mapping_fn": policy_mapping_fn
        },
        "model": {
                "fcnet_hiddens":[128],
                },
        
})
