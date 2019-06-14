import random
import gym
import ray
import numpy as np
import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print

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


# DQN TRAINER

ray.shutdown()
ray.init()

trainer = dqn.DQNAgent(env=MultiAgentFirmsPricing, config={
        "env_config": env_config,
        "horizon": 1000,
        "soft_horizon": True,
        "double_q": True,
        "dueling": True,
        "hiddens": [],
        "learning_starts":100000,
        "lr":0.1,
        "exploration_final_eps":0.001,
        "buffer_size": 10**5,
        "target_network_update_freq": 5000,
        "timesteps_per_iteration":10000,
        "sample_batch_size":16,
        "train_batch_size":128,
        "num_envs_per_worker": 16,
        "num_cpus_per_worker": 4,
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
                "fcnet_hiddens":[64],
                },
        
})

    
for i in range(10**6):
    result = trainer.train()
    print(pretty_print(result))
       
   
# APEX-DQN TRAINER


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
