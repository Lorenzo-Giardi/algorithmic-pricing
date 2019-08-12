import random
import argparse
import gym
import numpy as np
import ray
from ray import tune
from ray.tune import register_env, grid_search
from ray.rllib.agents.dqn_policy import DQNTFPolicy
import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print

# import environment. Files must be in the same directory!
# path='/home/lorenzo/Desktop'
# os.chdir(path)
from MA_Firms_Pricing import MultiAgentFirmsPricing

parser = argparse.ArgumentParser()
parser.add_argument("--num", type=int, default=2)
parser.add_argument("--steps", type=int, default=10**7)

args = parser.parse_args()

# initialize the environment with the given configs
env_config = {"num_agents": args.num,
              "max_steps":  args.steps,
             }
env=MultiAgentFirmsPricing(env_config=env_config)

# Function for generating new policies
def gen_policy():
    return(DQNTFPolicy, env.observation_space, env.action_space, {})

# Policy graphs dictionary {'policy_name': policy}
policy_graphs = dict() 
for i in range(args.num):
    policy_graphs['agent_'+str(i)]=gen_policy()

# Function for mapping agents to policies
def policy_mapping_fn(agent_id):
    return agent_id


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
        "sample_batch_size":128, # batches of this size are collected untile train_batch_size is met
        "train_batch_size":512, # batch size used for training the neural network
        "num_envs_per_worker": 16, # number of envs to evaluate vectorwise per worker
        "num_cpus_per_worker": 4,
        "num_cpus_for_driver": 2,
        # Additional options for multiagent. Map agents in the environment to policies,
        # i.e., 'agents' in the training algorithm.
        "multiagent": {
                "policy_graphs": policy_graphs,
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
                "policy_graphs": policy_graphs,
                "policy_mapping_fn": policy_mapping_fn
        },
        "model": {
                "fcnet_hiddens":[64],
                },
        
})
"""
### USING TUNE ###
Tune is a framework for hyperparameter search and optimization
with a focus on deep learning and deep reinforcement learning

If you get a connection error, run: apt-get install redis-server
"""
def env_creator(config=env_config):
    return MultiAgentFirmsPricing(config)
register_env("MultiAgentFirmsPricing", env_creator)

ray.shutdown()
ray.init()

tune.run(
    "DQN",
    name="exp_1",
    stop={timesteps_total: 10**6},
    config={
        "env": "MultiAgentFirmsPricing",
        "env_config": env_config,
        "horizon": 1000,
        "soft_horizon": True,
        "double_q": True,
        "dueling": True,
        "hiddens": [],
        "learning_starts":50000,
        "lr": tune.grid_search([0.1, 0.05, 0.01]),
        "exploration_final_eps": tune.grid_search([0.05, 0.01, 0.001]),
        "buffer_size": 10**5,
        "timesteps_per_iteration":25000,
        "target_network_update_freq": 50000,
        "sample_batch_size":256,
        "train_batch_size":1024,
        "num_envs_per_worker": 8,
        "num_cpus_per_worker": 2,
        "num_cpus_for_driver": 1,
        "multiagent": {
                "policy_graphs": policy_graphs,
                "policy_mapping_fn": tune.function(policy_mapping_fn)
        },
        "model": {
                "fcnet_hiddens":[64],
                },
    }
  )
