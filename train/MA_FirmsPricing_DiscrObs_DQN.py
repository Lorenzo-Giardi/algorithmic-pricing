#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:43:12 2019

@author: lorenzo
"""

# import libraries
import os
import numpy as np
import ray
from ray import tune


# import environment. set directory to find it.
path='/home/lorenzo/Desktop/FirmsPricing_DiscrObs'
os.chdir(path)
from MA_Firms_Pricing import MultiAgentFirmsPricing

# initialize the environment with the given configs
ENV_CONFIG = {
           "num_agents":2,
           "max_steps":10**9,
           "p_min":1.4315251,
           "p_max":1.9509807,
           "p_num":15,}
env=MultiAgentFirmsPricing(env_config=ENV_CONFIG)


### TUNE

def gen_policy():
    return(None, env.observation_space, env.action_space, {})

policy_graphs = dict() 
for i in range(env.num):
    policy_graphs['agent_'+str(i)]=gen_policy()

# Function for mapping agents to policies
def policy_mapping_fn(agent_id):
    return agent_id

"""
# callbacks for custom metrics
def on_episode_start(info):
    episode = info["episode"]
    episode.user_data["delta0"] = []
    episode.user_data["delta1"] = []
    episode.user_data["obs"] = []

def on_episode_step(info):
    episode = info["episode"]
    delta0 = (episode.prev_reward_for(agent_id='agent_0') - 0.22589)/(0.337472 - 0.22589)
    delta1 = (episode.prev_reward_for(agent_id='agent_1') - 0.22589)/(0.337472 - 0.22589)
    obs = episode.last_raw_obs_for(agent_id='agent_0')
    
    episode.user_data["delta0"].append(delta0)
    episode.user_data["delta1"].append(delta1)
    episode.user_data["obs"].append(obs)

def on_episode_end(info):
    episode = info["episode"]
    
    delta0 = np.mean(episode.user_data["delta0"])
    delta1 = np.mean(episode.user_data["delta1"])
    obs1, obs2 = np.mean(episode.user_data["obs"], axis=0)

    episode.custom_metrics["delta0"] = delta0
    episode.custom_metrics["delta1"] = delta1
    episode.custom_metrics["obs1"] = obs1
    episode.custom_metrics["obs2"] = obs2
"""


### USING TUNE
# Tune is a framework for hyperparameter search and optimization
# with a focus on deep learning and deep reinforcement learning

ray.init()

trial = tune.run(
    run_or_experiment= "APEX",
    name='10_disc_DQN',
    stop={"timesteps_total":10**9},
    checkpoint_freq=100,
    num_samples = 1,
    #resume = True,
    #restore= '/home/lorenzo/Desktop/DQN_MultiAgentFirmsPricing_1_2019-07-25_09-57-16d5zrwkrs/checkpoint_3200/checkpoint-3200',
    config={
            "env": MultiAgentFirmsPricing,
            "env_config": ENV_CONFIG,
            "horizon": 100,
            "soft_horizon": True,
            "double_q": True,
            "dueling": True,
            "hiddens": [16],
            "n_step": 3,
            "num_atoms": 10,
            #"noisy": True,
            #"sigma0": 0.5,
            "gamma": 0.975,
            "prioritized_replay": True,
            "prioritized_replay_alpha": 0.5,
            "beta_annealing_fraction": 0.2,
            "final_prioritized_replay_beta": 1.0,
            "learning_starts": 20000,
            "lr":0.0005,
            "adam_epsilon": 0.0015,
            "schedule_max_timesteps": 10**7,
            "exploration_final_eps":0.02,
            "exploration_fraction":0.1,
            "buffer_size": 10**5,
            "target_network_update_freq": 50000,
            "sample_batch_size":16,
            "train_batch_size":64,
            "observation_filter": "MeanStdFilter",
            "num_workers": 2,
            "num_envs_per_worker": 16,
            "num_cpus_per_worker": 2,
            "num_cpus_for_driver": 1,
            "num_gpus":0,
            "multiagent": {
                    "policy_graphs": policy_graphs,
                    "policy_mapping_fn": tune.function(policy_mapping_fn)
            },
            "model": {
                    "fcnet_activation": "tanh",
                    "fcnet_hiddens":[32, 32],
                    },
            #"callbacks": {
                    #"on_episode_start": tune.function(on_episode_start),
                    #"on_episode_step": tune.function(on_episode_step),
                    #"on_episode_end": tune.function(on_episode_end),
                    #},      
           },
)
