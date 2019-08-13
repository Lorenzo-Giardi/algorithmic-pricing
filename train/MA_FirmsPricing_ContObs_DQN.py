#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:32:57 2019

@author: lorenzo
"""

# import libraries
import os
import numpy as np
import ray
from ray import tune


ray.init(object_store_memory=2_000_000_000, redis_max_memory=1_000_000_000)

# import environment. set directory to find it.
path='/home/lorenzo/Desktop/FirmsPricing_ContObs'
os.chdir(path)
from MA_Firms_Pricing_ContObs import MultiAgentFirmsPricingContinuous


# initialize the environment with the given configs
ENV_CONFIG = {"num_agents": 2,
              "max_steps":  10**9,
              "p_min":1.2,
              "p_max":2,}
env=MultiAgentFirmsPricingContinuous(env_config=ENV_CONFIG)

# Define policies
def gen_policy():
    return(None, env.observation_space, env.action_space, {})

policy_graphs = dict() 
for i in range(env.num):
    policy_graphs['agent_'+str(i)]=gen_policy()

# Function for mapping agents to policies
def policy_mapping_fn(agent_id):
    return agent_id


# callbacks for custom metrics
def on_episode_start(info):
    episode = info["episode"]
    episode.user_data["delta0"] = []
    #episode.user_data["delta1"] = []
    episode.user_data["price0"] = []
    #episode.user_data["price1"] = []
    episode.user_data["rew0"] = []
    episode.user_data["info0"] = []

def on_episode_step(info):
    episode = info["episode"]
    delta0 = (episode.prev_reward_for(agent_id='agent_0') - 0.22589)/(0.337472 - 0.22589)
    delta1 = (episode.prev_reward_for(agent_id='agent_1') - 0.22589)/(0.337472 - 0.22589)
    price0 = episode.last_raw_obs_for(agent_id='agent_0')[0]
    price1 = episode.last_raw_obs_for(agent_id='agent_0')[1]
    rew0 = episode.prev_reward_for(agent_id='agent_0')
    info0 = episode.last_info_for(agent_id='agent_0')
    info0 = list(info0.values())[0]
    print(info0)
    episode.user_data["delta0"].append(delta0)
    episode.user_data["delta1"].append(delta1)
    episode.user_data["price0"].append(price0)
    episode.user_data["price1"].append(price1)
    episode.user_data["rew0"].append(rew0)
    episode.user_data["info0"].append(info0)

def on_episode_end(info):
    episode = info["episode"]
    delta0 = np.mean(episode.user_data["delta0"])
    delta1 = np.mean(episode.user_data["delta1"])
    price0 = np.mean(episode.user_data["price0"])
    price1 = np.mean(episode.user_data["price1"])
    rew0 = np.mean(episode.user_data["rew0"])
    episode.custom_metrics["delta0"] = delta0
    episode.custom_metrics["delta1"] = delta1
    episode.custom_metrics["price0"] = price0
    episode.custom_metrics["price1"] = price1
    episode.custom_metrics["rew0"] = rew0

trial = tune.run(
        run_or_experiment= 'APEX',
        name='21_cont_DQN',
        stop={"timesteps_total":10**7},
        checkpoint_freq=50,
        #resume=True,
        #num_samples = 2,
        config={
            "env": MultiAgentFirmsPricingContinuous,
            "env_config": ENV_CONFIG,
            "horizon": 100,
            "soft_horizon": True,
            "double_q": True,
            "dueling": True,
            "hiddens": [24],
            "n_step": 3,
            "num_atoms": 10,
            "gamma": 0.975,
            "prioritized_replay": True,
            "prioritized_replay_alpha": 0.5,
            "beta_annealing_fraction": 0.2,
            "final_prioritized_replay_beta": 1.0,
            "learning_starts": 20000,
            "lr":0.001,
            "adam_epsilon": 0.00015,
            "schedule_max_timesteps": 5*10**6,
            "exploration_final_eps":0.01,
            "exploration_fraction":0.05,
            "buffer_size": 10**5,
            "target_network_update_freq": 50000,
            "sample_batch_size":16,
            "train_batch_size":64,
            
            "observation_filter": "MeanStdFilter",
            "num_workers": 2,
            "num_envs_per_worker": 8,
            "num_cpus_per_worker": 2,
            #"num_cpus_for_driver": 1,
            "num_gpus":0,
            "multiagent": {
                    "policy_graphs": policy_graphs,
                    "policy_mapping_fn": tune.function(policy_mapping_fn)
            },
            "model": {
                    "fcnet_activation": "tanh",
                    "fcnet_hiddens":[32, 32],
                    },
            "callbacks": {
                    "on_episode_start": tune.function(on_episode_start),
                    "on_episode_step": tune.function(on_episode_step),
                    "on_episode_end": tune.function(on_episode_end),
                    },
            },
    )
