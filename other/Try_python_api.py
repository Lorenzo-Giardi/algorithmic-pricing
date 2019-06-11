#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:57:32 2019

@author: lorenzo
"""


# Simple training API for applying RLlib to custom problems.


import ray
import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print

ray.init(ignore_reinit_error=True)

config = dqn.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
trainer = dqn.DQNTrainer(config=config, env="CartPole-v0")

for i in range(1000):
    result = trainer.train()
    print(pretty_print(result))
    
    if i % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
        
        
# Using TUNE

import ray
from ray import tune

ray.shutdown()
ray.init()

tune.run(
        "DQN",
        stop={"episode_reward_mean": 200},
        config={"env": "CartPole-v0",
                "num_gpus": 0,
                "num_workers": 4,
                "num_envs_per_worker": 8,
                "lr": tune.grid_search([0.01, 0.001, 0.0001]),
                },
)

