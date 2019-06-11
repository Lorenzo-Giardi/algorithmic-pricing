#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:32:57 2019

@author: lorenzo
"""

### ENVIRONMENT ###

import gym
import ray
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class FourAgentsPrisonerDilemma(MultiAgentEnv):
    
    def __init__(self, num_agents=4, max_steps=100):
        self.agents = ['agent_0', 'agent_1', 'agent_2', 'agent_3']
        self.dones = set()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(2)
        self.local_steps = 0
        self.max_steps = max_steps
        self.obs = {'agent_0':1, 'agent_1':1, 'agent_2':1, 'agent_3':1,}

        
    def reset(self):
        self.dones = set()
        self.local_steps = 0
        self.obs = {'agent_0':1, 'agent_1':1, 'agent_2':1, 'agent_3':1,}
        return self.obs
    
    def step(self, action_dict):
        
        self.local_steps += 1
        
        act0 = action_dict['agent_0']
        act1 = action_dict['agent_1']
        act2 = action_dict['agent_2']
        act3 = action_dict['agent_3']
        
        if act0==0 and act1==0 and act2==0 and act3==0:
            rew0 = 1
            rew1 = 1
            rew2 = 1
            rew3 = 1
        if act0==1 and act1==1 and act2==1 and act3==1:
            rew0 = 5
            rew1 = 5
            rew2 = 5
            rew3 = 5
        else:
            rew0= -10 if act0==1 else 3
            rew1= -10 if act0==1 else 3
            rew2= -10 if act0==1 else 3
            rew3= -10 if act0==1 else 3
        
        rew = {'agent_0':rew0, 'agent_1':rew1,'agent_2':rew2,'agent_3':rew3,}
        
        self.obs = {'agent_0':act0, 'agent_1':act1, 'agent_2':act2, 'agent_3':act3,}
        
        if self.local_steps < self.max_steps:
            done = False
        else:
            done = True
       
        dones = {'agent_0':done, 'agent_1':done, 'agent_2':done, 'agent_3':done,'__all__':done,} 
        
        info = {'agent_0':{}, 'agent_1':{}, 'agent_2':{}, 'agent_3':{},}
        
        return self.obs, rew, dones, info

 


###      TRAINING 3

import ray.rllib.agents.dqn as dqn
import random
from ray.tune.logger import pretty_print

env=FourAgentsPrisonerDilemma()

ray.shutdown()
ray.init()

trainer = dqn.DQNAgent(env=FourAgentsPrisonerDilemma, config={
        #"num_workers": 1,
        "num_envs_per_worker": 4,
        "num_cpus_per_worker": 4,
        "num_cpus_for_driver": 2,
        #"sample_batch_size": 200,
        #"train_batch_size": 200,
        "multiagent": {
                "policy_graphs": {
                        "agent_0": (None, env.observation_space, env.action_space, {}),
                        "agent_1": (None, env.observation_space, env.action_space, {}),
                        "agent_2": (None, env.observation_space, env.action_space, {}),
                        "agent_3": (None, env.observation_space, env.action_space, {}),
                },
                "policy_mapping_fn":
                    lambda agent_id:
                        random.choice(["agent_0", "agent_1","agent_2", "agent_3"])
        },
})

    
for i in range(1000):
    result = trainer.train()
    print(pretty_print(result))
    
    if i % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
        