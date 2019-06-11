# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:48:54 2019

@author: Lorenzo
"""

import math
import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding

"""
REPEATED PRISONER DILEMMA

Agents:
    1: playing with RL
    2: playing tit-for-tat
    
Actions:
    0: not cooperate
    1: cooperate
    
State: actions in previous period
    [0,0]
    [1,0]
    [0,1]
    [1,1]
    
Rewards:
    [1,1]
    [0,5]
    [5,0]
    [4,4]

"""

class RepeatedPrisonerDilemma(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
                low=np.array([0,0]), high=np.array([1,1]), dtype=np.int32)
        self.state = np.array([1,1])
        self.steps_beyond_done = None
        self.viewer = None
        self.local_steps = 0
        self.current_reward = 0
        self.cumulated_reward = 0
        
    def reset(self):
        self.state = np.array([1,1])
        # for random start substitute with
        # something using np.random.randint(0,2)
        self.local_steps = 0
        self.current_reward = 0
        self.cumulated_reward = 0
        self.steps_beyond_done = None
        return self.state
    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        s1, s2 = state
        a1 = action
        a2 = s1
        
        self.state = (a1, a2)
        
        done = self.local_steps > 100
        done = bool(done)
        
        if not done:
            if a1==0 and a2==0:
                reward = 1
            elif a1==0 and a2==1:
                reward = 5
            elif a1==1 and a2==0:
                reward = 0
            else:
                reward = 4
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You should be calling reset rather than step!")
            reward = 0
        
        self.local_steps += 1
        self.current_reward = reward
        self.cumulated_reward += reward
        
        return np.array(self.state), reward, done, {}
    
    def render(self, mode='human', close=False):
        print(f'Local steps:    {self.local_steps}')
        print(f'Current Reward:     {self.current_reward}')
        print(f'Cumulated Reward:   {self.cumulated_reward}')
    
            
      
# TUNE  

import gym      
import ray
from ray import tune

ray.shutdown()
ray.init()

tune.run(
        "DQN",
        stop={"training_iteration": 400},
        config={"env": 'custom_envs:repeatedprisoner-v0',
                "num_gpus": 0,
                "num_workers": 4,
                "num_envs_per_worker": 8,
                },
)

# TRAINER

import ray.rllib.agents.dqn as dqn
import random
from ray.tune.logger import pretty_print

trainer = dqn.DQNAgent(env='custom_envs:repeatedprisoner-v0', config={
        #"num_workers": 1,
        "num_envs_per_worker": 4,
        "num_cpus_per_worker": 4,
        "num_cpus_for_driver": 2,
        #"sample_batch_size": 200,
        #"train_batch_size": 200,
        })

    
for i in range(1000):
    result = trainer.train()
    print(pretty_print(result))
    
    if i % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)