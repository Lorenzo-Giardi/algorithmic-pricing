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
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
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
import random   

def s_id(arg):
    if arg.all() == np.array([0, 0]).all():
        return 0
    elif arg.all() == np.array([0, 1]).all():
        return 1
    elif arg.all() == np.array([1, 0]).all():
        return 2
    elif arg.all() == np.array([1, 1]).all():
        return 3
    else:
        raise ValueError('Problem in the state to id loop')

def eps_greedy_tabular_q_learning(env,num_episodes=500):
    q_table = np.zeros((4,2))
    gamma = 0.95
    eps = 0.9
    lr = 0.05
    decay_factor = 0.99
    progress_rew = np.zeros(num_episodes)
    
    for i in range(num_episodes):
        state = env.reset()
        eps *= decay_factor
        done = False
        cumul_rew = 0
        
        while not done:
            if np.random.random() < eps or np.sum(q_table[s_id(state),:])==0:
                action = np.random.randint(0,2)
            else:
                action = np.argmax(q_table[s_id(state),:])
            new_state, reward, done, _ = env.step(action)
            q_table[s_id(state),action] += reward + lr * (
                    gamma * np.max(q_table[s_id(new_state),:]) - q_table[s_id(state),action])
            state = new_state
            cumul_rew += reward
            
        progress_rew[i] = cumul_rew 
    
    progress_ep = np.array(range(num_episodes))
        
    return q_table, progress_rew, progress_ep, eps

q_table, progress_rew, progress_ep, eps = eps_greedy_tabular_q_learning(RepeatedPrisonerDilemma(),num_episodes=500)

q_table
eps
plt.plot(progress_ep, progress_rew)