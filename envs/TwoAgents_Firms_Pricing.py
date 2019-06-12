import gym
import ray
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class TwoAgentsFirmsPricing(MultiAgentEnv):
    
    def __init__(self, num_agents=2, max_steps=10000000):
        self.agents = ['agent_0', 'agent_1']
        self.dones = set()
        self.action_space = gym.spaces.Discrete(15)
        self.observation_space = gym.spaces.Box(
                low=np.array([0,0]), high=np.array([14,14]), dtype=np.int32)
        self.local_steps = 0
        self.max_steps = max_steps
        self.obs = {'agent_0':np.array([1,1]), 'agent_1':np.array([1,1]),}

        
    def reset(self):
        self.dones = set()
        self.local_steps = 0
        self.obs = {'agent_0':np.array([1,1]), 'agent_1':np.array([1,1]),}
        return self.obs
    
    def step(self, action_dict):
        
        self.local_steps += 1
        
        act0 = action_dict['agent_0']
        act1 = action_dict['agent_1']
        
        prices = [1.42617, 1.479722, 1.533274, 1.586826, 1.640379, 1.693931, 
                  1.747486, 1.801035, 1,854587, 1.908139, 1.961691, 2.015244, 
                  2.068796, 2.122348, 2.1759]
        prices_dict = dict(zip(list(range(15)), prices))
        
        p0 = prices_dict[act0]
        p1 = prices_dict[act1]
        
        q0 = (np.exp((2-p0)/0.25))/(np.exp((2-p0)/0.25)+np.exp((2-p1)/0.25)+np.exp(0))
        q1 = (np.exp((2-p1)/0.25))/(np.exp((2-p1)/0.25)+np.exp((2-p0)/0.25)+np.exp(0))
        
        r0 = (p0-1)*q0
        r1 = (p1-1)*q1
        
        rew = {'agent_0':r0, 'agent_1':r1,}
        
        self.obs = {'agent_0':np.array([act0,act1]), 'agent_1':np.array([act0,act1]),}
        
        if self.local_steps < self.max_steps:
            done = False
        else:
            done = True
       
        dones = {'agent_0':done, 'agent_1':done,'__all__':done,} 
        
        info = {'agent_0':{}, 'agent_1':{},}
        
        return self.obs, rew, dones, info
    
