
import gym
import ray
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class MultiAgentPrisonerDilemma(MultiAgentEnv):
    
    def __init__(self, num_agents=2, max_steps=100):
        self.agents = ['agent_0', 'agent_1']
        self.dones = set()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(2)
        self.local_steps = 0
        self.max_steps = max_steps
        self.obs = {'agent_0':1, 'agent_1':1,}

        
    def reset(self):
        self.dones = set()
        self.local_steps = 0
        self.obs = {'agent_0':1, 'agent_1':1,}
        return self.obs
    
    def step(self, action_dict):
        
        self.local_steps += 1
        
        act0 = action_dict['agent_0']
        act1 = action_dict['agent_1']
        
        if act0==0 and act1==0:
            rew0 = 1
            rew1 = 1
        if act0==0 and act1==1:
            rew0 = 5
            rew1 = 0
        if act0==1 and act1==0:
            rew0 = 0
            rew1 = 5
        if act0==1 and act1==1:
            rew0 = 4
            rew1 = 4
        
        rew = {'agent_0':rew0, 'agent_1':rew1,}
        
        self.obs = {'agent_0':act0, 'agent_1':act1,}
        
        if self.local_steps < self.max_steps:
            done = False
        else:
            done = True
       
        dones = {'agent_0':done, 'agent_1':done,'__all__':done,} 
        
        info = {'agent_0':{}, 'agent_1':{},}
        
        return self.obs, rew, dones, info
    
