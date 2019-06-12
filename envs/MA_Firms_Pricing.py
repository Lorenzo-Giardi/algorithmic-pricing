
import gym
import ray
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class MultiAgentFirmsPricing(MultiAgentEnv):
    
    def __init__(self, num_agents=4, max_steps=10000000):
        self.dones = set()
        self.action_space = gym.spaces.Discrete(15)
        self.observation_space = gym.spaces.Box(
                low=np.zeros(num_agents), high=np.repeat(14, num_agents), dtype=np.int32)
        self.local_steps = 0
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.agents = list()
        self.obs = dict()
        
        for i in range(num_agents):
            self.agents.append('agent_'+str(i))
            self.obs.update({'agent_'+str(i):np.repeat(1, num_agents)})

        
    def reset(self):
        self.dones = set()
        self.local_steps = 0
        
        for i in self.agents:
            self.obs[i] = np.repeat(1, self.num_agents)
        
        return self.obs
    
    
    def step(self, action_dict):
        
        self.local_steps += 1
        
        prices = [1.42617, 1.479722, 1.533274, 1.586826, 1.640379, 1.693931, 
                  1.747486, 1.801035, 1,854587, 1.908139, 1.961691, 2.015244, 
                  2.068796, 2.122348, 2.1759]
        actions_to_prices_dict = dict(zip(list(range(15)), prices))
        
        agt_prices_dict = dict()
        for i in self.agents:
            p = actions_to_prices_dict[action_dict[i]]
            agt_prices_dict.update({i:p})
            
        actions_list = list(action_dict.values())
        prices_list = list(agt_prices_dict.values())
        prices_array = np.array(prices_list)
        
        q_dict = dict()
        for i in self.agents:
            q = (np.exp((2-agt_prices_dict[i])/0.25))/(np.sum((np.exp((2-prices_array)/0.25))+np.exp(0)))
            q_dict.update({i:q})
        
        rew_dict = dict()
        for i in self.agents:
            r = (agt_prices_dict[i] - 1) * q_dict[i]
            rew_dict.update({i:r})
        
        
        for i in self.agents:
            self.obs[i] = np.array(actions_list)
        
        if self.local_steps < self.max_steps:
            done = False
        else:
            done = True
       
        dones_dict = dict()
        for i in self.agents:
            dones_dict.update({i:done})
        dones_dict.update({'__all__':done})
        
        infos_dict = dict()
        for i in self.agents:
            infos_dict.update({i:{}})
        
        return self.obs, rew_dict, dones_dict, infos_dict
    
