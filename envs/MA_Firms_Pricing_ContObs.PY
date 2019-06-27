import gym
import ray
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

"""
Multiagent environment in which N firms have to set their price
with the goal of maximizing their profits.

Action space: [0, 1, 2, 3, 4] -> size: 5
    - 0: big decrease in price (-.03, -.08)
    - 1: small decrease in price (-.01, -.03)
    - 2: no change in price (0)
    - 3: small increase in price (.01, .03)
    - 4: big increase in price (.03, .08)

Observation space: array of continuous prices from 1.200 to 2.00 (float16)

The following methods are provided:

    env.reset()
    -> returns: observation

    env.step(actions_dictionary)
    -> returns: observation, reward, done, info

Everything is returned in dictionary format:
    obs = {
            'agent_0': obs0,
            'agent_1': obs1,
            'agent_2': obs2
    }
"""

class MultiAgentFirmsPricingContinuous(MultiAgentEnv):
    
    def __init__(self, env_config={"num_agents":2, "max_steps":10**9}):
        self.dones = set()
        self.max_steps = env_config["max_steps"]
        self.num = env_config["num_agents"]
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
                low=np.repeat(1.2, self.num), high=np.repeat(2, self.num), dtype=np.float16)
        self.local_steps = 0

        self.agents = list()
        self.obs = dict()
        
        for i in range(self.num):
            self.agents.append('agent_'+str(i))
            self.obs.update({'agent_'+str(i):np.repeat(1.75, self.num)})

        
    def reset(self):
        self.dones = set()
        self.local_steps = 0
        
        for i in self.agents:
            self.obs[i] = np.repeat(1.75, self.num)
        
        return self.obs
    
    
    def step(self, action_dict):
        
        self.local_steps += 1
        
        delta_p_dict = dict()
        for i in self.agents:
            a = action_dict[i]
            if a==0:
                delta_p = - np.random.uniform(.03, .08)
            elif a==1:
                delta_p = - np.random.uniform(.01, .03)
            elif a==2:
                delta_p = 0
            elif a==3:
                delta_p = np.random.uniform(.01, .03)
            elif a==4:
                delta_p = np.random.uniform(.03, .08)
            else:
                raise ValueError('Something is wrong with the action')
            delta_p_dict.update({i:delta_p})
        
        old_prices_array = self.obs['agent_0']
        delta_prices_array = np.array(list(delta_p_dict.values()))
        new_prices_array = old_prices_array + delta_prices_array
        new_prices_array = np.clip(new_prices_array, 1.2, 1.999)
        new_prices_array = new_prices_array.astype(np.float16)
        
        new_prices_dict = dict(zip(self.agents, new_prices_array))
        
        q_dict = dict()
        for i in self.agents:
            q = (np.exp((2-new_prices_dict[i])/0.25))/(np.sum((np.exp((2-new_prices_array)/0.25)))+np.exp(0))
            q_dict.update({i:q})
        
        rew = dict()
        for i in self.agents:
            r = (new_prices_dict[i] - 1) * q_dict[i]
            rew.update({i:r})
        
        
        for i in self.agents:
            self.obs[i] = new_prices_array
        
        if self.local_steps < self.max_steps:
            done = False
        else:
            done = True
       
        dones = dict()
        for i in self.agents:
            dones.update({i:done})
        dones.update({'__all__':done})
        
        info = dict()
        for i in self.agents:
            info.update({i:{}})
        
        return self.obs, rew, dones, info
