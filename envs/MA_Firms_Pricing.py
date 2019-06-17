
import itertools
import gym
import ray
import numpy as np
import matplotlib.pyplot as plt
from ray.rllib.env.multi_agent_env import MultiAgentEnv

"""
Multiagent environment in which N firms have to set their price
with the goal of maximizing their profits.

Action space: [0, 1, 2, ...., 14] -> size:15

Observation space: all possible combinations of [a0, a1, ..., aN] -> size 15^N

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

class MultiAgentFirmsPricing(MultiAgentEnv):
    
    # env initialization. configs should be provided using a dictionary
    # to avoid errors in RLLib
    def __init__(self, env_config={"num_agents":2, "max_steps":10**9}):
        self.dones = set()
        self.max_steps = env_config["max_steps"]
        self.num = env_config["num_agents"]
        self.action_space = gym.spaces.Discrete(15)
        self.observation_space = gym.spaces.Box(
                low=np.zeros(self.num), high=np.repeat(14, self.num), dtype=np.int32)
        self.local_steps = 0

        self.agents = list()
        self.obs = dict()
        
        # Name agents 'agent_0', 'agent_1' and so on
        # and initialize observation to nash equilibrium
        for i in range(self.num):
            self.agents.append('agent_'+str(i))
            self.obs.update({'agent_'+str(i):np.repeat(1, self.num)})

    # function for resetting the env    
    def reset(self):
        self.dones = set()
        self.local_steps = 0
        
        for i in self.agents:
            self.obs[i] = np.repeat(1, self.num)
        
        return self.obs
    
    # function for stepping the env
    def step(self, action_dict):
        
        self.local_steps += 1
        
        # 15 prices, equally spaced
        prices = [1.42617, 1.479722, 1.533274, 1.586826, 1.640379, 1.693931, 
                  1.747486, 1.801035, 1.854587, 1.908139, 1.961691, 2.015244, 
                  2.068796, 2.122348, 2.1759]
        # dictionary {actions:prices}
        actions_to_prices_dict = dict(zip(list(range(15)), prices))
        
        # dictionary {agent_id : agent_price}
        agt_prices_dict = dict()
        for i in self.agents:
            a = action_dict[i]
            p = actions_to_prices_dict[a]
            agt_prices_dict.update({i:p})
        
        # list of actions and prices that have been taken
        actions_list = list(action_dict.values())
        prices_list = list(agt_prices_dict.values())
        prices_array = np.array(prices_list)
        
        # dictionary {agent_id: agent_quantity}
        q_dict = dict()
        for i in self.agents:
            q = (np.exp((2-agt_prices_dict[i])/0.25))/(
                np.sum((np.exp((2-prices_array)/0.25))+np.exp(0)))
            q_dict.update({i:q})
        
        # dictionary {agent_id: agent_reward}
        rew = dict()
        for i in self.agents:
            r = (agt_prices_dict[i] - 1) * q_dict[i]
            rew.update({i:r})
        
        # update new_observation to new actions
        for i in self.agents:
            self.obs[i] = np.array(actions_list)
        
        # set env done=True after max_steps
        if self.local_steps < self.max_steps:
            done = False
        else:
            done = True
        # dictionary {agent_id: done, ..., '__all__': done}
        dones = dict()
        for i in self.agents:
            dones.update({i:done})
        dones.update({'__all__':done})
        
        # dictionary {agent_id: info}
        info = dict()
        for i in self.agents:
            info.update({i:{}})
        
        return self.obs, rew, dones, info

    
