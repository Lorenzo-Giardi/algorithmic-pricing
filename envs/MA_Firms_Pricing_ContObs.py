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
    
    def __init__(self, env_config={
                                   "num_agents":2,
                                   "max_steps":10**9,
                                   "p_min":1.2,
                                   "p_max":2,}):
        self.dones = set()
        self.local_steps = 0
        self.max_steps = env_config["max_steps"]
        self.num = env_config["num_agents"]
        self.p_min = env_config["p_min"]
        self.p_max = env_config["p_max"]
        
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
                low=np.repeat(self.p_min, self.num), high=np.repeat(self.p_max, self.num), dtype=np.float16)
        

        self.agents = list()
        self.obs = dict()
        self.info = dict()
        p = np.random.uniform(low=self.p_min, high=self.p_max)
        
        for i in range(self.num):
            self.agents.append('agent_'+str(i))
            self.info.update({'agent_'+str(i):None})
            obs = np.repeat(p, self.num)
            self.obs.update({'agent_'+str(i):obs.astype(np.float16)})
            
        self.avg_delta = 0

        
    def reset(self):
        self.dones = set()
        self.local_steps = 0
        p = np.random.uniform(low=self.p_min, high=self.p_max)
        
        for i in self.agents:
            obs = np.repeat(p, self.num)
            self.obs[i] = obs.astype(np.float16)
        
        return self.obs
    
    
    def step(self, action_dict):
        
        self.local_steps += 1
        
        delta_p_dict = dict()
        for i in self.agents:
            j = int(i[6]) # get agent numerical index
            a = action_dict[i] # get agent action
            x = np.random.uniform(.01, .03) # get a random number
            y = self.obs['agent_0'][j] # get previous price for agent
            # use log and exp to ease convergence while still allowing
            # for sudden changes in prices (e.g. to defect from cooperation)
            if a==0:
                delta_p = - (x + np.log(y)/4)
            elif a==1:
                delta_p = - x
            elif a==2:
                delta_p = 0
            elif a==3:
                delta_p = x
            elif a==4:
                delta_p = x + np.exp(-y)/2
            else:
                raise ValueError('Something is wrong with the action')
            delta_p_dict.update({i:delta_p})
        
        old_prices_array = self.obs['agent_0']
        delta_prices_array = np.array(list(delta_p_dict.values()))
        new_prices_array = old_prices_array + delta_prices_array
        new_prices_array = np.clip(new_prices_array, self.p_min, self.p_max - 0.001)
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
        
        # compute profit gains
        for i in self.agents:
            d = (rew[i] - 0.22589)/(0.337472 - 0.22589)
            self.info[i]={"delta":d}
        
        return self.obs, rew, dones, self.info
    
    def render(self):
        # Running average of profit gains
        d_sum = 0
        for i in self.agents:
            d_sum += self.info[i]['delta']
        d_avg = d_sum / self.num
        
        self.avg_delta = self.avg_delta * (self.local_steps -1) / self.local_steps + d_avg / self.local_steps 
        
        # print stuff every 100 steps
        if self.local_steps % 100==0:
            print(f'Step n: {self.local_steps}, Profit gains: {self.info}')
            print(f'Running avg profit gain: {self.avg_delta}')
