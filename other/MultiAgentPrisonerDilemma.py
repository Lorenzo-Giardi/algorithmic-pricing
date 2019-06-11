#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:56:52 2019

@author: lorenzo
"""

# SINGLE AGENT ENVIRONMENT 

import gym
import ray
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class RepeatedPrisonerDilemma(gym.Env):
        
    def __init__(self, episode_lenght):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(2)
        self.state = np.random.randint(0,2)
        self.viewer = None
        self.local_steps = 0
        self.episode_lenght = episode_lenght

        
    def reset(self):
        self.state = np.random.randint(0,2)
        self.local_steps = 0
        return self.state
    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        
        self.state = action
        self.local_steps += 1
        
        done = self.local_steps > self.episode_lenght
        done = bool(done)
        
        reward = 0
        
        return self.state, reward, done, {}

# MULTI-AGENTS ENVIRONMENT

class MultiAgentPrisonerDilemma(MultiAgentEnv):
    
    def __init__(self, num_agents):
        self.agents = [RepeatedPrisonerDilemma(10) for _ in range(num_agents)]
        self.dones = set()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(2)
        self.state = np.array([1,1])
        self.resetted = False
        
    def reset(self):
        self.resetted = True
        self.dones = set()
        return {i: a.reset() for i, a in enumerate(self.agents)}
    
    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}
        
        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
            if done[i]:
                self.dones.add[i]
            done["__all__"] = len(self.dones) == len(self.agents)
        
        if obs[0]==0 and obs[1]==0:
            rew[0] = 1
            rew[1] = 1
        if obs[0]==0 and obs[1]==1:
            rew[0] = 5
            rew[1] = 0
        if obs[0]==1 and obs[1]==0:
            rew[0] = 0
            rew[1] = 5
        if obs[0]==1 and obs[1]==1:
            rew[0] = 4
            rew[1] = 4
            
        return obs, rew, done, info

# TRAINING

import argparse
import random
from ray import tune
from ray.rllib.models import Model, ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

parser = argparse.ArgumentParser()

parser.add_argument("--num-agents", type=int, default=2)
parser.add_argument("--num-policies", type=int, default=2)
parser.add_argument("--num-iters", type=int, default=100)
parser.add_argument("--simple", action="store_true")

class CustomModel1(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        # Example of (optional) weight sharing between two different policies.
        # Here, we share the variables defined in the 'shared' variable scope
        # by entering it explicitly with tf.AUTO_REUSE. This creates the
        # variables for the 'fc1' layer in a global scope called 'shared'
        # outside of the policy's normal variable scope.
        with tf.variable_scope(
                tf.VariableScope(tf.AUTO_REUSE, "shared"),
                reuse=tf.AUTO_REUSE,
                auxiliary_name_scope=False):
            last_layer = tf.layers.dense(
                input_dict["obs"], 64, activation=tf.nn.relu, name="fc1")
        last_layer = tf.layers.dense(
            last_layer, 64, activation=tf.nn.relu, name="fc2")
        output = tf.layers.dense(
            last_layer, num_outputs, activation=None, name="fc_out")
        return output, last_layer


class CustomModel2(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        # Weights shared with CustomModel1
        with tf.variable_scope(
                tf.VariableScope(tf.AUTO_REUSE, "shared"),
                reuse=tf.AUTO_REUSE,
                auxiliary_name_scope=False):
            last_layer = tf.layers.dense(
                input_dict["obs"], 64, activation=tf.nn.relu, name="fc1")
        last_layer = tf.layers.dense(
            last_layer, 64, activation=tf.nn.relu, name="fc2")
        output = tf.layers.dense(
            last_layer, num_outputs, activation=None, name="fc_out")
        return output, last_layer


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    # Simple environment with `num_agents` independent cartpole entities
    register_env("multi_prisoner_dilemma", lambda _: MultiAgentPrisonerDilemma(args.num_agents))
    ModelCatalog.register_custom_model("model1", CustomModel1)
    ModelCatalog.register_custom_model("model2", CustomModel2)
    single_env = RepeatedPrisonerDilemma
    
    obs_space = gym.spaces.Discrete(2)
    act_space = gym.spaces.Discrete(2)

    # Each policy can have a different configuration (including custom model)
    def gen_policy(i):
        config = {
            "model": {
                "custom_model": ["model1", "model2"][i % 2],
            },
            "gamma": random.choice([0.95, 0.99]),
        }
        return (None, obs_space, act_space, config)

    # Setup PPO with an ensemble of `num_policies` different policies
    policies = {
        "policy_{}".format(i): gen_policy(i)
        for i in range(args.num_policies)
    }
    policy_ids = list(policies.keys())

    tune.run(
        "DQN",
        stop={"training_iteration": args.num_iters},
        config={
            "env": "multi_prisoner_dilemma",
            "log_level": "DEBUG",
            "num_workers": 2,
            "num_envs_per_worker": 4,
            #"simple_optimizer": args.simple,
            #"num_sgd_iter": 10,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": tune.function(
                    lambda agent_id: random.choice(policy_ids)),
            },
        },
    )












import ray.rllib.agents.dqn as dqn
import random

trainer = dqn.DQNAgent(env=MultiAgentPrisonerDilemma(2), config={
        "multiagent": {
                "policy_graphs": {
                        "agent_0": (None, gym.spaces.Discrete(2), gym.spaces.Discrete(2), {}),
                        "agent_1": (None, gym.spaces.Discrete(2), gym.spaces.Discrete(2), {}),
                },
                "policy_mapping_fn":
                    lambda agent_id:
                        random.choice(["agent_0", "agent_1"])
        },
})
    
while True:
    print(trainer.train())


import ray
from ray import tune
from ray.tune.registry import register_env

register_env("multi_prisoner_dilemma", lambda _: MultiAgentPrisonerDilemma(args.num_agents))

ray.shutdown()
ray.init()

tune.run(
        "DQN",
        stop={"episode_reward_mean": 350},
        config={"env": "multi_prisoner_dilemma",
                "num_gpus": 0,
                "num_workers": 4,
                "num_envs_per_worker": 8,
                "lr": 0.001,
                },
)
