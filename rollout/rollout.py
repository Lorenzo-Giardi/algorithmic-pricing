#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:06:04 2019

@author: lorenzo
"""
# From: https://github.com/ray-project/ray/blob/master/rllib/rollout.py

import argparse
import collections
import json
import os
import pickle

import gym
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.episode import _flatten_action
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.tune.util import merge_dicts
from ray.tune.registry import register_env

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
Example Usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
"""
#### HOW TO RUN 
"""
Add to command line options
checkpoint-path --run algname --env envname

Example
/home/lorenzo/ray_results/19_cont_DQN/DQN_MultiAgentFirmsPricingContinuous_1_2019-07-31_14-11-11sxig7ieu/checkpoint_3500/checkpoint-3500
 --run APEX --env firms_pricing_cont
"""

path='/home/lorenzo/algorithmic-pricing/rollout/'
os.chdir(path)
from MA_Firms_Pricing_ContObs import MultiAgentFirmsPricingContinuous
from MA_Firms_Pricing import MultiAgentFirmsPricing

ENV_CONFIG_1 = {"num_agents": 2,
              "max_steps":  10**9,
              "p_min":1.2,
              "p_max":2,}
ENV_CONFIG_2 = {
           "num_agents":2,
           "max_steps":10**9,
           "p_min":1.4315251,
           "p_max":1.9509807,
           "p_num":15,}

register_env("env_cont", lambda _: MultiAgentFirmsPricingContinuous(ENV_CONFIG_1))
register_env("env_disc", lambda _: MultiAgentFirmsPricing(ENV_CONFIG_2))

### PARSER ###
# Used to provide configs via command line
def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
        "given a checkpoint.",
        epilog=EXAMPLE_USAGE)

    parser.add_argument(
        "checkpoint",
        default="/tmp/ray/checkpoint_3500/checkpoint-3500",
        type=str,
        help="Checkpoint from which to roll out.")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        default = "DQN",
        #required=True,
        help="The algorithm or model to train. This may refer to the name "
        "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
        "user-defined trainable function or class registered in the "
        "tune registry.")
    required_named.add_argument(
        "--env",
        default = "firms_pricing_cont",
        type=str,
        help="The gym environment to use.")
    parser.add_argument(
        "--no-render",
        default=True,
        action="store_const",
        const=True,
        help="Surpress rendering of the environment.")
    parser.add_argument(
        "--steps", default=10000, help="Number of steps to roll out.")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Surpresses loading of configuration from checkpoint.")
    parser.add_argument(
            "--irfs",
            default=True,
            help = "Compute and plot impulse response functions")
    return parser

### RUNNER FUNCTION ###
# Used for loading configs, environment and policies
# by restoring the selected checkpoint
def run(args, parser, noplot=False, num_episodes=1):
    config = {}
    # Load configuration from file
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        if not args.config:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory.")
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    if "num_workers" in config:
        config["num_workers"] = min(2, config["num_workers"])
    config = merge_dicts(config, args.config)
    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    cls = get_agent_class(args.run)
    agent = cls(env=args.env, config=config)
    agent.restore(args.checkpoint)
    num_steps = int(args.steps)
    
    # call to rollout function
    deltas, del_irf, obs_irf = rollout(
            agent, args.env, num_steps, args.out, args.no_render, args.irfs, noplot, num_episodes)
    
    return deltas, del_irf, obs_irf


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID

### ROLLOUT FUNCTION ###
def rollout(agent, env_name, num_steps, out=None, no_render=False, irfs=True, noplot=False, num_episodes=1):
    Deltas = []
    Del_irf = []
    Obs_irf = []
    
    policy_agent_mapping = default_policy_agent_mapping

    if hasattr(agent, "workers"):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        action_init = {
            p: m.action_space.sample()
            for p, m in policy_map.items()
        }
    else:
        env = gym.make(env_name)
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    if out is not None:
        rollouts = []
    
    episode = 0

    while episode < num_episodes:
        print(f'Episode {episode} of {num_episodes}')
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        if out is not None:
            rollout = []
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        steps = 0
        deltas = []
        
        while not done and steps < (num_steps or steps + 1):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                    a_action = _flatten_action(a_action)  # tuple actions
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict
            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, info = env.step(action)
            
            #print(f'Step: {env.local_steps}, action: {action} info: {info}')
            deltas.append([info['agent_0']['delta'],info['agent_1']['delta']])
            
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            if not no_render:
                env.render()
            if out is not None:
                rollout.append([obs, action, next_obs, reward, done])
            steps += 1
            obs = next_obs
        if out is not None:
            rollouts.append(rollout)
        print("Episode reward", reward_total)
        
        if noplot == False:
            plt.plot(deltas)
            plt.show()
        
            sns.kdeplot(deltas, shade=True, cbar=True, cmap='Blues')
            plt.show()
            
        # === Code for impulse response functions ===
        
        if irfs == True:
            del_irf = []
            obs_irf = []
            del_irf.append([info['agent_0']['delta'],info['agent_1']['delta']])
            obs_irf.append(obs['agent_0'])
            
            for count in range(100):
                multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
                action_dict = {}
                for agent_id, a_obs in multi_obs.items():
                    if a_obs is not None:
                        policy_id = mapping_cache.setdefault(
                            agent_id, policy_agent_mapping(agent_id))
                        p_use_lstm = use_lstm[policy_id]
                        if p_use_lstm:
                            a_action, p_state, _ = agent.compute_action(
                                a_obs,
                                state=agent_states[agent_id],
                                prev_action=prev_actions[agent_id],
                                prev_reward=prev_rewards[agent_id],
                                policy_id=policy_id)
                            agent_states[agent_id] = p_state
                        else:
                            a_action = agent.compute_action(
                                a_obs,
                                prev_action=prev_actions[agent_id],
                                prev_reward=prev_rewards[agent_id],
                                policy_id=policy_id)
                        a_action = _flatten_action(a_action)  # tuple actions
                        if count < 1 and agent_id=='agent_0':
                            action_dict[agent_id] = 2
                            prev_actions[agent_id] = 2
                        else:
                            action_dict[agent_id] = a_action
                            prev_actions[agent_id] = a_action
                action = action_dict
                
                action = action if multiagent else action[_DUMMY_AGENT_ID]
                next_obs, reward, done, info = env.step(action)
                
                del_irf.append([info['agent_0']['delta'],info['agent_1']['delta']])
                obs_irf.append(obs['agent_0'])
            
            if noplot==False:
                plt.plot(del_irf)
                plt.show()
                
                plt.plot(obs_irf)
                plt.show()
            
        episode += 1
        Deltas.append(deltas)
        Del_irf.append(del_irf)
        Obs_irf.append(obs_irf)
    
    if out is not None:
        pickle.dump(rollouts, open(out, "wb"))
    
    return Deltas, Del_irf, Obs_irf

### EXECUTE CODE ###

ray.init()
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    
Deltas, Del_irf, Obs_irf = run(args, parser, noplot=True, num_episodes=100)

d_array = np.array(Deltas)
d_array_avgsess = d_array.mean(axis=0)
d_array_avgts = d_array.mean(axis=1)
dirf_array = np.array(Del_irf)
dirf_array = dirf_array.mean(axis=0)
obs_array = np.array(Obs_irf)
obs_array = obs_array.mean(axis=0)

# Some general results
print(f'Overall deltas mean: {d_array_avgts.mean():,.4f} and std: {d_array_avgts.std():,.4f}')
print(f'Agent0 deltas mean: {d_array_avgts[:,0].mean():,.4f} and std: {d_array_avgts[:,0].std():,.4f}')
print(f'Agent1 deltas mean: {d_array_avgts[:,1].mean():,.4f} and std: {d_array_avgts[:,1].std():,.4f}')

sns.set_style("ticks")

# plots for general rollout
sns.kdeplot(d_array_avgts, shade=True, cbar=True, cmap='Blues')
plt.xlabel('Agent_0')
plt.ylabel('Agent_1')
plt.savefig('/home/lorenzo/Desktop/bivariate-density-deltas.png', dpi=600)
plt.show()

sns.kdeplot(d_array_avgts[:,0], shade=True, label='Agent_0')
sns.kdeplot(d_array_avgts[:,1], shade=True, label='Agent_1')
plt.xlabel('Profit gains (deltas)')
plt.legend()
plt.savefig('/home/lorenzo/Desktop/bivariate-density-deltas-2.png', dpi=600)
plt.show()

plt.plot(d_array_avgsess[0:1000,0], label='Agent_0', lw=0.8, alpha=0.9)
plt.plot(d_array_avgsess[0:1000,1], label='Agent_1', lw=0.8, alpha=0.9)
plt.xlabel('Timesteps')
plt.ylabel('Profit gains (deltas)')
plt.legend()
plt.savefig('/home/lorenzo/Desktop/plot-deltas.png', dpi=600)
plt.show()

# plots for impulse response functions
plt.plot(dirf_array[:,0], label='Agent_0', c="#247afd")
plt.plot(dirf_array[:,1], label='Agent_1', c="#fd8d49")
plt.axhline(1, linestyle='dashed', c="#929591")
plt.xlabel('Timesteps')
plt.ylabel('Profit gains (deltas)')
plt.legend()
plt.savefig('/home/lorenzo/Desktop/deltas-irf.png', dpi=600)
plt.show()


plt.plot(obs_array[:,0], label='Agent_0', c="#247afd")
plt.plot(obs_array[:,1], label='Agent_1', c="#fd8d49")
plt.xlabel('Timesteps')
plt.ylabel('Prices')
plt.legend()
plt.savefig('/home/lorenzo/Desktop/prices-irf.png', dpi=600)
plt.show()

# save some data as pandas dataframe
import pandas as pd
Deltas_df = pd.DataFrame(d_array)
Deltas_df.columns = ['Agent_0', 'Agent_1']

# save dataframe to disk
# Deltas_df.to_csv('Deltas_df.csv')
