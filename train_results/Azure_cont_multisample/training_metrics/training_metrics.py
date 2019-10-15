#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:05:47 2019

@author: lorenzo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")

import glob
import os

path = '/home/lorenzo/algorithmic-pricing/train_results/Azure_cont_multisample/training_metrics' # use your path
all_files = glob.glob(os.path.join(path, "*.csv"))

j=0
dfs = dict()
for i in all_files:
    dfs.update({'data_'+str(j): pd.read_csv(i) })
    j += 1

var_list = ['episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 'timesteps_total', \
            'policy_reward_mean/agent_0', 'policy_reward_mean/agent_1', 'info/learner/agent_0/mean_td_error', 'info/learner/agent_1/mean_td_error']
new_column_names = ['rew_max', 'rew_mean', 'rew_min', 'ts_total', 'rew_agt0', 'rew_agt1', 'tderr_agt0', 'tderr_agt1']

for i in dfs:
    dfs[i] = dfs[i][var_list]
    dfs[i].columns = new_column_names

# coop profit 67.5
# nash profit 45.18

def deltas(data):
    x = np.array(data)
    x = (x-45.18)/(67.5-45.18)
    return x

# coop profit 33.75
# nash profit 22.59

def deltas1(data):
    x = np.array(data)
    x = (x-22.59)/(33.75-22.59)
    return x

# create new columns for deltas
for i in dfs:
    dfs[i]['delta_min'] = deltas(dfs[i]['rew_min'])
    dfs[i]['delta_mean'] = deltas(dfs[i]['rew_mean'])
    dfs[i]['delta_max'] = deltas(dfs[i]['rew_max'])
    dfs[i]['delta_agt0'] = deltas1(dfs[i]['rew_agt0'])
    dfs[i]['delta_agt1'] = deltas1(dfs[i]['rew_agt1'])

# dataframes in dfs have different lenghts!
for i in dfs:
    dfs[i] = dfs[i][0:106]

data_mean = pd.concat(dfs.values()).groupby(level=0).mean()
data_max = pd.concat(dfs.values()).groupby(level=0).min()
data_min = pd.concat(dfs.values()).groupby(level=0).max()
data_std = pd.concat(dfs.values()).groupby(level=0).std()

# Aggregate reward
plt.plot(data_max['ts_total'], data_max['delta_max'], linestyle='--', color='grey')
plt.plot(data_min['ts_total'], data_min['delta_min'], linestyle='--', color='grey')
plt.plot(data_mean['ts_total'], data_mean['delta_mean'])
plt.fill_between(data_mean['ts_total'], (data_mean['delta_mean']-1.96*data_std['delta_mean']), (data_mean['delta_mean']+1.96*data_std['delta_mean']), alpha=.2)
plt.xlabel('Timesteps')
plt.ylabel('Profit gain')
plt.savefig('/home/lorenzo/Desktop/training_metrics_2.png', dpi=600)
plt.show()

# Per agent reward
plt.plot(data_mean['ts_total'], data_mean['delta_agt0'], label='Agent_0')
plt.plot(data_mean['ts_total'], data_mean['delta_agt1'], label='Agent_1')
plt.xlabel('Timesteps')
plt.ylabel('Average profit gains')
plt.legend()
plt.savefig('/home/lorenzo/Desktop/training_metrics_4.png', dpi=600)
plt.show()
