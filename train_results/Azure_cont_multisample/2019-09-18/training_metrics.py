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

data1 = pd.read_csv("/home/lorenzo/algorithmic-pricing/train_results/Azure_cont_multisample/APEX_MultiAgentFirmsPricingContinuous_0_2019-09-18_13-15-58uc731pvg/progress.csv") 
data2 = pd.read_csv("/home/lorenzo/algorithmic-pricing/train_results/Azure_cont_multisample/APEX_MultiAgentFirmsPricingContinuous_1_2019-09-18_14-11-57dep9xakf/progress.csv")
data3 = pd.read_csv("/home/lorenzo/algorithmic-pricing/train_results/Azure_cont_multisample/APEX_MultiAgentFirmsPricingContinuous_2_2019-09-18_15-06-11ivqb33bz/progress.csv")
data4 = pd.read_csv("/home/lorenzo/algorithmic-pricing/train_results/Azure_cont_multisample/APEX_MultiAgentFirmsPricingContinuous_3_2019-09-18_16-01-27796dtdhp/progress.csv")
data5 = pd.read_csv("/home/lorenzo/algorithmic-pricing/train_results/Azure_cont_multisample/APEX_MultiAgentFirmsPricingContinuous_4_2019-09-18_16-57-15l_ueptaq/progress.csv")

var_list = ['episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 'timesteps_total', \
            'policy_reward_mean/agent_0', 'policy_reward_mean/agent_1', 'info/learner/agent_0/mean_td_error', 'info/learner/agent_1/mean_td_error']

data1 = data1[var_list]
data2 = data2[var_list]
data3 = data3[var_list]
data4 = data4[var_list]
data5 = data5[var_list]

new_column_names = ['rew_max', 'rew_mean', 'rew_min', 'ts_total', 'rew_agt0', 'rew_agt1', 'tderr_agt0', 'tderr_agt1']

data1.columns = new_column_names
data2.columns = new_column_names
data3.columns = new_column_names
data4.columns = new_column_names
data5.columns = new_column_names

data_mean = pd.concat([data1, data2, data3, data4, data5]).groupby(level=0).mean()

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

# Aggregate reward
plt.plot(data1['ts_total'], deltas(data1['rew_max']))
plt.plot(data2['ts_total'], deltas(data2['rew_max']))
plt.plot(data3['ts_total'], deltas(data3['rew_max']))
plt.plot(data4['ts_total'], deltas(data4['rew_max']))
plt.plot(data5['ts_total'], deltas(data5['rew_max']))
plt.xlabel('Timesteps')
plt.ylabel('Max profit gain')
plt.savefig('/home/lorenzo/Desktop/training_metrics_1.png', dpi=600)
plt.show()

plt.plot(data1['ts_total'], deltas(data1['rew_mean']))
plt.plot(data2['ts_total'], deltas(data2['rew_mean']))
plt.plot(data3['ts_total'], deltas(data3['rew_mean']))
plt.plot(data4['ts_total'], deltas(data4['rew_mean']))
plt.plot(data5['ts_total'], deltas(data5['rew_mean']))
plt.xlabel('Timesteps')
plt.ylabel('Mean profit gain')
plt.savefig('/home/lorenzo/Desktop/training_metrics_2.png', dpi=600)
plt.show()

plt.plot(data1['ts_total'], deltas(data1['rew_min']))
plt.plot(data2['ts_total'], deltas(data2['rew_min']))
plt.plot(data3['ts_total'], deltas(data3['rew_min']))
plt.plot(data4['ts_total'], deltas(data4['rew_min']))
plt.plot(data5['ts_total'], deltas(data5['rew_min']))
plt.xlabel('Timesteps')
plt.ylabel('Min profit gain')
plt.savefig('/home/lorenzo/Desktop/training_metrics_3.png', dpi=600)
plt.show()

# Per agent reward
plt.plot(data_mean['ts_total'], deltas1(data_mean['rew_agt0']), label='Agent_0')
plt.plot(data_mean['ts_total'], deltas1(data_mean['rew_agt1']), label='Agent_1')
plt.xlabel('Timesteps')
plt.ylabel('Average profit gains')
plt.legend()
plt.savefig('/home/lorenzo/Desktop/training_metrics_4.png', dpi=600)
plt.show()

# Per agent td error
plt.plot(data_mean['ts_total'], data_mean['tderr_agt0'], label='Agent_0')
plt.plot(data_mean['ts_total'], data_mean['tderr_agt1'], label='Agent_1')
plt.xlabel('Timesteps')
plt.ylabel('Mean TD error')
plt.legend()
plt.savefig('/home/lorenzo/Desktop/training_metrics_5.png', dpi=600)
plt.show()