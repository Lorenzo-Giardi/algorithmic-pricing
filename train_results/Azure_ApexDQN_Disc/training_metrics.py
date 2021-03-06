#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:51:07 2019

@author: lorenzo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")

data1 = pd.read_csv("/home/lorenzo/algorithmic-pricing/train_results/Azure_ApexDQN_Disc/azure_disc_10_res0/APEX_MultiAgentFirmsPricing_0_2019-09-21_08-33-42mdvqcgyv/progress.csv") 
data2 = pd.read_csv("/home/lorenzo/algorithmic-pricing/train_results/Azure_ApexDQN_Disc/azure_disc_10_res1/APEX_MultiAgentFirmsPricing_0_2019-09-21_10-29-54_19h51zm/progress.csv")
data3 = pd.read_csv("/home/lorenzo/algorithmic-pricing/train_results/Azure_ApexDQN_Disc/azure_disc_10_res2/APEX_MultiAgentFirmsPricing_0_2019-09-21_14-03-453aaxi5sb/progress.csv")

var_list = ['episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 'timesteps_total', \
            'policy_reward_mean/agent_0', 'policy_reward_mean/agent_1', 'info/learner/agent_0/mean_td_error', 'info/learner/agent_1/mean_td_error']

data1 = data1[var_list]
data2 = data2[var_list]
data3 = data3[var_list]

new_column_names = ['rew_max', 'rew_mean', 'rew_min', 'ts_total', 'rew_agt0', 'rew_agt1', 'tderr_agt0', 'tderr_agt1']

data1.columns = new_column_names
data2.columns = new_column_names
data3.columns = new_column_names

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

# plot mean-min-max
plt.plot(data1['ts_total'], deltas(data1['rew_mean']), alpha = 0.9, lw=1, c='C0', label = 'mean')
plt.plot(data2['ts_total'], deltas(data2['rew_mean']), alpha = 0.9, lw=1, c='C0')
plt.plot(data3['ts_total'], deltas(data3['rew_mean']), alpha = 0.9, lw=1, c='C0')

plt.plot(data1['ts_total'], deltas(data1['rew_max']), alpha = 0.75, lw=0.75, linestyle=':', color='C2', label = 'max')
plt.plot(data2['ts_total'], deltas(data2['rew_max']), alpha = 0.75, lw=0.75, linestyle=':', color='C2')
plt.plot(data3['ts_total'], deltas(data3['rew_max']), alpha = 0.75, lw=0.75, linestyle=':', color='C2')      

plt.plot(data1['ts_total'], deltas(data1['rew_min']), alpha = 0.75, lw=0.75, linestyle=':', color='C1', label = 'min')
plt.plot(data2['ts_total'], deltas(data2['rew_min']), alpha = 0.75, lw=0.75, linestyle=':', color='C1')
plt.plot(data3['ts_total'], deltas(data3['rew_min']), alpha = 0.75, lw=0.75, linestyle=':', color='C1')  

plt.legend(loc = 'lower right')       
plt.xlabel('Timesteps')
plt.ylabel('Profit gain (delta)')
plt.savefig('/home/lorenzo/Desktop/training_metrics_0d.png', dpi=600)
plt.show()

# Aggregate reward
plt.plot(data1['ts_total'], deltas(data1['rew_max']), label='res0')
plt.plot(data2['ts_total'], deltas(data2['rew_max']), label='res1')
plt.plot(data3['ts_total'], deltas(data3['rew_max']), label='res2')
plt.xlabel('Timesteps')
plt.ylabel('Max profit gain')
plt.savefig('/home/lorenzo/Desktop/training_metrics_1d.png', dpi=600)
plt.show()

plt.plot(data1['ts_total'], deltas(data1['rew_mean']))
plt.plot(data2['ts_total'], deltas(data2['rew_mean']))
plt.plot(data3['ts_total'], deltas(data3['rew_mean']))
plt.xlabel('Timesteps')
plt.ylabel('Mean profit gain')
plt.savefig('/home/lorenzo/Desktop/training_metrics_2d.png', dpi=600)
plt.show()

plt.plot(data1['ts_total'], deltas(data1['rew_min']))
plt.plot(data2['ts_total'], deltas(data2['rew_min']))
plt.plot(data3['ts_total'], deltas(data3['rew_min']))
plt.xlabel('Timesteps')
plt.ylabel('Min profit gain')
plt.savefig('/home/lorenzo/Desktop/training_metrics_3d.png', dpi=600)
plt.show()

# Per agent reward
plt.plot(data1['ts_total'], deltas1(data1['rew_agt0']), label='Agent_0', c="C0", alpha=0.9, lw=1)
plt.plot(data2['ts_total'], deltas1(data2['rew_agt0']), c="C0", alpha=0.9, lw=1)
plt.plot(data3['ts_total'], deltas1(data3['rew_agt0']), c="C0", alpha=0.9, lw=1)
plt.plot(data1['ts_total'], deltas1(data1['rew_agt1']), label='Agent_1', c='C1', alpha=0.9, lw=1)
plt.plot(data2['ts_total'], deltas1(data2['rew_agt1']), c='C1', alpha=0.9, lw=1)
plt.plot(data3['ts_total'], deltas1(data3['rew_agt1']), c='C1', alpha=0.9, lw=1)
plt.xlabel('Timesteps')
plt.ylabel('Average profit gains')
plt.legend()
plt.savefig('/home/lorenzo/Desktop/training_metrics_4d.png', dpi=600)
plt.show()

# Per agent td error
plt.plot(data1['ts_total'], data1['tderr_agt0'], label='Agent_0', c="C0", alpha=0.9, lw=1)
plt.plot(data2['ts_total'], data2['tderr_agt0'], c="C0", alpha=0.9, lw=1)
plt.plot(data3['ts_total'], data3['tderr_agt0'], c="C0", alpha=0.9, lw=1)
plt.plot(data1['ts_total'], data1['tderr_agt1'], label='Agent_1', c='C1', alpha=0.9, lw=1)
plt.plot(data2['ts_total'], data2['tderr_agt1'], c='C1', alpha=0.9, lw=1)
plt.plot(data3['ts_total'], data3['tderr_agt1'], c='C1', alpha=0.9, lw=1)
plt.xlabel('Timesteps')
plt.ylabel('Mean TD error')
plt.legend()
plt.savefig('/home/lorenzo/Desktop/training_metrics_5d.png', dpi=600)
plt.show()

