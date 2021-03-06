#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:00:14 2019

@author: lorenzo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")

import glob
import os

# ================ #
#       DATA      ##
# ================ #


#### deltas averaged over agents ####
path = '/home/lorenzo/algorithmic-pricing/train_results/Azure_cont_multisample/rollout/deltas_avgagt_csv'
all_files = glob.glob(os.path.join(path, "*.csv"))

j=0
deltas_avgagt = dict()
for i in all_files:
    deltas_avgagt.update({'data_'+str(j): pd.read_csv(i, index_col=0)})
    j += 1
    
deltas_avgagt = pd.concat(deltas_avgagt.values())


#### deltas averaged over timesteps ####
path = '/home/lorenzo/algorithmic-pricing/train_results/Azure_cont_multisample/rollout/deltas_avgts_csv'
all_files = glob.glob(os.path.join(path, "*.csv"))

j=0
deltas_avgts = dict()
for i in all_files:
    deltas_avgts.update({'data_'+str(j): pd.read_csv(i, index_col=0)})
    j += 1
    
deltas_avgts = pd.concat(deltas_avgts.values())


#### deltas averaged over sessions ####
path = '/home/lorenzo/algorithmic-pricing/train_results/Azure_cont_multisample/rollout/deltas_avgsess_csv'
all_files = glob.glob(os.path.join(path, "*.csv"))

j=0
deltas_avgsess = dict()
for i in all_files:
    deltas_avgsess.update({'data_'+str(j): pd.read_csv(i, index_col=0)})
    j += 1
    
deltas_avgsess = pd.concat(deltas_avgsess.values()).groupby(level=0).mean()


#### deltas IRFs averaged over sessions ####
path = '/home/lorenzo/algorithmic-pricing/train_results/Azure_cont_multisample/rollout/deltas_irf_csv'
all_files = glob.glob(os.path.join(path, "*.csv"))

j=0
deltas_irf = dict()
for i in all_files:
    deltas_irf.update({'data_'+str(j): pd.read_csv(i, index_col=0)})
    j += 1
    
deltas_irf = pd.concat(deltas_irf.values()).groupby(level=0).mean()


#### prices IRFs averaged over sessions ####
path = '/home/lorenzo/algorithmic-pricing/train_results/Azure_cont_multisample/rollout/prices_irf_csv'
all_files = glob.glob(os.path.join(path, "*.csv"))

j=0
prices_irf = dict()
for i in all_files:
    prices_irf.update({'data_'+str(j): pd.read_csv(i, index_col=0)})
    j += 1
    
prices_irf = pd.concat(prices_irf.values()).groupby(level=0).mean()

# Print some general results
deltas_mean = deltas_avgsess.mean()
print(f'Overall deltas mean: {np.mean(np.array(deltas_avgts)):,.4f} and std: {np.std(np.array(deltas_avgts), axis=None):,.4f}')
print(f'Agent0 deltas mean: {deltas_avgts.mean()[0]:,.4f} and std: {deltas_avgts.std()[0]:,.4f}')
print(f'Agent1 deltas mean: {deltas_avgts.mean()[1]:,.4f} and std: {deltas_avgts.std()[1]:,.4f}')


# ================ #
#      PLOTS      ##
# ================ #

### Plot distribution of deltas

sns.kdeplot(deltas_avgts, shade=True, cbar=True, cmap='Blues')
plt.xlabel('Agent_0')
plt.ylabel('Agent_1')
plt.savefig('/home/lorenzo/Desktop/multisample-eval_density_deltas.png', dpi=600)
plt.show()

sns.kdeplot(deltas_avgts['Agent_0'], shade=True, label='Agent_0')
sns.kdeplot(deltas_avgts['Agent_1'], shade=True, label='Agent_1')
plt.xlabel('Profit gains (deltas)')
plt.legend()
plt.savefig('/home/lorenzo/Desktop/multisample-eval_density_deltas_2.png', dpi=600)
plt.show()

plt.plot(deltas_avgsess['Agent_0'], label='Agent_0', lw=0.8, alpha=0.9)
plt.plot(deltas_avgsess['Agent_1'], label='Agent_1', lw=0.8, alpha=0.9)
plt.xlabel('Timesteps')
plt.ylabel('Profit gains (deltas)')
plt.legend()
plt.savefig('/home/lorenzo/Desktop/multisample-eval_plot_deltas.png', dpi=600)
plt.show()



### plot impulse response functions

plt.plot(deltas_irf['Agent_0'], label='Agent_0', c="#247afd")
plt.plot(deltas_irf['Agent_1'], label='Agent_1', c="#fd8d49")
#plt.axhline(1, linestyle='dashed', c="#929591")
plt.xlabel('Timesteps')
plt.ylabel('Profit gains (deltas)')
plt.legend()
plt.savefig('/home/lorenzo/Desktop/multisample_deltas_irf.png', dpi=600)
plt.show()


plt.plot(prices_irf['Agent_0'], label='Agent_0', c="#247afd")
plt.plot(prices_irf['Agent_1'], label='Agent_1', c="#fd8d49")
plt.xlabel('Timesteps')
plt.ylabel('Prices')
plt.legend()
plt.savefig('/home/lorenzo/Desktop/multisample_prices-irf.png', dpi=600)
plt.show()
