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







### plot impulse response functions
plt.plot(deltas_irf['Agent_0'], label='Agent_0', c="#247afd")
plt.plot(deltas_irf['Agent_1'], label='Agent_1', c="#fd8d49")
plt.axhline(1, linestyle='dashed', c="#929591")
plt.xlabel('Timesteps')
plt.ylabel('Profit gains (deltas)')
plt.legend()
plt.savefig('/home/lorenzo/Desktop/multisample-deltas-irf.png', dpi=600)
plt.show()


plt.plot(prices_irf['Agent_0'], label='Agent_0', c="#247afd")
plt.plot(prices_irf['Agent_1'], label='Agent_1', c="#fd8d49")
plt.xlabel('Timesteps')
plt.ylabel('Prices')
plt.legend()
plt.savefig('/home/lorenzo/Desktop/multisample-prices-irf.png', dpi=600)
plt.show()
