#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:07:13 2019

@author: lorenzo
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")

x = [1.2, 2]
fx = [0, 0.2]
gx = [0.2,0]

plt.plot(x, fx, label='f(p)', c="#247afd")
plt.plot(x, gx, label='g(p)', c="#fd8d49")
plt.xlabel('Price')
#plt.ylabel('...')
plt.legend()
plt.savefig('/home/lorenzo/Desktop/function_plots.png', dpi=600)
plt.show()