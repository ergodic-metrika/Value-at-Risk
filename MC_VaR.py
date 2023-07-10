# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 16:13:16 2022

@author: sigma
"""

# Data manipulation
import pandas as pd
import numpy as np
from numpy.linalg import multi_dot

from scipy.stats import norm
from tabulate import tabulate

# Import matplotlib for visualization
import matplotlib
import matplotlib.pyplot as plt

import yfinance as yf

# Plot settings
plt.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = [12.0, 8.0]
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['lines.linewidth'] = 2.0

#from helper import plot_var


# Set seed for reproducibility
np.random.seed(42)

# Number of simulations
n_sims = 2500

#Get data 
#ProShare Bitcoin Strategy ETF
asset=yf.Ticker("BITO")
BITO = asset.history(period="3y")
BITO['Close'].plot(title="BITO trend")

BITO_returns=BITO['Close'].pct_change()

mean=np.mean(BITO_returns)
             
stdev=np.std(BITO_returns)

# Simulate returns and sort
sim_returns = np.random.normal(mean, stdev, n_sims)

# Use percentile function for MCVaR
MCVaR_90 = np.percentile(sim_returns,10)
MCVaR_95 = np.percentile(sim_returns, 5)
MCVaR_99 = np.percentile(sim_returns,1)

mctable = [['90%', MCVaR_90],['95%', MCVaR_95],['99%', MCVaR_99]]
mctable


header = ['Confidence Level', 'Value At Risk']

print(tabulate(mctable,headers=header))