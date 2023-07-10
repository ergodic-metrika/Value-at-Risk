# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:38:23 2022

@author: sigma
"""

# Import 
import pandas as pd
import yfinance as yf 

import numpy as np
from numpy import *
from numpy.linalg import multi_dot

from scipy.stats import norm
from tabulate import tabulate

# Plot settings
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.rcParams['figure.figsize'] = 16, 8

plt.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = [12.0, 8.0]
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['lines.linewidth'] = 2.0


def VaR(BITO, cl=0.95):
    mean = np.mean(returns)
    stdev = np.std(returns)
    
    return np.around(100*norm.ppf(1-cl,mean,stdev),4)

# Portfolio stocklist
symbols = ['AAPL', 'BA', 'JPM', 'PG', 'CAT', 'XOM', 'AMAT', 'WMT', 'CRM', 'MRK' ]

# Number of assets
numofasset = len(symbols)

# Number of portfolio for optimization
numofportfolio = 5000
# Fetch data from yahoo finance for last six years
stocks_port = yf.download(symbols, start='2019-11-21', end='2022-11-21', progress=False)['Adj Close']

# Fetch data from yahoo finance for last six years
stocks_port = yf.download(symbols, start='2019-11-21', end='2022-11-21', progress=False)['Adj Close']

# Verify the output
stocks_port.tail()

#Retrive data
# Let's save the data for future use
stocks_port.to_csv('D:/Python for Quant Model/stocks_port.csv')

# Load locally stored data
df = pd.read_csv('D:/Python for Quant Model/stocks_port.csv', index_col=0, parse_dates=True)

df.head()

#Calculate returns
returns = df.pct_change().dropna()

#Assign weights
#The following weights are the optimal weights which come from another optimization model which 
#can be found in: https://github.com/UltimaMetrics/Optimization/blob/main/PortOpti.py
wts = np.array([0.5278,0.0,0.0,0.073,0.0,0.0,0.1278,0.0,0.0232,0.2482])[:,np.newaxis]
wts

# Stock returns
returns[:5]

port_ret = np.dot(returns,wts)
port_ret.flatten()

port_mean = port_ret.mean()
port_mean


# Covariance matrix
returns.cov()

# Portfolio volatility
port_stdev = np.sqrt(multi_dot([wts.T, returns.cov(), wts]))
port_stdev.flatten()[0]

# Portfolio Position
num_of_shares = 1000
port_pos = (df.iloc[-1] * num_of_shares).sum()
port_pos

# Calculate Portfolio VaR at difference confidence level
pVaR_90 = norm.ppf(1-0.90,port_mean,port_stdev).flatten()[0]
pVaR_95 = norm.ppf(1-0.95,port_mean,port_stdev).flatten()[0]
pVaR_99 = norm.ppf(1-0.99,port_mean,port_stdev).flatten()[0]

# Ouput results in tabular format
ptable = [['90%', pVaR_90],['95%', pVaR_95],['99%', pVaR_99]]
header = ['Confidence Level', 'Value At Risk']
print(tabulate(ptable,headers=header))

# Iterate over symbols
for stock in df.columns:
    pos = df[stock].iloc[-1] * num_of_shares
    pvar = pos * VaR(stock)
    
    print(f'{stock} Holding Value: {pos:0.4}') 
    print(f'{stock} VaR at 95% confidence level: {pvar:0.4}')
    print()

    
print(f'Portfolio Holding Value: {port_pos:0.4}')
print(f'Portoflio VaR at 95% confidence level: {port_pos * pVaR_95:0.4}')
