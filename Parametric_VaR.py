# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 16:36:23 2022

@author: sigma
"""

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

asset=yf.Ticker("BITO")
BITO = asset.history(period="3y")
BITO['Close'].plot(title="BITO trend")

BITO['Close'].head()
BITO_returns=BITO['Close'].pct_change()

mean=np.mean(BITO_returns)
             
stdev=np.std(BITO_returns)

# Calculate VaR at difference confidence level
VaR_90 = norm.ppf(1-0.90,mean,stdev)
VaR_95 = norm.ppf(1-0.95,mean,stdev) #norm.ppf(0.05)
VaR_99 = norm.ppf(1-0.99,mean,stdev)


table = [['90%', VaR_90],['95%', VaR_95],['99%', VaR_99] ]
header = ['Confidence Level', 'Value At Risk']
print(tabulate(table,headers=header))


returns = BITO['Close'].pct_change().dropna()


#VaR function for individual stock/index
# VaR function
def VaR(BITO, cl=0.95):
    mean = np.mean(returns)
    stdev = np.std(returns)
    
    return np.around(100*norm.ppf(1-cl,mean,stdev),4)

VaR(BITO)



#Now assume holdings certain number of shares
num_of_shares = 1000
price = BITO.iloc[-1]
position = num_of_shares * price 

BITO_var = position * VaR_99


BITO_var

print(f'Bito Holding Value: {position}')
print(f'Bito VaR at 99% confidence level is: {Bito_var}')

# VaR calculation by appling direct formulae
position * (mean + norm.ppf(1-0.99) * stdev)         # mean-2.33*stdev
