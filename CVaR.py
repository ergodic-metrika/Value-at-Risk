# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 16:56:35 2022

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
from drawtoolset import plot_var

#pip install yfinance


header = ['Confidence Level', 'Value At Risk']

asset=yf.Ticker("BITO")
BITO = asset.history(period="3y")
BITO['Close'].plot(title="BITO trend")

symbol=BITO

BITO_returns=BITO['Close'].pct_change()

mean=np.mean(BITO_returns)
             
stdev=np.std(BITO_returns)

asset=yf.Ticker("BITO")
BITO = asset.history(period="3y")
BITO['Close'].plot(title="BITO trend")
BITO['Close']

price=BITO['Close']



returns = price.pct_change().dropna()

#Historical VaR
# Use quantile function for Historical VaR
hVaR_90 = BITO_returns.quantile(0.10)
hVaR_95 = BITO_returns.quantile(0.05)
hVaR_99 = BITO_returns.quantile(0.01)


#Scaling VaR
VaR_90 = norm.ppf(1-0.90,mean,stdev)
VaR_95 = norm.ppf(1-0.95,mean,stdev) #norm.ppf(0.05)
VaR_99 = norm.ppf(1-0.99,mean,stdev)

forecast_days = 5
f_VaR_90 = VaR_90*np.sqrt(forecast_days)
f_VaR_95 = VaR_95*np.sqrt(forecast_days)
f_VaR_99 = VaR_99*np.sqrt(forecast_days)

ftable = [['90%', f_VaR_90],['95%', f_VaR_95],['99%', f_VaR_99] ]
fheader = ['Confidence Level', '5-Day Forecast Value At Risk']
print(tabulate(ftable,headers=fheader))

num_of_shares = 1000
price = BITO['Close'].iloc[-1]
position = num_of_shares * price 

BITO_var_5days = position * f_VaR_99

print(f'BITO Holding Value: {position}')
print(f'BITO VaR at 99% confidence level is: {BITO_var_5days}')

# Scaled VaR over different time horizon
plt.figure(figsize=(8,6))
plt.plot(range(100),[-100*VaR_95*np.sqrt(x) for x in range(100)])
plt.xlabel('Horizon')
plt.ylabel('Var 95 (%)')
plt.title('VaR_95 Scaled by Time');


# Calculate CVar
CVaR_90 = (BITO_returns)[(BITO_returns)<=hVaR_90].mean()
CVaR_95 =(BITO_returns)[(BITO_returns)<=hVaR_95].mean()
CVaR_99 = (BITO_returns)[(BITO_returns)<=hVaR_99].mean()


ctable = [['90%', CVaR_90],['95%', CVaR_95],['99%', CVaR_99] ]
cheader = ['Confidence Level', 'Conditional Value At Risk']
print(tabulate(ctable,headers=cheader))