# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 13:40:43 2022

@author: sigma
"""



import pandas as pd
import numpy as np
from numpy import *
import mibian
import yfinance as yf

#pip install mibian
#http://code.mibian.net/
import matplotlib.pyplot as pp

#Quick way to load data from Yahoo Finance
asset=yf.Ticker("UTEN")
UTEN_data= asset.history(period="5y")
UTEN_data['Close'].plot(title="10 YR T-note ETF")
#However the data must be sorted from the lastest to oldest

db=pd.read_csv(r'D:\Python for Quant Model\Bond_Data.csv')


UTEN=db.loc[:,'UTEN'] #Use UTEN as proxy for fixed income performance
AAPL=db.loc[:,'AAPL'] #Use AAPL as proxy for equity index performance


log_UTEN=np.log(UTEN)-np.log(UTEN.shift(1))
log_AAPL=np.log(AAPL)-np.log(AAPL.shift(1))
log_UTEN=log_UTEN[1:61]
log_AAPL=log_AAPL[1:61]


lamda=np.zeros(60)
for i in range(0,59):
    lamda[i]=0.94**i

var_UTEN=(1-0.94)*np.sum(lamda*log_UTEN*log_UTEN)
var_AAPL=(1-0.94)*np.sum(lamda*log_AAPL*log_AAPL)

#cov = pd.DataFrame(columns=['SPX','DJX','VIX','VXD'])
cov_UTEN_AAPL=(1-0.94)*np.sum(lamda*log_UTEN*log_AAPL)
cov_AAPL_UTEN=(1-0.94)*np.sum(lamda*log_UTEN*log_AAPL)


y=np.array([[var_UTEN,cov_UTEN_AAPL], [cov_AAPL_UTEN, var_AAPL]])


#UTEN is the ETF of the US 10-yr note


#Pricing models and IV generation
#BS([underlyingPrice, strikePrice, interestRate, daysToExpiration], volatility=x, callPrice=y, putPrice=z)
#MertonMe([underlyingPrice, strikePrice, interestRate, annualDividends, daysToExpiration], volatility=x, callPrice=y, putPrice=z)
#c_UTEN=mibian.ME([UTEN[0], 40, 0.07, 0, 25], callPrice=(80+82)/2)
#p_UTEN=mibian.Me([UTEN[0], 40, 0.07, 0, 25], putPrice=(65+68)/2)


#Now use BS rather than Merton of the above
c_UTEN=mibian.BS([UTEN[0], 40, 0.03, 25], volatility=0.07, callPrice=(80+82)/2)
p_UTEN=mibian.BS([UTEN[0], 40, 0.03, 25], volatility=0.07, putPrice=(65+68)/2)


c_AAPL=mibian.BS([UTEN[0],160, 0.03, 25], volatility=0.30, callPrice=(20+25)/2)
p_AAPL=mibian.BS([UTEN[0],160, 0.03, 25], volatility=0.30, putPrice=(30+35)/2)

print('implied Volatility')
print('UTEN call:',c_UTEN.impliedVolatility)
print('UTEN put:',p_UTEN.impliedVolatility)


# Question 3 & 4
#UTEN_initial=mibian.Me([UTEN[0], 40, 0.07, 0, 25], volatility=(c_UTEN.impliedVolatility+p_UTEN.impliedVolatility)/2)
UTEN_initial=mibian.BS([UTEN[0], 40, 0.03, 25], volatility=(c_UTEN.impliedVolatility+p_UTEN.impliedVolatility)/2)
AAPL_initial=mibian.BS([AAPL[0], 160, 0.03, 25], volatility=(c_AAPL.impliedVolatility+p_AAPL.impliedVolatility)/2)
initial_value=-50*UTEN_initial.callPrice-50*UTEN_initial.putPrice+50*AAPL_initial.callPrice+50*AAPL_initial.putPrice
print(initial_value)
iteration=1000
A=np.linalg.cholesky(y)
A=mat(A)
A
PL=np.zeros(iteration+1)
for i in range(0,iteration):
    e = np.random.standard_normal(size=(2,2))
    x=A*e
    UTENmc=UTEN[1]*np.exp(x[0])
    AAPLmc=AAPL[1]*np.exp(x[1])
    UTEN_montecarlo=mibian.Me([UTENmc, 40, 0.03, 0, 25], volatility=0.07)
    AAPL_montecarlo=mibian.Me([AAPLmc, 160, 0.03, 0, 25], volatility=0.30)
    mc_value=-50*UTEN_montecarlo.callPrice-50*UTEN_montecarlo.putPrice+50*AAPL_montecarlo.callPrice+50*AAPL_montecarlo.putPrice
    PL[i]=100*(mc_value-initial_value)
    e = np.random.standard_normal((4, 1))
    x=A*e
    UTENmc=UTEN[1]*np.exp(x[0])

    UTEN_montecarlo=mibian.Me([UTENmc, 405, 0.07, 0, 25], volatility=0.07)
    
    mc_value=-50*UTEN_montecarlo.callPrice-50*UTEN_montecarlo.putPrice
    PL[i]=100*(mc_value-initial_value)

# print(PL)
VaR=np.percentile(PL,5)
CVaR=np.mean(PL[PL<= VaR])

print('Monte Carlo VaR:',VaR)
print('CVaR (Expected Shortfall):',CVaR)