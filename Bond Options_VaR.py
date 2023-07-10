# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 13:40:43 2022

@author: sigma
"""



import pandas as pd
import numpy as np
from numpy import *
import mibian

#pip install mibian
#http://code.mibian.net/
import matplotlib.pyplot as pp


index=pd.read_csv(r'D:\Python for Quant Model\Bond_Data.csv')

#df=pd.read_excel(r'D:\Derivatives Trading\ResearchRecord.xlsx')

#print(index.head(100))

#SPX=index.iloc[:,1]
#UTEN is the ETF of the US 10-yr note
UTEN=index.loc[:,'UTEN']


daily_return_uten=UTEN.pct_change()


sd_20=daily_return_uten.rolling(20).std()

vol_20=sd_20*(250**0.5)

vol_20

mean(vol_20)

#Question 1-a
log_UTEN=np.log(UTEN)-np.log(UTEN.shift(1))


lamda=np.zeros(200)
for i in range(0,199):
    lamda[i]=0.94**i

var_UTEN=(1-0.94)*np.sum(lamda*log_UTEN*log_UTEN)



# Question 2
#BS([underlyingPrice, strikePrice, interestRate, daysToExpiration], volatility=x, callPrice=y, putPrice=z)
#MertonMe([underlyingPrice, strikePrice, interestRate, annualDividends, daysToExpiration], volatility=x, callPrice=y, putPrice=z)
#c_UTEN=mibian.ME([UTEN[0], 40, 0.07, 0, 25], callPrice=(80+82)/2)
#p_UTEN=mibian.Me([UTEN[0], 40, 0.07, 0, 25], putPrice=(65+68)/2)


#Now use BS rather than Merton of the above
c_UTEN=mibian.BS([40, 42, 0.04, 18], volatility=0.11, callPrice=(5+5.2)/2)
p_UTEN=mibian.BS([40, 38, 0.04, 18], volatility=0.11, putPrice=(4.5+4.8)/2)

print('implied Volatility')
print('UTEN call:',c_UTEN.impliedVolatility)
print('UTEN put:',p_UTEN.impliedVolatility)


# Question 3 & 4
#UTEN_initial=mibian.Me([UTEN[0], 40, 0.07, 0, 25], volatility=(c_UTEN.impliedVolatility+p_UTEN.impliedVolatility)/2)
UTEN_initial=mibian.BS([UTEN[0], 40, 0.03, 25], volatility=(c_UTEN.impliedVolatility+p_UTEN.impliedVolatility)/2)

initial_value=-50*UTEN_initial.callPrice-50*UTEN_initial.putPrice
print(initial_value)
iteration=1000
A=np.linalg.cholesky(y)
A=mat(A)
PL=np.zeros(iteration+1)
for i in range(0,iteration):
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
