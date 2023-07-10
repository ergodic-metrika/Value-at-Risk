# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 10:27:27 2023

@author: sigma
"""

import numpy as np
import pandas as pd
import scipy.stats as stats

TX_data= pd.read_csv(r'E:\Python for Quant Model\TXVaR.csv', parse_dates=['Date'], index_col='Date')

def calculate_var(prices, holding_period, num_futures, num_calls, num_puts, call_delta, put_delta, contract_multiplier, alpha=0.99):
    # Calculate daily returns
    daily_returns = prices.pct_change().dropna()
    print("Daily returns:\n", daily_returns)

    # Calculate the holding period returns
    holding_period_returns = daily_returns.rolling(holding_period).apply(lambda x: (1 + x).prod() - 1, raw=True)
    print("Holding period returns:\n", holding_period_returns)

    # Calculate the holding period volatility
    holding_period_volatility = holding_period_returns.std()
    print("Holding period volatility:", holding_period_volatility)

    # Calculate the holding period mean return
    holding_period_mean_return = holding_period_returns.mean()
    print("Holding period mean return:", holding_period_mean_return)

    # Calculate the z-score for the given confidence level
    z = -stats.norm.ppf(alpha)

    # Calculate the VaR for the futures, short calls, and long puts
    futures_var = num_futures * contract_multiplier * (holding_period_mean_return - z * holding_period_volatility)
    calls_var = -num_calls * contract_multiplier * call_delta * (holding_period_mean_return - z * holding_period_volatility)
    puts_var = num_puts * contract_multiplier * -put_delta * (holding_period_mean_return - z * holding_period_volatility)

    # Calculate the total VaR for the portfolio
    total_var = (futures_var + calls_var + puts_var)*TX_data.iloc[-1]

    return total_var

# Replace these placeholders with your specific data
prices = TX_data  # A pandas Series with historical daily prices for the underlying asset
holding_period = 25  # The holding period in days
num_futures = 1  # The number of long futures contracts
num_calls = 1  # The number of short call options
num_puts = 1  # The number of long put options
call_delta = 0.5  # The delta of the short call options
put_delta = 0.3  # The delta of the long put options
contract_multiplier = 50  # The contract multiplier (e.g., $1000 for E-mini S&P 500 futures)
alpha = 0.99  # Confidence level

VaR = calculate_var(prices, holding_period, num_futures, num_calls, num_puts, call_delta, put_delta, contract_multiplier, alpha)
print("Value at Risk (VaR) for the portfolio:", VaR)



daily_returns = prices.pct_change().dropna()
holding_period_returns = daily_returns.rolling(holding_period).apply(lambda x: (1 + x).prod() - 1, raw=True)
holding_period_volatility = holding_period_returns.std()
holding_period_mean_return = holding_period_returns.mean()
z = -stats.norm.ppf(alpha)


futures_var = num_futures * contract_multiplier * (holding_period_mean_return - z * holding_period_volatility)
calls_var = -num_calls * contract_multiplier * call_delta * (holding_period_mean_return - z * holding_period_volatility)
puts_var = num_puts * contract_multiplier * -put_delta * (holding_period_mean_return - z * holding_period_volatility)
futures_var 
calls_var 
puts_var 

net=(futures_var +calls_var+puts_var)*TX_data.iloc[-1]
net


