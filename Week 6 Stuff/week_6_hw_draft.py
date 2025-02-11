import pandas as pd
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
from datetime import datetime
import matplotlib.pyplot as plt


# Stock portfolio
tickers = ['QQQ', 'IWB', 'VONE']
start_date = '2013-01-01'
end_date = '2024-01-01'

# Download historical price data
stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

stock_data.reset_index(inplace=True)
stock_data['Year']=stock_data['Date'].dt.year
stock_data['Month']=stock_data['Date'].dt.month


stock_data.rename(columns={'QQQ':'fund1','IWB':'fund2','VONE':'fund3'},inplace=True)
stock_data[['fund1_ret_d','fund2_ret_d','fund3_ret_d']]=stock_data[['fund1','fund2','fund3']].pct_change()
stock_data.sort_values(by=['Date'], inplace=True)

stock_data[['fund1_ret_d+1','fund2_ret_d+1','fund3_ret_d+1']]=stock_data[['fund1_ret_d','fund2_ret_d','fund3_ret_d']]+1


stock_data1=stock_data[['fund1_ret_d+1','fund2_ret_d+1','fund3_ret_d+1','Year','Month']].groupby(['Year','Month']).prod()
stock_data1[['fund1_ret_m','fund2_ret_m','fund3_ret_m']]=stock_data1[['fund1_ret_d+1','fund2_ret_d+1','fund3_ret_d+1']]-1
stock_data2=stock_data1[['fund1_ret_m','fund2_ret_m','fund3_ret_m']]

print(stock_data2)

returns=stock_data2

weight=[0.33,0.33,0.33]  

# Multiply each fund's return with its weight
weighted_returns = returns.multiply(weight)

# Calculate the weighted sum of returns for each year-month
portfolio_returns = weighted_returns.sum(axis=1)

# Print or use the resulting Series as needed
print('\n', portfolio_returns)


#find the average portfolio monthly return
def pret(weight):
      pret1=returns.multiply(weight).sum(axis=1)
      pret1_mean=pret1.mean() #find the average portfolio return
      return pret1_mean



#find the volatililty of portfolio returns
def pvol(weight):
      pret1=returns.multiply(weight).sum(axis=1)           
      pret1_vol=pret1.std()*np.sqrt(12)#annualize volatlitiy based on monthly returns
      return pret1_vol
  
############ Question 3
target_vol = 0.35  #
no_fund = 3  
weight=[0.1,0.1,0.1]  

returns=stock_data2    

def search_weight(weight, returns, target_vol, no_fund):
    
    def pret(weight):
        pret1 = returns.multiply(weight).sum(axis=1)
        pret1_mean_flip = -pret1.mean()
        return pret1_mean_flip

    def pvol(weight, target_vol):
        pret1 = returns.multiply(weight).sum(axis=1)
        pret1_vol = pret1.std() * np.sqrt(12) - target_vol
        return pret1_vol

    def sumweight(weight):
        return weight.sum() - 1

    solve1 = minimize(pret, weight,
                  constraints=[{"fun": pvol, "type": "eq", 'args': (target_vol,)},
                               {"fun": sumweight, "type": "eq"}],
                  bounds=[(0, 1)] * no_fund)
                      
    weight_select = solve1.x
    portfolio_ret = -solve1.fun * 12
    success = solve1.success

    return portfolio_ret, weight_select, success

portfolio_ret, weight_select, success = search_weight(weight, returns, target_vol, no_fund)

print("Success:", success)
print("Portfolio Return:", portfolio_ret)
print("Selected Weights:", weight_select)

    
"""Simulate returns using random numbers from normal distribution"""

plt.plot(stock_data['fund1_ret_d'])

plt.hist(stock_data['fund1_ret_d'], bins = 1000, range=(-0.1,0.1))

stock_data['fund1_ret_d'].mean()

stock_data['fund1_ret_d'].std()

pret_sim1=np.random.normal(0.000456,0.01195,size=(10000))

pret_sim1.mean()

pret_sim1.std()


plt.hist(pret_sim1, bins = 1000, range=(-0.1,0.1))

#simulate monthly portfolio returns

portfolio_ret=0.1085

target_vol=0.15

no_month=5

no_simulation=100

#simulate portfolio returns
pret_sim1=np.random.normal(portfolio_ret/12,target_vol/np.sqrt(12),size=(no_month,no_simulation))

annualfee=0.0035

pret_sim2=pret_sim1-annualfee/12




    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
