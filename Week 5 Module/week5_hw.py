"""
Robert Gorman 
Week 6 Homework"""


import pandas as pd
import yfinance as yf
import numpy as np
import statsmodels.api as sm

#stock portfolio
tickers = ['MSGS', 'NKE', 'CHDN', 'DKS', 'UAA']
start_date = '2017-01-01'
end_date = '2023-01-01'

stock_data = pd.concat([yf.download(ticker, start=start_date, end=end_date)['Adj Close'] for ticker in tickers], axis=1, keys=tickers)

pe_ratio = stock_data / stock_data.shift(1)
pe_ratio = pe_ratio.fillna(0.0)  

#ranking stocks based on pe ratio - #1
pe_ratio_rank = pe_ratio.groupby([pe_ratio.index.year, pe_ratio.index.month]).rank(pct=True)

#find the stocks with >= 90th or <=10th percentile
top_portfolio = pe_ratio_rank >= 0.9
bottom_portfolio = pe_ratio_rank <= 0.1

#equal-weighted portfolio returns
port_returns = (stock_data.pct_change() * np.logical_xor(top_portfolio, bottom_portfolio)).sum(axis=1)

sp500_data = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
excess_returns = port_returns - sp500_data.pct_change()

print('\nMonthly Return of Equal-Weight Portfolio')
print('\n',port_returns)

#drop/fill missing values so # of observations are the same
portfolio_returns_clean = port_returns.dropna()
excess_returns_clean = excess_returns.fillna(0)  

#print to confirm 
print (len(portfolio_returns_clean))
print (len(excess_returns_clean))

#regression for average portfolio return
model_portfolio = sm.OLS(portfolio_returns_clean, sm.add_constant(np.ones_like(portfolio_returns_clean)))
results_portfolio = model_portfolio.fit().get_robustcov_results(cov_type= 'HC0')

#regression for average excess portfolio return in excess of sp500 index returns
model_excess = sm.OLS(excess_returns_clean, sm.add_constant(np.ones_like(excess_returns_clean)))
results_excess = model_excess.fit().get_robustcov_results(cov_type= 'HC0')

# Print or analyze the regression results
print("\nRegression for Average Portfolio Return:")
print(results_portfolio.summary())
print("\nRegression  for Average Excess Portfolio Return:")
print(results_excess.summary())


pe_ratio_rank_reset = pe_ratio_rank.stack().reset_index().rename(columns={0: 'ret_top'})
print(pe_ratio_rank_reset.columns)

#convert year/month to date
pe_ratio_rank_reset['Year'] = pe_ratio_rank_reset['Date'].dt.year
pe_ratio_rank_reset['Month'] = pe_ratio_rank_reset['Date'].dt.month

#download factor data
Factor = pd.read_excel(r"C:\Users\rdg83\OneDrive - Rutgers University\Course Investment Portfolio Management\Week 5 Stuff\Factors-1.xlsx")

Factor['Year'] = Factor['Year'].astype(int)
Factor['Month'] = Factor['Month'].astype(int)
#print to see exact column names
print(Factor.columns)

#merge with right column names
prank_top_factor1 = pd.merge(pe_ratio_rank_reset, Factor, left_on=['Year', 'Month'], right_on=['Year', 'Month'])

#excess returns
prank_top_factor1['ret_top_rf'] = prank_top_factor1['ret_top'] - prank_top_factor1['RF']

#add constant
prank_top_factor1 = sm.add_constant(prank_top_factor1)
reg_top_factor1 = sm.OLS(prank_top_factor1['ret_top_rf'],
                         prank_top_factor1[['const', 'MktRF', 'SMB', 'HML', 'MOM', 'RMW', 'CMA']]
                         ).fit().get_robustcov_results(cov_type='HC0')

print("\nRegression for portfolio excess of the risk-free rate on the six factors:")
print('\n',reg_top_factor1.summary())
















