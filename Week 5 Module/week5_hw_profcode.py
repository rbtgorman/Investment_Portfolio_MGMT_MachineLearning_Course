import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
plt.rcParams['figure.figsize'] = [20, 15] 
import statsmodels.api as sm

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'WMT']
start_date = '2017-01-01'
end_date = '2023-01-01'

# Download stock data
stock_data = yf.download(tickers, start=start_date, end=end_date)

pe_ratio = stock_data / stock_data.shift(1)

# Rank stocks based on P/E ratio for each year-month
pe_ratio_rank = pe_ratio.groupby([pe_ratio.index.year, pe_ratio.index.month]).rank(pct=True)

# Construct portfolios based on percentiles
top_portfolio = pe_ratio_rank >= 0.9
bottom_portfolio = pe_ratio_rank <= 0.1

# Calculate equal-weighted portfolio returns
portfolio_returns = (stock_data.pct_change() * np.logical_xor(top_portfolio, bottom_portfolio)).sum(axis=1)

# Download S&P 500 data
sp500_data = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']

# Calculate excess returns
excess_returns = portfolio_returns - sp500_data.pct_change()

print(portfolio_returns)

""" rank stocks based on a certain variable and construct portfolios """


prank1=pd.read_stata(r"C:\Users\rdg83\OneDrive - Rutgers University\Course Investment Portfolio Management\Week 5 Stuff\stockreturnprocessed.dta")


prank1['var_rank']=prank1['dlttq']/prank1['atq']
#rank based on long-term debt ratio

prank1['var_rank'].describe()


prank1.dropna(subset=['var_rank'], inplace=True)
#remove observations (rows) with var_rank as missing value


prank1a=prank1[['gvkey','datadate','ret','var_rank']]


#remove penny stocks
stockprice1=pd.read_stata(r"C:\Users\rdg83\OneDrive - Rutgers University\Course Investment Portfolio Management\Week 4 Stuff/stockreturnprocessed.dta")

lagprc1=stockprice1[['gvkey', 'datadate','prccm']]

lagprc1.rename(columns={'datadate':'lagdate'}, inplace=True)


#for each gvkey-datadate merge with price of the same stock in all the months
prank1b=pd.merge(prank1a,lagprc1,left_on=['gvkey'], right_on=['gvkey'], how='inner')


prank1b['datediff1']=prank1b['datadate']-prank1b['lagdate']


#Calculating current monthly return assuming we invest at the closing price at the end of the previous month
#But we only know the closing price after the market closes at the end of the previous month.
#Since we use monthly data, we get the closing price two month ago
prank1b=prank1b[(prank1b['datediff1']>pd.Timedelta(days=55))
                    &(prank1b['datediff1']<pd.Timedelta(days=65))]


prank1c=prank1b[prank1b['prccm']>=5]



prank1c['Year']=prank1c['datadate'].dt.year

prank1c['Month']=prank1c['datadate'].dt.month


top1=prank1c[['var_rank','Year','Month']].groupby(['Year','Month'], as_index=False).quantile(0.9)
#find the 90th percentile 

top1.rename(columns={'var_rank':'var_rank_top'},inplace=True)



bot1=prank1c[['var_rank','Year','Month']].groupby(['Year','Month'], as_index=False).quantile(0.1)
#find the 10th percentile

bot1.rename(columns={'var_rank':'var_rank_bot'},inplace=True)



prank2=pd.merge(prank1c,top1,left_on=['Year','Month'], right_on=['Year','Month'],how='inner')



prank3=pd.merge(prank2,bot1,left_on=['Year','Month'], right_on=['Year','Month'],how='inner')



#Find firms with top 10% and bottom 10% 

prank_top1=prank3[prank3['var_rank']>=prank3['var_rank_top']]

prank_top1['datadate'].value_counts()


prank_bot1=prank3[prank3['var_rank']<=prank3['var_rank_bot']]

#Construct the equal-weighted portfolios for top and bot



prank_top2=prank_top1[['Year','Month', 'ret']].groupby(['Year','Month'],as_index=False).mean()

prank_top2.rename(columns={'ret':'ret_top'}, inplace=True)


prank_top2['date']= pd.to_datetime({'year': prank_top2['Year'],'month': prank_top2['Month'],'day':28})



prank_bot2=prank_bot1[['Year','Month', 'ret']].groupby(['Year','Month'],as_index=False).mean()

prank_bot2.rename(columns={'ret':'ret_bot'}, inplace=True)

prank_bot2['date']= pd.to_datetime({'year': prank_bot2['Year'],'month': prank_bot2['Month'],'day':28})




# Average portfolio monthly return is positive. But the monthly returns distribute around zero


prank_top2['ret_top'].mean()


plt.hist(prank_top2['ret_top'], bins = 100, range=(-0.5,0.5), rwidth=0.8, color='green')




#Does the portfolio generates statistically significant returns? Is the average return truly different from zero?

prank_top3=sm.add_constant(prank_top2)

prank_bot3=sm.add_constant(prank_bot2)
#regressing returns on the constant varable returns the statistical significance of the mean of the returns

reg_top1=sm.OLS(prank_top3[['ret_top']],prank_top3[['const']]).fit().get_robustcov_results(cov_type='HC0')

reg_top1.summary()


reg_bot1=sm.OLS(prank_bot3[['ret_bot']],prank_bot3[['const']]).fit().get_robustcov_results(cov_type='HC0')

reg_bot1.summary()



#Does the portfolio outperform the market ?

indexret1=pd.read_stata(r"C:\Users\rdg83\OneDrive - Rutgers University\Course Investment Portfolio Management\Week 5 Stuff\index return-5.dta")


prank_top4=pd.merge(prank_top3,indexret1,left_on=['Year','Month'], right_on=['Year','Month'],how='inner')

prank_top4['ret_top_ex']=prank_top4['ret_top']-prank_top4['sp500_ret_m']


reg_top_ex1=sm.OLS(prank_top4[['ret_top_ex']],prank_top4[['const']]).fit().get_robustcov_results(cov_type='HC0')

reg_top_ex1.summary()



prank_bot4=pd.merge(prank_bot3,indexret1,left_on=['Year','Month'], right_on=['Year','Month'],how='inner')

prank_bot4['ret_bot_ex']=prank_bot4['ret_bot']-prank_bot4['sp500_ret_m']


reg_bot_ex1=sm.OLS(prank_bot4[['ret_bot_ex']],prank_bot4[['const']]).fit().get_robustcov_results(cov_type='HC0')

reg_bot_ex1.summary()



#Could the factors explain the performance of the portfolio?

Factor=pd.read_excel(r"C:\Users\rdg83\OneDrive - Rutgers University\Course Investment Portfolio Management\Week 5 Stuff/Factors-1.xlsx")

prank_top_factor1=pd.merge(prank_top2, Factor, left_on=['Year','Month'], right_on=['Year','Month'])


prank_top_factor1['ret_top_rf']=prank_top_factor1['ret_top']-prank_top_factor1['RF']

prank_top_factor1=sm.add_constant(prank_top_factor1)


reg_top_factor1=sm.OLS(prank_top_factor1['ret_top_rf'],
                         prank_top_factor1[['const','MktRF','SMB','HML','MOM','RMW','CMA']]).fit().get_robustcov_results(cov_type='HC0')

reg_top_factor1.summary()




prank_bot2['ret_bot'].mean()

prank_bot_factor1=pd.merge(prank_bot2, Factor, left_on=['Year','Month'], right_on=['Year','Month'])


prank_bot_factor1['ret_bot_rf']=prank_bot_factor1['ret_bot']-prank_bot_factor1['RF']

prank_bot_factor1=sm.add_constant(prank_bot_factor1)

reg_bot_factor1=sm.OLS(prank_bot_factor1['ret_bot_rf'],
                         prank_bot_factor1[['const','MktRF','SMB','HML','MOM','RMW','CMA']]).fit().get_robustcov_results(cov_type='HC0')

reg_bot_factor1.summary()


