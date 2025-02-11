import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

start_date = datetime(1990, 1, 1)  # Set start date
end_date = datetime(2023, 12, 31)

index_w1 = yf.download(['AAPL', 'WMT'], start=start_date, end=end_date)['Adj Close']

index_w1.reset_index(inplace=True)

index_w1['Year'] = index_w1['Date'].dt.year
index_w1['Month'] = index_w1['Date'].dt.month

index_w1.rename(columns={'AAPL': 'Apple', 'WMT': 'Walmart'}, inplace=True)

index_w1.sort_values(by=['Date'], inplace=True)

#Find the daily return
index_w1['Apple_ret_d'] = index_w1['Apple'].pct_change()
index_w1['Walmart_ret_d'] = index_w1['Walmart'].pct_change()


#next day's return
index_w1['Apple_ret_d+1'] = index_w1['Apple_ret_d'].pct_change() + 1
index_w1['Walmart_ret_d+1'] = index_w1['Walmart_ret_d'].pct_change() + 1

#Find the monthly return
index_w2 = index_w1[['Apple_ret_d+1', 'Walmart_ret_d+1', 'Year', 'Month']].groupby(['Year', 'Month'], as_index=False).prod()
index_w2[['Apple_ret_y', 'Walmart_ret_y']] = index_w2[['Apple_ret_d+1', 'Walmart_ret_d+1']] - 1


index_w3 = index_w2[['Apple_ret_y', 'Walmart_ret_y', 'Year', 'Month']]
index_w3['Date'] = pd.to_datetime({'year': index_w3['Year'], 'month': index_w3['Month'], 'day': 28})

index_w3.plot(x='Date', y=['Apple_ret_y', 'Walmart_ret_y'], kind='line', linewidth=2, color=['red', 'blue'])
plt.xlabel('Date', fontsize=20)
plt.ylabel('Monthly return', fontsize=20)
plt.title('Index Monthly Return', fontsize=20)
plt.legend(['Apple_ret_m', 'Walmart_ret_m'], fontsize=20)
plt.show()


