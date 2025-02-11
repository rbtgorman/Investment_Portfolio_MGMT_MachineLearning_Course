import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates

plt.rcParams['figure.figsize'] = [20, 15]
"""
Individual Stock Market Data
Monthly
2001-
"""
st1 = pd.read_stata(r"C:\Users\rdg83\OneDrive - Rutgers University\Course Investment Portfolio Management\Week 3 Stuff\hw3-problem1-data.dta")

st1.columns
st1.head()
"""
gvkey: stock id
iid: issue ID
datadate: month end
tic: ticker
cusip: stock id
conm: firm name
primiss: identify primary issue
cshtrm: trading volumn
curcdm: currency
prccm: closing price
trt1m: monthly return
cshom: shares outstanding
exchg: stock exchange code.
tpci: identify common stocks
fic: identifies the country in which the company is incorporated or legally
registered
sic: industry classification code
"""

st2=st1[st1['fic']=='USA']
st3 = st2.dropna()
st3.head()


st3['share turnover']=st3['cshtrm']/st3['cshom']
#share turnover= trading volumn/ shares outstanding

st3["Year"]=st3["datadate"].dt.year
st3["Month"]=st3["datadate"].dt.month

st4=st3[st3["share turnover"] > 0.5]
pd.set_option("display.max_rows", None)

stock_counts = st4.groupby(['Year', 'Month']).size().reset_index(name='Number of stocks w turnover > 0.5')
print(stock_counts)
