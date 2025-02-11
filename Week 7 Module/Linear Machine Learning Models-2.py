

"""

Investment Management and Machine Learning

Week 7

Instructor:Wei Jiao

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data
import matplotlib.dates as dates
import statsmodels.api as sm
plt.rcParams['figure.figsize'] = [20, 15] 




""" Example for multicolinearity problem"""

data1=pd.read_excel("D:/Investment Management and Machine Learning/Week 7/Multicollinearity.xlsx")

data1[['X1','X2']].corr()


x=data1[['X1','X2']]

y=data1[['Y']]

x_cons=sm.add_constant(x)

reg1=sm.OLS(y,x_cons).fit().get_robustcov_results(cov_type='HC0')

reg1.summary()




















"""Prepare the Dataset"""

"""If you are interested, you could study the prepare the dataset part.
Otherwise, you could jump into the next part: Set up samples for machine learning algorithms
"""


#Read monthly stock return, price, and traing volumn information

stock1=pd.read_stata("D:/Investment Management and Machine Learning/Week 3/stock return processed.dta")
#the path for the data should be updated based on where the data are saved in the computer


stock1['sic_2']=stock1['sic'].str[:2] #get two digit SIC code

stock2=stock1[stock1['sic_2']!='']


"""
If you find your computer memory is not enough,
you could use only recent a few years to do the program
"""


#Convert datatype to save memory

stock1.memory_usage(deep=True)

stock2['gvkey']=stock2['gvkey'].astype(float)

stock2['sic_2']=stock2['sic_2'].astype(float)

stock2.memory_usage(deep=True)


#merge with lagged return information

laginfo1=stock2[['gvkey', 'ret','datadate','cshtrm','prccm','mv_million','share_turnover']]


#rename variables
laginfo1.rename(columns={'datadate':'lagdate','ret':'lagRet','cshtrm':'lagVOL','prccm':'lagPrice',
                        'mv_million':'lagMV','share_turnover':'lagShareturnover'},inplace=True)


#first merge by date
datadate1=stock2[['datadate']]


datadate1.drop_duplicates(subset =['datadate'], keep = 'first', inplace = True) 


datadate1['link']=1


lagdate1=laginfo1[['lagdate']]

lagdate1.drop_duplicates(subset =['lagdate'], keep = 'first', inplace = True) 

lagdate1['link']=1



dmerge1=pd.merge(datadate1, lagdate1, left_on=['link'], right_on=['link'], how='inner')

dmerge1['datediff1']=dmerge1['datadate']-dmerge1['lagdate']


#find information two month ago
sret_lag2_1=dmerge1[(dmerge1['datediff1']>pd.Timedelta(days=55))
                    &(dmerge1['datediff1']<pd.Timedelta(days=65))]

sret_lag2_1.drop(columns=['link','datediff1'],inplace=True)

slist1=stock2[['gvkey','datadate','sic_2']]


sret_lag2_2=pd.merge(slist1, sret_lag2_1, left_on=['datadate'], right_on=['datadate'], how='inner')

sret_lag2_3=pd.merge(sret_lag2_2, laginfo1, left_on=['gvkey','lagdate'], right_on=['gvkey','lagdate'], how='inner')


sret_lag2_3.rename(columns={'lagRet':'lagRet2', 'lagVOL':'lagVOL2','lagPrice':'lagPrice2',
                            'lagMV':'lagMV2','lagShareturnover':'lagShareturnover2'},inplace=True)

sic_lag2=sret_lag2_3[['sic_2','lagRet2','datadate']].groupby(['sic_2','datadate'],as_index=False).mean()

sic_lag2.rename(columns={'lagRet2':'lagRet2_sic'},inplace=True)

sret_lag2_4=pd.merge(sret_lag2_3,sic_lag2,left_on=['sic_2','datadate'],right_on=['sic_2','datadate'], how='inner')


#find the information in the previous 12 months skiping the most recent month

sret_lag12_1=dmerge1[(dmerge1['datediff1']>pd.Timedelta(days=55))
                    &(dmerge1['datediff1']<pd.Timedelta(days=380))]


sret_lag12_1.drop(columns=['link','datediff1'],inplace=True)


sret_lag12_2=pd.merge(slist1, sret_lag12_1, left_on=['datadate'], right_on=['datadate'], how='inner')

sret_lag12_3=pd.merge(sret_lag12_2, laginfo1, left_on=['gvkey','lagdate'], right_on=['gvkey','lagdate'], how='inner')

sret_lag12_3[['gvkey','datadate']].value_counts()

#calculate the volatility
sret_lag12_std=sret_lag12_3[['gvkey','datadate','lagRet']].groupby(['gvkey','datadate'],as_index=False).std()

sret_lag12_std.rename(columns={'lagRet':'lagRet12_std'},inplace=True)

#get max and min monthly return
sret_lag12_min=sret_lag12_3[['gvkey','datadate','lagRet']].groupby(['gvkey','datadate'],as_index=False).min()

sret_lag12_min.rename(columns={'lagRet':'lagRet12_min'},inplace=True)

sret_lag12_max=sret_lag12_3[['gvkey','datadate','lagRet']].groupby(['gvkey','datadate'],as_index=False).max()

sret_lag12_max.rename(columns={'lagRet':'lagRet12_max'},inplace=True)


#average return in the previous 12 months
sret_lag12_4=sret_lag12_3.groupby(['gvkey','datadate','sic_2'],as_index=False).mean()


sret_lag12_4.rename(columns={'lagRet':'lagRet12', 'lagVOL':'lagVOL12',
                            'lagShareturnover':'lagShareturnover12'},inplace=True)



sret_lag12_5=sret_lag12_4[['gvkey','datadate','sic_2','lagRet12','lagVOL12','lagShareturnover12']]


#industry average return in the previous 12 months
sic_lag12=sret_lag12_3[['sic_2','lagRet','datadate']].groupby(['sic_2','datadate'],as_index=False).mean()


sic_lag12.rename(columns={'lagRet':'lagRet12_sic'},inplace=True)


#merge all files related to past 12-month info

sret_lag12_6=pd.merge(sret_lag12_5,sret_lag12_std,
                      left_on=['gvkey','datadate'], right_on=['gvkey','datadate'], how='inner')

sret_lag12_7=pd.merge(sret_lag12_6,sret_lag12_min,
                      left_on=['gvkey','datadate'], right_on=['gvkey','datadate'], how='inner')

sret_lag12_8=pd.merge(sret_lag12_7,sret_lag12_max,
                      left_on=['gvkey','datadate'], right_on=['gvkey','datadate'], how='inner')


sret_lag12_9=pd.merge(sret_lag12_8,sic_lag12,
                      left_on=['sic_2','datadate'], right_on=['sic_2','datadate'], how='inner')

sret_lag12_9.drop(columns=['sic_2'], inplace=True)

sret_lag_all_1=pd.merge(sret_lag2_4,sret_lag12_9, left_on=['gvkey','datadate'], right_on=['gvkey','datadate'], how='inner')




#financial statement information

fslist1=stock2[['gvkey','datadate']]


fslist1['lagdatadate']=fslist1['datadate']+ pd.Timedelta(days=-40)+pd.tseries.offsets.BusinessMonthEnd(0)
#find the last business day of the previous month



fs1=pd.read_stata("D:/Investment Management and Machine Learning/Week 7/financial statement.dta")


fs2=fs1[fs1['curncdq']=='USD']

#keep obs with positive assets
fs3=fs2[fs2['atq']>0]


#Drop duplicated obs
fs3.drop_duplicates(subset =['gvkey' ,'datadate'], keep = 'first', inplace = True) 



fs4=fs3[['gvkey', 'datadate','atq','ceqq','cheq','dlttq','epspiq','saleq','dvpspq']]

fs4.rename(columns={'datadate':'date_fs'}, inplace=True)


fs4['gvkey']=fs4['gvkey'].astype(float)


fsmerge1=pd.merge(fslist1, fs4, left_on=['gvkey'], right_on=['gvkey'], how='inner')
#merge based on gvkey

fsmerge1['datediff']=fsmerge1['lagdatadate']-fsmerge1['date_fs']


fsmerge2=fsmerge1[(fsmerge1['datediff']>pd.Timedelta(days=90))
              &(fsmerge1['datediff']<pd.Timedelta(days=200))]



fsmerge2.sort_values(by=['gvkey' ,'datadate','date_fs'], inplace=True)

fsmerge2.drop_duplicates(subset =['gvkey' ,'datadate'], keep = 'last', inplace = True) 



fsmerge2.drop(columns=(['datediff','date_fs']),inplace=True)




finalsample1=pd.merge(sret_lag_all_1,fsmerge2,left_on=['gvkey','datadate'], right_on=['gvkey','datadate'], how='inner')


finalsample1.to_stata("D:/Investment Management and Machine Learning/Week 7/1finalsample1.dta")


#Merge information with daily frequency

#index return

import yfinance as yf
from datetime import datetime

start_date = datetime(2000, 1, 1) 
end_date = datetime(2021, 4, 30) 

index_w1 = yf.download(['^GSPC', '^IXIC','^RUT','DX-Y.NYB'], start = start_date, end = end_date)['Adj Close']


#^GSPC:S&P500; IXIC: Nasdaq composite; ^RUT: Rusell 2000; DX-Y.NYB: US Dollar index

index_w1.reset_index(inplace=True)

index_w1.rename(columns={'^GSPC':'sp500','^IXIC':'nasdaq','^RUT':'r2000','DX-Y.NYB':'dollar'},inplace=True)

index_w1.sort_values(by=['Date'], inplace=True)


#find the daily return

index_w1[['sp500_ret_d','nasdaq_ret_d','r2000_ret_d','dollar_ret_d']]=index_w1[['sp500','nasdaq','r2000','dollar']].pct_change()
#pct_change() calculates the percentage change from the immediately previous row.

index_w2=index_w1[['Date','sp500_ret_d','nasdaq_ret_d','r2000_ret_d','dollar_ret_d']]


#Vix index

vix1 = yf.download(['^VIX'], start = start_date, end = end_date)['Adj Close'].to_frame(name='VIX')


vix1.reset_index(inplace=True)

# interest rate

ir1=pd.read_excel("D:/Investment Management and Machine Learning/Week 3/treasury bill.xlsx")

ir1.dropna(inplace=True)

ir1['yield_3m']=ir1['DGS3MO']/1200
ir1['yield_10y']=ir1['DGS10']/1200

ir2=ir1[['Date','yield_3m','yield_10y']]

daily1=pd.merge(index_w2,vix1, left_on=['Date'],right_on=['Date'],how='inner')


daily2=pd.merge(daily1,ir2, left_on=['Date'],right_on=['Date'],how='inner')


lagdatadate1=finalsample1[['lagdatadate']]

lagdatadate1.drop_duplicates(subset =['lagdatadate'], keep = 'first', inplace = True) 

lagdatadate1['link']=1

daily2['link']=1

daily2.rename(columns={'Date':'Date_daily'},inplace=True)

daily3=pd.merge(lagdatadate1,daily2,left_on=['link'], right_on=['link'],how='inner')

daily3['datediff']=daily3['lagdatadate']-daily3['Date_daily']

#find the average value in the previous 30 days
daily4=daily3[(daily3['datediff']>=pd.Timedelta(days=1))
              &(daily3['datediff']<pd.Timedelta(days=31))]

daily5=daily4.groupby(['lagdatadate'], as_index=False).mean()

daily5.drop(columns=['link'],inplace=True)


finalsample2=pd.merge(finalsample1,daily5,left_on=['lagdatadate'], right_on=['lagdatadate'], how='inner')



#merge GDP

gdp1=pd.read_excel("D:/Investment Management and Machine Learning/Week 3/gdp quarterly growth rate.xlsx", sheet_name='real gdp growth rate')

gdp1['gdp_growth']=gdp1['GDP_Growth']/100 #reported values are in percentage. convert them

gdp2=gdp1[['Date','gdp_growth']]

gdp2['link']=1

gdp2.rename(columns={'Date':'Date_gdp'},inplace=True)

gdp3=pd.merge(lagdatadate1,gdp2,left_on=['link'], right_on=['link'],how='inner')

gdp3['datediff_gdp']=gdp3['lagdatadate']-gdp3['Date_gdp']

##The advance estimate of quartelry GDP is released about one month after the quarter end
gdp4=gdp3[(gdp3['datediff_gdp']>=pd.Timedelta(days=25))
              &(gdp3['datediff_gdp']<pd.Timedelta(days=130))]


gdp4.sort_values(by=['lagdatadate','Date_gdp'], inplace=True)

gdp4.drop_duplicates(subset =['lagdatadate'], keep = 'last', inplace = True) 

gdp5=gdp4[['lagdatadate','gdp_growth']]

finalsample3=pd.merge(finalsample2,gdp5,left_on=['lagdatadate'], right_on=['lagdatadate'], how='inner')



#merge with investor sentiment survey, released weekly
#https://www.aaii.com/sentimentsurvey

sent1=pd.read_excel("D:/Investment Management and Machine Learning/Week 7/AAII Sentiment.xlsx", sheet_name='Sheet1')

sent1.dropna(inplace=True)

sent1.rename(columns={'Date':'Date_sent'},inplace=True)
sent1['link']=1

sent2=pd.merge(lagdatadate1,sent1,left_on=['link'], right_on=['link'],how='inner')

sent2['datediff_sent']=sent2['lagdatadate']-sent2['Date_sent']

#merge with the sentiment survey data released at least one day before

sent3=sent2[(sent2['datediff_sent']>=pd.Timedelta(days=1))
              &(sent2['datediff_sent']<pd.Timedelta(days=10))]


sent3.sort_values(by=['lagdatadate','Date_sent'], inplace=True)

sent3.drop_duplicates(subset =['lagdatadate'], keep = 'last', inplace = True) 

sent4=sent3[['lagdatadate','Bull_ave','Bull_Bear']]

finalsample4=pd.merge(finalsample3,sent4,left_on=['lagdatadate'], right_on=['lagdatadate'], how='inner')



#merge with current monthly return

sret1=stock2[['gvkey','datadate','ret']]

finalsample5=pd.merge(finalsample4,sret1, left_on=['gvkey','datadate'], 
                      right_on=['gvkey','datadate'], how='inner')



finalsample5.to_stata("D:/Investment Management and Machine Learning/Week 7/finalsample5.dta")





finalsample5=pd.read_stata("D:/Investment Management and Machine Learning/Week 7/finalsample5.dta")


#scale variables

finalsample5['debt']=finalsample5['dlttq']/finalsample5['atq']

finalsample5['cash']=finalsample5['cheq']/finalsample5['atq']

finalsample5['sale']=finalsample5['saleq']/finalsample5['atq']

finalsample5['BM']=finalsample5['ceqq']/finalsample5['lagMV2'] #book to market ratio

finalsample5['PE']=finalsample5['lagPrice2']/finalsample5['epspiq'] #PE ratio

finalsample5['div_p']=finalsample5['dvpspq']/finalsample5['lagPrice2'] #dividend yield


#winsorize ratio variables to mitigate the impact of outliers
from scipy.stats.mstats import winsorize


plt.hist(finalsample5['BM'], bins = 100, range=(-5,5))

winsorize(finalsample5['BM'],limits=[0.01,0.01], inplace=True)#winsorize at top 1% and bottom 1%

plt.hist(finalsample5['BM'], bins = 100, range=(-5,5))


winsorize(finalsample5['sale'],limits=[0.01,0.01], inplace=True) #winsorize at top 1% and bottom 1%

winsorize(finalsample5['div_p'],limits=[0.01,0.01], inplace=True)

winsorize(finalsample5['PE'],limits=[0.01,0.01], inplace=True)

winsorize(finalsample5['cash'],limits=[0.01,0.01], inplace=True)

winsorize(finalsample5['debt'],limits=[0.01,0.01], inplace=True)

winsorize(finalsample5['lagShareturnover2'],limits=[0.01,0.01], inplace=True)

winsorize(finalsample5['lagShareturnover12'],limits=[0.01,0.01], inplace=True)




#we take natural logrithm for those variable with exponential distribution

plt.hist(finalsample5['lagPrice2'], bins = 100, range=(0,1000))

finalsample5['loglagPrice2']=np.log(finalsample5['lagPrice2'])

plt.hist(finalsample5['loglagPrice2'], bins = 100, range=(-5,20))



finalsample5['loglagVOL2']=np.log(finalsample5['lagVOL2'])

finalsample5['loglagVOL12']=np.log(finalsample5['lagVOL12'])

finalsample5['loglagMV2']=np.log(finalsample5['lagMV2'])

finalsample5['logatq']=np.log(finalsample5['atq'])



# Changing option to use infinite as nan/missing
pd.set_option('mode.use_inf_as_na', True) 


finalsample5.dropna(inplace=True)
#Drop all the obs with any missing variable

finalsample5.to_stata("D:/Investment Management and Machine Learning/Week 7/finalsample.dta")








"""Set up samples for machine learning algorithms"""

sample1 = pd.read_stata(r"C:\Users\rdg83\OneDrive - Rutgers University\Course Investment Portfolio Management\Week 7 Stuff\1finalsample.dta")



sample1.sort_values(by=['datadate'], inplace=True)


#sample2=sample1[sample1['lagPrice2']>=5]#remove penny stocks

var_remove = ['Bull_ave', 'Bull_Bear']
sample2 = stock1.drop(var_remove, axis=1)
sample2['Year']=sample2['datadate'].dt.year

sample2['Month']=sample2['datadate'].dt.month


#set gvkey and datadate as the index
sample2=sample2.set_index(['gvkey','datadate'])



"""

IF you find the sample data is too big to process, you could select part of the data for your exercise.

For example: only use the sample after 2010

newsample=sample2[sample2['Year']>=2010]
"""


#split training and testing samples


Train1=sample2[sample2['Year']<2016] #feel free to use another year to split the sample. 

Test1=sample2[sample2['Year']>=2016]

X_train=Train1[['lagRet2','loglagVOL2','loglagPrice2', 'loglagMV2','lagShareturnover2','lagRet2_sic',
                'lagRet12','loglagVOL12','lagShareturnover12','lagRet12_std','lagRet12_min',
                'lagRet12_max','lagRet12_sic','epspiq','dvpspq','sale','BM','div_p','PE', 'cash',
                'debt','logatq',
                'sp500_ret_d','nasdaq_ret_d','r2000_ret_d','dollar_ret_d','VIX',
                'yield_3m','yield_10y','gdp_growth']]

Y_train=Train1[['ret']]



X_test=Test1[['lagRet2','loglagVOL2','loglagPrice2', 'loglagMV2','lagShareturnover2','lagRet2_sic',
                'lagRet12','loglagVOL12','lagShareturnover12','lagRet12_std','lagRet12_min',
                'lagRet12_max','lagRet12_sic','epspiq','dvpspq','sale','BM','div_p','PE', 'cash',
                'debt','logatq',
                'sp500_ret_d','nasdaq_ret_d','r2000_ret_d','dollar_ret_d','VIX',
                'yield_3m','yield_10y','gdp_growth']]

Y_test=Test1[['ret']]


rf1 = pd.read_excel(r"C:\Users\rdg83\OneDrive - Rutgers University\Course Investment Portfolio Management\Week 7 Stuff\Treasury bill.xlsx")


rf1['rf']=rf1['DGS3MO']/1200

rf2=rf1[['Date','rf']].dropna()

rf2['Year']=rf2['Date'].dt.year

rf2['Month']=rf2['Date'].dt.month

rf3=rf2[['Year','Month','rf']].groupby(['Year','Month'], as_index=False).mean()

indexret1=pd.read_stata(r"C:\Users\rdg83\OneDrive - Rutgers University\Course Investment Portfolio Management\Week 7 Stuff\Index return-2.dta")



"""
Dependent variable:'ret': current monthly return

Independent variable:
'lagRet2': lag return
'loglagVOL2': lag trading volumn
'loglagPrice2': lag share price
'loglagMV2': lag market value
'lagShareturnover2': lag share turnover
'lagRet2_sic': lag industry return
'lagRet12': past 12-month return
'loglagVOL12': past 12-month trading volumn
'lagShareturnover12':past 12-month share turnover
'lagRet12_std': past 12-month return volatility
'lagRet12_min': minimum monthly return in the past 12 months
'lagRet12_max': maximum monthly return in the past 12 months
'lagRet12_sic': past 12-month industry return
'epspiq': lag earnings per share
'dvpspq': lag dividend per share
'sale': lag sale to asset ratio
'BM': lag book to market
'div_p': lag dividend yield
'PE': lag PE ratio
'cash': lag cash to asset ratio
'debt': lag debt to asset ratio
'logatq': lag total assets
'sp500_ret_d': past month sp500 return
'nasdaq_ret_d': past month nasdaq index return
'r2000_ret_d': past month russell 2000 index return
'dollar_ret_d': past month US dollar index
'VIX': past month VIX index
'yield_3m': past month 3-month US treasury yield
'yield_10y' past month 10-year US treasury yield
'gdp_growth': lag quarterly US GDP growth rate
'Bull_ave': lag AAII investor sentiment survery Bullish level
'Bull_Bear': lag AAII investor sentiment survery Bullish-bearish level

"""








"""""""""""""""""""""
Build Machine Learning Models
"""""""""""""""""""""



"""Linear regression"""

from sklearn.linear_model import LinearRegression

LR_m=LinearRegression() #define the model

LR_m.fit(X_train,Y_train) #train the model and get coefficients on training sample


LR_m.coef_

coefficients_LR=pd.DataFrame(LR_m.coef_).T #save all the coefficients in a dataframe

variable_name=X_test.columns.tolist() #get the independent variable names

coefficients_LR.index=variable_name

print (coefficients_LR)


#predict returns based on the trained model
Y_predict=pd.DataFrame(LR_m.predict(X_test), columns=['Y_predict']) 

#merge the predicted returns with corresponding actual returns

Y_test1=pd.DataFrame(Y_test).reset_index()

Comb1=pd.merge(Y_test1, Y_predict, left_index=True,right_index=True,how='inner')


Comb1['Year']=Comb1['datadate'].dt.year

Comb1['Month']=Comb1['datadate'].dt.month


#rank stock based on predicted returns in each year-month    
rank1=Comb1[['Y_predict','Year', 'Month']].groupby(['Year','Month'],as_index=False).rank(ascending=False)
rank1.rename(columns={'Y_predict':'Y_predict_rank'},inplace=True)
    
stock_long1=pd.merge(Comb1,rank1,left_index=True, right_index=True)
    
#select the N stocks with top predicted returns in each year-month     
stock_long2=stock_long1[stock_long1['Y_predict_rank']<=100]

#count the number of stocks selected in each month
stock_long2['datadate'].value_counts()



#calculate the real returns on selected stocks. equal weight
stock_long3=stock_long2[['ret','Year','Month']].groupby(['Year','Month']).mean()

#merge with RF and Index return
stock_long4=pd.merge(stock_long3, rf3, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5=pd.merge(stock_long4, indexret1, left_on=['Year','Month'], right_on=['Year','Month'], how='left')


stock_long5['ret_rf']=stock_long5['ret']-stock_long5['rf']


stock_long5['ret_sp500']=stock_long5['ret']-stock_long5['sp500_ret_m']


stock_long5=sm.add_constant(stock_long5)

sm.OLS(stock_long5[['ret']],stock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()
#0.0168 per month




#Sharpe Ratio
Ret_rf=stock_long5[['ret_rf']]

SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)

SR
#0.738


# Plot monthly returns
plt.bar(stock_long5['Date'],stock_long5['ret'], color='green', width=20, label='Portfolio')
plt.plot(stock_long5['Date'],stock_long5['sp500_ret_m'], color='red',linestyle='dashed', linewidth=2, label='SP500')
plt.xlabel('Date', fontsize=20)
plt.ylabel('Portfolio return', fontsize=20)
plt.title('Portfolio return', fontsize=20)
plt.legend( fontsize=20)
plt.gca().xaxis.set_major_locator(dates.MonthLocator(interval=3))






""" Lasso regression"""
from sklearn.linear_model import Lasso
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html


Lasso_m = Lasso(alpha=0.001)

Lasso_m.fit(X_train,Y_train)#train the model

coefficients=pd.DataFrame(Lasso_m.coef_)

coefficients.index=variable_name

#many coefficients set to zero
print (coefficients)


#Using cross-validation to search alpha value

from sklearn.linear_model import LassoCV
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html


from sklearn.model_selection import TimeSeriesSplit
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
#time series split the training sample
#Use time series split because we always want to use the data in old days as the training
#and the data in recent days to test the model
#gap is the number of samples to exclude from the end of each train set before the test set.
#Using gap, because we do not want to split the sample in the middle of a year-month into training and tesing


tsplit=TimeSeriesSplit(n_splits=5,test_size=10000, gap=5000)
#n_splits=5: we split into 5 subsamples. For each value of alpha, we will train the model 5 times
#there are 5000 obs gap between training and testing sample


Lasso_m = LassoCV(cv=tsplit) #define the model


Lasso_m.fit(X_train,Y_train)#train the model


#show the selected alpha
Lasso_m.alpha_

coefficients_Lasso=pd.DataFrame(Lasso_m.coef_, columns=['coef'])

coefficients_Lasso.index=variable_name

print (coefficients_Lasso)

coef_select=coefficients_Lasso.query("coef!=0")




#predict returns based on the trained model
Y_predict=pd.DataFrame(Lasso_m.predict(X_test), columns=['Y_predict']) 

#merge the predicted returns with corresponding actual returns

Y_test1=pd.DataFrame(Y_test).reset_index()

Comb1=pd.merge(Y_test1, Y_predict, left_index=True,right_index=True,how='inner')


Comb1['Year']=Comb1['datadate'].dt.year

Comb1['Month']=Comb1['datadate'].dt.month


#rank stock based on predicted returns in each year-month    
rank1=Comb1[['Y_predict','Year', 'Month']].groupby(['Year','Month'],as_index=False).rank(ascending=False)
rank1.rename(columns={'Y_predict':'Y_predict_rank'},inplace=True)
    
stock_long1=pd.merge(Comb1,rank1,left_index=True, right_index=True)
    
#select the N stocks with top predicted returns in each year-month     
stock_long2=stock_long1[stock_long1['Y_predict_rank']<=100]

#count the number of stocks selected in each month
stock_long2['datadate'].value_counts()



#calculate the real returns on selected stocks
stock_long3=stock_long2[['ret','Year','Month']].groupby(['Year','Month']).mean()

#merge with RF and Index return
stock_long4=pd.merge(stock_long3, rf3, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5=pd.merge(stock_long4, indexret1, left_on=['Year','Month'], right_on=['Year','Month'], how='left')


stock_long5['ret_rf']=stock_long5['ret']-stock_long5['rf']


stock_long5['ret_sp500']=stock_long5['ret']-stock_long5['sp500_ret_m']


stock_long5=sm.add_constant(stock_long5)


sm.OLS(stock_long5[['ret']],stock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()
#0.0172

#Sharpe Ratio

Ret_rf=stock_long5[['ret_rf']]

SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)

SR
#0.802








"""Ridge regression"""

from sklearn.linear_model import RidgeCV
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html

tsplit=TimeSeriesSplit(n_splits=5,test_size=10000, gap=5000)

#create the candidates of alpha parameter for RidgeCV to select
alpha_candidate=np.linspace(0.01,0.1,20)
#use cross validation to select the best alpha parameter

Ridge_m = RidgeCV(alphas=alpha_candidate, cv=tsplit)

Ridge_m.fit(X_train,Y_train)#train the model



Ridge_m.alpha_



coefficients_Ridge=pd.DataFrame(Ridge_m.coef_).T

coefficients_Ridge.index=variable_name

print (coefficients_Ridge)





#predict returns based on the trained model

Y_predict=pd.DataFrame(Ridge_m.predict(X_test), columns=['Y_predict']) 

#merge the predicted returns with corresponding actual returns

Y_test1=pd.DataFrame(Y_test).reset_index()

Comb1=pd.merge(Y_test1, Y_predict, left_index=True,right_index=True,how='inner')


Comb1['Year']=Comb1['datadate'].dt.year

Comb1['Month']=Comb1['datadate'].dt.month


#rank stock based on predicted returns in each year-month    
rank1=Comb1[['Y_predict','Year', 'Month']].groupby(['Year','Month'],as_index=False).rank(ascending=False)
rank1.rename(columns={'Y_predict':'Y_predict_rank'},inplace=True)
    
stock_long1=pd.merge(Comb1,rank1,left_index=True, right_index=True)
    
#select the N stocks with top predicted returns in each year-month     
stock_long2=stock_long1[stock_long1['Y_predict_rank']<=100]

#count the number of stocks selected in each month
stock_long2['datadate'].value_counts()



#calculate the real returns on selected stocks
stock_long3=stock_long2[['ret','Year','Month']].groupby(['Year','Month']).mean()

#merge with RF and Index return
stock_long4=pd.merge(stock_long3, rf3, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5=pd.merge(stock_long4, indexret1, left_on=['Year','Month'], right_on=['Year','Month'], how='left')


stock_long5['ret_rf']=stock_long5['ret']-stock_long5['rf']


stock_long5['ret_sp500']=stock_long5['ret']-stock_long5['sp500_ret_m']


stock_long5=sm.add_constant(stock_long5)


sm.OLS(stock_long5[['ret']],stock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()
#0.0166

#Sharpe Ratio

Ret_rf=stock_long5[['ret_rf']]

SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)

SR
#0.723











""" ElasticNet regression"""

from sklearn.linear_model import ElasticNetCV
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html

tsplit=TimeSeriesSplit(n_splits=5,test_size=10000, gap=5000)

#l1_ratio between 0 and 1. close to 1 =Lasso
#create the candidates of l1_ratio

l1_ratio_candidate=np.linspace(0.05,0.95,10)

Elastic_m = ElasticNetCV(l1_ratio=l1_ratio_candidate, cv=tsplit)

Elastic_m.fit(X_train,Y_train)


Elastic_m.l1_ratio_


coefficients_Elas=pd.DataFrame(Elastic_m.coef_,columns=['coef'])

coefficients_Elas.index=variable_name

print (coefficients_Elas)


coef_select_Elas=coefficients_Lasso.query("coef!=0")




#predict returns based on the trained model

Y_predict=pd.DataFrame(Elastic_m.predict(X_test), columns=['Y_predict']) 

#merge the predicted returns with corresponding actual returns

Y_test1=pd.DataFrame(Y_test).reset_index()

Comb1=pd.merge(Y_test1, Y_predict, left_index=True,right_index=True,how='inner')


Comb1['Year']=Comb1['datadate'].dt.year

Comb1['Month']=Comb1['datadate'].dt.month


#rank stock based on predicted returns in each year-month    
rank1=Comb1[['Y_predict','Year', 'Month']].groupby(['Year','Month'],as_index=False).rank(ascending=False)
rank1.rename(columns={'Y_predict':'Y_predict_rank'},inplace=True)
    
stock_long1=pd.merge(Comb1,rank1,left_index=True, right_index=True)
    
#select the N stocks with top predicted returns in each year-month     
stock_long2=stock_long1[stock_long1['Y_predict_rank']<=100]

#count the number of stocks selected in each month
stock_long2['datadate'].value_counts()



#calculate the real returns on selected stocks
stock_long3=stock_long2[['ret','Year','Month']].groupby(['Year','Month']).mean()

#merge with RF and Index return
stock_long4=pd.merge(stock_long3, rf3, left_on=['Year','Month'], right_on=['Year','Month'], how='left')
stock_long5=pd.merge(stock_long4, indexret1, left_on=['Year','Month'], right_on=['Year','Month'], how='left')


stock_long5['ret_rf']=stock_long5['ret']-stock_long5['rf']


stock_long5['ret_sp500']=stock_long5['ret']-stock_long5['sp500_ret_m']


stock_long5=sm.add_constant(stock_long5)


sm.OLS(stock_long5[['ret']],stock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()
#0.0173

#Sharpe Ratio


Ret_rf=stock_long5[['ret_rf']]

SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)

SR
#0.8055






