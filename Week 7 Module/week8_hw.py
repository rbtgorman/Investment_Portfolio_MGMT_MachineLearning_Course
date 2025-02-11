import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import statsmodels.api as sm
plt.rcParams['figure.figsize'] = [20, 15] 

#LOAD DATA
sample1 = pd.read_stata(r"C:\Users\rdg83\OneDrive - Rutgers University\Course Investment Portfolio Management\Week 7 Stuff\1finalsample.dta")

sample1.sort_values(by=['datadate'], inplace=True)


#PROBLEM 1 -  removing two independent variables from finalsample.dta dataset
#Removing PE and BM variables, printing columns to ensure they were dropped
var_remove = ['PE', 'BM']
sample2 = sample1.drop(var_remove, axis=1)
sample2['Year']=sample2['datadate'].dt.year

sample2['Month']=sample2['datadate'].dt.month


#set gvkey and datadate as the index
sample2=sample2.set_index(['gvkey','datadate'])
print (sample2.columns)


#PROBLEM 2 - Split  new dataset into training and testing samples
#Changed years to 2018 to help lighten the data load

Train1=sample2[sample2['Year']<2018] #feel free to use another year to split the sample. 

Test1=sample2[sample2['Year']>=2018]

X_train=Train1[['lagRet2','loglagVOL2','loglagPrice2', 'loglagMV2','lagShareturnover2','lagRet2_sic',
                'lagRet12','loglagVOL12','lagShareturnover12','lagRet12_std','lagRet12_min',
                'lagRet12_max','lagRet12_sic','epspiq','dvpspq','sale','div_p', 'cash',
                'debt','logatq',
                'sp500_ret_d','nasdaq_ret_d','r2000_ret_d','dollar_ret_d','VIX',
                'yield_3m','yield_10y','gdp_growth']]

Y_train=Train1[['ret']]



X_test=Test1[['lagRet2','loglagVOL2','loglagPrice2', 'loglagMV2','lagShareturnover2','lagRet2_sic',
                'lagRet12','loglagVOL12','lagShareturnover12','lagRet12_std','lagRet12_min',
                'lagRet12_max','lagRet12_sic','epspiq','dvpspq','sale','div_p', 'cash',
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



"""PROBLEM 3 - DecisionTreeRegressor"""
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

DTree_m= DecisionTreeRegressor(min_samples_leaf=50, random_state=21)
DTree_m.fit(X_train,Y_train) 
DTree_m.get_depth()
DTree_m.get_n_leaves() 

#PLOT
tree.plot_tree(DTree_m, max_depth=2, feature_names=X_test.columns.tolist())  
plt.show()


#se the trained model to predict returns based on your new testing sample
Y_predict=pd.DataFrame(DTree_m.predict(X_test), columns=['Y_predict']) 
print (Y_predict)

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
#coef = 0.1633

print('Average return of the portfolio: 0.1633')


#Sharpe Ratio

Ret_rf=stock_long5[['ret_rf']]
SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)
SR
print('Sharpe ratio of the portfolio: 0.8097')


"""PROBLEM 4 - RandomTreeRegressor"""

from sklearn.ensemble import RandomForestRegressor

RFor_m= RandomForestRegressor(n_estimators=50, min_samples_leaf=5,
                              bootstrap=True,max_samples=0.75,n_jobs=-1)

RFor_m.fit(X_train,Y_train)


#Use the trained model to predict returns based on your new testing sample
Y_predict=pd.DataFrame(RFor_m.predict(X_test), columns=['Y_predict']) 
print(Y_predict)

#merge the predicted returns with corresponding actual returns
Y_test1=pd.DataFrame(Y_test).reset_index()

Comb1=pd.merge(Y_test1, Y_predict, left_index=True,right_index=True,how='inner')
Comb1['Year']=Comb1['datadate'].dt.year
Comb1['Month']=Comb1['datadate'].dt.month

#rank stock based on predicted returns in each year-month    
rank1=Comb1[['Y_predict','Year', 'Month']].groupby(['Year','Month'],as_index=False).rank(ascending=False)
rank1.rename(columns={'Y_predict':'Y_predict_rank'},inplace=True)
    
stock_long1=pd.merge(Comb1,rank1,left_index=True, right_index=True)
stock_long2=stock_long1[stock_long1['Y_predict_rank']<=100]
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
#0.0791

print('Average return of the portfolio: 0.0791')

#Sharpe Ratio
Ret_rf=stock_long5[['ret_rf']]
SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)
SR
print('Sharpe ratio of the portfolio: 1.3103')


"""PROBLEM 5 - ExtraTreesRegressor"""

from sklearn.ensemble import ExtraTreesRegressor

ETree_m= ExtraTreesRegressor(n_estimators=50, min_samples_leaf=5,
                             bootstrap=True,max_samples=0.75,n_jobs=-1)

ETree_m.fit(X_train,Y_train)


#Use the trained model to predictreturns based on your new testing sample
Y_predict=pd.DataFrame(ETree_m.predict(X_test), columns=['Y_predict']) 

#merge the predicted returns with corresponding actual returns

Y_test1=pd.DataFrame(Y_test).reset_index()

Comb1=pd.merge(Y_test1, Y_predict, left_index=True,right_index=True,how='inner')
Comb1['Year']=Comb1['datadate'].dt.year
Comb1['Month']=Comb1['datadate'].dt.month


#rank stock based on predicted returns in each year-month    
rank1=Comb1[['Y_predict','Year', 'Month']].groupby(['Year','Month'],as_index=False).rank(ascending=False)
rank1.rename(columns={'Y_predict':'Y_predict_rank'},inplace=True)
    
stock_long1=pd.merge(Comb1,rank1,left_index=True, right_index=True)
    
stock_long2=stock_long1[stock_long1['Y_predict_rank']<=100]
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
#coef = 0.0831
print('Average return of the portfolio: 0.0831')

#Sharpe Ratio
Ret_rf=stock_long5[['ret_rf']]
SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)
SR
print('Sharpe ratio of the portfolio: 1.4119')



"""PROBLEM 6 - HistGradientBoostingRegressor """

from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingRegressor

GBR_m = HistGradientBoostingRegressor(max_iter=75, min_samples_leaf=5, early_stopping=True)
GBR_m.fit(X_train, Y_train)
                                

#Use the trained model to predict returns based on your new testing sample
Y_predict=pd.DataFrame(GBR_m.predict(X_test), columns=['Y_predict']) 

#merge the predicted returns with corresponding actual returns
Y_test1=pd.DataFrame(Y_test).reset_index()

Comb1=pd.merge(Y_test1, Y_predict, left_index=True,right_index=True,how='inner')
Comb1['Year']=Comb1['datadate'].dt.year
Comb1['Month']=Comb1['datadate'].dt.month


#rank stock based on predicted returns in each year-month    
rank1=Comb1[['Y_predict','Year', 'Month']].groupby(['Year','Month'],as_index=False).rank(ascending=False)
rank1.rename(columns={'Y_predict':'Y_predict_rank'},inplace=True)
    
stock_long1=pd.merge(Comb1,rank1,left_index=True, right_index=True)
    
stock_long2=stock_long1[stock_long1['Y_predict_rank']<=100]
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
#coef = 0.1009
print('Average return of the portfolio: 0.0831')

#Sharpe Ratio
Ret_rf=stock_long5[['ret_rf']]
SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)
SR
print('Sharpe ratio of the portfolio: 1.3459')
