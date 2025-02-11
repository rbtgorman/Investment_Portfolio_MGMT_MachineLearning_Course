

"""

Investment Management and Machine Learning

Week 8

Instructor:Wei Jiao

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import statsmodels.api as sm
#Change figure size
plt.rcParams['figure.figsize'] = [25, 15] 


sample1=pd.read_stata("D:/Investment Management and Machine Learning/Week 7/finalsample.dta")


sample1.sort_values(by=['datadate'], inplace=True)


sample2=sample1[sample1['lagPrice2']>=5]#remove penny stocks

sample2['Year']=sample2['datadate'].dt.year

sample2['Month']=sample2['datadate'].dt.month


#set gvkey and datadate as the index
sample2=sample2.set_index(['gvkey','datadate'])


#split training and testing samples
Train1=sample2[sample2['Year']<2016] #feel free to use another year to split the sample. 

Test1=sample2[sample2['Year']>=2016]

X_train=Train1[['lagRet2','loglagVOL2','loglagPrice2', 'loglagMV2','lagShareturnover2','lagRet2_sic',
                'lagRet12','loglagVOL12','lagShareturnover12','lagRet12_std','lagRet12_min',
                'lagRet12_max','lagRet12_sic','epspiq','dvpspq','sale','BM','div_p','PE', 'cash',
                'debt','logatq',
                'sp500_ret_d','nasdaq_ret_d','r2000_ret_d','dollar_ret_d','VIX',
                'yield_3m','yield_10y','gdp_growth','Bull_ave','Bull_Bear']]

Y_train=Train1[['ret']]



X_test=Test1[['lagRet2','loglagVOL2','loglagPrice2', 'loglagMV2','lagShareturnover2','lagRet2_sic',
                'lagRet12','loglagVOL12','lagShareturnover12','lagRet12_std','lagRet12_min',
                'lagRet12_max','lagRet12_sic','epspiq','dvpspq','sale','BM','div_p','PE', 'cash',
                'debt','logatq',
                'sp500_ret_d','nasdaq_ret_d','r2000_ret_d','dollar_ret_d','VIX',
                'yield_3m','yield_10y','gdp_growth','Bull_ave','Bull_Bear']]

Y_test=Test1[['ret']]



rf1=pd.read_excel("D:/Investment Management and Machine Learning/Week 3/treasury bill.xls")

rf1['rf']=rf1['DGS3MO']/1200

rf2=rf1[['Date','rf']].dropna()

rf2['Year']=rf2['Date'].dt.year

rf2['Month']=rf2['Date'].dt.month

rf3=rf2[['Year','Month','rf']].groupby(['Year','Month'], as_index=False).mean()


indexret1=pd.read_stata("D:\Investment Management and Machine Learning\Week 3\index return.dta")




"""Decision tree"""
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor

DTree_m= DecisionTreeRegressor(min_samples_leaf=100)
#min_samples_leaf=N: when there are less that N obs in each of the left and right branches,
# we do not split that leaf further

# Individual stock's returns could be very noisy
#requiring minimum number of obs in each branch helps
#prevent the predicted returns for one branch are based on only very few stocks
#(e.g. 1 or 2 stocks) and thus very noisy.

DTree_m.fit(X_train,Y_train) #train the model

DTree_m.get_depth()#the number of levels

DTree_m.get_n_leaves() #the number of leaves

#plot the decision tress
#max_depth=2; Show only first two levels of the three
tree.plot_tree(DTree_m, max_depth=2, feature_names=X_test.columns.tolist())  
plt.show()



#predict returns based on the trained model
Y_predict=pd.DataFrame(DTree_m.predict(X_test), columns=['Y_predict']) 

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
#0.0841


#Sharpe Ratio

Ret_rf=stock_long5[['ret_rf']]

SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)

SR
#1.032



"""Random Forest"""

from sklearn.ensemble import RandomForestRegressor
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html



RFor_m= RandomForestRegressor(n_estimators=100, min_samples_leaf=100,
                              bootstrap=True,max_samples=0.5,n_jobs=-1)
#n_estimators:The number of trees in the forest.
#bootstrap: whether use a different subsample of training sample to train each tree
#max_samples=0.5: randomly draw 50% of the training sample to train each tree
#n_jobs=-1 means using all CPU processors

RFor_m.fit(X_train,Y_train)


#predict returns based on the trained model

Y_predict=pd.DataFrame(RFor_m.predict(X_test), columns=['Y_predict']) 

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


#Sharpe Ratio


Ret_rf=stock_long5[['ret_rf']]

SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)

SR





"""Extra Tree"""

from sklearn.ensemble import ExtraTreesRegressor
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html



ETree_m= ExtraTreesRegressor(n_estimators=100, min_samples_leaf=100,
                             bootstrap=True,max_samples=0.5,n_jobs=-1)


ETree_m.fit(X_train,Y_train)



#predict returns based on the trained model

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

#Sharpe Ratio


Ret_rf=stock_long5[['ret_rf']]

SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)

SR









"""GradientBoostingRegressor"""

from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingRegressor
#have to run the above two lines to import this algorithm
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html#sklearn.ensemble.HistGradientBoostingRegressor

GBR_m= HistGradientBoostingRegressor(max_iter=100, min_samples_leaf=100, early_stopping='True')
#max_iter: The maximum number of iterations of the boosting process
#early_stopping: If Yes, the algorithm use internal cross-validation to determine max_iter                                   

GBR_m.fit(X_train,Y_train)


#predict returns based on the trained model

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

#Sharpe Ratio


Ret_rf=stock_long5[['ret_rf']]

SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)

SR




"""Decision tree algorithms involve randomness in the computation.

Thus, we would get slightly different results each time we run the same algorithm.

"""




