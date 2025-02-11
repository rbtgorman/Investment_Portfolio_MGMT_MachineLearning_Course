"""

Investment Management and Machine Learning

Week 9

Instructor:Wei Jiao

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import statsmodels.api as sm
plt.rcParams['figure.figsize'] = [20, 15] 


sample1=pd.read_stata("D:/Investment Management and Machine Learning/Week 7/finalsample.dta")

sample1.sort_values(by=['datadate'], inplace=True)

sample2=sample1[sample1['lagPrice2']>=5]#remove penny stocks

sample2['Year']=sample2['datadate'].dt.year

sample2['Month']=sample2['datadate'].dt.month


#set gvkey and datadate as the index
sample2=sample2.set_index(['gvkey','datadate'])


#split training and testing samples

Train1=sample2[sample2['Year']<2016]

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



Factor=pd.read_excel("D:/Investment Management and Machine Learning/Week 5/factors.xlsx")


rf1=pd.read_excel("D:/Investment Management and Machine Learning/Week 3/treasury bill.xls")

rf1['rf']=rf1['DGS3MO']/1200

rf2=rf1[['Date','rf']].dropna()

rf2['Year']=rf2['Date'].dt.year

rf2['Month']=rf2['Date'].dt.month

rf3=rf2[['Year','Month','rf']].groupby(['Year','Month'], as_index=False).mean()


indexret1=pd.read_stata("D:\Investment Management and Machine Learning\Week 3\index return.dta")





"""Improve model through tuning hyperparameters"""


from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit


#Grid Search
from sklearn.model_selection import GridSearchCV


tsplit=TimeSeriesSplit(n_splits=5, test_size=50000, gap=5000)


#specify the candidate hyperparameter settings/values
param_candidate = {'max_iter': [ 50, 100, 200]}
#for max_iter, we try three numbers: 50, 100, and 200


#define the model
model= HistGradientBoostingRegressor(min_samples_leaf=100,  early_stopping='True')
#we should not specify the value for the hyperparamter we are searching
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html


#define the searching
grid = GridSearchCV(estimator=model, param_grid=param_candidate,
                    cv=tsplit,scoring='neg_mean_squared_error')


grid.fit(X_train, Y_train) 
#execute the search


grid.cv_results_
#show the results for each value of the hyperparameter


grid.best_params_
#report the best value of the hyperparameter




#search for multiple hyperparameter
param_candidate = {'max_iter': [ 50, 100], 'min_samples_leaf':[50,100]}

2*2

5*5

5*5*5*5*5


model= HistGradientBoostingRegressor(early_stopping='True')

grid = GridSearchCV(estimator=model, param_grid=param_candidate,
                    cv=tsplit,scoring='neg_mean_squared_error')

grid.fit(X_train, Y_train) 
#execute the search


grid.cv_results_


grid.best_params_
#report the best value of the parameter














#Randomized Search
from sklearn.model_selection import RandomizedSearchCV
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

tsplit=TimeSeriesSplit(n_splits=5, test_size=50000, gap=5000)


from scipy.stats import randint
#randint generates random integers with equal probability


#define the distribution from which we draw random numbers
param_dist = {'max_iter': randint(25,200)}
#draw random integers between 25 and 200 with equal chance

model= HistGradientBoostingRegressor(min_samples_leaf=100, early_stopping='False')
                            
       
rgrid = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=5,
                           cv=tsplit,scoring='neg_mean_squared_error')
#n_iter: Number of hyperparameter settings to try.
#n_iter=5, draw 5 random numbers from the distribution as the candidate values of hyperparameter


rgrid.fit(X_train, Y_train)

rgrid.cv_results_

rgrid.best_params_





"""For Machine learning algorithms with a large number of hyperparameters,
a grid search becomes computationally intractable. 

For random search, the number of attempts/iterations can be chosen independent 
of the number of parameters and possible values.

Randomized search offers more efficient search for multiple hyperparameters.



"""

from scipy.stats import uniform

#Randomized search for multiple hyperparameters
param_dist = {'max_iter': randint(100,170), 'min_samples_leaf': randint(70,120), 'learning_rate': uniform(0.01,0.05)}

#uniform(0.01, 0.2): draw random numbers between 0.01 and 0.2+0.01
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html


model= HistGradientBoostingRegressor( early_stopping='False')
                                                       
#n_iter: Number of parameter settings to try. 

rgrid = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=5,
                           cv=tsplit,scoring='neg_mean_squared_error')
#n_iter: try 5 sets of max_iter-min_sample_leaf-learning_rate

rgrid.fit(X_train, Y_train)


rgrid.cv_results_

rgrid.best_params_






#Retrain the model

model= HistGradientBoostingRegressor(max_iter=139,min_samples_leaf=71, learning_rate=0.0218, early_stopping='False')
                                   

model.fit(X_train,Y_train)


#predict returns based on the trained model

Y_predict=pd.DataFrame(model.predict(X_test), columns=['Y_predict']) 

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






"""
For the candidate values in grid search or the distribution settings in random search

1. start with some initial values to do the search

2. based on the best values from the search, fine tune the candidate values or the distribution settings.
The best values help us narrow down the appropriate range for a hyperparameter. 

3. start a new search using those refined candidate values or distrbution settings

"""




"""
Because of the randomness in the models and the searching process,

you would not get the same best hyperparameter values as I found in the video lectures.


"""

















""" Feature importance"""

model= HistGradientBoostingRegressor(max_iter=100, min_samples_leaf=100, early_stopping='False')
                             

model.fit(X_train,Y_train)


from sklearn.inspection import permutation_importance
#https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html

#need to train the model before using permutation_importance
FIM = permutation_importance(model, X_train, Y_train, n_repeats=5, scoring='neg_mean_squared_error')
#n_repeats: Number of times to randomly re-ordering a feature.
#repeating multiple times, we could get the average improvement in MSE by one feature


FIM_score_mean=pd.DataFrame(FIM.importances_mean, columns=['Feature Importance'])
#get the average change in MSE. 
#The change in MSE for one variable reflects 
#the decrease in MSE if we turn the information of that variable into noise

FIM_score_std=pd.DataFrame(FIM.importances_std, columns=['Feature Importance_std'])
#the standard deviation of changes in MSE for the 5 times we shuffle the data



FIM_score=pd.merge(FIM_score_mean, FIM_score_std, left_index=True,right_index=True)
#merge the mean and the standard deviation

FIM_score['Feature']=X_test.columns.tolist()
#get the feature names


FIM_score.sort_values(by=['Feature Importance'],inplace=True)


FIM_score.plot(kind = "barh",x='Feature', y = 'Feature Importance',  title = "Feature Importance", 
               xerr = 'Feature Importance_std', fontsize=25, color='red')


#scaled graph
#If it is difficult to explain to clients about mean squared errors
#We could scale feature importance
#use the biggest MSE change for one feature as the denominator
FIM_score['benchmark']=FIM_score['Feature Importance'].max()


FIM_score['Feature Importance%']=FIM_score['Feature Importance']/FIM_score['benchmark']
#Feature Importance% shows the importance of a feature relative to the most importance feature
#the most important feature would have a 100% feature importance

FIM_score.plot(kind = "barh",x='Feature', y = 'Feature Importance%',  title = "Feature Importance", 
                fontsize=20, color='red')











"""Partial Dependence Plots"""




model= HistGradientBoostingRegressor(max_iter=100, min_samples_leaf=100, early_stopping='False')
                             

model.fit(X_train,Y_train)


from sklearn.inspection import PartialDependenceDisplay
#https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay.html


PartialDependenceDisplay.from_estimator(model, features=[['VIX']], X=X_train, grid_resolution=1000, 
                                        line_kw={"color": "green",
                                                 "linestyle":"dashed",
                                                 "linewidth":"5"}) 
plt.xlabel('VIX', fontsize=30)
plt.ylabel('RET', fontsize=30)
plt.title('Partial Dependence Plot', fontsize=30, fontstyle='italic', fontname='Times New Roman')








