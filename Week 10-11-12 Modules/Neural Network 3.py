
"""

Investment Management and Machine Learning

Author: Wei Jiao

Neural network 3

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
plt.rcParams['figure.figsize'] = [20, 15] 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import optuna
from sklearn.metrics import mean_squared_error

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



#model without dropout

def deep_network1():
    model = Sequential() 

    model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
    #Multiple hidden layers
    
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    return model


deep_m=deep_network1()
deep_m.fit(X_train, Y_train, epochs=15, batch_size=50000,verbose=0)



#predict returns based on the trained model

Y_predict=pd.DataFrame(deep_m.predict(X_test), columns=['Y_predict']) 

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
Dropout
"""


from tensorflow.keras.layers import Dropout
#Import the dropout layer

50*(1-0.2)


def deep_network1():
    model = Sequential() 
    model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    #output from 20% of the neurons in the first hidden layer will be discarded 
    
    model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer="Adam")
    return model

deep_m=deep_network1()
deep_m.fit(X_train, Y_train, epochs=15, batch_size=50000,verbose=0)




#predict returns based on the trained model

Y_predict=pd.DataFrame(deep_m.predict(X_test), columns=['Y_predict']) 

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
Weight Regularization
"""
from tensorflow.keras.regularizers import l1, l2, l1_l2
#import the regularizers 



regularizer=l1(1e-8)
#1e-8 is the L1 regularizer ratio

regularizer=l2(1e-8)
#1e-8 is the L2 regularizer ratio

regularizer=l1_l2(l1=1e-9,l2=1e-9)
#a regularizer that applies both L1 and L2 penalties.

#1e-1 means 0.1

1e-1

1e-2


def deep_network1():
    model = Sequential() 

    model.add(Dense(50, kernel_initializer='uniform', activation='relu',kernel_regularizer=regularizer))

    model.add(Dense(50, kernel_initializer='uniform', activation='relu',kernel_regularizer=regularizer))

    model.add(Dense(50, kernel_initializer='uniform', activation='relu',kernel_regularizer=regularizer))

    model.add(Dense(50, kernel_initializer='uniform', activation='relu',kernel_regularizer=regularizer))

    model.add(Dense(50, kernel_initializer='uniform', activation='relu',kernel_regularizer=regularizer)) 

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    return model


deep_m=deep_network1()
deep_m.fit(X_train, Y_train, epochs=15, batch_size=50000,verbose=0)




#predict returns based on the trained model

Y_predict=pd.DataFrame(deep_m.predict(X_test), columns=['Y_predict']) 

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
regularization + dropout
"""

regularizer=l2(1e-8)



def deep_network1():
    model = Sequential() 

    model.add(Dense(50, kernel_initializer='uniform', activation='relu',kernel_regularizer=regularizer))
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu',kernel_regularizer=regularizer))
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu',kernel_regularizer=regularizer))
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu',kernel_regularizer=regularizer))
    model.add(Dropout(0.2)) 
    model.add(Dense(50, kernel_initializer='uniform', activation='relu',kernel_regularizer=regularizer))    
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    return model


deep_m=deep_network1()
deep_m.fit(X_train, Y_train, epochs=15, batch_size=50000,verbose=0)



#predict returns based on the trained model

Y_predict=pd.DataFrame(deep_m.predict(X_test), columns=['Y_predict']) 

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








"""Optuna search for dropout ratio and regularizer ratio"""

Train_new1=sample2[sample2['Year']<2010]

Val_new1=sample2[(sample2['Year']>=2010)&(sample2['Year']<2016)]

X_train_new=Train_new1[['lagRet2','loglagVOL2','loglagPrice2', 'loglagMV2','lagShareturnover2','lagRet2_sic',
                'lagRet12','loglagVOL12','lagShareturnover12','lagRet12_std','lagRet12_min',
                'lagRet12_max','lagRet12_sic','epspiq','dvpspq','sale','BM','div_p','PE', 'cash',
                'debt','logatq',
                'sp500_ret_d','nasdaq_ret_d','r2000_ret_d','dollar_ret_d','VIX',
                'yield_3m','yield_10y','gdp_growth','Bull_ave','Bull_Bear']]

Y_train_new=Train_new1[['ret']]

X_val=Val_new1[['lagRet2','loglagVOL2','loglagPrice2', 'loglagMV2','lagShareturnover2','lagRet2_sic',
                'lagRet12','loglagVOL12','lagShareturnover12','lagRet12_std','lagRet12_min',
                'lagRet12_max','lagRet12_sic','epspiq','dvpspq','sale','BM','div_p','PE', 'cash',
                'debt','logatq',
                'sp500_ret_d','nasdaq_ret_d','r2000_ret_d','dollar_ret_d','VIX',
                'yield_3m','yield_10y','gdp_growth','Bull_ave','Bull_Bear']]

Y_val=Val_new1[['ret']]



def objective(trial):   
    
    model = Sequential()     
    model.add(Dense(50, kernel_initializer='uniform', activation='relu',
                    kernel_regularizer=l2(trial.suggest_float('l2',1e-10,1e-5))))
    
    model.add(Dropout(trial.suggest_float('ratio',0.05, 0.2)))
    
    model.add(Dense(50, kernel_initializer='uniform', activation='relu',
                    kernel_regularizer=l2(trial.suggest_float('l2',1e-10,1e-5))))
    model.add(Dropout(trial.suggest_float('ratio',0.05, 0.2)))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu',
                    kernel_regularizer=l2(trial.suggest_float('l2',1e-10,1e-5))))
    model.add(Dropout(trial.suggest_float('ratio',0.05, 0.2)))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu',
                    kernel_regularizer=l2(trial.suggest_float('l2',1e-10,1e-5))))
    model.add(Dropout(trial.suggest_float('ratio',0.05, 0.2)))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu',
                    kernel_regularizer=l2(trial.suggest_float('l2',1e-10,1e-5))))
    model.add(Dropout(trial.suggest_float('ratio',0.05, 0.2)))
   
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    
    model.fit(X_train_new,Y_train_new,epochs=15, batch_size=50000,verbose=0)
    
    score =mean_squared_error(Y_val, model.predict(X_val))
    return score

search1 = optuna.create_study(direction='minimize')

search1.optimize(objective, n_trials=10)

search1.best_params
# show the best hyperparameter values

optuna.visualization.matplotlib.plot_param_importances(search1)
#visualize the importance of hyperparameters


























regularizer=l2(6.83500808446983e-06)


def deep_network1():
    model = Sequential() 

    model.add(Dense(50, kernel_initializer='uniform', activation='relu',kernel_regularizer=regularizer))
    model.add(Dropout(0.18))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu',kernel_regularizer=regularizer))
    model.add(Dropout(0.18))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu',kernel_regularizer=regularizer))
    model.add(Dropout(0.18))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu',kernel_regularizer=regularizer))
    model.add(Dropout(0.18)) 
    model.add(Dense(50, kernel_initializer='uniform', activation='relu',kernel_regularizer=regularizer))    
    model.add(Dropout(0.18))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    return model



deep_m=deep_network1()
deep_m.fit(X_train, Y_train, epochs=15, batch_size=50000,verbose=0)





#predict returns based on the trained model

Y_predict=pd.DataFrame(deep_m.predict(X_test), columns=['Y_predict']) 

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


























""" Feature importance 

Which features have the biggest impact on predictions of Y?

"""


from sklearn.inspection import permutation_importance
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
    


regularizer=l2(6.83500808446983e-06)


def deep_network1():
    model = Sequential() 

    model.add(Dense(50, kernel_initializer='uniform', activation='relu',kernel_regularizer=regularizer))
    model.add(Dropout(0.18))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu',kernel_regularizer=regularizer))
    model.add(Dropout(0.18))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu',kernel_regularizer=regularizer))
    model.add(Dropout(0.18))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu',kernel_regularizer=regularizer))
    model.add(Dropout(0.18)) 
    model.add(Dense(50, kernel_initializer='uniform', activation='relu',kernel_regularizer=regularizer))    
    model.add(Dropout(0.18))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    return model


model = KerasRegressor(build_fn=deep_network1, epochs=15, batch_size=50000, verbose=0)
#we wrap neural network using the kerasregressor function.
#So we turn neural network function into a model recognized by sklearn library

model.fit(X_train,Y_train)




#need to train the model before using permutation_importance
FIM = permutation_importance(model, X_train, Y_train, n_repeats=3, scoring='neg_mean_squared_error')
#n_repeats: Number of times to randomly re-ordering a feature.
#repeating multiple times, we could get the average improvement in MSE by one feature


FIM_score_mean=pd.DataFrame(FIM.importances_mean, columns=['Feature Importance'])
#get the average change in MSE. 
#The change in MSE for one variable reflects 
#the decrease in MSE if we turn the information of that variable into noise

FIM_score_std=pd.DataFrame(FIM.importances_std, columns=['Feature Importance_std'])
#the standard deviation of changes in MSE for the 3 times we shuffle the data


FIM_score=pd.merge(FIM_score_mean, FIM_score_std, left_index=True,right_index=True)
#merge the mean and the standard deviation
FIM_score['Feature']=X_train.columns.tolist()
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





