

"""

Investment Management and Machine Learning

Author: Wei Jiao

Neural network 2 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
plt.rcParams['figure.figsize'] = [25,20] 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

"""
Install optuna library

pip install optuna

https://optuna.org/

"""


"""A simple example"""

import optuna

def objective(trial):
    x = trial.suggest_float('x', -1, 1)
    #trial.suggest_float suggests and gives numbers between -1 and 1
    return x**2
    #the returning value of this objective function is the square of X

search1 = optuna.create_study(direction='minimize')
#we create a search to minimize the value of the objective function
search1.optimize(objective, n_trials=100)
#the search will try 100 potential values of x to minimize the objective function


search1.best_params  
# show the best value found by the search







"""Hyperparameter tuning using optuna"""



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




# define a new training and a validation dataset
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


""" 
Search for epoch

"""

import optuna

from sklearn.metrics import mean_squared_error


def objective(trial):   
    #start the neural network model
    model = Sequential() 
    
    model.add(Dense(10,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(10,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    
    model.fit(X_train_new,Y_train_new,epochs=trial.suggest_int("epochs",10,25),
              batch_size=10000,verbose=0)
    #trial.suggest_int gives integers as candidate values for epochs
    
    score =mean_squared_error(Y_val, model.predict(X_val))
    #the whole objective function returns the mean squared error on the validation dataset
    
    return score

search1 = optuna.create_study(direction='minimize')
#define a search and minimize the objective function (namely the mean squared error)

search1.optimize(objective, n_trials=5)
#executive the search. n_trials indicate how many time we ask optuna to search for the best hyperparameter values


search1.best_params
# show the best hyperparameter values





""" 
Search for epoch and batch_size

"""

def objective(trial):   
    #start the neural network model
    model = Sequential() 
    
    model.add(Dense(10,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(10,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    
    model.fit(X_train_new,Y_train_new,epochs=trial.suggest_int("epochs",10,25),
              batch_size=trial.suggest_int("batch_size",10000,100000),verbose=0)
    #search for the epochs and batch_size
    
    score =mean_squared_error(Y_val, model.predict(X_val))
    #the whole objective function returns the mean squared error on the validation dataset
    
    return score

search1 = optuna.create_study(direction='minimize')
#define a search and minimize the objective function (namely the mean squared error)

search1.optimize(objective, n_trials=5)
#executive the search. n_trials indicate how many time we ask optuna to search for the best hyperparameter values


search1.best_params
# show the best hyperparameter values


optuna.visualization.matplotlib.plot_param_importances(search1)
#visualize the importance of hyperparameters




""" 
Search for epoch, batch_size, kernel_initializer

kernel_initializer is a categorical variable

"""


def objective(trial):   
    #start the neural network model
    model = Sequential() 
    
    model.add(Dense(10,kernel_initializer=trial.suggest_categorical('kernel_initializer', ['uniform', 'normal']),
                    activation='relu'))
    #trial.suggest_categorical gives categorical variable values
    model.add(Dense(10,kernel_initializer=trial.suggest_categorical('kernel_initializer', ['uniform', 'normal']),
                    activation='relu'))
    
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    
    model.fit(X_train_new,Y_train_new,epochs=trial.suggest_int("epochs",10,25),
              batch_size=trial.suggest_int("batch_size",10000,100000),verbose=0)
    #search for the epochs and batch_size
    
    score =mean_squared_error(Y_val, model.predict(X_val))
    #the whole objective function returns the mean squared error on the validation dataset
    
    return score

search1 = optuna.create_study(direction='minimize')
#define a search and minimize the objective function (namely the mean squared error)

search1.optimize(objective, n_trials=5)
#executive the search. n_trials indicate how many time we ask optuna to search for the best hyperparameter values


search1.best_params
# show the best hyperparameter values

optuna.visualization.matplotlib.plot_param_importances(search1)
#visualize the importance of hyperparameters





""" 
Search for epoch, batch_size, and number of hidden layers

"""

def objective(trial):   
    
    num_layers=trial.suggest_int('num_layers', 2, 10)
    #gives values for the number of hidden layers
    
    #start the neural network model
    model = Sequential() 
       
    #generate the hidden layers
    for i in range(num_layers):        
        model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
    #e.g., if num_layers=2, then this for-loop will generate 2 hidden layers with 50 neurons   
        
    
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    
    model.fit(X_train_new,Y_train_new,epochs=trial.suggest_int("epochs",10,25),
              batch_size=trial.suggest_int("batch_size",10000,100000),verbose=0)
    #search for the epochs and batch_size
    
    score =mean_squared_error(Y_val, model.predict(X_val))
    #the whole objective function returns the mean squared error on the validation dataset
    
    return score

search1 = optuna.create_study(direction='minimize')
#define a search and minimize the objective function (namely the mean squared error)

search1.optimize(objective, n_trials=5)
#executive the search. n_trials indicate how many time we ask optuna to search for the best hyperparameter values


search1.best_params
# show the best hyperparameter values


optuna.visualization.matplotlib.plot_param_importances(search1)
#visualize the importance of hyperparameters










""" 
Search for almost all the hyperparameters

"""

def objective(trial):   
    
    num_layers=trial.suggest_int('num_layers', 2, 10)
    #gives values for the number of hidden layers
        
    #start the neural network model
    model = Sequential() 
        
    #generate the hidden layers
    for i in range(num_layers):
        
        num_neuron = trial.suggest_int(f'hidden_layer{i}', 10, 100)
        #give values for the number of neurons in each hidden layer
        
        model.add(Dense(num_neuron, 
                        kernel_initializer=trial.suggest_categorical('kernel_initializer', ['uniform', 'normal']),
                        #we can also search for categorical variable
                        activation='relu'))
                    
        
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    
    model.fit(X_train_new,Y_train_new,epochs=trial.suggest_int("epochs",10,25),
              batch_size=trial.suggest_int("batch_size",10000,100000),verbose=0)
    #search for the epochs and batch_size
    
    score =mean_squared_error(Y_val, model.predict(X_val))
    #the whole objective function returns the mean squared error on the validation dataset
    
    return score

search1 = optuna.create_study(direction='minimize')
#define a search and minimize the objective function (namely the mean squared error)

search1.optimize(objective, n_trials=25)
#executive the search. n_trials indicate how many time we ask optuna to search for the best hyperparameter values



search1.best_params
# show the best hyperparameter values

optuna.visualization.matplotlib.plot_param_importances(search1)
#visualize the importance of hyperparameters







#retrain the model 

def deep_network1():
    model = Sequential() 

    model.add(Dense(31, kernel_initializer='normal', activation='relu'))

    model.add(Dense(47, kernel_initializer='normal', activation='relu'))
    
    model.add(Dense(58, kernel_initializer='normal', activation='relu'))
    
    model.add(Dense(76, kernel_initializer='normal', activation='relu'))
    
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    
    model.add(Dense(99, kernel_initializer='normal', activation='relu'))

    model.add(Dense(98, kernel_initializer='normal', activation='relu'))
    
    model.add(Dense(41, kernel_initializer='normal', activation='relu'))
    
    model.add(Dense(93, kernel_initializer='normal', activation='relu'))
    
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    return model


deep_m=deep_network1()

deep_m.fit(X_train, Y_train, epochs=10, batch_size=49675,verbose=0)



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







""" Save Trained Model"""


deep_m.save("D:/Investment Management and Machine Learning/Week 11/deep_network_m")
#save the trained model


from tensorflow.keras.models import load_model

deep_m_load=load_model("D:/Investment Management and Machine Learning/Week 11/deep_network_m")
#load the trained model



#predict returns based on the trained model

Y_predict=pd.DataFrame(deep_m_load.predict(X_test), columns=['Y_predict']) 

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






















