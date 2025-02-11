import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import statsmodels.api as sm
plt.rcParams['figure.figsize'] = [20, 15] 

sample1 = pd.read_stata(r"C:\Users\rdg83\OneDrive - Rutgers University\Course Investment Portfolio Management\Week 9 Stuff\finalsample.dta")

sample1.sort_values(by=['datadate'], inplace=True)

var_remove = ['PE', 'BM']
sample2 = sample1.drop(var_remove, axis=1)
sample2['Year']=sample2['datadate'].dt.year

sample2['Month']=sample2['datadate'].dt.month

sample2=sample1[sample1['lagPrice2']>=5]#remove penny stocks

sample2['Year']=sample2['datadate'].dt.year

sample2['Month']=sample2['datadate'].dt.month


#set gvkey and datadate as the index
sample2=sample2.set_index(['gvkey','datadate'])


#split training and testing samples

Train1=sample2[sample2['Year']<2019]

Test1=sample2[sample2['Year']>=2019]

X_train=Train1[['lagRet2','loglagVOL2','loglagPrice2', 'loglagMV2','lagShareturnover2','lagRet2_sic',
                'lagRet12','loglagVOL12','lagShareturnover12','lagRet12_std','lagRet12_min',
                'lagRet12_max','lagRet12_sic','epspiq','dvpspq','sale','div_p', 'cash',
                'debt','logatq',
                'sp500_ret_d','nasdaq_ret_d','r2000_ret_d','dollar_ret_d','VIX',
                'yield_3m','yield_10y','gdp_growth','Bull_ave','Bull_Bear']]

Y_train=Train1[['ret']]



X_test=Test1[['lagRet2','loglagVOL2','loglagPrice2', 'loglagMV2','lagShareturnover2','lagRet2_sic',
                'lagRet12','loglagVOL12','lagShareturnover12','lagRet12_std','lagRet12_min',
                'lagRet12_max','lagRet12_sic','epspiq','dvpspq','sale','div_p', 'cash',
                'debt','logatq',
                'sp500_ret_d','nasdaq_ret_d','r2000_ret_d','dollar_ret_d','VIX',
                'yield_3m','yield_10y','gdp_growth','Bull_ave','Bull_Bear']]

Y_test=Test1[['ret']]


Factor = pd.read_excel(r"C:\Users\rdg83\OneDrive - Rutgers University\Course Investment Portfolio Management\Week 5 Stuff\Factors-1.xlsx")


rf1 = pd.read_excel(r"C:\Users\rdg83\OneDrive - Rutgers University\Course Investment Portfolio Management\Week 7 Stuff\Treasury bill.xlsx")

rf1['rf']=rf1['DGS3MO']/1200

rf2=rf1[['Date','rf']].dropna()

rf2['Year']=rf2['Date'].dt.year

rf2['Month']=rf2['Date'].dt.month

rf3=rf2[['Year','Month','rf']].groupby(['Year','Month'], as_index=False).mean()


indexret1=pd.read_stata(r"C:\Users\rdg83\OneDrive - Rutgers University\Course Investment Portfolio Management\Week 9 Stuff\Index return-1.dta")

"""PROBLEM #3"""

"""Set the training data from the year 2010 to the year 2015 as the validation dataset and the data
in years <2010 as the new training set for tuning hyperparameters"""

#define a new training and a validation dataset
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


"""PROBLEM #4"""

"""Build a neural network with three hidden layers and 50 neurons in each hidden layer. Set
kernel_initializer=uniform and activation=relu. Use Optuna to tune and search the values of
epochs and batch_size. Please feel free to choose the number of trials for the search. Report the
values of epochs and batch_size found by the search"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
plt.rcParams['figure.figsize'] = [25,20] 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import optuna
from sklearn.metrics import mean_squared_error

def objective(trial):
    #neural network model
    model = Sequential()
    model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='Adam')

    #epochs and batch_size using Optuna
    epochs = trial.suggest_int("epochs", 10, 100)
    batch_size = trial.suggest_int("batch_size", 32, 512)
    model.fit(X_train_new, Y_train_new, epochs=epochs, batch_size=batch_size, verbose=0)
    score = mean_squared_error(Y_val, model.predict(X_val))
    return score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)
print(f"Best epochs: {study.best_params['epochs']}")
print(f"Best batch_size: {study.best_params['batch_size']}")


"""PROBLEM #5"""

"""Build another neural network and use Optuna to tune and search the values for (1) number of
hidden layers, (2) number of neurons in each hidden layer, (3) kernel_initializer, (4) epochs, and
(5) batch_size. Please feel free to choose the number of trials for the search."""

import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error

def objective(trial):
    #tune # of hidden layers
    num_layers = trial.suggest_int("num_layers", 2, 6)

    #neural network model
    model = Sequential()

    #hidden layers
    for i in range(num_layers):
        #tune # of neurons in each hidden layer
        num_neurons = trial.suggest_int(f"hidden_layer_{i}", 32, 128)
        #tune kernel_initializer
        kernel_initializer = trial.suggest_categorical("kernel_initializer", ["uniform", "normal"])
        model.add(Dense(num_neurons, kernel_initializer=kernel_initializer, activation="relu"))

    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="Adam")

    #epochs and batch_size
    epochs = trial.suggest_int("epochs", 10, 100)
    batch_size = trial.suggest_int("batch_size", 32, 512)
    
    model.fit(X_train_new, Y_train_new, epochs=epochs, batch_size=batch_size, verbose=0)
    score = mean_squared_error(Y_val, model.predict(X_val))
    return score

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)

print(f"Best number of hidden layers: {study.best_params['num_layers']}")
for i in range(study.best_params['num_layers']):
    print(f"Best number of neurons in hidden layer {i}: {study.best_params[f'hidden_layer_{i}']}")
print(f"Best kernel_initializer: {study.best_params['kernel_initializer']}")
print(f"Best epochs: {study.best_params['epochs']}")
print(f"Best batch_size: {study.best_params['batch_size']}")

#neural network using best hyperparameters
best_model = Sequential()

for i in range(study.best_params['num_layers']):
    best_model.add(Dense(study.best_params[f'hidden_layer_{i}'], kernel_initializer=study.best_params['kernel_initializer'], activation="relu"))

best_model.add(Dense(1))
best_model.compile(loss="mean_squared_error", optimizer="Adam")

#train the neural network using best hyperparameters
best_model.fit(X_train_new, Y_train_new, epochs=study.best_params['epochs'], batch_size=study.best_params['batch_size'], verbose=1)


"""Save the trained neural network to your computer.
Load this saved neural network and use it to predict returns based on your testing sample. Report
the average return of the portfolio that consists of the 100 stocks with the highest predicted returns
in each year-month. Also, report the Sharpe ratio of the portfolio."""


import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model


#save trained neural network
best_model.save('best_model.keras')

#load saved neural network
deep_m_load = load_model('best_model.keras')

#predict returns based on the testing sample
Y_predict = pd.DataFrame(deep_m_load.predict(X_test), columns=['Y_predict'])
Y_test1 = pd.DataFrame(Y_test).reset_index()
Comb1 = pd.merge(Y_test1, Y_predict, left_index=True, right_index=True, how='inner')
Comb1['Year'] = Comb1['datadate'].dt.year
Comb1['Month'] = Comb1['datadate'].dt.month

#rank stocks by predicte returns
rank1 = Comb1[['Y_predict', 'Year', 'Month']].groupby(['Year', 'Month'], as_index=False).rank(ascending=False)
rank1.rename(columns={'Y_predict': 'Y_predict_rank'}, inplace=True)
stock_long1 = pd.merge(Comb1, rank1, left_index=True, right_index=True)
stock_long2 = stock_long1[stock_long1['Y_predict_rank'] <= 100]

#calculate  real returns
stock_long3 = stock_long2[['ret', 'Year', 'Month']].groupby(['Year', 'Month']).mean()

#merge with risk-free rate and index return
stock_long4 = pd.merge(stock_long3, rf3, left_on=['Year', 'Month'], right_on=['Year', 'Month'], how='left')
stock_long5 = pd.merge(stock_long4, indexret1, left_on=['Year', 'Month'], right_on=['Year', 'Month'], how='left')

#excess returns
stock_long5['ret_rf'] = stock_long5['ret'] - stock_long5['rf']
stock_long5['ret_sp500'] = stock_long5['ret'] - stock_long5['sp500_ret_m']

#average return of the portfolio
avg_return = stock_long5['ret'].mean()
print(f"The average return of the portfolio is: {avg_return:.4f}")

#Sharpe ratio of the portfolio
Ret_rf = stock_long5[['ret_rf']]
SR = (Ret_rf.mean()[0] / Ret_rf.std()[0]) * np.sqrt(12)
print(f"Sharpe ratio of the portfolio: {SR:.4f}")





























