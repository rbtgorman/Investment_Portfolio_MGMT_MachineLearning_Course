import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

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

X_train, X_val, Y_train, Y_val = train_test_split(Train1[['lagRet2','loglagVOL2','loglagPrice2', 'loglagMV2','lagShareturnover2','lagRet2_sic',
                'lagRet12','loglagVOL12','lagShareturnover12','lagRet12_std','lagRet12_min',
                'lagRet12_max','lagRet12_sic','epspiq','dvpspq','sale','div_p', 'cash',
                'debt','logatq',
                'sp500_ret_d','nasdaq_ret_d','r2000_ret_d','dollar_ret_d','VIX',
                'yield_3m','yield_10y','gdp_growth','Bull_ave','Bull_Bear']],
                Train1[['ret']], test_size=0.2, random_state=42)

X_test = Test1[['lagRet2','loglagVOL2','loglagPrice2', 'loglagMV2','lagShareturnover2','lagRet2_sic',
                'lagRet12','loglagVOL12','lagShareturnover12','lagRet12_std','lagRet12_min',
                'lagRet12_max','lagRet12_sic','epspiq','dvpspq','sale','div_p', 'cash',
                'debt','logatq',
                'sp500_ret_d','nasdaq_ret_d','r2000_ret_d','dollar_ret_d','VIX',
                'yield_3m','yield_10y','gdp_growth','Bull_ave','Bull_Bear']]
Y_test = Test1[['ret']]

Y_train = Y_train.values.reshape(-1, 1)
Y_val = Y_val.values.reshape(-1, 1)


Factor = pd.read_excel(r"C:\Users\rdg83\OneDrive - Rutgers University\Course Investment Portfolio Management\Week 5 Stuff\Factors-1.xlsx")


rf1 = pd.read_excel(r"C:\Users\rdg83\OneDrive - Rutgers University\Course Investment Portfolio Management\Week 7 Stuff\Treasury bill.xlsx")

rf1['rf']=rf1['DGS3MO']/1200

rf2=rf1[['Date','rf']].dropna()

rf2['Year']=rf2['Date'].dt.year

rf2['Month']=rf2['Date'].dt.month

rf3=rf2[['Year','Month','rf']].groupby(['Year','Month'], as_index=False).mean()


indexret1=pd.read_stata(r"C:\Users\rdg83\OneDrive - Rutgers University\Course Investment Portfolio Management\Week 9 Stuff\Index return-1.dta")

import optuna
import tensorflow as tf

def objective(trial):
    n_neurons_wide = trial.suggest_int('n_neurons_wide', 32, 256, step=32)
    n_neurons_deep = trial.suggest_int('n_neurons_deep', 32, 256, step=32)
    epochs = trial.suggest_int('epochs', 10, 50, step=10)
    batch_size = trial.suggest_int('batch_size', 32, 256, step=32)

    wide_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(n_neurons_wide, activation='relu', kernel_initializer='uniform', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(n_neurons_wide, activation='relu', kernel_initializer='uniform'),
        tf.keras.layers.Dense(30)
    ])

    deep_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(n_neurons_deep, activation='relu', kernel_initializer='uniform', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(n_neurons_deep, activation='relu', kernel_initializer='uniform'),
        tf.keras.layers.Dense(n_neurons_deep, activation='relu', kernel_initializer='uniform'),
        tf.keras.layers.Dense(n_neurons_deep, activation='relu', kernel_initializer='uniform'),
        tf.keras.layers.Dense(n_neurons_deep, activation='relu', kernel_initializer='uniform'),
        tf.keras.layers.Dense(n_neurons_deep, activation='relu', kernel_initializer='uniform'),
        tf.keras.layers.Dense(n_neurons_deep, activation='relu', kernel_initializer='uniform'),
        tf.keras.layers.Dense(n_neurons_deep, activation='relu', kernel_initializer='uniform'),
        tf.keras.layers.Dense(n_neurons_deep, activation='relu', kernel_initializer='uniform'),
        tf.keras.layers.Dense(n_neurons_deep, activation='relu', kernel_initializer='uniform'),
        tf.keras.layers.Dense(30)
    ])

    model = tf.keras.models.Sequential([
        wide_model,
        deep_model
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val))
    test_loss = model.evaluate(X_test, Y_test)
    return test_loss


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

print('Best hyperparameter values: \n')
print(f'n_neurons_wide: {study.best_params["n_neurons_wide"]}')
print(f'n_neurons_deep: {study.best_params["n_neurons_deep"]}')
print(f'epochs: {study.best_params["epochs"]}')
print(f'batch_size: {study.best_params["batch_size"]}')


#PROBLEM 3

n_neurons_wide = study.best_params['n_neurons_wide']
n_neurons_deep = study.best_params['n_neurons_deep']
epochs = study.best_params['epochs']
batch_size = study.best_params['batch_size']

wide_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(n_neurons_wide, activation='relu', kernel_initializer='uniform', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(n_neurons_wide, activation='relu', kernel_initializer='uniform'),
    tf.keras.layers.Dense(30)
])

deep_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(n_neurons_deep, activation='relu', kernel_initializer='uniform', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(n_neurons_deep, activation='relu', kernel_initializer='uniform'),
    tf.keras.layers.Dense(n_neurons_deep, activation='relu', kernel_initializer='uniform'),
    tf.keras.layers.Dense(n_neurons_deep, activation='relu', kernel_initializer='uniform'),
    tf.keras.layers.Dense(n_neurons_deep, activation='relu', kernel_initializer='uniform'),
    tf.keras.layers.Dense(n_neurons_deep, activation='relu', kernel_initializer='uniform'),
    tf.keras.layers.Dense(n_neurons_deep, activation='relu', kernel_initializer='uniform'),
    tf.keras.layers.Dense(n_neurons_deep, activation='relu', kernel_initializer='uniform'),
    tf.keras.layers.Dense(n_neurons_deep, activation='relu', kernel_initializer='uniform'),
    tf.keras.layers.Dense(n_neurons_deep, activation='relu', kernel_initializer='uniform'),
    tf.keras.layers.Dense(30)
])

model = tf.keras.models.Sequential([
    wide_model,
    deep_model
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val))

y_pred = model.predict(X_test)
portfolio_returns = np.mean(np.argsort(y_pred, axis=1)[:, -100:], axis=1)
sharpe_ratio = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8)  # Add a small constant to the denominator

print(f'Average return of the top 100 stocks portfolio: {np.mean(portfolio_returns)}')
print(f'Sharpe ratio of the top 100 stocks portfolio: {sharpe_ratio}')

