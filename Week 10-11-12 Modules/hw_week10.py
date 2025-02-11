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

"""Build a neural network with one hidden layer and 20 neurons in the hidden layer. Set batch
size=10,000. Feel free to pick the values for other hyperparameters for this shallow network (e.g.,
epochs, kernel_initializer, etc.). Train this neural network and use the trained neural network to
predict returns based on your new testing sample. Report the average return of the portfolio that
consists of the 100 stocks with the highest predicted returns in each year-month. Also, report the
Sharpe ratio of the portfolio """

#pip install tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import statsmodels.api as sm

#build neural network using a function
def network1():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(20, input_dim=X_train.shape[1], kernel_initializer='glorot_uniform', activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

#define the model after
network_m = network1()

#set batch size to 10,000, pick any hyperprams
batch_size = 10000
epochs = 50
network_m.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1)

#predict returns on the test set
Y_predict = pd.DataFrame(network_m.predict(X_test), columns=['Y_predict'])
Y_test1 = pd.DataFrame(Y_test).reset_index()
Comb1 = pd.merge(Y_test1, Y_predict, left_index=True, right_index=True, how='inner')
Comb1['Year'] = Comb1['datadate'].dt.year
Comb1['Month'] = Comb1['datadate'].dt.month

#rank stocks by predicted returns in each year-month
rank1 = Comb1[['Y_predict', 'Year', 'Month']].groupby(['Year', 'Month'], as_index=False).rank(ascending=False)
rank1.rename(columns={'Y_predict': 'Y_predict_rank'}, inplace=True)
stock_long1 = pd.merge(Comb1, rank1, left_index=True, right_index=True)
stock_long2 = stock_long1[stock_long1['Y_predict_rank'] <= 100]

#calculate the real returns on selected stocks
stock_long3 = stock_long2[['ret', 'Year', 'Month']].groupby(['Year', 'Month']).mean()

#merge with risk-free rate and index return
stock_long4 = pd.merge(stock_long3, rf3, left_on=['Year', 'Month'], right_on=['Year', 'Month'], how='left')
stock_long5 = pd.merge(stock_long4, indexret1, left_on=['Year', 'Month'], right_on=['Year', 'Month'], how='left')

#calculate excess returns
stock_long5['ret_rf'] = stock_long5['ret'] - stock_long5['rf']
stock_long5['ret_sp500'] = stock_long5['ret'] - stock_long5['sp500_ret_m']

#perform robust regression analysis
stock_long5 = sm.add_constant(stock_long5)
model_summary = sm.OLS(stock_long5[['ret']], stock_long5[['const']]).fit().get_robustcov_results(cov_type='HC0').summary()
print(model_summary)

#report the average return of the portfolio that consists of the 100 stocks with the highest predicted returns in each year-month
avg_return = stock_long5['ret'].mean()
print(f"The average return of the portfolio is: {avg_return:.4f}")

#report the Sharpe ratio of the portfolio
Ret_rf = stock_long5[['ret_rf']]
SR = (Ret_rf.mean()[0] / Ret_rf.std()[0]) * np.sqrt(12)
print(f"Sharpe ratio of the portfolio: {SR:.4f}")


"""PROBLEM 4"""

"""Build a deep neural network with more than 2 hidden layers. Feel free to pick the number of
hidden layers and the number of neurons in each hidden layer. Also, pick your own values for
other hyperparameters for this deep neural network (e.g., epochs, batch size, kernel_initializer,
etc.). Train this deep neural network and use the trained deep neural network to predict returns
based on your new testing sample. Report the average return of the portfolio that consists of the
100 stocks with the highest predicted returns in each year-month. Also, report the Sharpe ratio of
the portfolio."""

#build a deep neural network with more than 2 hidden layers
#mine has 3 hidden layers (64,32,16)
def deep_network1():
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
    return model

#Train - kept batch size at 10,000 and epochs at 50
deep_m = deep_network1()
deep_m.fit(X_train, Y_train, epochs=50, batch_size=10000, verbose=1)

#predict returns based on the trained model
Y_predict = pd.DataFrame(deep_m.predict(X_test), columns=['Y_predict'])
Y_test1 = pd.DataFrame(Y_test).reset_index()
Comb1 = pd.merge(Y_test1, Y_predict, left_index=True, right_index=True, how='inner')
Comb1['Year'] = Comb1['datadate'].dt.year
Comb1['Month'] = Comb1['datadate'].dt.month

#rank stocks by predicted returns in each year-month
rank1 = Comb1[['Y_predict', 'Year', 'Month']].groupby(['Year', 'Month'], as_index=False).rank(ascending=False)
rank1.rename(columns={'Y_predict': 'Y_predict_rank'}, inplace=True)
stock_long1 = pd.merge(Comb1, rank1, left_index=True, right_index=True)
stock_long2 = stock_long1[stock_long1['Y_predict_rank'] <= 100]

#calculate the real returns on selected stocks
stock_long3 = stock_long2[['ret', 'Year', 'Month']].groupby(['Year', 'Month']).mean()

#merge with risk-free rate and index return
stock_long4 = pd.merge(stock_long3, rf3, left_on=['Year', 'Month'], right_on=['Year', 'Month'], how='left')
stock_long5 = pd.merge(stock_long4, indexret1, left_on=['Year', 'Month'], right_on=['Year', 'Month'], how='left')

#calculate excess returns
stock_long5['ret_rf'] = stock_long5['ret'] - stock_long5['rf']
stock_long5['ret_sp500'] = stock_long5['ret'] - stock_long5['sp500_ret_m']

#report the average return of the portfolio that consists of the100 stocks with the highest predicted returns in each year-mont
avg_return = stock_long5['ret'].mean()
print(f"The average return of the portfolio is: {avg_return:.4f}")

#report the Sharpe ratio of the portfolio.
Ret_rf = stock_long5[['ret_rf']]
SR = (Ret_rf.mean()[0] / Ret_rf.std()[0]) * np.sqrt(12)
print(f"Sharpe ratio of the portfolio: {SR:.4f}")
