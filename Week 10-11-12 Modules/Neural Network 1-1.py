

"""

Investment Management and Machine Learning

Author: Wei Jiao

Neural network 1

"""
#just introduce tensorflow and neural network


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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





""" Install tensorflow

pip install tensorflow

may need to run this install code twice. and restart python

"""


 
"""Build networks"""


from tensorflow.keras.models import Sequential
#sequential is for starting the neural network
from tensorflow.keras.layers import Dense
#dense is for adding layers


#build the network
def network1():
    model = Sequential() #this is to start the network. And this is the input layer
    model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
    #Add one hidden layer with 10 neurons
    #The initial values for weights are drawn from distributions
    #kernel_initializer is to define the distrbuition
    #here, the initial values for weights are drawn from uniform disttribution
    model.add(Dense(1))
    #this is the output layer. 
    #we try to predict one variable (return), thus we give 1 neuron in the output layer
    model.compile(loss='mean_squared_error', optimizer='Adam')
    #We use mean sqaured error as the loss/objective function to assess models
    #the otpimizer we use Adam
    return model

#A neural network with one hidden layers is called a shallow network 
#optimizer options: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
#kernel initializer options:https://www.tensorflow.org/api_docs/python/tf/keras/initializers
    

network_m=network1()
#define the model
network_m.fit(X_train, Y_train, epochs=10, batch_size=10000,verbose=0)
#Train the model
#verbose=0, no display on detailed training info


#predict returns based on the trained model

Y_predict=pd.DataFrame(network_m.predict(X_test), columns=['Y_predict']) 

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













"""A deep network"""

def deep_network1():
    model = Sequential() 

    model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
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
deep_m.fit(X_train, Y_train, epochs=20, batch_size=10000,verbose=0)


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
















