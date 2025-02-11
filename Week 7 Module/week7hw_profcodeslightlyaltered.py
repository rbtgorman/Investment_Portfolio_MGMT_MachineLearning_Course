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
#Removed 'Bull_ave' and 'Bull_Bear'

var_remove = ['Bull_ave', 'Bull_Bear']
sample2 = sample1.drop(var_remove, axis=1)
sample2['Year']=sample2['datadate'].dt.year

sample2['Month']=sample2['datadate'].dt.month


#set gvkey and datadate as the index
sample2=sample2.set_index(['gvkey','datadate'])



#PROBLEM 2 - Split  new dataset into training and testing samples
#Changed years to 2018 to help lighten the data load

Train1=sample2[sample2['Year']<2018] #feel free to use another year to split the sample. 

Test1=sample2[sample2['Year']>=2018]

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



"""Linear regression"""

from sklearn.linear_model import LinearRegression

LR_m=LinearRegression() #define the model

LR_m.fit(X_train,Y_train) #train the model and get coefficients on training sample


LR_m.coef_

coefficients_LR=pd.DataFrame(LR_m.coef_).T #save all the coefficients in a dataframe

var_lasso=X_test.columns.tolist() #get the independent variable names

coefficients_LR.index=var_lasso

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
#0.0986 per month


#Sharpe Ratio
Ret_rf=stock_long5[['ret_rf']]

SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)

SR
#1.648



#PROBLEM 3 - Use LassoCV to run lasso regression to predict stock monthly returns
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

tsplit = TimeSeriesSplit(n_splits=5,test_size=10000, gap=5000)

alphas = np.logspace(-4, 4, 100)
lasso_cv = LassoCV(alphas=alphas, cv=tsplit)
lasso_cv.fit(X_train_scaled, Y_train.values.ravel())

Lasso_m = 0.001
lasso_final = Lasso(alpha=Lasso_m)
lasso_final.fit(X_train_scaled, Y_train)

# Get coefficients
coefficients_Lasso = pd.DataFrame(lasso_final.coef_, index=var_lasso, columns=["Coefficient"])
print(coefficients_Lasso)

coef_select=coefficients_Lasso.query("Coefficient != 0")
print (coef_select)


#PROBLEM 4 - Use RidgeCV to run ridge regression to predict stock monthly returns.

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

tsplit=TimeSeriesSplit(n_splits=5,test_size=10000, gap=5000)

alpha_candidate=np.linspace(0.01,0.1,20)

Ridge_m = RidgeCV(alphas=alpha_candidate, cv=tsplit)
Ridge_m.fit(X_train,Y_train)#train the model
Ridge_m.alpha_

print('Optimal Alpha Chosen: ',Ridge_m.alpha_)

coefficients_Ridge=pd.DataFrame(Ridge_m.coef_).T
coefficients_Ridge.index=var_lasso
coef_select_Ridge=coefficients_Lasso.query("Coefficient != 0")


print (coefficients_Ridge)




#PROBLEM 5 - Use ElasticNetCV to run elasticnet regression to predict stock monthly returns

from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

tsplit = TimeSeriesSplit(n_splits=5, test_size=10000, gap=5000)

l1_ratios = np.linspace(0.05, 0.95, 10)

elasticnet_cv = ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas, cv=tsplit, max_iter=10000)
elasticnet_cv.fit(X_train_scaled, Y_train.values.ravel())

selected_l1_ratio = elasticnet_cv.l1_ratio_
print("Optimal l1_ratio Chosen: " , selected_l1_ratio)

elasticnet_final = ElasticNet(alpha=elasticnet_cv.alpha_, l1_ratio=selected_l1_ratio, max_iter=10000)
elasticnet_final.fit(X_train_scaled, Y_train)
coef_select_Elas=coefficients_Lasso.query("Coefficient != 0")


coefficients_elasticnet = pd.DataFrame(elasticnet_final.coef_, index=var_lasso, columns=["Coefficient"])
print(coefficients_elasticnet)



#compare
Y_predict=pd.DataFrame(elasticnet_cv.predict(X_test), columns=['Y_predict']) 

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
#0.0998

#Sharpe Ratio
Ret_rf=stock_long5[['ret_rf']]

SR=(Ret_rf.mean()/Ret_rf.std())*np.sqrt(12)

SR
#1.507














