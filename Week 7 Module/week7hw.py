import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import statsmodels.api as sm
plt.rcParams['figure.figsize'] = [20, 15]

stock1 = pd.read_stata(r"C:\Users\rdg83\OneDrive - Rutgers University\Course Investment Portfolio Management\Week 7 Stuff\finalsample.dta")


stock1.sort_values(by=['datadate'], inplace=True)

#removing 2 variables
var_remove = ['Bull_ave', 'Bull_Bear']
stock2 = stock1.drop(var_remove, axis=1)

stock2['Year']=stock2['datadate'].dt.year
stock2['Month']=stock2['datadate'].dt.month

#set gvkey and datadate as the index
stock2=stock2.set_index(['gvkey','datadate'])

Train1 = stock2[stock2['Year'] < 2018]
Test1 = stock2[stock2['Year'] >= 2018]

X_train = Train1[['lagRet2', 'loglagVOL2', 'loglagPrice2', 'loglagMV2', 'lagShareturnover2', 'lagRet2_sic',
                  'lagRet12', 'loglagVOL12', 'lagShareturnover12', 'lagRet12_std', 'lagRet12_min',
                  'lagRet12_max', 'lagRet12_sic', 'epspiq', 'dvpspq', 'sale', 'BM', 'div_p', 'PE', 'cash',
                  'debt', 'logatq', 'sp500_ret_d', 'nasdaq_ret_d', 'r2000_ret_d', 'dollar_ret_d', 'VIX',
                  'yield_3m', 'yield_10y', 'gdp_growth']]

Y_train=Train1[['ret']]


X_test = Test1[['lagRet2', 'loglagVOL2', 'loglagPrice2', 'loglagMV2', 'lagShareturnover2', 'lagRet2_sic',
                 'lagRet12', 'loglagVOL12', 'lagShareturnover12', 'lagRet12_std', 'lagRet12_min',
                 'lagRet12_max', 'lagRet12_sic', 'epspiq', 'dvpspq', 'sale', 'BM', 'div_p', 'PE', 'cash',
                 'debt', 'logatq', 'sp500_ret_d', 'nasdaq_ret_d', 'r2000_ret_d', 'dollar_ret_d', 'VIX',
                 'yield_3m', 'yield_10y', 'gdp_growth']]

Y_test=Test1[['ret']]



rf1=pd.read_excel(r"C:\Users\rdg83\OneDrive - Rutgers University\Course Investment Portfolio Management\Week 7 Stuff\Treasury bill.xlsx")

rf1['rf']=rf1['DGS3MO']/1200
rf2=rf1[['Date','rf']].dropna()
rf2['Year']=rf2['Date'].dt.year
rf2['Month']=rf2['Date'].dt.month
rf3=rf2[['Year','Month','rf']].groupby(['Year','Month'], as_index=False).mean()


indexret1=pd.read_stata(r"C:\Users\rdg83\OneDrive - Rutgers University\Course Investment Portfolio Management\Week 7 Stuff\Index return-2.dta")



from sklearn.linear_model import LinearRegression

LR_m=LinearRegression() #define the model
LR_m.fit(X_train,Y_train) #train the model and get coefficients on training sample
LR_m.coef_

LR_m = LinearRegression()  # Define the model
LR_m.fit(X_train, Y_train)  # Train the model and get coefficients on the training sample
LR_coefs = LR_m.coef_

# Reshape coefficients into a DataFrame
coefficients_LR = pd.DataFrame({'Variable': X_train.columns, 'Coefficient': LR_coefs.flatten()})
print(coefficients_LR)



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

from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit

#LASSO Regression
Lasso_m = Lasso(alpha=0.001)
Lasso_m.fit(X_train, Y_train.values.ravel())  

coefficients = pd.DataFrame(Lasso_m.coef_)
variable_name = X_train.columns.tolist()

#Column names = index
coefficients.index = variable_name
print(coefficients)

# Using cross-validation to search for alpha value
tsplit = TimeSeriesSplit(n_splits=5, test_size=10000, gap=5000)

Lasso_m_cv = LassoCV(cv=tsplit)  # Define the model
Lasso_m_cv.fit(X_train, Y_train.values.ravel())  # Ensure Y_train is a 1D array

# Display the selected alpha
selected_alpha = Lasso_m_cv.alpha_
print(f"Selected Alpha: {selected_alpha}")

# Get the coefficients on all the selected independent variables
coefficients_Lasso = pd.DataFrame({'Variable': variable_name, 'Coefficient': Lasso_m_cv.coef_})
selected_variables = coefficients_Lasso[coefficients_Lasso['Coefficient'] != 0]

# Display coefficients on selected independent variables
print("Coefficients on Selected Independent Variables:")
print(selected_variables)

coef_select = coefficients_Lasso.query("Coefficient != 0")


#LASSO Regression
from sklearn.linear_model import Lasso
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html


Lasso_m = Lasso(alpha=0.001)
Lasso_m.fit(X_train,Y_train)#train the model

coefficients=pd.DataFrame(Lasso_m.coef_)

coefficients.index=variable_name

#many coefficients set to zero
print (coefficients)



#Using cross-validation to search alpha value

from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit

tsplit=TimeSeriesSplit(n_splits=5,test_size=10000, gap=5000)

Lasso_m = LassoCV(cv=tsplit) #define the model
Lasso_m.fit(X_train,Y_train)#train the model
Lasso_m.alpha_

coefficients_Lasso=pd.DataFrame(Lasso_m.coef_, columns=['coef'])
coefficients_Lasso.index=variable_name
print (coefficients_Lasso)

coef_select=coefficients_Lasso.query("coef!=0")