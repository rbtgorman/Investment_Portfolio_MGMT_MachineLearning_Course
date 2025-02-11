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


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit

# Load your stock data (stock2) and preprocess it as you've done before

# Create time slices
tscv = TimeSeriesSplit(n_splits=10)  # Adjust the number of splits as needed

# Fit LassoCV model
lasso_cv = LassoCV(cv=tscv)
lasso_cv.fit(X_train, Y_train)

# Get the selected alpha
selected_alpha = lasso_cv.alpha_

# Get the coefficients
coefficients = lasso_cv.coef_

# Report the results
print(f"Selected Alpha: {selected_alpha}")
print("Coefficients:")
print("\n".join(f"{feature}: {coef:.4f}" for feature, coef in zip(X_train.columns, coefficients)))

#Problem 3 - LASSO
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit

# Concatenate training and testing datasets for convenience
X_all = pd.concat([X_train, X_test])
Y_all = pd.concat([Y_train, Y_test])

# Set up time series split
tscv = TimeSeriesSplit(n_splits=5)  # You can adjust the number of splits as needed

# Initialize LassoCV with alphas to test and increase max_iter
alphas = np.logspace(-4, 4, 100)
lasso_cv = LassoCV(alphas=alphas, cv=tscv, max_iter=10000)

# Fit LassoCV model
lasso_cv.fit(X_all, Y_all.values.ravel())

# Display the selected alpha
selected_alpha = lasso_cv.alpha_
print(f"Selected Alpha: {selected_alpha}")

# Get the coefficients on all the selected independent variables
lasso_coefficients = pd.DataFrame({'Variable': X_all.columns, 'Coefficient': lasso_cv.coef_})
selected_variables = lasso_coefficients[lasso_coefficients['Coefficient'] != 0]

# Display coefficients on selected independent variables
print("Coefficients on Selected Independent Variables:")
print(selected_variables)



