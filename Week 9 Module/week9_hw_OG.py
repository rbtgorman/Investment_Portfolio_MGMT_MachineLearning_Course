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
model= HistGradientBoostingRegressor(min_samples_leaf=100,  early_stopping='auto')
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

###########################
"""PROBLEM 3"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint

# Define the parameter distribution
param_dist = {"n_estimators": randint(25, 200),
              "min_samples_leaf": randint(25, 200)}

# Define the model
model = RandomForestRegressor()

# Define the TimeSeriesSplit
tsplit = TimeSeriesSplit(n_splits=5, test_size=50000, gap=5000)

# Create the RandomizedSearchCV object
rgrid = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=5,
                           cv=tsplit, scoring='neg_mean_squared_error')

# Fit the RandomizedSearchCV object to the data
rgrid.fit(X_train, Y_train)

# Print the cross-validation results
print(rgrid.cv_results_)

# Print the best parameters
print(rgrid.best_params_)


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint

# Define the parameter distribution
param_dist = {"n_estimators": randint(25, 200),
              "min_samples_leaf": randint(25, 200)}

# Define the model
model = GradientBoostingRegressor()

# Define the TimeSeriesSplit
tsplit = TimeSeriesSplit(n_splits=5, test_size=50000, gap=5000)

# Create the RandomizedSearchCV object
rgrid = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=5,
                           cv=tsplit, scoring='neg_mean_squared_error')

# Fit the RandomizedSearchCV object to the data
rgrid.fit(X_train, Y_train)

# Print the mean cross-validation scores for each parameter combination
print("Mean cross-validation scores:")
print(rgrid.cv_results_['mean_test_score'])

# Print the best parameters
print("\nBest parameters:")
print(rgrid.best_params_)



#########################################

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint

param_dist = {"n_estimators": randint(25, 200),
              "min_samples_leaf": randint(25, 200)}

model = ExtraTreesRegressor()

tsplit = TimeSeriesSplit(n_splits=5, test_size=50000, gap=5000)

rgrid = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=5,
                           cv=tsplit, scoring='neg_mean_squared_error')

rgrid.fit(X_train, Y_train)
print(rgrid.cv_results_)
print(rgrid.best_params_)


#######################
"""PROBLEM 4"""

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

model = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=50)
model.fit(X_train, Y_train.values.ravel())

result = permutation_importance(model, X_test, Y_test.values.ravel(), n_repeats=10, random_state=42)

sorted_idx = result.importances_mean.argsort()

plt.figure(figsize=(12, 6))
plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=X_test.columns[sorted_idx])
plt.title("Permutation Importances (test set)")
plt.show()

###################################
"""PROBLEM 5"""

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import plot_partial_dependence

model = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=50)
model.fit(X_train, Y_train)

fig, ax = plt.subplots(figsize=(12, 6))
plot_partial_dependence(model, X_train, ['lagRet2'], ax=ax)
plt.show()



















