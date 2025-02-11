
"""Investment Management and Machine Learning

Week 6

Author: Wei Jiao

"""

import numpy as np
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = [20, 15] 


"""
US market: iShares Core S&P 500 ETF (IVV); iShares Russell 2000 ETF (IWM);

International market: iShares MSCI EAFE ETF (EFA); ishares emerging markets (EEM)

Bond market: iShares Core U.S. Aggregate Bond ETF (AGG)

(EEM and AGG started in 2003)
"""

"""
If you find the pandas_datareader is not working, you could use the following yfinance library

"""

#Download price information 


#install yfinance library
#after you run the above pip install yfinance, if you still find the yfinance is not working, please run the following line of code
pip install yfinance --upgrade





import yfinance as yf

from datetime import datetime

start_date = datetime(2004, 1, 1) #set start date

end_date = datetime(2021, 6, 30) #set end date

etf1 = yf.download(['IVV','IWM','EFA','EEM','AGG'], start = start_date, end = end_date)['Adj Close']


etf1.reset_index(inplace=True)

etf1['Year']=etf1['Date'].dt.year

etf1['Month']=etf1['Date'].dt.month


etf1.rename(columns={'IVV':'fund1','IWM':'fund2','EFA':'fund3','EEM':'fund4','AGG':'fund5'},inplace=True)
#rename funds, so we could use the code for other ETFs and funds

etf1.sort_values(by=['Date'], inplace=True)


etf1[['fund1_ret_d','fund2_ret_d','fund3_ret_d','fund4_ret_d','fund5_ret_d']]=etf1[['fund1','fund2','fund3','fund4','fund5']].pct_change()


#find the monthly return

etf1[['fund1_ret_d+1','fund2_ret_d+1','fund3_ret_d+1','fund4_ret_d+1','fund5_ret_d+1']]=etf1[['fund1_ret_d','fund2_ret_d','fund3_ret_d','fund4_ret_d','fund5_ret_d']]+1


etf2=etf1[['fund1_ret_d+1','fund2_ret_d+1','fund3_ret_d+1','fund4_ret_d+1','fund5_ret_d+1','Year','Month']].groupby(['Year','Month']).prod()


etf2[['fund1_ret_m','fund2_ret_m','fund3_ret_m','fund4_ret_m','fund5_ret_m']]=etf2[['fund1_ret_d+1','fund2_ret_d+1','fund3_ret_d+1','fund4_ret_d+1','fund5_ret_d+1']]-1

etf3=etf2[['fund1_ret_m','fund2_ret_m','fund3_ret_m','fund4_ret_m','fund5_ret_m']]

print(etf3)


"""Calculate portfolio average return and volatility"""
  
returns=etf3

weight=[0.2,0.2,0.2,0.2,0.2]   


returns.multiply(weight) #each year-month, multiply each fund's return with its weight


returns.multiply(weight).sum(axis=1) #each year-month, multiply each fund's return with its weight and sum the weighted returns to get portfoio returns



#find the average portfolio monthly return
def pret(weight):
           
      pret1=returns.multiply(weight).sum(axis=1)
     
      pret1_mean=pret1.mean() #find the average portfolio return
      return pret1_mean

 

pret(weight)


  
#find the volatililty of portfolio returns
def pvol(weight):
      pret1=returns.multiply(weight).sum(axis=1)
            
      pret1_vol=pret1.std()*np.sqrt(12)#annualize volatlitiy based on monthly returns
      return pret1_vol
  
    
  
pvol(weight)    
  



"""Search for the optimal asset allocation"""

#For a given level of volatility, we search for the weights that maximize the portfolio average returns


"""Examples"""
from scipy.optimize import minimize


def fun1(x):
    return x**2+x+1

fun1(0.3)


x=[0] #give an initial value of x

solve1=minimize(fun1, x) #use minimize to search for the value of x minimzing the value of fun1

solve1.success

solve1.x




#with additional constraint
def fun2(x):
    return x**2+x+1

def fun3(x,y):
    return x*y-y**3


x=[0]

y=[0.5]


solve1=minimize(fun2,x,
                
                constraints=({"fun": fun3, "type": "eq", 'args':y}),
                
                bounds=[(-1,1)])

#constraints=({"fun": fun3, "type": "eq", 'args':y}): require fun3=0. additional argument is y
#bounds: set boundary for x


solve1.success

solve1.x  


fun3(0.25,0.5)














"""Search weight function"""


#set values for parameters
target_vol=[0.1]
#target_vol is the target annualized portfolio volatility

no_fund=5
#no_fund indicates the number of ETFs in the portfolio

#give initial values for weights
weight=[0.2,0.2,0.2,0.2,0.2]  

returns=etf3



def search_weight(weight, returns, target_vol, no_fund):
    
  def pret(weight):
      pret1=returns.multiply(weight).sum(axis=1)
      #find the average portfolio return
      #we use the minimize function, so we mutiply portfolio average return with negative one
      pret1_mean_flip=-pret1.mean()
      return pret1_mean_flip

  def pvol(weight,target_vol):
      pret1=returns.multiply(weight).sum(axis=1)
      #find the volatililty of portfolio returns equals to the target_vol
      pret1_vol=pret1.std()*np.sqrt(12)-target_vol
      return pret1_vol
  
  # For any portfolio, the sum of all weights should be 1
  #a function to make sure the sum of weights =1  
  def sumweight(weight):
      return weight.sum()-1
  
  #use scipy library-minimize to search for weights 

  solve1=minimize(pret, weight,
                  constraints=({"fun": pvol, "type": "eq", 'args': target_vol},{"fun": sumweight, "type": "eq"}),
                  bounds=[(0,1)]*no_fund)
  #the weight of each of the give ETFs should be between 0 and 1; bounds=[(0,1)]*no_fund
  
  
  #report selected weights
  weight_select=solve1.x
  #report annualized average portfolio return based on selected weights
  portfolio_ret=-solve1.fun*12
  #report whether the search is successful
  success=solve1.success

  return portfolio_ret, weight_select, success;




#Get the returning values from the function
portfolio_ret,weight_select,success=search_weight(weight,returns,target_vol,no_fund)

success

portfolio_ret

weight_select












"""Simulate returns using random numbers from normal distribution"""


plt.plot(etf1['fund1_ret_d'])


plt.hist(etf1['fund1_ret_d'], bins = 1000, range=(-0.1,0.1))

etf1['fund1_ret_d'].mean()

etf1['fund1_ret_d'].std()



pret_sim1=np.random.normal(0.000456,0.01195,size=(10000))

pret_sim1.mean()

pret_sim1.std()


plt.hist(pret_sim1, bins = 1000, range=(-0.1,0.1))




#simulate monthly portfolio returns

portfolio_ret=0.1085

target_vol=0.15

no_month=5

no_simulation=100

#simulate portfolio returns
pret_sim1=np.random.normal(portfolio_ret/12,target_vol/np.sqrt(12),size=(no_month,no_simulation))

annualfee=0.0035

pret_sim2=pret_sim1-annualfee/12






"""Simulate account balance"""

#account balance at the end of the first month
value=0

monthlypayment=500

value=(value+monthlypayment)*(1+pret_sim2[0,:])

balance=[]

balance.append(value)

balance1=pd.DataFrame(balance)

        

#account balance at the end of each month

value=0
balance=[]
no_month=50
    
for i in range (no_month):
    value=(value+monthlypayment)*(1+pret_sim2[i,:])
    balance.append(value)
    balance1=pd.DataFrame(balance)


    balance1['month_no']=balance1.index+1

    
500*50        
    
    
    
        
        
        
#The function to calculate accountbalance        
def accountbalance(age_current, age_retire, monthlypayment, no_simulation, annualfee, portfolio_ret, target_vol):

    no_month=(age_retire-age_current)*12


#simulate returns using random numbers from normal distribution
    pret_sim1=np.random.normal(portfolio_ret/12,target_vol/np.sqrt(12),size=(no_month,no_simulation))

#take into account advisory fees
    pret_sim2=pret_sim1-annualfee/12

#simulate account balance over time
    
    value=0
    balance=[]
    
    for i in range (no_month):
        value=(value+monthlypayment)*(1+pret_sim2[i,:])
        balance.append(value)

    balance1=pd.DataFrame(balance)

    balance1['month_no']=balance1.index+1
    
    #Reshape the balance1 file
    balance2=pd.melt(balance1, id_vars=['month_no'], var_name='Sim_no', value_name='balance')
    
#We set the median account balance in each month as the balance under normal market condition
#With 50% chance, the account balance is at least this amount
    normal1=balance2[['month_no','balance']].groupby(['month_no']).quantile(0.5)

    normal1['balance_m']=normal1['balance']/1000000
#We set the 10th percentile account balance in each month as the balance under weak market condition
#With 90% chance, the account balance is at least this amount
    weak1=balance2[['month_no','balance']].groupby(['month_no']).quantile(0.1)

    weak1['balance_m']=weak1['balance']/1000000
    
    return normal1,weak1;







#Before we run the accountbalance function, We should set the target_vol and run the search_weight function 
no_fund=5
weight=[0.2,0.2,0.2,0.2,0.2]  
returns=etf3



target_vol=[0.1] #set target volatility
#target_vol is the annualized portfolio volatility


portfolio_ret,weight_select,success=search_weight(weight,returns,target_vol,no_fund)

success

portfolio_ret 
#Run the search_weight function help use obtain the portfolio_ret for a specific target_vol 
#We give the portfolio_ret value to the accountbalance function



#accountbalance(age_current, age_retire, monthlypayment, no_simulation, annualfee, portfolio_ret, target_vol)
normal1, weak1=accountbalance(25, 67, 500, 1000, 0.0035,0.084, 0.1)



plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
#show Y axis lable to the right
plt.plot(normal1['balance_m'],label="normal market")
plt.plot(weak1['balance_m'],label='weak market')
plt.xlabel("No. of months", size=15) 
plt.title("Account Balance ($million)",size=36) 
plt.xticks(size=22)
plt.yticks(size=22)
plt.legend(fontsize=22)







