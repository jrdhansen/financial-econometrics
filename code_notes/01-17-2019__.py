# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:30:40 2019

@author: jrdha
"""

#==== Notes for Tuesday, 01/15/2019



import numpy as np
import matplotlib.pyplot as plt

# Showing how E[eps] = 0 and Var[eps] = sigma**2
eps = np.random.normal(size = 100)
plt.plot(eps, linewidth = 2.5, color = "purple")
plt.show




# Simulate a random walk (this is not a stationary time series).
y= np.zeros(100)
y[0] = 0.25

z = np.random.normal(scale = 50, size = 100)  # z is an error term
for t in range(1,100):
    y[t] = y[t-1] + z[t]
    
plt.plot(y, color = "orange", linewidth = 2)
plt.plot(np.zeros(100), color = "black", linewidth = 2)
plt.show

np.random.normal?








#==============================================================================
#==== TUESDAY, 01/22/2019 =====================================================
#==============================================================================

# simulating correlated data

# 1. Draw z_1 ~ Normal(0,1)
# 2. Draw z_2 ~ Normal(0,1)
# 3. Set epsilon_1 = z_1
# 4. Set epsilon_2 = rho*z_1 + (1 - rho^2)^(1/2) * z_2

import numpy as np
M = 10000
z_1 = np.random.normal(size=M)
z_2 = np.random.normal(size=M)
rho = 0.75  # You can pick how closely you want the two z's to be correlated.
e1 = z_1
e2 = rho * z_1 + np.sqrt(1 - rho**2) * z_2
np.corrcoef(e1,e2)   # this should return values that are very close to the
# rho value that we specified above in the script.























#==============================================================================
#==== THURSDAY, 01/24/2019 ====================================================
#==============================================================================

# Stationarity, AR(p) 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#======= 100 draws of a stationary AR(1) model.=======================
# This will be bouncing around 0 (with constant variance)
M = 1000
#=== phi is our AR coefficient 
phi = -0.75  ## Try a few values: 0.5, 0.8, -0.8 ( to be stationary must have
# values such that abs(phi) < 1.)
# A negative phi value gives an ACF plot that bounces between positive and then
# negative values. So at half of the lags we have positive autocorrelation, and
# half of the lags have negative autocorrelation (run the plot with a negative
# phi value and observe this.))
# Positive phi values gives an ACF plot that only decays
# from positive values down toward 0.
y = np.zeros(M)
y[0] = np.abs(np.random.normal())  # Give the starting value a random draw
a = np.random.normal(size=M)   # Like in Tsay, this "a" is our white noise 

for t in range(1,M):
    y[t] = phi * y[t-1] + a[t]

# this data structure called a Series is for time series
# we can do time series operations on the arry directly.
y = pd.Series(y) 
y.plot(grid=True, color="orange", linewidth=2.5)


pd.Series.autocorr?
g1 = y.autocorr(lag=1)
g1   #=== this value should be very close to phi (this is a pt estimate for phi)

phi * g1
# What's the difference here?  autocorr uses the empirical values, whereas
# (phi * g1) gives the true value since we use the specifically defined value of phi
g2 = y.autocorr(lag=2)
g2  # this should be close to the qty (phi * g1)


#=== We're not getting all of statsmodels, just the spcified tools (acf, pacf)
from statsmodels.tsa.stattools import acf, pacf

y_acf = acf(y, nlags=10)
pd.Series(y_acf).plot(kind="bar", grid=False, color="blue")




























#==============================================================================
#==== THURSDAY, 01/31/2019 ====================================================
#==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate some random data
x1 = np.random.normal(loc = -5, scale = 2, size = 5000)
x1.mean()
t_ratio = x1.mean() / (x1.std(ddof=1)/np.sqrt(5000))
t_ratio

# Time series in pandas
ts = pd.Series(x1)
whos
pd.Series.autocorr(ts)   # -0.00041124851751352993

ts = pd.Series(np.random.normal(loc = -5, scale = 2, size = 5000))
ts.plot()
ts.autocorr(lag=2)
ts.autocorr(lag=1)













#==============================================================================
#==== TUESDAY, 02/05/2019 =====================================================
#==============================================================================

import numpy as np

# These are the "true" parameters
mu = 156.0
sigma = 11.0

# Generate some data from these parameters
samp_size = 1000
x = np.random.normal(loc=mu, scale=sigma, size=samp_size)

# Point estimate for the mean
mean_pt_est = x.mean()
mean_pt_est

# Estimate of the standard error.
std_error = np.std(x, ddof=1)/np.sqrt(samp_size)  # ddof = denom deg of freedom
std_error

# Is the standard deviation of the x data similar to the true stDev?
x_sd = x.std(ddof=1)
print("The true sd is", sigma, ". The sd of the generated data is", x_sd)



# BOOSTRAPPING with time series data, AR(1) model, stationary, phi = 0.7
num_draws = 100
phi = 0.7

# set up the empty array
draws = np.empty(num_draws)
draws[0] = np.random.normal()

# vector of error terms
noises = np.random.normal(loc=0.0, scale=2.5, size=num_draws)

for t in range(1, num_draws):
    draws[t] = phi * draws[t-1] + noises[t]

import pandas as pd
import matplotlib.pyplot as plt

s = pd.Series(draws)
s.shape
s.plot(grid=True)
#s = s.values  # this gives you back the original numpy array


# We want to estimate phi as a frequentist statistician.
# Since the stationarity assumption holds, it makes sense to estimate phi
# using the OLS estimation method.
# x_t = (phi)*(x_{t-1}) + epsilon_t
# y_t = mu + (phi)*(y_{t-1}) + u_t

from sklearn.linear_model import LinearRegression

LinearRegression?
LinearRegression.fit?

# This gives the 1-lagged series
s.shift(periods=1)

Y = draws[1:].reshape(99,1)
X = draws[0:-1].reshape(99,1)
Y.shape
X.shape

reg = LinearRegression().fit(X=draws[:-1].reshape(99,1), y=draws[1:].reshape(99,1))
reg = LinearRegression().fit(X=X, y=Y)

phi_hat = reg.coef_[0]
phi_hat  # Notice that this is fairly close to the true value of 0.7


# Now, let's estimate the epsilon term (distn of the epsilon terms).
# We'll do this using bootstrapping on the estimated residuals:
# eps_t = y - phi_hat*y

# Take x_0 as given, and then use a sample w/replacment series of residuals
# to generate m pseudo time series.
# Then re-estimate the phi_hat parameter for that pseudo time series, and you
# can create a sampling distn (via histogram) of the phi parameter.



















#==============================================================================
#==== TUESDAY, 02/19/2019 =====================================================
#==============================================================================

# BOOTSTRAPPING in Python
#=== VERY SIMILAR TO HW2, PROBLEM1 ============================================

import numpy as np
from scipy import stats

M = 100

x = np.random.uniform(size=M)

u = np.random.normal(size=M)
a = 0.22
b=2.50

y = a + b * x + u

results = stats.linregress(x,y)

whos

ahat = results.intercept
bhat = results.slope

yhat = ahat * bhat * x
resids = y - yhat

B = 25000
vals = np.empty(B)

for b in range(B): 
    z = np.random.choice(resids, size=M, replace=True)
    ysim = ahat + bhat * x + z
    res = stats.linregress(x, ysim)
    vals[b] = res.slope

import matplotlib.pyplot as plt

plt.hist(vals)   # this looks fairly normal
















#==== THURSDAY, 02/28/2019 ====================================================



















































