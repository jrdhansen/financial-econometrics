"""
Code from Thursday, 03/07/2019

BOOTSTRAP
See Sheppard's Bootstrap notes on Canvas
"""


#=== import statements =============
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
import pandas as pd
from scipy import stats



np.random.seed(12345)

# Draw a sample of 
N = 250
x = np.random.normal(loc=10, scale=25, size=N)

x.mean()
x.std()

# Number of bootstrap samples ====== generally, the more samples the better
B = 10000
mean_values = np.empty(B)
std_values = np.empty(B)

for b in range(B):
    sample_x = np.random.choice(a=x, size=N, replace=True)
    mean_values[b] = sample_x.mean()
    std_values[b] = sample_x.std()
    

plt.hist(mean_values, bins=50);
plt.hist(std_values, bins=50);

mean_values.mean()
std_values.mean()

# Standard error of the bootstrap will be
values.std(ddof=1)
# This should be really close to the result from the line above
x.std() / math.sqrt(N)
25 / math.sqrt(N)







#=== BOOTSTRAPPING the AR(1) for PREDICTIVE DENSITY ===========================

# Simulate an AR(1) model
N = 52 * 5  # 5 trading days for 52 weeks in the year
phi = 0.8
y = np.empty(N)
y[0] = 0.0   # first value of the year is 0
# Error terms
u = np.random.normal(size=N, loc=0.0, scale=2.5)

# Simulate the years-worth of data
for t in range(1,N):
    y[t] = phi * y[t-1] + u[t]

ts = pd.Series(y)

ts.plot(grid=True)


# Now estimate, will use bootstrap to generate predictive density
reg = stats.linregress(y[1:], y[:-1])
reg.slope

# Calculate the residuals
intercept = reg.intercept
slope = reg.slope

fitted_pts = intercept + slope*y
resids = y[1:] - (intercept - slope*y[:-1])
# I think the line below is wrong, should be same as the line above, but isn't
# resids = y[1:] - fitted_pts[1:]

plt.hist(resids, bins=50);
pd.Series(resids).plot(grid=True)



# This function will generate a path for the whole year
# WE'RE IID-BOOTSTRAPPING THE RESIDUALS not THE PHI'S
#--------------------------------------------------------------
def gen_path(resids, phi=0.8, y0=0.0, num_obs=260):
    # Initialize empty array for path
    path = np.empty(num_obs)
    # Initial starting value (value for first trading day)
    path[0] = y0
    
    # Draw residuals for creating the path
    u = np.random.choice(resids, size=num_obs, replace=True)
    
    for t in range(1, num_obs):
        path[t] = phi * path[t-1] + u[t]

    return path


N = 52*5
B = 1000
paths = np.empty((B, N))
paths.shape

for b in range(B):
    paths[b] = gen_path(resids, reg.slope)

# paths = gen_path(resids)

# If we up the B value higher, we should get back a normal-looking thing
# Here, we are plotting the values of the last value for each path
plt.hist(paths[:,-1]);





#=== THE MOVING BLOCK BOOTSTRAP: just need the indices of the time series for 
# this algorithm, not the series itself (we just sample indices)


#=== THE CIRCULAR BLOCK BOOTSTRAP: similar to the moving block boostrap, but
# circles back to the front once we get to the end.
#=== In practice, it's hard to pin down this block. If the time series is close
# to non-stationary we want a small block (since the data might change quickly
# instead of being stationary with consecutive values being very similar to
# each other).


#=== THE STATIONARY BOOTSTRAP: instead of having an optimal fixed block size m,
# we just treat this m as random drawn from an exponential bootstrap. The other
# two algorithms are better IF we know the block size. But since we usually
# don't in practice, we use this and it gives better empirical results with
# less initial headache, tuning, analysis, etc.
#===  Many times we set m=10, and smoothing parameter q=1/m
#=== See sheppard notes on bootstrap on canvas for the algorithm

























