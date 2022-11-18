
"""
TUESDAY, 03/19/2019
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression


# This bootstrap just returns indices, doesn't even need the original time 
# series data.
# Preserves the time series nature of the data, but assumes nothing else about
# the data. Which is GREAT, since most financial data doesn't follow 
# distributional assumptions.
# We're going to be modeling returns (we use returns instead of prices since
# returns have nicer statistical properties than prices.) Spreads (in addition
# to returns) should be amenable to bootstrapping.
def stationaryBootstrap(numObs, numReps, blockSize):
    m = 1.0/blockSize
    u = np.empty(numObs, dtype='int64')
    indices = np.empty((numReps, numObs), dtype='int64')
    
    for b in range(numReps):
        u[0] = np.random.randint(low=0, high=numObs, size=1, dtype='int64')
        v = np.random.uniform(size=numObs)
        
        for t in range(1, numObs):
            if(v[t] < m):
                u[t] = np.random.randint(low=0, high=numObs, size=1, dtype='int64')
            else:
                u[t] = u[t-1] + 1
                if(u[t] >= numObs):
                    u[t] = u[t] = numObs-1
                    
        indices[b] = u
        
    return indices



# This function generates artificial data from an AR(1) process.
def gen_AR1(numObs = 100, theta = 0.5):
    T = numObs
    # Generate the epislon terms vector
    epsilon_t = np.random.normal(size = T-1)
    # Create a vector to store the series {y_t}
    y_t = np.empty(T)
    # Intialize the first value, y_0, outside of the loop
    y_t[0] = 0
    for i in range(T-1):
        y_t[i+1] = theta * y_t[i] + epsilon_t[i]
    return y_t


# WE NEED A STATIONARY TIME SERIES TO BE STATIONARY IN ORDER TO USE THE 
# STATIONARY BOOTSTRAP (duh, but I'll say it anyway).

np.random.seed(seed=123456)
yraw = gen_AR1(numObs=500, theta=0.6)

numObs = yraw.shape[0]
# This number should be very high, 10000+ in order to get good results.
numReps = 10000
# This being 20 is a good number; sometimes requires tuning
blockSize = 10
values = np.empty(numReps)
ii = stationaryBootstrap(numObs, numReps, blockSize)

for b in range(numReps):
    yart = yraw[ii[b]]
    y = yart[1:]
    x = yart[0:-1]
    reg = LinearRegression(fit_intercept=False)
    reg.fit(x.reshape(numObs-1,1), y.reshape(numObs-1,1))
    values[b] = reg.coef_[0][0]
    
mu = np.round(np.mean(values), 2)
sigma = np.round(np.std(values, ddof=1), 2)

# Looks very NORMAL even WITHOUT CLT. AMAZING!
plt.hist(values, bins=100);

























'''
TUESDAY, 04/02/2019

'''






