'''
    File name: JHansen_fin5330_hw2.py
    Author: Jared Hansen
    Date created: 03/02/2019
    Date last modified: 03/02/2019
    Python Version: 3.6.4
'''




# IMPORT STATEMENTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab
import scipy.stats as stats
import math
import statsmodels
from statsmodels.tsa.stattools import adfuller
from sklearn import linear_model




# Set the seed
np.random.seed(123456)









#==============================================================================
#==============================================================================
#==============================================================================
#==== PROBLEM 1 ===============================================================
#==============================================================================
#==============================================================================
#==============================================================================


"""
1. Simulate T = 500 observations from an AR(1) process for
   phi = {0.25, 0.5, 0.75, 0.8, 0.9}
   y_t = phi*y_{t-1} + eps_t
"""


#***** Comment this later
def gen_AR1(phi_val):
    
    T = 500
    
    # Generate the epislon terms vector
    epsilon_t = np.random.normal(size = T-1)
    
    # Create a vector to store the series {y_t}
    y_t = np.empty(T)
    
    # Intialize the first value, y_0, outside of the loop
    y_t[0] = 0
    
    for i in range(T-1):
        y_t[i+1] = phi_val * y_t[i] + epsilon_t[i]
    return y_t
    
# Generate the AR(1) series for each value of phi
y_phi_25 = gen_AR1(0.25)
y_phi_5  = gen_AR1(0.5)
y_phi_75 = gen_AR1(0.75)
y_phi_8  = gen_AR1(0.8)
y_phi_9  = gen_AR1(0.9)





"""
3. Estimate each model via OLS.
"""

# **** COMMENT THIS LATER
def fit_OLS(orig_series):
    # Create the one-value-shifted series
    series_shifted = orig_series[1:]
    # Cut off the first value in the original series
    orig_series = orig_series[0:-1]
    # Regress the original series on the shifted series
    ols_reg = stats.linregress(series_shifted, orig_series)
    return ols_reg

# Fit an OLS model for each of the series.
ols_25 = fit_OLS(y_phi_25)
ols_5  = fit_OLS(y_phi_5)
ols_75 = fit_OLS(y_phi_75)
ols_8  = fit_OLS(y_phi_8)
ols_9  = fit_OLS(y_phi_9)





"""
4. Test the standard null hypothesis.....

"""

# CRITICAL VALUES FOUND AT THE URL BELOW
# We have to account for doing a two-sided test.
# https://resources.saylor.org/wwwresources/archived/site/wp-content/uploads/2015/07/BUS204-FinalExamAid-TandZDistributionTables-CCBY.pdf
crit_val_01 = 2.58570
crit_val_05 = 1.96472
crit_val_1  = 1.64791


# *** COMMENT THIS LATER
def gen_test_results(ols_reg):
    # Here, test statistic is (estimated phi - 0)/(stderr_phi)
    test_stat = ols_reg[0] / ols_reg[-1]
    # standard error
    std_error = ols_reg[-1]
    # p-value
    p_val = ols_reg[3]

    
    return test_stat, std_error, p_val
    


hyp_test_phi_25 = gen_test_results(ols_25)
hyp_test_phi_5  = gen_test_results(ols_5)
hyp_test_phi_75 = gen_test_results(ols_75)
hyp_test_phi_8  = gen_test_results(ols_8)
hyp_test_phi_9  = gen_test_results(ols_9)



"""
5. Pick one of the parameter values .....
   PHI = 0.9
"""

#==============================================================================
#=== PART A: Plotting the CLT-derived sampling distribution
#==============================================================================
mean_phi_distn = ols_9[0]
stDev_phi_distn = ols_9[-1]

x_vals_plot = np.linspace(mean_phi_distn - 5*stDev_phi_distn,
                          mean_phi_distn + 5*stDev_phi_distn,
                          100)
plt.plot(x_vals_plot,
         stats.norm.pdf(x_vals_plot, mean_phi_distn, stDev_phi_distn),
         c = "g")
plt.title("CLT-derived Sampling Distribution of $\hat{\phi}$")




#==============================================================================
#=== PART b: parametric Monte Carlo
#==============================================================================

# This function will generate predicted y-hat values (a series) for a given
# slope (phi_hat) parameter. It is perfectly linear, no white nosie error term
def gen_y_hats(series, phi_hat):
    y_hats = phi_hat*series
    return y_hats
    

# Set the phi parameter
phi_hat = ols_9[0]
# Generate predictions using fctn
y_hats = gen_y_hats(y_phi_9, phi_hat)
# Drop the first value
y_hats = y_hats[1:]
# Calculate the residuals of the predicted values VS the true values
residuals = y_phi_9 - y_hats

# For the "parametric" part of parametric Monte Carlo, we now take the residuals
# to estimate parameters for a normal distn for the white noise part of the model
mu_eps = np.mean(residuals)
stDev_eps = np.std(residuals)

#***** Comment this later
def gen_AR1_MoCarl(phi_hat, mu_eps, stDev_eps):
    # Number of observations in the series
    T = 500
    # Generate the epislon terms vector
    eps_t = np.random.normal(loc=mu_eps, scale=stDev_eps, size=T-1)
    # Create a vector to store the series {y_t}
    y_t = np.empty(T)
    # Intialize the first value, y_0, outside of the loop
    y_t[0] = 0
    # Create the path
    for i in range(T-1):
        y_t[i+1] = phi_hat * y_t[i] + eps_t[i]
    return y_t



# Set the number of simulation repititions
M = 10000

# Simulate 100 different series and compute the regression (and regr params)
# for each of them
phi_MC = np.empty(M)
for i in range(M):
    # The inner part generates a parametric Monte Carlo series based on the parameters
    # we pass (our phi_hat, and the center and stDev of white noise process).
    # The outer part then fits an OLS regression onto this MC series, and uses
    # the function above to access the new predicted slope (phi_hat for that
    # series).
    # This then gets stored in the array of all Monte Carlo phi's (phi_MC).
    phi_MC[i] = fit_OLS(gen_AR1_MoCarl(phi_hat, mu_eps, stDev_eps))[0]
    
# Use the sample mean and standard deviation to estimate parameter values of 
# the distribtuion (of PHI's).
est_mu_phi_MC = np.mean(phi_MC)
est_sd_phi_MC = np.std(phi_MC)
    
# Plot a histogram
plt.hist(phi_MC, bins=100)
plt.title("Histogram of $\hat{\phi}$ sampling dist'n from parametric Monte Carlo")




#==============================================================================
#=== PART c: use IID bootstrap to simulate sampling distn of PHI
#==============================================================================

# *** COMMENT LATER
def gen_AR1_boot(phi_hat):
    # Number of observations in the series
    T = 500
    # Generate the epislon terms vector by IID bootstrapping 499 observations
    # from the residuals generated above (during Monte Carlo).
    eps_t = np.random.choice(residuals, size=T-1)
    # Create a vector to store the series {y_t}
    y_t = np.empty(T)
    # Intialize the first value, y_0, outside of the loop
    y_t[0] = 0
    # Create the path
    for i in range(T-1):
        y_t[i+1] = phi_hat * y_t[i] + eps_t[i]
    return y_t

# Set the number of simulation repititions
B = 10000

# Simulate 100 different series and compute the regression (and regr params)
# for each of them
phi_boot = np.empty(B)
for i in range(B):
    # The inner part generates a parametric Monte Carlo series based on the parameters
    # we pass (our phi_hat, and the center and stDev of white noise process).
    # The outer part then fits an OLS regression onto this MC series, and uses
    # the function above to access the new predicted slope (phi_hat for that
    # series).
    # This then gets stored in the array of all Monte Carlo phi's (phi_MC).
    phi_boot[i] = fit_OLS(gen_AR1_boot(phi_hat))[0]
    
# Use the sample mean and standard deviation to estimate parameter values of 
# the distribtuion (of PHI's).
est_mu_phi_boot = np.mean(phi_boot)
est_sd_phi_boot = np.std(phi_boot)
    
# Plot a histogram
plt.hist(phi_boot, bins=100)
plt.title("Histogram of $\hat{\phi}$ sampling dist'n from IID bootstrap")












"""
6. Return to the problem in 5 above and redo the simulation from step one, but
   replace the error distribution with a student-T distribution with df=5.
   Even though we know at the generation stage that the errors....

"""


#==============================================================================
#==== SIMULATE USING STUDENT-T INSTEAD OF STD-NORMAL white noise ==============
#==============================================================================

#***** Comment this later
def gen_AR1_studentT(phi_val):
    # Number of obs in the series
    T = 500
    # Generate the epislon terms vector
    epsilon_t = np.random.standard_t(size = T-1, df=5)
    # Create a vector to store the series {y_t}
    y_t = np.empty(T)
    # Intialize the first value, y_0, outside of the loop
    y_t[0] = 0
    for i in range(T-1):
        y_t[i+1] = phi_val * y_t[i] + epsilon_t[i]
    return y_t
    
# Generate the new series (different method for generating white noise: student-T)
y_phi_9_student = gen_AR1_studentT(0.9)
# Do the regression for this new series.
ols_9_student = fit_OLS(y_phi_9_student)


#==============================================================================
#=== PART A: Plotting the CLT-derived sampling distribution
#==============================================================================
mean_phi_distn_stud = ols_9_student[0]
stDev_phi_distn_stud = ols_9_student[-1]

x_vals_plot = np.linspace(mean_phi_distn - 5*stDev_phi_distn,
                          mean_phi_distn + 5*stDev_phi_distn,
                          100)
plt.plot(x_vals_plot,
         stats.norm.pdf(x_vals_plot, mean_phi_distn_stud, stDev_phi_distn_stud),
         c = "g")
plt.title("CLT-derived Sampling Distribution of $\hat{\phi}$, student-T version")




#==============================================================================
#=== PART b: parametric Monte Carlo
#==============================================================================

# This function will generate predicted y-hat values (a series) for a given
# slope (phi_hat) parameter. It is perfectly linear, no white nosie error term
def gen_y_hats(series, phi_hat):
    y_hats = phi_hat*series
    return y_hats
    

# Set the phi parameter
phi_hat = ols_9_student[0]
# Generate predictions using fctn
y_hats_student = gen_y_hats(y_phi_9_student, phi_hat)
# Drop the first value
y_hats_student = y_hats_student
# Calculate the residuals of the predicted values VS the true values
residuals_student = y_phi_9_student - y_hats_student

# For the "parametric" part of parametric Monte Carlo, we now take the residuals
# to estimate parameters for a normal distn for the white noise part of the model
mu_eps_student = np.mean(residuals_student)
stDev_eps_student = np.std(residuals_student)

#***** Comment this later
def gen_AR1_MoCarl_student(phi_hat, mu_eps_student, stDev_eps_student):
    # Number of observations in the series
    T = 500
    # Generate the epislon terms vector
    eps_t = np.random.normal(loc=mu_eps_student,
                             scale=stDev_eps_student,
                             size=T-1)
    # Create a vector to store the series {y_t}
    y_t = np.empty(T)
    # Intialize the first value, y_0, outside of the loop
    y_t[0] = 0
    # Create the path
    for i in range(T-1):
        y_t[i+1] = phi_hat * y_t[i] + eps_t[i]
    return y_t



# Set the number of simulation repititions
M = 10000

# Simulate 100 different series and compute the regression (and regr params)
# for each of them
phi_MC_student = np.empty(M)
for i in range(M):
    # The inner part generates a parametric Monte Carlo series based on the parameters
    # we pass (our phi_hat, and the center and stDev of white noise process).
    # The outer part then fits an OLS regression onto this MC series, and uses
    # the function above to access the new predicted slope (phi_hat for that
    # series).
    # This then gets stored in the array of all Monte Carlo phi's (phi_MC).
    phi_MC_student[i] = fit_OLS(gen_AR1_MoCarl_student(phi_hat,
                  mu_eps_student,
                  stDev_eps_student))[0]
    
# Use the sample mean and standard deviation to estimate parameter values of 
# the distribtuion (of PHI's).
est_mu_phi_MC_student = np.mean(phi_MC_student)
est_sd_phi_MC_student = np.std(phi_MC_student)
    
# Plot a histogram
plt.hist(phi_MC_student, bins=100)
plt.title("Histogram of $\hat{\phi}$ sampling dist from parametric Monte Carlo, student-T")




#==============================================================================
#=== PART c: use IID bootstrap to simulate sampling distn of PHI
#==============================================================================

# *** COMMENT LATER
def gen_AR1_boot_stud(phi_hat):
    # Number of observations in the series
    T = 500
    # Generate the epislon terms vector by IID bootstrapping 499 observations
    # from the residuals generated above (during Monte Carlo).
    eps_t = np.random.choice(residuals_student, size=T-1)
    # Create a vector to store the series {y_t}
    y_t = np.empty(T)
    # Intialize the first value, y_0, outside of the loop
    y_t[0] = 0
    # Create the path
    for i in range(T-1):
        y_t[i+1] = phi_hat * y_t[i] + eps_t[i]
    return y_t

# Set the number of simulation repititions
B = 10000

# Simulate 100 different series and compute the regression (and regr params)
# for each of them
phi_boot_stud = np.empty(B)
for i in range(B):
    # The inner part generates a parametric Monte Carlo series based on the parameters
    # we pass (our phi_hat, and the center and stDev of white noise process).
    # The outer part then fits an OLS regression onto this MC series, and uses
    # the function above to access the new predicted slope (phi_hat for that
    # series).
    # This then gets stored in the array of all Monte Carlo phi's (phi_MC).
    phi_boot_stud[i] = fit_OLS(gen_AR1_boot_stud(phi_hat))[0]
    
# Use the sample mean and standard deviation to estimate parameter values of 
# the distribtuion (of PHI's).
est_mu_phi_boot_stud = np.mean(phi_boot_stud)
est_sd_phi_boot_stud = np.std(phi_boot_stud)
    
# Plot a histogram
plt.hist(phi_boot_stud, bins=100)
plt.title("Histogram of $\hat{\phi}$ sampling dist'n from IID bootstrap, student-T")


# *****************************************************************************
#!!!!!! DON'T FORGET TO "Compare all three methods" again !!!!!!!!!!!!!!!!!!!!!!!!!!
# *****************************************************************************



















































#==============================================================================
#==============================================================================
#==============================================================================
#==== PROBLEM 2 ===============================================================
#==============================================================================
#==============================================================================
#==============================================================================


"""
Simulate T= 500 time steps for the following two equations:
y_t = y_{t-1} + u_{1,t}
x_t = x_{t-1} + u_{2,t}

where u_{j,t} are independent standard white noise processes.
"""

# The number of time steps
T = 500
# This creates the series {y_t}
y = np.cumsum(np.random.normal(size=T))
# Turn the y numpy array into a Pandas series
ts1 = pd.Series(y)
# Plot the series
ts1.plot(grid=True, color="purple")

# This creates the series {x_t}
x = np.cumsum(np.random.normal(size=T))
# Turn the x numpy array into a Pandas series
ts2 = pd.Series(x)
# Plot the series
ts2.plot(grid=True, color="orange")

# Regress {y_t} on {x_t} and estimate beta (slope coefficient) via the 
# OLS equation: y_t = alpha + beta*x_t + eps_t
reg = stats.linregress(x,y)
reg

#=== ESTIMATE OF BETA (this is an answer)
reg[0]

#=== TEST THE NULL HYPOTHESIS (built in to the regression function)
p_val = reg[3]  
p_val
# We reject H_0 for significance levels 0.1 and 0.05, but fail to reject for
# sign_level = 0.01. WE SHOULD have found we failed to reject at all levels of
# significance (since the series aren't correlate) but we didn't find this.
# This is because both series are related to time similarly.
# SPURRIOUS REGRESSION. DAMMIT.

# REPEAT THE PROCES M = 50,000 TIMES AND STORE THE ..... (BULLET POINT)
T = 500
M = 50000
betas = np.empty(M)
r_sqrd = np.empty(M)

for i in range(M):
    # Create the two series
    y = np.cumsum(np.random.normal(size=T))
    x = np.cumsum(np.random.normal(size=T))
    # Regress y on x
    reg = stats.linregress(x,y)
    # Store the beta coefficient for that regression in the array
    betas[i] = reg.slope
    r_sqrd[i] = reg.rvalue**2
    # Theoretically, all of the r-squared values should be 0
    
    
#=== SUMMARIZE THE SIMULATE SAMPLING DISTRIBUTION FOR beta
mean_betas = np.mean(betas)
stDev_betas = np.std(betas)
# These are the numeric summaries of the histogram plotted below.
    
# Make a histogram of the betas values (very normal looking)
# The semicolon prevents the array being histogram-ed from printing in console.
plt.hist(betas, bins=100);
# Make a histogram of the r-squared values. WE WOULD HAVE EXPECTED TO HAVE FAR
# MORE VALUES NEAR 0. TOO MANY HIGH R-SQUARED VALUES FOR SERIES THAT SHOULD NOT
# BE CORRELATED.
# This illustrates the problem of "spurrious regression": we can perform 
# regressions that aren't valid (and shouldn't be done, shouldn't draw 
# conclusions from). Regression can tell us that series that are distinct are
# related when they aren't actually related.
# THEY'RE NOT ACTUALLY RELATED TO EACH OTHER; THEY'RE BOTH RELATED TO TIME!!!
plt.hist(r_sqrd, bins=100);














































#==============================================================================
#==============================================================================
#==============================================================================
#==== PROBLEM 4 ===============================================================
#==============================================================================
#==============================================================================
#==============================================================================



# *****************************************************************************
# *** Any price in a highly-arbitraged market should be close to having a  ***
#     unit root
# *****************************************************************************




#==============================================================================
#==== PROBLEM 4, part A =======================================================
#==============================================================================


# Number of time steps (number of observations)
T = 500
# Simulate the T = 500 time teps from the random walk model: x_t = x_{t-1} + u_{1,t}
x = np.cumsum(np.random.normal(size=T))

# Parameters for the I(1) process simulated below.
alpha = 0.22
beta = 2.5
# Cointegration data-generating process (white noise ~N(0,1) in the equation below)
eps_t = np.random.normal()
# Simulate the T = 500 time steps from the model: y_t = alpha + beta*x_t + eps_t
y = alpha + beta*x + eps_t

# Let's look at them. Nice.
plt.plot(x)
plt.plot(y)







#==============================================================================
#==== PROBLEM 4, part B =======================================================
#==============================================================================

'''
Use the augmented Dickey-Fuller test to check for the presence of a unit root
in both {y_t} and {x_t}.
What do you find? What should you find?
'''

adf_y = adfuller(y, regression = "ct", maxlag = 10)
adf_x = adfuller(x, regression = "ct", maxlag = 10)

adf_y
adf_x
















#==============================================================================
#==== PROBLEM 4, part C =======================================================
#==============================================================================

'''
Implement the Engle-Granger two-step method by:
    ** First, test for cointegration by submitting eps_hat_t to the ADF test.
        What do you find?
    ** Obtain beta_hat via OLS.
    ** Estimate the error-correction model with p=1 and include contemporaneous x_t
'''

#==============================================================================
# FIRST, TEST FOR COINTEGRATION BY SUBMITTING eps_hat_t TO THE ADF TEST. 
# WHAT DO YOU FIND?
#==============================================================================

# Residuals eps (typically epsilon)
reg = stats.linregress(x,y)   # regress y on x
eps = y - reg.intercept - reg.slope*x

# Submit eps to the ADF test
adf_eps = adfuller(eps, maxlag = 10)
adf_eps   # The small p-value ==> no presence of unit-root in epsilon series


#==============================================================================
# OBTAIN beta_hat VIA OLS
#==============================================================================
beta_hat = reg.slope   # This comes out to be 2.5, which is what we said (worked).







#==============================================================================
# ESTIMATING THE ERROR-CORRECTION MODEL WITH p=1 AND INCLUDE CONTEMPORANEOUS x_t
#==============================================================================

# To fix the problem of spurrious regression, we have to do differencing to 
# make the series stationary
"""
change: y_t = y_{t-1} + eps_t ===> [y_t - y_{t-1} = eps_t ] = [deltaY = eps_t]

Then regress deltaY on deltaX: deltaY_t = alpha + betas*deltaX_t + error_t
Here betas is the [Cov(deltaY, deltaX)]/[Var(deltaX)] = rho*(sigma_deltaY/sigma_deltaX)


-------------------------------------------------------------------------------------
For:
    y_t ~ I(1)
    x_t ~ I(1)
    eps_hat_t ~ I(0)
    y_t = alpha + betas*x_t + eps_t
    
    ** We'll get the CoInt vector [1, -beta] ===> beta gives an indication of how
       strong the relationship is between the series {x_t} and {y_t}. Larger beta
       means stronger CoInt (larger the ABS VALUE of beta).
    ** With strong CoInt, the asymptotic properties of the beta estimator 
       are even better than for beta in a standard, cross-sectional OLS regr
    
    
------------------------------------------------------------------------------------
Every CoInt relationship implies an ERROR CORRECTION FORM:
    deltaY_t 
    deltaX_t
    y_t, x_t  ==> eps_hat_t = [y_{t-1} - alpha_hat - betas_hat * x_{t-1}]
    
    ECM(1):
    -------
        deltaY_t = mu + lambda*(y_{t-1} - alpha_hat - betas_hat * x_{t-1})
                   + gamma*deltaY_{t-1} + del*deltaX_{t-1} + nu_t
        
        (y_t - y_{t-1}) = mu + lambda*(y_{t-1} - alpha_hat - betas_hat * x_{t-1})
                          + gamma*(y_{t-1} - y_{t-2}) + del*(y_{t-1} - x_{t-2}) _ nu_t
        
        ** where lambda is the speed of return to equilibrium: want large abs(lambda),
           and we want high variation also for pairs trading.
        ** Ideal pairs trading has high volatility (large variance in eps_hat_t),
           large abs(lambda) and large abs(betas_hat)
    
"""

# This creates the deltaY and deltaX series
deltaY = y[1:] - y[:-1]
deltaX = x[1:] - x[:-1]

# These arrays need to have the same length(s) and cut off the first value
# in order to do the ECM model above.
deltaY.shape
deltaX.shape
eps.shape

deltaY = deltaY[1:]
deltaX = deltaX[1:]
eps = eps[2:]

# Design matrix from our regression: we need N-2 rows, and 4 columns (4 terms in the regr)
X = np.ones((T-2,4))
X[:,1] = eps
X[:,2] = deltaY
X[:,3] = deltaX
Y = deltaY

# Now run regression of Y on X
# Use statsmodels or sk-learn to do the regression (statsmodels)
clf = linear_model.LinearRegression()
fitted_VECM_1 = clf.fit(X, Y)

# This gives parameter estimates when regression Y on X, in the order:
#  mu, lambda, gamma, del
fitted_VECM_1.coef_
fitted_VECM_1.intercept_




#==============================================================================
#=== We have to do this again, but for estimating the deltaX series by regressing
#=== X on Y (since they're related to each other).

# Now run regression of X on Y
# Use statsmodels or sk-learn to do the regression (statsmodels)
fitted_VECM_2 = clf.fit(X, deltaX)

# This gives parameter estimates when regression Y on X, in the order:
#  mu, lambda, gamma, del
fitted_VECM_2.coef_
fitted_VECM_2.intercept_



