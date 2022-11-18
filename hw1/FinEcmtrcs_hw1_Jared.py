'''
    File name: FinEcmtrcs_hw1_Jared.py
    Author: Jared Hansen
    Date created: 01/28/2019
    Date last modified: 01/28/2019
    Python Version: 3.6
'''


'''
CODE FOR HW1 IN [FINANCIAL ECONOMETRICS]

PROMPT:
    
    Consider the daily stock returns of American Express (AXP), Caterpillar
    (CAT), and Starbucks (SBUX) from January 1999 to December 2008. The data
    are daily prices in the file stock-data-hwk1.txt.
    
        (a) Calculate simple returns for the three series.
        
        (b) Express the simple returns in percentages. Compute the sample mean,
        standard deviation, skewness, excess kurtosis, minimum, and maximum of
        the percentage simple returns.
        
        (c) Transform the simple returns to log returns.
        
        (d) Express the log returns in percentages. Compute the sample mean,
        standard deviation, skewness, excess kurtosis, minimum, and maximum of
        the percentage log returns.
        
        (e) Test the null hypothesis that the mean of the log returns of each
        stock is zero. That is, perform three separate tests. Use 5%
        significance level to draw your conclusion.
        
        (f) Plot histograms for each of the three series (both simple and log
        returns - so six graphs total).
        
        (g) Test the null hypothesis that the lag-$2$ autocorrelation is zero
        for log returns.
'''





#==============================================================================
#==============================================================================
#==== Import statements =======================================================
#==============================================================================
#==============================================================================

import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt





#==============================================================================
#==============================================================================
#==== Function definitions ====================================================
#==============================================================================
#==============================================================================

def compute_rtns_series(log_bool, simp_v_pct, orig_series):
    """
    This function computes the series of returns for the passed-in time series
    array. See the Returns section below for the 4 types of returns we can
    calculate and output with this function.
    
    Parameters
    ----------
    log_bool : bool
        This indicates whether or not the computed series of returns should be
        log-transformed (True) or not (False).
    simp_v_pct : string
        This indicates whether or not the computed series should be in the 
        form of simple (net) returns ("net") or simple returns expressed as
        percentages ("pct").
    orig_series : numpy ndarray
        The original 1-dimensional time series array.
        
    Returns
    -------
    rtns_series : numpy ndarray
        The final 1-dimensional time series array: either simple gross returns,
        simple net (pctg) returns, simple gross log returns, or simple
        net (pctg) log returns.
    """
    # Based on definitions in notes, I'm assuming the following:
    # == The time series [t_0, t_1, t_2, t_3, t_4] would have simple (net)
    # == returns of the form: [(t_1/t_0)-1,(t_2/t_1)-1,(t_3/t_2)-1,(t_4/t_3)-1]
    # == To convert these to pctgs we just multiiply each value in the series
    # == of simple net returns by 100.
    # == The log-transformed series will look like the following:
    # == [ln(t_1/t_0),ln(t_2/t_1),ln(t_3/t_2),ln(t_4/t_3)] and the log pctg
    # == returns: 100*[ln(t_1/t_0),ln(t_2/t_1),ln(t_3/t_2),ln(t_4/t_3)]
    
    # This deletes the first element of the series to create the array of 
    # numerator values for the gross returns (above, [t_1,t_2,t_3,t_4]).
    grs_rtns_numrtr = np.delete(orig_series, 0)
    # This deletes the last element of the series to create the array of 
    # denominator values for the gross returns (above, [t_0,t_,t_2,t_3]).
    grs_rtns_denmtr = np.delete(orig_series, len(orig_series) - 1)
    # The below chunk will calculate the series of returns (rtns_series)
    # according to whether we want "simp" (outer if chunk) or "pct" (outer
    # else chunk) and log returns (inner if for each outer chunk) or not.
    grs_rtns = grs_rtns_numrtr/grs_rtns_denmtr
    if(simp_v_pct == "simp"):
        rtns_series = grs_rtns - 1
        if(log_bool):
            rtns_series = np.log(grs_rtns)
    elif(simp_v_pct == "pct"):
        rtns_series = 100 * (grs_rtns - 1)
        if(log_bool):
            rtns_series = 100 * np.log(grs_rtns)
    return rtns_series


def compute_print_stats(log_bool, orig_series):
    """
    This function computes and prints the desired statistics (sample mean,
    standard deviation, skewness, excess kurtosis, minimum, maximum) of the
    given time series array that has been converted to percentages.
    The input time series array should be in its original form.
    
    Parameters
    ----------
    log_bool : bool
        This indicates whether or not the computed series of returns should be
        log-transformed (True) or not (False).
    orig_series : numpy ndarray
        The original 1-dimensional array of time series values (stock prices).
    
    Returns
    -------
    Nothing (prints information instead)
    """
    # Set the gross_v_net arg = to "net" since we're calculating pctg returns.
    rtns_as_pctgs = compute_rtns_series(log_bool, "pct", orig_series)
    rtns_mean = np.mean(rtns_as_pctgs)
    rtns_stDev = np.std(rtns_as_pctgs)
    rtns_skew = sp.stats.skew(rtns_as_pctgs)
    rtns_kurt = sp.stats.kurtosis(rtns_as_pctgs)
    rtns_min = np.min(rtns_as_pctgs)
    rtns_max = np.max(rtns_as_pctgs)
    print("The mean of this series is            :", rtns_mean)
    print("The stDev of this series is           :", rtns_stDev)
    print("The skewness of this series is        :", rtns_skew)
    print("The excess kurtosis of this series is :", rtns_kurt)
    print("The minimum value of this series is   :", rtns_min)
    print("The maximum value of this series is   :", rtns_max)
    #== NOTE: I looked up the difference between kurtosis and excess kurtosis:
    #==       [excess kurtosis = kurtosis - 3]. The default value of the
    #==       "fisher" argument is True for the sp.stats.kurtosis() function.
    #==       This default calculates EXCESS kurtosis, so my code is correct.


def hyp_test_lag2(series, rho_hat, ticker):
    """
    This function calculates the test statistic for testing whether or not the
    lagged-l autocorrelation for a series is 0.
    --- H_0: the series has lagged-l autocorrelation = 0
    --- H_A: the series has lagged-l autocorrelation != 0
    Since the sample size for each of the series is 2514, this is a large 
    enough number that we can just use the standard normal distribution instead
    of a t-distn for computing the threshold for reject/not reject.
        That being the case, the threshold for rejecting H_0 is a test statistic
    with an absolute value of > 1.96. The function prints a message giving the
    computed test statistic and the result relative to reject/not H_0.
    
    Parameters
    ----------
    series : numpy ndarray
        This is the original series. We just use it for obtaining sample size
    rho_hat : float
        This is the empirically estimated value of the autocorrelation (in this
        case, lag-2 autocorr).
    ticker : string
        The ticker symbol of the series for which we're testing the hypothesis.
        
    Returns
    -------
        Nothing (prints message).
    """
    test_stat = (np.sqrt(len(series)) * rho_hat)
    if(abs(test_stat) > 1.96):
        print("\nSince the value of the test statistic [ sqrt(T)*\hat{rho_2} =",
              round(test_stat, 4), "] > |1.96| we reject the null hypothesis",
              "that the lag-2 autocorrelation is 0 for ", ticker, ". This",
              "implies that the data give evidence that lag-2 autocorrelation",
              "is a value other than 0.")
    else:
        print("\nSince the value of the test statistic [ sqrt(T)*\hat{rho_2} = ",
              round(test_stat, 4), "] < |1.96| we fail to reject the null",
              "hypothesis that the lag-2 autocorrelation is 0 for", ticker, ".")


    
    
    
#==============================================================================
#==============================================================================
#==== Procedural code =========================================================
#==============================================================================
#==============================================================================
    


#==== DATA PRE-PROCESSING =====================================================
#==============================================================================
# Import the data -- NOTE: CHANGE FILE PATH TO THAT OF THE LOCAL DATA.
# Set the path of the file
data_loc = "C:/Users/jrdha/OneDrive/Desktop/_USU_Sp2019/_Brough_FinEcon/hw1/"
data_fileName = "stock-data-hwk1.csv"
data_path = data_loc + data_fileName

# Import the full CSV file.
raw_data = pd.read_csv(data_path)

# Brough's code (thanks!) for separating into the data for each stock.
tickers = ['AXP', 'CAT', 'SBUX']

ind = raw_data.TICKER == tickers[0]
axp_data = raw_data[ind]

ind = raw_data.TICKER == tickers[1]
cat_data = raw_data[ind]

ind = raw_data.TICKER == tickers[2]
sbux_data = raw_data[ind]



#==== QUESTION A ==============================================================
#==============================================================================
#==== Calculate the simple returns for the three series
#==== NOTE: I take "simple returns" to mean the simple, gross returns. Also, 
#====       since the problem doesn't specify whether or not to display these
#====       calculations, I'll just show the first few values and last few
#====       values of each series using the NumPy view function.
axp_prcs = np.array(axp_data.PRC)
cat_prcs = np.array(cat_data.PRC)
sbux_prcs = np.array(sbux_data.PRC)

simp_net_rtns_AXP = compute_rtns_series(False, "simp", axp_prcs)
print("\nThe first and last values for the series of simple (net) returns for",
      "the AXP (American Express) stock series:\n", simp_net_rtns_AXP.view())

simp_net_rtns_CAT = compute_rtns_series(False, "simp", cat_prcs)
print("\nThe first and last values for the series of simple (net) returns for",
      "the CAT (Caterpillar) stock series:\n", simp_net_rtns_CAT.view())

simp_net_rtns_SBUX = compute_rtns_series(False, "simp", sbux_prcs)
print("\nThe first and last values for the series of simple (net) returns for",
      "the SBUX (Starbucks) stock series:\n", simp_net_rtns_SBUX.view())



#==== QUESTION B ==============================================================
#==============================================================================
#==== Express the simple returns in percentages. Compute the sample mean,
#==== standard deviation, skewness, excess kurtosis, minimum, and maximum of
#==== the percentage simple returns.
simp_net_pctg_AXP = compute_rtns_series(False, "pct", axp_prcs)
print("\nThe first and last values for the series of simple (net) returns in",
      "percentages for the AXP (American Express) stock series:\n",
      simp_net_pctg_AXP.view())
compute_print_stats(False, axp_prcs)

simp_net_pctg_CAT = compute_rtns_series(False, "pct", cat_prcs)
print("\nThe first and last values for the series of simple (net) returns in",
      "percentages for the CAT (Caterpillar) stock series:\n",
      simp_net_pctg_CAT.view())
compute_print_stats(False, cat_prcs)

simp_net_pctg_SBUX = compute_rtns_series(False, "pct", sbux_prcs)
print("\nThe first and last values for the series of simple (net) returns in",
      "percentages for the SBUX (Starbucks) stock series:\n",
      simp_net_pctg_SBUX.view())
compute_print_stats(False, sbux_prcs)



#==== QUESTION C ==============================================================
#==============================================================================
#==== Transform the simple returns to log returns.
#==== NOTE: I just use my function to caclulate this rather than explicitly
#====       converting the simple net returns series.

simp_net_log_rtns_AXP = compute_rtns_series(True, "simp", axp_prcs)
print("\nThe first and last values for the series of simple (net)",
      "log-transformed returns for the AXP (American Express) stock series:\n",
      simp_net_log_rtns_AXP.view())

simp_net_log_rtns_CAT = compute_rtns_series(True, "simp", cat_prcs)
print("\nThe first and last values for the series of simple (net)",
      "log-transformed returns for the CAT (Caterpillar) stock series:\n",
      simp_net_log_rtns_CAT.view())

simp_net_log_rtns_SBUX = compute_rtns_series(True, "simp", sbux_prcs)
print("\nThe first and last values for the series of simple (net)",
      "log-transformed returns for the SBUX (Starbucks) stock series:\n",
      simp_net_log_rtns_SBUX.view())



#==== QUESTION D ==============================================================
#==============================================================================
#==== Express the log returns in percentages. Compute the sample mean,
#==== standard deviation, skewness, excess kurtosis, minimum, and maximum of
#==== the percentage log returns.

simp_net_log_pctg_AXP = compute_rtns_series(True, "pct", axp_prcs)
print("\nThe first and last values for the series of simple, percentage, "
      "log-transformed returns for the AXP (American Express) stock series:\n",
      simp_net_log_pctg_AXP.view())
compute_print_stats(True, axp_prcs)

simp_net_log_pctg_CAT = compute_rtns_series(True, "pct", cat_prcs)
print("\nThe first and last values for the series of simple, percentage, "
      "log-transformed returns for the CAT (Caterpillar) stock series:\n",
      simp_net_log_pctg_CAT.view())
compute_print_stats(True, cat_prcs)

simp_net_log_pctg_SBUX = compute_rtns_series(True, "pct", sbux_prcs)
print("\nThe first and last values for the series of simple, percentage, "
      "log-transformed returns for the SBUX (Starbucks) stock series:\n",
      simp_net_log_pctg_SBUX.view())
compute_print_stats(True, sbux_prcs)



#==== QUESTION E ==============================================================
#==============================================================================
#==== Test the null hypothesis that the mean of the log returns of each
#==== stock is zero. That is, perform three separate tests. Use 5%
#==== significance level to draw your conclusion.
#==== NOTE: from scipy.stats I use the ttest_1samp(a, popmean) function. It's a
#====       two-sided test for the null hypothesis that the expected value
#====       (mean) of a sample of independent observations a is equal to the
#====       given population mean, popmean.
#====       EVEN THOUGH this says "...sample of independent observations" we
#====       are going to use it as-is since we aren't worrying about time
#====       dependence structure for this problem (or I'm assuming we're not.)

sp.stats.ttest_1samp(simp_net_log_rtns_AXP, 0)
# pvalue=0.297298 ==> We FAIL TO REJECT the null hypothesis that the mean of 
# the log return series for the AXP stock is 0.

sp.stats.ttest_1samp(simp_net_log_rtns_CAT, 0)
# pvalue=0.963371 ==> We FAIL TO REJECT the null hypothesis that the mean of 
# the log return series for the CAT stock is 0.

sp.stats.ttest_1samp(simp_net_log_rtns_SBUX, 0)
# pvalue=0.32415 ==> We FAIL TO REJECT the null hypothesis that the mean of 
# the log return series for the SBUX stock is 0.



#==== QUESTION F ==============================================================
#==============================================================================
#==== Plot histograms for each of the three series (both simple and log
#==== returns - so six graphs total).

#==== NOTE: it appears that there are some outliers for each of the 6 plots.
#====       We know this because the x-axis scale for each plot extends a ways
#====       past where we'd (visually) expect it to end based on where the mass
#====       of the respective distributions are centered.

plt.figure(1)

plt.subplot(2, 3, 1)
plt.hist(simp_net_rtns_AXP,
         bins = np.linspace(min(simp_net_rtns_AXP),
                            max(simp_net_rtns_AXP),
                            40))
plt.xlabel("Simple Net return values")
plt.ylabel("Count (frequency)")
plt.title("Simple Net Returns: AXP")
plt.show()

plt.subplot(2, 3, 2)
plt.hist(simp_net_rtns_CAT,
         bins = np.linspace(min(simp_net_rtns_CAT),
                            max(simp_net_rtns_CAT),
                            40))
plt.xlabel("Simple Net return values")
plt.ylabel("Count (frequency)")
plt.title("Simple Net Returns: CAT")
plt.show()

plt.subplot(2, 3, 3)
plt.hist(simp_net_rtns_SBUX,
         bins = np.linspace(min(simp_net_rtns_SBUX),
                            max(simp_net_rtns_SBUX),
                            40))
plt.xlabel("Simple Net return values")
plt.ylabel("Count (frequency)")
plt.title("Simple Net Returns: SBUX")
plt.show()

plt.subplot(2, 3, 4)
plt.hist(simp_net_log_rtns_AXP,
         bins = np.linspace(min(simp_net_log_rtns_AXP),
                            max(simp_net_log_rtns_AXP),
                            40))
plt.xlabel("Simple Net LOG return values")
plt.ylabel("Count (frequency)")
plt.title("Simple Net LOG Returns: AXP")
plt.show()

plt.subplot(2, 3, 5)
plt.hist(simp_net_log_rtns_CAT,
         bins = np.linspace(min(simp_net_log_rtns_CAT),
                            max(simp_net_log_rtns_CAT),
                            40))
plt.xlabel("Simple Net LOG return values")
plt.ylabel("Count (frequency)")
plt.title("Simple Net LOG Returns: CAT")
plt.show()

plt.subplot(2, 3, 6)
plt.hist(simp_net_log_rtns_SBUX,
         bins = np.linspace(min(simp_net_log_rtns_SBUX),
                            max(simp_net_log_rtns_SBUX),
                            40))
plt.xlabel("Simple Net LOG return values")
plt.ylabel("Count (frequency)")
plt.title("Simple Net LOG Returns: SBUX")
plt.show()

plt.subplots_adjust(top=0.85, bottom=0.12, left=0.10, right=0.95, hspace=0.55,
                    wspace=0.35)

plt.suptitle("Top Row: simple returns, Bottom Row: simple LOG returns")
plt.show()



#==== QUESTION G ==============================================================
#==============================================================================
#==== Test the null hypothesis that the lag-2 autocorrelation is zero
#==== for log returns.

# Calculate the value of the lag-2 autocorrelations for each stock.
rho_axp = pd.Series(simp_net_log_rtns_AXP).autocorr(lag=2)
rho_cat = pd.Series(simp_net_log_rtns_CAT).autocorr(lag=2)
rho_sbux = pd.Series(simp_net_log_rtns_SBUX).autocorr(lag=2)

# Is test statistic big enough to reject H_0?
hyp_test_lag2(simp_net_log_rtns_AXP, rho_axp, "AXP")
hyp_test_lag2(simp_net_log_rtns_CAT, rho_cat, "CAT")
hyp_test_lag2(simp_net_log_rtns_SBUX, rho_sbux, "SBUX")