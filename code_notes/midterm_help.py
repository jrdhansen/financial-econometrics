# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:55:49 2019

@author: jrdha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

path = "C:\\__JARED\\_USU_Sp2019\\_Brough_FinEcon\\WaterStocks.csv"
data = pd.read_csv(path, parse_dates=True)


ind = data.PERMNO == 26463
temp = data.loc[ind]
temp.head()


sm.tsa.stattools.adfuller?
result = sm.tsa.stattools.adfuller(tmp.PRC.apply(np.abs).apply(np.log), maxlag=1, regression="nc")




















