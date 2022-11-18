'''
File name: model1_RF.py
Author: Jared Hansen
Date created: 01/12/2019
Date last modified: 03/21/2019
Python version: 3.6.4

DESCRIPTION:
    

'''



import os
import pandas as pd
#import inspect

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics





# (1.) Save this script in the same directory as the data.
# (2.) Set the working directory using the line below, to whatever file path
#      points to this directory.
os.chdir("C:/__JARED/_TradeSignResearch")


# Set the path of the file
#FANG_location = "C:/Users/jrdha/OneDrive/Desktop/_TradeSignResearch"
#FANG_fileName = "/FANG.txt"
#FANG_path = FANG_location + FANG_fileName

# Import the full file (cleaned data)
df = pd.read_csv("FANG_030118_ITCH.csv")
new_df = df



#==============================================================================
#=== Exploratory Data Analysis
#==============================================================================

# What is the relative proportion of buys VS sells? (Class imbalance?)
len(new_df[new_df["buysell"] == "B"]) / len(new_df)
# Roughly 0.39 buys and 0.61 sells



#==============================================================================
#=== Initial Random Forest model to see what kind of results we get
#==============================================================================

#=== Data Pre-processing ======================================================

# Allows us to see other columns in console (need it in this instance).
pd.set_option('display.max_columns', None)
# Look at summaries for each of the numeric columns.
new_df.describe()
# Resets the option so that console won't get filled up for future commands.
pd.reset_option('display.max_columns')

# One-hot encode the data. This will turn the "msg_type" and "buysell" columns
# each into two binary columns (since each feature has only two values).
new_df = pd.get_dummies(new_df)
#new_df.iloc[:, 5:].head(5)
#new_df.head(5)

# The values we want to predict are buys and sells. The one-hot encoding turned
# the single buysell column into two columns: buysell_B and buysell_S.
# For sake of ease, let's simply remove the buysell_S column and use the
# buysell_B column as our response. For this column, a value of 1 signals a buy
# and a value of 0 signals a sell.
new_df = new_df.drop(columns = "buysell_S")

# The labels (in the buysell_B column) are what we want to predict.
labels = np.array(new_df["buysell_B"])
# Remove the labels from the dataframe. Axis 1 refers to the column(s).
features = new_df.drop("buysell_B", axis = 1)
# Save the feature names for future use.
feature_list = list(features.columns)
# Convert the features to a numpy array.
features = np.array(features)

# Split into training and test sets.
train_features,test_features,train_labels,test_labels = train_test_split(features,
                               labels,
                               test_size = 0.25,
                               random_state = 42)

# Make sure we did things correctly. We'd expect the training features number
# of columns to match the testing feature number of columns, and the number of
# rows to match for the respective training and testing features and the labels
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
# Training Features Shape: (3745, 10)
# Training Labels Shape: (3745,)
# Testing Features Shape: (1249, 10)
# Testing Labels Shape: (1249,)


#=== Model fitting ============================================================

# Create a Gaussian classifier.
clf = RandomForestClassifier(n_estimators = 100)

# Train the model using the training sets.
clf.fit(train_features, train_labels)

# Predict onto the test features

test_pred = clf.predict(test_features)

# Check the accuracy ------ 86.6% right out of the box!!
print("Accuracy:", metrics.accuracy_score(test_pred, test_labels))

# Check feature importance, visualize.
feature_imp = pd.Series(clf.feature_importances_,
                        index = feature_list
                        ).sort_values(ascending = False)
# %matplotlib inline
sns.barplot(x = feature_imp, y = feature_imp.index)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Visualizing Random Forest Feature Importance")
plt.legend()
plt.show()
