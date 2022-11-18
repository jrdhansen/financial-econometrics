'''
File name: clean_singleFile_FINAL.py
Author: Jared Hansen
Date created: 03/21/2019
Date last modified: 04/11/2019
Python version: 3.6.4

DESCRIPTION:
    Finalized script for cleaning ITCH data.
'''





#==============================================================================
#==== IMPORT STATEMENTS =======================================================
#==============================================================================

import os
import pandas as pd
import numpy as np
import re  # use this for one regular expressions function




#==============================================================================
#==== DATA-CLEANING FUNCTION DEFINITION =======================================
#==============================================================================


def clean_data(df, date):
    """
    This function takes in the raw data (as a pandas dataframe, pDF), and 
    performs necessary cleaning tasks.
    
    Parameters
    ----------
    df : pandas dataframe
        The original, uncleaned data 
    date : int
        Should be in the format "YYYYMMDD", e.g. March 1, 2018 should be 
        entered as 20180301
    
    Returns
    -------
    new_df : pandas dataframe
        The cleaned data
    """
    new_df = df
    # Only keep the lines with msg_type "E" or "C" or "P"
    new_df = new_df.loc[df["msg_type"].isin(["E", "P"])]
    
    # Let executed_shares = shares for missing values of executed_shares
    new_df['executed_shares'] = np.where(new_df.executed_shares.isna(),
          new_df.shares,
          new_df.executed_shares )
    # Let execution_price = price for missing values of execution_price
    new_df['execution_price'] = np.where(new_df.execution_price.isna(),
          new_df.price,
          new_df.execution_price )
    # To get decimal value for prices
    new_df["execution_price"] = new_df.execution_price / 10000.0
    new_df['price'] = new_df.price / 10000.0
    
    # Keep only these columns (keeping these based on exploratory data analysis
    # that I'd done)
    cols_to_keep = ['nanoseconds',
                    'buysell',
                    'executed_shares',
                    'execution_price',
                    'msg_type',
                    'match_number',
                    'order_number',
                    'price',
                    'shares',
                    'tracking_number']
    new_df = new_df[cols_to_keep]
    
    # One-hot encode the 'msg_type' and 'buysell' columns so that they can be
    # used in all ML algorithms
    new_df = pd.get_dummies(new_df)
    # Get rid of the 'buysell_B' column so we have only one column for the
    # response with [1=SELL] and [0=BUY]
    new_df = new_df.drop(columns=['buysell_B'])
    # Add the date column
    new_df['date'] = date

    return new_df



















#==============================================================================
#==== AUTOMATED DATE CLEANING USING clean_data FUNCTION =======================
#==============================================================================


# Specify the path where the data is located (this is the upper-most directory)
files_loc = 'C:/__JARED/_TradeSignResearch/all_Data/ITCH_CSV'

# Read all of the trading days (folders) into a list
folders = []
for folder in os.listdir(files_loc): folders.append(folder)

# Initialize total number of trades in the dataset
total_trades = 0

# We will clean the data within each folder (folder == trading day) before 
# moving onto the next folder
folder_count = 0
for folder in folders:    
    #print(folder, " : ", folder_count)    DELETE THIS AFTER DE-BUGGING
    # Initialize an empty list to store the names of the text files which
    # contain the trading info.
    txt_files = []
    # Define the folder path
    folder_path = files_loc + '/' + folders[folder_count]
    # This for loop creates the list of text files in whatever folder the
    # first for loop is currently in.
    for txt_name in os.listdir(folder_path): txt_files.append(txt_name)
    # Need to reset this counter variable for every new folder (in order to 
    # loop thru all .txt files in the current folder)
    txt_count = 0
    # Go thru all txt files in the current folder
    for txt_file in txt_files:
        # Set the path of the .txt file
        txt_path = folder_path + '/' + txt_files[txt_count]
        print(txt_path)
        #print(folders[folder_count])
        print(txt_files[txt_count])
        
        # Now, clean the data in each text file and write out to CSV.
        # Read in the dataframe
        df = pd.read_csv(filepath_or_buffer=txt_path, sep=',', header=0, low_memory=False)
        #cleaned_data = pd.DataFrame()

        # Create the value used to populate the 'date' column of the cleaned df
        orig_date = folders[folder_count]
        year = '20' + orig_date[4:]
        month_day = orig_date[:4]
        new_date = year + month_day

        
        # Run the data cleaning function on the dataframe of interest, store clean data
        cleaned_data = clean_data(df, new_date)
        
        # Increment the total number of trades by the amount in this data set.
        total_trades += len(cleaned_data)
        
        # Create the name of the cleaned folder we will be writing to
        ticker_name = txt_files[txt_count]
        ticker_name = re.sub(".txt", "", ticker_name)
        # Create the new path
        cleaned_path = 'C:/__JARED/_TradeSignResearch/all_Data/cleaned_data/' + ticker_name + '/' + new_date + ticker_name +'.csv'
        
        #print(cleaned_data)
        # This creates the name of the CSV file (nearly identical to the original
        # text file) and specifies the path to store it.
        #csv_path = re.sub('txt', 'csv', txt_path)
        # Write the newly cleaned data to a CSV in the specifed path.
        cleaned_data.to_csv(cleaned_path, index=False)
        print(cleaned_path)
        
        
        # Increment the counter variable to go to the next txt file in folder
        txt_count += 1
    # Increment the folder counter variable to go to the next folder.
    folder_count += 1




print("total # of data points: ", total_trades)
