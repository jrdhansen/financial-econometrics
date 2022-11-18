'''
    File name: automated_BRNN_v0.py
    Author: Jared Hansen
    Date created: 04/19/2019
    Date last modified: 04/19/2019
    Python Version: 3.6.4

======================================================================================
CODE FOR INITIAL BIDIRECTIONAL RECURRENT NEURAL NETWORK TO CLASSIFY TRADE SIGNS (ITCH)
======================================================================================
'''






# This site is where I initially got the code from. Looks like there are some
# other fantastic scripts!
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py



#==============================================================================
#==== IMPORT STATEMENTS =======================================================
#==============================================================================

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn import preprocessing

     




        
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#==============================================================================
#==== CLASS DEFINITION FOR BRNN (many-to-one) =================================
#==============================================================================
# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
        # Inherit activation functions for designing network.
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = self.Sigmoid(out)
        return out





#==============================================================================
#==== FUNCTION TO FIT MODEL, GENERATE TEST ACCURACY ===========================
#==============================================================================
    
def calc_test_acc(df_path):
    '''
    Takes in a dataframe's path, fits a BRNN and returns the test accuracy.
    '''
    
    
    # Read into a Pandas dataframe
    df = pd.read_csv(df_path)
    # Define which column is the response ('target') variable
    target = pd.DataFrame(df['buysell_S'])
    # Now remove the target column from the dataframe
    del df['buysell_S']
    
    #----- NORMALIZE THE DATA ACROSS EACH FEATURE ---------------------------------
    def standardize_cols(data, colnames):
        """
        Standardizes ((x - mean)/sd) the columns of <df> specified by <indices>
        
        Parameters
        ----------
        df : a pandas dataframe
        colnames : a list indicating the names of the columns to be standardized
        
        Returns
        --------
        the resulting dataframe with specified columns standardized
        """
        scaler = preprocessing.StandardScaler()
        for name in colnames:
            col_scaled = scaler.fit_transform(np.array(data[name]).reshape(-1,1))
            data[name] = col_scaled
        return(data)
        
    # Example
    # 'df' in the example below should be a pandas dataframe
    df_scaled = standardize_cols(data = df, colnames = ['date',
                                                       'executed_shares',
                                                       'execution_price',
                                                       'match_number',
                                                       'nanoseconds',
                                                       'order_number',
                                                       'price',
                                                       'shares',
                                                       'tracking_number'])
    #df_scaled.head(5)
    
    '''
    # Tried dropping columns that aren't as predictive
    cols_to_drop = ['msg_type_C', 'msg_type_E', 'msg_type_P', 'execution_price']
    df_scaled = df_scaled.drop(cols_to_drop, axis=1)
    '''
    
    df = df_scaled    
    
    # Hyper-parameters
    sequence_length = 1
    #input_size = 11
    input_size = len(df.columns)
    hidden_size = 128
    num_layers = 2
    num_classes = 1
    batch_size = 50
    num_epochs = 4
    learning_rate = 0.003
    
    #train = torch.utils.data.TensorDataset(torch.Tensor(np.array(df)), torch.LongTensor(np.array(target)))
    # Convert the dataframe to a tensor dataset, first giving the predictors array then the target array
    train = torch.utils.data.TensorDataset(torch.Tensor(np.array(df)), torch.Tensor(np.array(target)))
    # Now use the DataLoader function to store the data in the train_loader object so that we can re-use
    # code from one of the MNIST assignment problems.
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
    
    #===== Create [validation set + new training set + test set] ==================
    # Per the prompt instructions we need to create a validation data set.
    # For this, I'm just going to take the 70% of training samples as train_small,
    # leaving the remaining 30% as the validation data. After iteratively tuning
    # using the training and validation sets, I'll train on the full train_dataset.
    # Since the train_loader shuffled the data, we don't need to re-shuffle again.
    
    # How many observations in the data we've read in?
    len_train = len(train)
    # Define how many obs will be in the train_small dataset
    len_train_big = round(len_train * 0.85)
    # Define the length of the val_test dataset (together, will split further)
    len_test = len_train - len_train_big
    # Split the original training data into a train_big set and a test set
    train_big, test = torch.utils.data.random_split(train, [len_train_big, len_test])
    
    # Define how many obs will be in the validation dataset
    len_val = round(len_train * 0.15)
    # Define how many obs will be in the train_small dataset
    len_train_small = len_train_big - len_val
    # Now split the train_big set into validation and train_small sets
    validation, train_small = torch.utils.data.random_split(train_big, [len_val, len_train_small]) 
    
    
    # Load in the new (smaller) training data
    new_train_loader = torch.utils.data.DataLoader(dataset=train_small,
                                                   batch_size=batch_size,
                                                   shuffle=False)
    # Load in the validation data
    val_loader = torch.utils.data.DataLoader(dataset=validation,
                                             batch_size=batch_size,
                                             shuffle=False)
    # Load in the larger training data (for final training after validation)
    train_loader = torch.utils.data.DataLoader(dataset=train_big,
                                                   batch_size=batch_size,
                                                   shuffle=False)
    # Load in the test data
    test_loader = torch.utils.data.DataLoader(dataset=test,
                                             batch_size=batch_size,
                                             shuffle=False)
    
    
    # Instantiate the model
    model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
    
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100*(correct/total)
    
        print('Test Accuracy of the model on the test images: {:.4f} %'.format(accuracy)) 
    
    return accuracy
        





#==============================================================================
#==== AUTOMATE BRNN FITTING FOR EACH TRADING DAY, WRITE OUT ACCURACIES ========
#==============================================================================
  
# Specify the path where the data is located (this is the upper-most directory)
files_loc = 'C:/__JARED/_TradeSignResearch/all_Data/cleaned_data'

# Read all of the trading days (folders) into a list
folders = []
for folder in os.listdir(files_loc): folders.append(folder)


# We will clean the data within each folder (folder == trading day) before 
# moving onto the next folder
folder_count = 0
for folder in folders:
    # Initialize an empty list to store the accuracies for each day.
    accs = []    
    # Initialize an empty list to store the names of the csv files which
    # contain the trading info.
    csv_files = []
    # Define the folder path
    folder_path = files_loc + '/' + folders[folder_count]
    # This for loop creates the list of text files in whatever folder the
    # first for loop is currently in.
    for csv_name in os.listdir(folder_path): csv_files.append(csv_name)
    # Need to reset this counter variable for every new folder (in order to 
    # loop thru all .csv files in the current folder)
    csv_count = 0
    # Go thru all txt files in the current folder
    for csv_file in csv_files:
        # Set the path of the .txt file
        csv_path = folder_path + '/' + csv_files[csv_count]
        print(csv_path)
        #print(folders[folder_count])
        print(csv_files[csv_count])
        # Read in the dataframe
        df = pd.read_csv(filepath_or_buffer=csv_path, sep=',', header=0, low_memory=False)
        # Fit the model, calculate accuracy, store accuracy
        acc = calc_test_acc(df_path)  
        # Append acc to accs list
        accs.append(acc)

        # Increment the counter variable to go to the next txt file in folder
        csv_count += 1
        
    # Increment the folder counter variable to go to the next folder.
    folder_count += 1
    # Write out the values of [the acc at the end of each csv model] to a file
    file_path = folder_path
    with open(file_path + '_test_ACCs' + '.txt', 'w') as f:
        for item in accs:
            f.write("%s\n" % item) 
    
    
    
    
#==============================================================================
#==== AUTOMATE BRNN FITTING FOR EACH AGGREGATED TICKER, WRITE OUT ACCURACIES ==
#==============================================================================
  
# Specify the path where the data is located (this is the upper-most directory)
files_loc = 'C:/__JARED/_TradeSignResearch/all_Data/aggregated_data'


# Initialize an empty list to store the accuracies for each ticker.
accs = []    
# Initialize an empty list to store the names of the csv files which
# contain the trading info.
csv_files = []
# Define the folder path
folder_path = files_loc
# This for loop creates the list of text files in whatever folder the
# first for loop is currently in.
for csv_name in os.listdir(folder_path): csv_files.append(csv_name)
# Need to reset this counter variable for every new folder (in order to 
# loop thru all .csv files in the current folder)
csv_count = 0
# Go thru all txt files in the current folder
for csv_file in csv_files:
    # Set the path of the .txt file
    csv_path = folder_path + '/' + csv_files[csv_count]
    print(csv_path)
    #print(folders[folder_count])
    print(csv_files[csv_count])
    # Read in the dataframe
    df = pd.read_csv(filepath_or_buffer=csv_path, sep=',', header=0, low_memory=False)
    # Fit the model, calculate accuracy, store accuracy
    acc = calc_test_acc(df_path)  
    # Append acc to accs list
    accs.append(acc)

    # Increment the counter variable to go to the next txt file in folder
    csv_count += 1
    
# Increment the folder counter variable to go to the next folder.
folder_count += 1
# Write out the values of [the acc at the end of each csv model] to a file
file_path = folder_path
with open(files_loc + '/agg_test_ACCs' + '.txt', 'w') as f:
    for item in accs:
        f.write("%s\n" % item) 




