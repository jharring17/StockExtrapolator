###################################################################################################
#   Name        : extrapolator.py                                                                 #
#   Authors     : Jack Harrington && Sagarbir Bandesha                                            #
#   Date        : June 10, 2022                                                                   #
#   Version     : 1.0                                                                             #
#   Description : Stock extrapolation using Long Short Term Memory (LSTM) to create a graphical   #
#                 prediction of stock closing price.                                              #
###################################################################################################
from urllib import response
import numpy
import pandas
import math 
import datetime
import requests 
import urllib.request
import pandas_datareader as web # install current version using 'pip install pandas-datareader'
import numpy as np 
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler # install current version using 'pip install scikit-learn'
from sklearn.datasets import load_iris
from keras.models import Sequential 
from keras.layers import Dense, LSTM 
from IPython.display import display
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Checks to see if the date is of the correct format. 
def  dateValid(date): 
    #  Sets date format.
    date_string = date
    format = "%Y-%m-%d"
    
    # Checks the validity of the time format.
    try:
        datetime.datetime.strptime(date_string, format)
    # Quits if the date format is incorrect.
    except ValueError:
        print("Error: data must be of form: (yyyy-mm-dd)")
        quit()


def  stockValid(stockName): 
    #Queries yahoo finance url for given stock.
    stockURL = 'http://www.finance.yahoo.com/quote/' + stockName
    response = requests.get(stockURL)
    #Quits program if page not found (invalid stock name).
    if response.status_code != 200:
        print("Error: stock unavaliable from Yahoo Finance.")
        quit()



# Gets user information for stock extrapolation.
stockName = input("\nEnter a stock to extrapolate: ")
stockValid(stockName)
startDate = input("Enter start date (yyyy-mm-dd): ")
dateValid(startDate)
endDate = input("Enter an end date (yyyy-mm-dd): ")
dateValid(endDate)
print('\n')

# Prints the dataframe to screen between given dates. 
load_data = load_iris()
df = web.DataReader(stockName, data_source='yahoo', start=startDate, end=endDate);
display(df)

# Visualize closing price history.
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'], color="green")
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

# Create new dataframe for closing.
data = df.filter(['Close'])

# Convert the dataframe to numpy array.
dataset = data.values

# Get number of rows.
training_data_len = math.ceil(len(dataset) * 0.80)

# Scaling data.
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)   #! Watch for possible errors around data here (surrounding the input for this).
#print(scaled_data)

# Create the training dataset and scaled trainging dataset.
train_data = scaled_data[0:training_data_len:]

# Split data into x_train and y_train.
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60, 0])
    y_train.append(train_data[i, 0])
