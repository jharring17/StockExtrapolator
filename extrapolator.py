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
import pandas_datareader as web 
import numpy as np 
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
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

# Checks to see if teh stock entered is a stock.
def  stockValid(stockName): 
    # Queries yahoo finance url for given stock.
    stockURL = 'http://www.finance.yahoo.com/quote/' + stockName
    response = requests.get(stockURL)
    # Quits program if page not found (invalid stock name).
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
train_data = scaled_data[0:training_data_len, :]

# Split data into x_train and y_train.
x_train = []
y_train = []

# Creates the x and y trains.
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 60:
        print("Here are the data trains: ")
        print(x_train)  # Contains the [0, 60] values.
        print(y_train)  # Contains the 61st value.
        print()

# Convert x and  y train to numpy arrays.
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data (converts to a 3D array).
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Establish LSTM network 
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False)) # No additonal models needed.
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model 
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing dataset.
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]     # All values we want the model to predict.

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert the data to numpy array.
x_test = np.array(x_test)

# Reshape the data to make it 3D.
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get models predicted price values.
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)  # Value of zero means predictions of model are exact.
print("RMSE: ", rmse)

# Plot the data.
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Visualize the data.
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Data', fontsize=18)
plt.xlabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# Show the actual and predicted prices.
display(valid)
