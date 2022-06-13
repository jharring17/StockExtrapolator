###################################################################################################
#   Name        : extrapolator.py                                                                 #
#   Authors     : Jack Harrington && Sagarbir Bandesha                                            #
#   Date        : June 10, 2022                                                                   #
#   Version     : 1.0                                                                             #
#   Description : Stock extrapolation using Long Short Term Memory (LSTM) to create a graphical   #
#                 prediction of stock closing price.                                              #
###################################################################################################
import numpy
import pandas
import math 
import datetime
import urllib.request
import pandas_datareader as web # install current version using 'pip install pandas-datareader'
import numpy as np 
import pandas as pd
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
    format = "%Y-%m-d"
    
    # Checks the validity of the time format.
    try:
        datetime.datetime.strptime(date_string, format)
    # Quits if the date format is incorrect.
    except ValueError:
        print("Data must be of form: (yyyy-mm-dd)")
        quit()

# Gets user information for stock extrapolation.
stockName = input("Enter a stock to extrapolate: ")
startDate = input("Enter start date (yyyy-mm-dd): ")
dateValid(startDate)
endDate = input("Enter an end date (yyyy-mm-dd): ")
dateValid(endDate)

# Prints the data to screen between given dates. 
data = load_iris()
df = web.DataReader(stockName, data_source='yahoo', start=startDate, end=endDate);
display(df)

# start : 2012-01-05
# end : 2019-01-04