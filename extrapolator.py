###################################################################################################
#   Name        : extrapolator.py                                                                 #
#   Authors     : Jack Harrington && Sagarbir Bandesha                                            #
#   Date        : June 10, 2022                                                                   #
#   Version     : 1.0                                                                             #
#   Description : Stock extrapolation using Long Short Term Memory (LSTM) to create a graphical   #
#                 prediction of stock closing price.                                              #
###################################################################################################
from msilib import sequence
import numpy
import pandas
import math 
import pandas_datareader as web # install current version using 'pip install pandas-datareader'
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler # install current version using 'pip install scikit-learn'
from keras.models import Sequential 
from keras.layers import Dense, LSTM 
import matplotlib.pyplot as plt
plt.style.use('darkmode')