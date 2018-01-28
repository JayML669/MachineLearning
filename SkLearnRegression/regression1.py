import pandas as pd
import quandl
import math
import datetime
import os
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

# Get the dataframe of google stock prices
df = quandl.get('WIKI/GOOGL')

# Get rid of non-necessary parts of the dataframe
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# Compute the change from high to close and open to close stock prices
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100.0

# Again getting rid of unneccesary parts of the dataframe
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# Setting the vakues to be forecasted
forecast_col = 'Adj. Close'

# Filling empty spots in the dataset
df.fillna(-99999, inplace=True)

# Getting the number of days to forecat out
forecast_out = int(math.ceil(0.01*len(df)))

# Creating a new column of labels wghich is the adj. close for forecast_out days in the future
df['label'] = df[forecast_col].shift(-forecast_out)

# Setting the features as everything but the lable column
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Removing empty datapoints
df.dropna(inplace=True)

# Creating an array of the lables
y = np.array(df['label'])

# Setting the training and testing values for X and y
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Testing to see if a pickle of the model allready exists and loading it if so
if os.path.exists('SkLearnRegression\linearRegression.pickle'):
    print('Loading The Model...')
    pickle_in = open('SkLearnRegression\linearRegression.pickle', 'rb')
    clf = pickle.load(pickle_in)
else:
    print('Training The Model...')
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    with open('SkLearnRegression\linearRegression.pickle', 'wb') as f:
        pickle.dump(clf, f)


# Determining the accuracy of the model
accuracy = clf.score(X_test, y_test)

# Predicting the forecats using the model
forecast_set = clf.predict(X_lately)

# printing predictions and model data
print(forecast_set, accuracy, forecast_out)

# Creating an empty column of forecasts
df['Forecast'] = np.nan

# Getting the unix value of the last entry in the dataframe
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# Creating new entries in the dataset of predictions and filling everything but the forecast column witt NaN
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

# Plotting the data
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
