import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)	# skipping column names
        i=1
        for row in csvFileReader:
            dates.append((row[0].replace('-','')))
            prices.append(float(row[1]))
            
    return

def predict_price(dates, prices, x):
    dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1

    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
  
    svr_rbf.fit(dates, prices) # fitting the data points in the models

    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')
   
    price=svr_rbf.predict(dates)
    plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints 
    plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
    #plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
   # plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
   
 
   
 
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Machine')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0]


get_data("GOOG.csv") # calling get_data method by passing the csv file to it

print ("Dates- ", dates)
print ("Prices- ", prices)

predicted_price = predict_price(dates, prices, 290)  
print ("\nThe stock open price for 29th Feb is:")
print ("RBF kernel: $", str(predicted_price[0]))
#print ("Linear kernel: $", str(predicted_price[1]))
#print ("Polynomial kernel: $", str(predicted_price[2]))
