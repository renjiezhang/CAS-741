import pandas
import numpy as np
#from sklearn import preprocessing
from sklearn.svm import SVC
#from sklearn import cross_validation
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


# read the data
startYear=2012
endYear=2018
companyList = ['AMZN','FB','GOOG','NFLX']
companyDataSet=[]

# calculate price volatility array given company
def GetPriceVolatility(daysAhead,numDays, priceArray):
	# make price volatility array
	volatilityArray = []
	movingVolatilityArray = []
	for i in range(1, numDays+1):
		percentChange = 100 * (priceArray[i] - priceArray[i-1]) / priceArray[i-1]
		movingVolatilityArray.append(percentChange)
	volatilityArray.append(np.mean(movingVolatilityArray))
	for i in range(numDays + 1, len(priceArray) - daysAhead):
		del movingVolatilityArray[0]
		percentChange = 100 * (priceArray[i] - priceArray[i-1]) / priceArray[i-1]
		movingVolatilityArray.append(percentChange)
		volatilityArray.append(np.mean(movingVolatilityArray))

	return volatilityArray

# calculate momentum array
def GetMomentum(daysAhead,numDays, priceArray):
	#global daysAhead
	# now calculate momentum
	momentumArray = []
	movingMomentumArray = []
	for i in range(1, numDays + 1):
		movingMomentumArray.append(1 if priceArray[i] > priceArray[i-1] else -1)
	momentumArray.append(np.mean(movingMomentumArray))
	for i in range(numDays+1, len(priceArray) - daysAhead):
		del movingMomentumArray[0]
		movingMomentumArray.append(1 if priceArray[i] > priceArray[i-1] else -1)
		momentumArray.append(np.mean(movingMomentumArray))

	return momentumArray

def Predict(companyPrices,daysAhead, numDays, ndaqVolatility, ndaqMomentum):
	global stockData
	global sc

	#global daysAhead
	# get price volatility and momentum for this company
	volatilityArray =  GetPriceVolatility(daysAhead,numDays, companyPrices)
	momentumArray =  GetMomentum(daysAhead,numDays, companyPrices)

	splitIndex =int(len(companyPrices)*0.7)
	splitIndex=splitIndex- numDays

	# since they are different lengths, find the min length
	if len(volatilityArray) > len(ndaqVolatility):
		difference = len(volatilityArray) - len(ndaqVolatility)
		del volatilityArray[:difference]
		del momentumArray[:difference]

	elif len(ndaqVolatility) > len(volatilityArray):
		difference = len(ndaqVolatility) - len(volatilityArray)
		del ndaqVolatility[:difference]
		del ndaqMomentum[:difference]

	# create the feature vectors X
	featureX = np.transpose(np.array([volatilityArray, momentumArray, ndaqVolatility, ndaqMomentum]))

	# create the feature vectors Y
	featureY = []
	for i in range(numDays, len(companyPrices) - daysAhead):
		featureY.append(1 if companyPrices[i+daysAhead] > companyPrices[i] else -1)

	# fix the length of Y if necessary
	if len(featureY) > len(featureX):
		difference = len(featureY) - len(featureX)
		del featureY[:difference]

	# split into training and testing sets, 70% for training and 30% for testing
	X_train = np.array(featureX[0:splitIndex]).astype('float64')
	X_test = np.array(featureX[splitIndex:]).astype('float64')
	y_train = np.array(featureY[0:splitIndex]).astype('float64')
	y_test = np.array(featureY[splitIndex:]).astype('float64')

	# fit the model and calculate its accuracy
	rbf_svm = SVC(kernel='rbf')
	rbf_svm.fit(X_train, y_train)
	score = rbf_svm.score(X_test, y_test)
	print ('Accuracy : %f',(score))
	print('Result : %d ', (rbf_svm.predict([featureX[-1]])))
	return score
	
def main():
	companyPrices=[1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]
	ndxtVolatilityArray=[8.45, 7.78, 7.217, 6.72, 6.299]
	ndxtMomentumArray=[1.0, 1.0, 1.0, 1.0, 1.0]
	Predict(companyPrices,daysAhead,numDayStock,ndxtVolatilityArray,ndxtMomentumArray)
if __name__ == "__main__": 
	main()
