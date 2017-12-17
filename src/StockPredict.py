import pandas
import numpy as np
#from sklearn import preprocessing
from sklearn.svm import SVC
#from sklearn import cross_validation
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pyspark

# read the data
def DataInput(filename):
	dataSet = pandas.read_csv(filename)
	return dataSet

startYear=2012
endYear=2018
companyList = ['AMZN','FB','GOOG','NFLX']
companyDataSet=[]
sc = pyspark.SparkContext('local[*]')
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

def Predict(company,daysAhead, numDays, ndaqVolatility, ndaqMomentum):
	global stockData
	global sc

	#global daysAhead
	# get price volatility and momentum for this company
	
	companyData = DataInput('dataset/'+company+'.csv')
	companyData = companyData.sort_values(by='Date', ascending=True)
	
	#companyDataSet.append(companyData)
	companyPrices = list(companyData['Close'])
	companyDates= list(companyData['Date'])
	volatilityArray =  sc.parallelize(GetPriceVolatility(daysAhead,numDays, companyPrices)).collect()
	momentumArray =  sc.parallelize(GetMomentum(daysAhead,numDays, companyPrices)).collect()

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

def Plot():
	global companyList
	global companyDataSet
	
	years = mdates.YearLocator()   # every year
	months = mdates.MonthLocator()  # every month
	yearsFmt = mdates.DateFormatter('%Y')	
	fig, ax = plt.subplots()
	
	for company in companyList:
		companyData=DataInput('dataset/'+company+'.csv')
		dateList=list(companyData['Date'])
		prices=list(companyData['Close'])
		dates=[]
		for dat in dateList:
			dates.append(datetime.datetime.strptime(dat, "%Y-%m-%d"))	
		ax.plot(dates, prices,label=company)	
		
	legend = ax.legend(loc='best', shadow=False, fontsize='large')
	# format the ticks
	ax.xaxis.set_major_locator(years)
	ax.xaxis.set_major_formatter(yearsFmt)
	ax.xaxis.set_minor_locator(months)
	plt.xlabel('Date')
	plt.ylabel('Price')	
	datemin = datetime.date(startYear, 11, 1)
	datemax = datetime.date(endYear, 2, 1)
	ax.set_xlim(datemin, datemax)
	ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
	
	ax.grid(True)
	
	# rotates and right aligns the x labels, and moves the bottom of the
	# axes up to make room for them
	fig.autofmt_xdate()
	
	plt.show()
	
def main():
	global stockData
	global companyList
	global sc

	# read the tech sector data
	ndxtdf = DataInput('dataset/NDAQ.csv')
	ndxtdf = ndxtdf.sort_values(by='Date', ascending=True)
	ndxtPrices = list(ndxtdf['Close'])
	ndxtDates=list(ndxtdf['Date'])

	# we want to predict where it will be on the next day based on X days previous
	numDaysArray = [5,20,60,270] # day, week, month, quarter, year
	numDayAheadArray=[1,5,20,60]

	# iterate the company and days
	for numDayIndex in numDaysArray:
		for numDayStock in numDaysArray:
			for daysAhead in numDayAheadArray:
				print ('days Ahead: %d &  #days NASDAQ :%d  &  #days Stock %d' % (daysAhead,numDayIndex,numDayStock))
				ndxtVolatilityArray =  sc.parallelize(GetPriceVolatility(daysAhead,numDayStock, ndxtPrices)).collect()
				ndxtMomentumArray =  sc.parallelize(GetMomentum(daysAhead,numDayStock, ndxtPrices)).collect()
				
				for company in companyList:
					print ('Company : '+company)
					Predict(company,daysAhead,numDayStock,ndxtVolatilityArray,ndxtMomentumArray)

	Plot()
if __name__ == "__main__": 
	main()
