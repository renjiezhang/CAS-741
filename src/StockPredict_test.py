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
company_data_set=[]
TRANING_TESTING_RATIO=0.7
def data_input(file_name):
	data_set = pandas.read_csv(file_name)
	return data_set
# calculate price volatility array given company
def get_price_volatility(days_ahead,numDays, priceArray):
	# make price volatility array
	volatility_array = []
	moving_volatility_array = []
	for i in range(1, numDays+1):
		percent_change = 100 * (priceArray[i] - priceArray[i-1]) / priceArray[i-1]
		moving_volatility_array.append(percent_change)
	volatility_array.append(np.mean(moving_volatility_array))
	for i in range(numDays + 1, len(priceArray) - days_ahead):
		del moving_volatility_array[0]
		percent_change = 100 * (priceArray[i] - priceArray[i-1]) / priceArray[i-1]
		moving_volatility_array.append(percent_change)
		volatility_array.append(np.mean(moving_volatility_array))

	return volatility_array

# calculate momentum array
def get_momentum(days_ahead,numDays, priceArray):
	#global days_ahead
	# now calculate momentum
	momentum_array = []
	moving_momentum_array = []
	for i in range(1, numDays + 1):
		moving_momentum_array.append(1 if priceArray[i] > priceArray[i-1] else -1)
	momentum_array.append(np.mean(moving_momentum_array))
	for i in range(numDays+1, len(priceArray) - days_ahead):
		del moving_momentum_array[0]
		moving_momentum_array.append(1 if priceArray[i] > priceArray[i-1] else -1)
		momentum_array.append(np.mean(moving_momentum_array))

	return momentum_array

def predict(company,days_ahead, numDays, ndaq_volatility, ndaq_momentum):
	global stock_data
	global sc
	global TRANING_TESTING_RATIO
	#global days_ahead
	# get price volatility and momentum for this company
	
	company_data = data_input('dataset/'+company+'.csv')
	company_data = company_data.sort_values(by='Date', ascending=True)
	
	#company_data_set.append(company_data)
	company_prices = list(company_data['Close'])
	company_dates= list(company_data['Date'])
	volatility_array = get_price_volatility(days_ahead,numDays, company_prices)
	momentum_array =  get_momentum(days_ahead,numDays, company_prices)

	split_index =int(len(company_prices)*TRANING_TESTING_RATIO)
	split_index=split_index- numDays

	# since they are different lengths, find the min length
	if len(volatility_array) > len(ndaq_volatility):
		difference = len(volatility_array) - len(ndaq_volatility)
		del volatility_array[:difference]
		del momentum_array[:difference]

	elif len(ndaq_volatility) > len(volatility_array):
		difference = len(ndaq_volatility) - len(volatility_array)
		del ndaq_volatility[:difference]
		del ndaq_momentum[:difference]

	# create the feature vectors X
	feature_x = np.transpose(np.array([volatility_array, momentum_array, ndaq_volatility, ndaq_momentum]))

	# create the feature vectors Y
	feature_y = []
	for i in range(numDays, len(company_prices) - days_ahead):
		feature_y.append(1 if company_prices[i+days_ahead] > company_prices[i] else -1)

	# fix the length of Y if necessary
	if len(feature_y) > len(feature_x):
		difference = len(feature_y) - len(feature_x)
		del feature_y[:difference]

	# split into training and testing sets, 70% for training and 30% for testing
	X_train = np.array(feature_x[0:split_index]).astype('float64')
	X_test = np.array(feature_x[split_index:]).astype('float64')
	y_train = np.array(feature_y[0:split_index]).astype('float64')
	y_test = np.array(feature_y[split_index:]).astype('float64')

	# fit the model and calculate its accuracy
	rbf_svm = SVC(kernel='rbf')
	rbf_svm.fit(X_train, y_train)
	score = rbf_svm.score(X_test, y_test)
	print ('Accuracy : %f',(score))
	print('Result : %d ', (rbf_svm.predict([feature_x[-1]])))
	return score
	
def main():
	ndxtdf = data_input('dataset/NDAQ.csv')
	ndxtdf = ndxtdf.sort_values(by='Date', ascending=True)
	ndxt_prices = list(ndxtdf['Close'])
	ndxt_volatility_array =  get_price_volatility(1,5, ndxt_prices)
	ndxt_momentum_array =  get_momentum(1,5, ndxt_prices)
	result=predict('GOOG',1,5,ndxt_volatility_array,ndxt_momentum_array)
	print(result)
if __name__ == "__main__": 
	main()
