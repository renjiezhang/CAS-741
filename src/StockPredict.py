#the coding style of this file is based on Google Python Style
import pandas
import numpy as np
#from sklearn import preprocessing
from sklearn.svm import SVC
#from sklearn import cross_validation
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pyspark
import time
# read the data
def data_input(file_name):
	data_set = pandas.read_csv(file_name)
	return data_set

START_YEAR=2012
END_YEAR=2018
COMPANY_LIST = ['AMZN','FB','GOOG','NFLX']
company_data_set=[]
sc = pyspark.SparkContext('local[*]')
TRANING_TESTING_RATIO=0.7
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
	volatility_array =  sc.parallelize(get_price_volatility(days_ahead,numDays, company_prices)).collect()
	momentum_array =  sc.parallelize(get_momentum(days_ahead,numDays, company_prices)).collect()

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

def plot():
	global COMPANY_LIST
	global company_data_set
	
	years = mdates.YearLocator()   # every year
	months = mdates.MonthLocator()  # every month
	yearsFmt = mdates.DateFormatter('%Y')	
	fig, ax = plt.subplots()
	
	for company in COMPANY_LIST:
		company_data=data_input('dataset/'+company+'.csv')
		date_list=list(company_data['Date'])
		prices=list(company_data['Close'])
		dates=[]
		for dat in date_list:
			dates.append(datetime.datetime.strptime(dat, "%Y-%m-%d"))	
		ax.plot(dates, prices,label=company)	
		
	legend = ax.legend(loc='best', shadow=False, fontsize='large')
	# format the ticks
	ax.xaxis.set_major_locator(years)
	ax.xaxis.set_major_formatter(yearsFmt)
	ax.xaxis.set_minor_locator(months)
	plt.xlabel('Date')
	plt.ylabel('Price')	
	datemin = datetime.date(START_YEAR, 11, 1)
	datemax = datetime.date(END_YEAR, 2, 1)
	ax.set_xlim(datemin, datemax)
	ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
	
	ax.grid(True)
	
	# rotates and right aligns the x labels, and moves the bottom of the
	# axes up to make room for them
	fig.autofmt_xdate()
	
	plt.show()
	
def main():
	global stock_data
	global COMPANY_LIST
	global sc
	start_time = time.time()
	# read the tech sector data
	ndxtdf = data_input('dataset/NDAQ.csv')
	ndxtdf = ndxtdf.sort_values(by='Date', ascending=True)
	ndxt_prices = list(ndxtdf['Close'])
	

	# we want to predict where it will be on the next day based on X days previous
	NUM_DAYS_ARRAY = [5,20,60,270] # day, week, month, quarter, year
	NUM_DAYS_AHEAD_ARRAY=[1,5,20,60]

	# iterate the company and days
	for num_day_index in NUM_DAYS_ARRAY:
		for num_day_stock in NUM_DAYS_ARRAY:
			for days_ahead in NUM_DAYS_AHEAD_ARRAY:
				print ('days Ahead: %d &  #days NASDAQ :%d  &  #days Stock %d' % (days_ahead,num_day_index,num_day_stock))
				ndxt_volatility_array =  sc.parallelize(get_price_volatility(days_ahead,num_day_stock, ndxt_prices)).collect()
				ndxt_momentum_array =  sc.parallelize(get_momentum(days_ahead,num_day_stock, ndxt_prices)).collect()
				
				for company in COMPANY_LIST:
					print ('Company : '+company)
					predict(company,days_ahead,num_day_stock,ndxt_volatility_array,ndxt_momentum_array)
	print("--- %s seconds ---" % (time.time() - start_time))
	plot()
if __name__ == "__main__": 
	main()
