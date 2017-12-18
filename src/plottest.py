import pandas
import numpy as np
#from sklearn import preprocessing
from sklearn.svm import SVC
#from sklearn import cross_validation
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


# read the data
def DataInput(file_name):
	data_set = pandas.read_csv(file_name)
	return data_set

startYear=2012
endYear=2018
companyList = ['AMZN','FB','GOOG','NFLX']
company_data_set=[]
def Plot():
	global companyList
	global company_data_set
	
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
	Plot()
if __name__ == "__main__": 
	main()
