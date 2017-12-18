import pandas
# read the data
def DataInput(filename):
	dataSet = pandas.read_csv(filename)
	return dataSet
def main():
	# read the tech sector data
	print(DataInput('dataset/NDAQ.csv'))
if __name__ == "__main__": 
	main()