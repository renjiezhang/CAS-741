import pandas
# read the data
def DataInput(file_name):
	data_set = pandas.read_csv(file_name)
	return data_set
def main():
	# read the tech sector data
	print(DataInput('dataset/NDAQ.csv'))
if __name__ == "__main__": 
	main()