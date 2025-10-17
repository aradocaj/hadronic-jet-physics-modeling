import logging
from multiprocessing import Pool

def square(x):
	logging.basicConfig(level=logging.INFO)
	logging.info(f"argument {x} in progress..")
	return x * x

def main():
	# List of arguments to pass to the function
	input_data = [1, 2, 3, 4, 5]

	# Create a pool of worker processes
	with Pool(processes=4) as pool:
		# Use map to pass each element of data
		# as an argument to the function
		squared = pool.map(square, input_data)

	# Print results
	print("Input:", input_data)
	print("Squared:", squared)

if __name__ == "__main__":
	main()
