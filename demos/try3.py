import logging
from multiprocessing import Pool

def add(x, y):
	logging.basicConfig(level=logging.INFO)
	logging.info(f"argument ({x},{y}) in progress..")
	return x + y

def main():
	# List of tuples, where each tuple contains two arguments of the function
	input_data = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]

	# Create a pool of worker processes
	with Pool(processes=4) as pool:
		# Use starmap to pass each tuple from data as
		# arguments to the function
		sums = pool.starmap(add, input_data)

	# Print results
	print("Input:", input_data)
	print("Sums:", sums)

if __name__ == "__main__":
	main()
