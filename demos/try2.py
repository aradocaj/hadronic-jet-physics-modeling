import logging
from multiprocessing import Pool

def compute(x):
	logging.basicConfig(level=logging.INFO)
	logging.info(f"argument {x} in progress..")
	return x * x, x + x

def main():
	# List of arguments to pass to the function
	input_data = [1, 2, 3, 4, 5]

	# Create a pool of worker processes
	with Pool(processes=4) as pool:
		# Use map to pass each element of data as
		# an argument to the function
		results = pool.map(compute, input_data)

	# Create empty lists to store the results separately
	squares = []
	sums = []

	# Unpack the results manually using a for loop
	for square,summation in results:
		squares.append(square)
		sums.append(summation)

	# Print results
	print("Input:", input_data)
	print("Results:", results)
	print("Squares:", squares)
	print("Sums:", sums)

if __name__ == "__main__":
	main()
