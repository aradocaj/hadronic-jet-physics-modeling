import logging
from multiprocessing import Pool

def compute(x, y):
	logging.basicConfig(level=logging.INFO)
	logging.info(f"argument ({x},{y}) in progress..")
	return x * y, x + y

def main():
	# List of tuples, where each tuple contains two arguments of the function
	input_data = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]

	# Create a pool of worker processes
	with Pool(processes=4) as pool:
		# Use starmap to pass each tuple from data
		# as arguments to the function
		results = pool.starmap(compute, input_data)

	# Create empty lists to store the results separately
	products = []
	sums = []

	# Unpack the results manually using a for loop
	for product,summation in results:
		products.append(product)
		sums.append(summation)

	# Print results
	print("Input:", input_data)
	print("Results:", results)
	print("Products:", products)
	print("Sums:", sums)

if __name__ == "__main__":
	main()
