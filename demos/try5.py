import logging
import numpy as np
from multiprocessing import Pool

# Function to compute the matrices
def compute_matrix(x, y, n):

	logging.basicConfig(level=logging.INFO)
	logging.info(f"argument ({x},{y}) in progress..")

	# Create a (n x n) matrix filled with the same value x * y
	product_matrix = np.full((n, n), x * y)

	# Create a (n x n) matrix filled with the same value x + y
	sum_matrix = np.full((n, n), x + y)

	return product_matrix, sum_matrix

def main():
	# List of tuples,
	# where each tuple contains the first two arguments of the function
	input_data = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
	input_data_len=len(input_data)

	# Size of each matrix (2x2 matrix)
	n = 2

	# Prepare the list of full arguments (x, y, n) for each function call
	args=[(*input_data[i],n) for i in range(input_data_len)]

	# Create a pool of worker processes
	with Pool(processes=4) as pool:
		# Use starmap to distribute work across the pool of processes
		results=pool.starmap(compute_matrix,args)

	# Create a big matrix where each 'product_matrix'
	# will be placed as a separate block on the diagonal
	# (note: small matrices are already computed and stored in 'results')
	# - allocating extra memory here for combining into one large matrix
	final_product_matrix=np.zeros([input_data_len*n,input_data_len*n])

	# Create a big matrix where each 'sum_matrix'
	# will be placed as a separate block on the diagonal
	# (note: small matrices are already computed and stored in 'results')
	# - allocating extra memory here for combining into one large matrix
	final_sum_matrix=np.zeros([input_data_len*n,input_data_len*n])

	# Placing product_matrices and sum_matrices
	# as blocks on the diagonal of a corresponding matrix
	idx=0
	for product_matrix,sum_matrix in results:
		index1=idx*n
		index2=(idx+1)*n

		# Copy product_matrix into the diagonal block of final_product_matrix
		final_product_matrix[index1:index2,index1:index2]=product_matrix

		# Copy sum_matrix into the diagonal block of final_sum_matrix
		final_sum_matrix[index1:index2,index1:index2]=sum_matrix
		idx+=1

	# Print the final matrices
	print("Input:", input_data)
	print("Product block-diagonal matrix:")
	print(final_product_matrix)
	print("Sum block-diagonal matrix:")
	print(final_sum_matrix)

if __name__ == "__main__":
	main()

