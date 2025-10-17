import logging
import numpy as np
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory

# Function to compute the matrices
def compute_matrix(x, y, idx, n, input_data_len ,shm_product_matrix_name, shm_sum_matrix_name):

	logging.basicConfig(level=logging.INFO)
	logging.info(f"argument ({x},{y}) in progress..")

	# Attach to the existing shared memory block
	existing_shm_product_matrix = SharedMemory(name=shm_product_matrix_name)

	# Create a NumPy array that wraps the shared memory buffer
	product_matrix = np.ndarray((n*input_data_len,n*input_data_len),
			dtype=np.float64,buffer=existing_shm_product_matrix.buf)

	# Attach to the existing shared memory block
	existing_shm_sum_matrix = SharedMemory(name=shm_sum_matrix_name)

	# Create a NumPy array that wraps the shared memory buffer
	sum_matrix = np.ndarray((n*input_data_len,n*input_data_len),
			dtype=np.float64,buffer=existing_shm_sum_matrix.buf)

	index1=idx*n
	index2=(idx+1)*n
	# Fill the block on the diagonal with the same value x * y
	product_matrix[index1:index2,index1:index2] = np.full((n, n), x * y)

	# Fill the block on the diagonal with the same value x + y
	sum_matrix[index1:index2,index1:index2] = np.full((n, n), x + y)

def main():
	# List of tuples,
	# where each tuple contains the first two arguments of the function
	input_data = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
	input_data_len=len(input_data)

	# Size of each matrix (2x2 matrix)
	n = 2

	# Create shared memory for the big matrix
	# Each float64 takes 8 bytes
	shm_product_matrix = SharedMemory(
		create=True,size=n*input_data_len*n*input_data_len*8)

	# View into the shared memory
	product_matrix = np.ndarray((n*input_data_len,n*input_data_len),
		dtype=np.float64,buffer=shm_product_matrix.buf)

	# Create shared memory for the big matrix
	# Each float64 takes 8 bytes
	shm_sum_matrix = SharedMemory(
		create=True,size=n*input_data_len*n*input_data_len*8)

	# View into the shared memory
	sum_matrix = np.ndarray((n*input_data_len, n*input_data_len),
		dtype=np.float64,buffer=shm_sum_matrix.buf)

	# Prepare the list of full arguments
	# (x,y,idx,n,input_data_len,shm_product_matrix.name,shm_sum_matrix.name)
	# for each function call:
	args=[(*input_data[idx],idx,n,input_data_len,
		shm_product_matrix.name,shm_sum_matrix.name)
		for idx in range(input_data_len)]

	# Create a Pool of processes
	with Pool(processes=4) as pool:
		# Use starmap to distribute work across the pool of processes
		pool.starmap(compute_matrix,args)

	# Print the final matrices
	print("Input:", input_data)
	print("Product block-diagonal matrix:")
	print(product_matrix)
	print("Sum block-diagonal matrix:")
	print(sum_matrix)

	# Clean up the shared memory once everything is done
	shm_product_matrix.close()  # Detach from the shared memory
	shm_product_matrix.unlink()  # Release the shared memory block

	# Clean up the shared memory once everything is done
	shm_sum_matrix.close()  # Detach from the shared memory
	shm_sum_matrix.unlink()  # Release the shared memory block

if __name__ == "__main__":
	main()
