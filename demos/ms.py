import os
import sys
sys.path.append('./machine/') #downloaded files are here

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from mcmc import * #downloaded mcmc.py file
from parallel import * #downloaded parallel.py file
from fit_prior import read_prior_par #downloaded read_prior_par.py file

print(sys.version) #3.12.3
print(f"\n",flush=True)

num_points = 20
x1 = np.random.uniform(-5, 5, num_points)
x2 = np.random.uniform(-5, 5, num_points)

# True function
y_true = 0.4 * x1**2 - 3.2167 * x2**3

# Add noise (Gaussian noise with mean 0 and standard deviation 1)
noise = np.random.normal(0, 1, size=num_points)
y_noisy = y_true + noise

x=np.zeros([num_points,2])
x[:,0]=x1
x[:,1]=x2

x=pd.DataFrame(data=x,columns=['x1','x2'])
y=pd.Series(data=y_noisy)
XLABS=['x1','x2']

# Read the hyperparameters for the prior from downloaded file
file='./machine/final_prior_param_sq.named_equations.nv13.np13.2016-09-01 17_05_57.196882.dat'
prior_par = read_prior_par(file)

# Set the temperatures for the parallel tempering
Ts = [1] + [1.04**k for k in range(1, 20)]

# Initialize the parallel machine scientist
pms = Parallel(
	Ts,
	variables=XLABS,
	parameters=['a%d' % i for i in range(13)],
	x=x, y=y,
	prior_par=prior_par,
)

# Number of MCMC steps
nstep = 3000

# MCMC
description_lengths, mdl, mdl_model = [], np.inf, None
for i in range(nstep):
	# MCMC update
	pms.mcmc_step() # MCMC step within each T
	pms.tree_swap() # Attempt to swap two randomly selected consecutive temps
	# Add the description length to the trace
	description_lengths.append(pms.t1.E)
	# Check if this is the MDL expression so far
	if pms.t1.E < mdl:
		mdl, mdl_model = pms.t1.E, deepcopy(pms.t1)
	# Keep track of the progress
	#if (i+1) % 100 == 0:
	#    print(f"Step {i+1}/{nstep}")
	#    print(f"\n",flush=True)

print("Best model:\t", mdl_model)
print("Desc. length:\t", mdl)
print("Parameter values:")
print(mdl_model.par_values)
print(f"\n",flush=True)

if not os.path.exists("./Plots"):
	os.makedirs("./Plots")

plt.figure(figsize=(6, 6))
plt.scatter(mdl_model.predict(x), y)
plt.plot((y_noisy.min(),y_noisy.max()),(y_noisy.min(),y_noisy.max()))
plt.xlabel('MDL model predictions', fontsize=14)
plt.ylabel('Actual values', fontsize=14)
plt.savefig("./Plots/model.jpg")
