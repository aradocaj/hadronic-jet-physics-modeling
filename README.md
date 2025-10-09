# hadronic-jet-physics-modeling
Implementation of data processing, visualization, and probabilistic modeling of hadronic jet momentum distributions using simulated and measured LHC datasets.

## Project Summary
First, I implemented the code to read JetClass, Rodem and Aspen Open Jets datasets of simulated and measured hadronic jets at the LHC.
Visualization of the data is given too.

Next, I constructed probabilistic model descriptions of momentum distributions within jets, both in 1D and 2D projections, for both binned and unbinned data. 
In particular, I implemented the full covariance matrix formalism for Poissonian distributed data.

I then explored several machine learning methods for performing inference on these distributions, including
Bayesian Machine Scientist, Quantile transformations, Box-Cox transformations and Kernel density estimation. 
The final solution involved a fit to the parametric family of exponential distributions.

I performed several cross-checks of my methodology and compiled a detailed report of my work and results.
The main deliverables are functioning python code packages performing the required tasks on preprocessed data.
The packages were run on the local computing cluster and a local desktop computer.
