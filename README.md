# hadronic-jet-physics-modeling
Implementation of data processing, visualization, and probabilistic modeling of hadronic jet momentum distributions using simulated and measured LHC datasets.

## Project Summary
First, I implemented the code to read JetClass, Rodem and Aspen Open Jets datasets of simulated and measured hadronic jets at the LHC. Visualization of the data is given too.

Next, I constructed probabilistic model descriptions of momentum distributions within jets, both in 1D and 2D projections, for both binned and unbinned data. In particular, I implemented the full covariance matrix formalism for Poissonian distributed data.

I then explored several data preprocessing tools, including Quantile Transformations and Box-Cox Transformation, to try fit the data in an unconventional way. Kernel Density Estimation was tested as well to see how it works and if it can be useful in any way. Advanced tool for finding analytic function of the data, Bayesian Machine Scientist, was used to explore potential options. The final solution involved a fit to the parametric family of exponential distributions.

I performed several cross-checks of my methodology and compiled a detailed report of my work and results. The main deliverables are functioning Python code packages performing the required tasks on preprocessed data. The packages were run on the local computing cluster and a local desktop computer.
