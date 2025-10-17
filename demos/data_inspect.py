#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np

# **READING FROM A FILE:**

def load_cnsts_rodem(ifile: str, mask_pt: list):
    """Load constituents from an HDF5 file."""
    with h5py.File(ifile, "r") as f:

        cnsts=f["objects/jets/jet1_cnsts"][mask_pt,:,:3] #first three features

    mask = np.repeat(cnsts[:, :, 0] == 0, cnsts.shape[2])
    mask = mask.reshape(-1, cnsts.shape[1], cnsts.shape[2])
    cnsts = np.ma.masked_where(mask, cnsts)

    print(f"cnsts.shape={cnsts.shape}")
    #(num_jets,num_particles,num_particle_features)
    print(f"\n")

    return cnsts[:,:,0],cnsts[:,:,1],cnsts[:,:,2] #pt,eta,phi

def load_jets_rodem(ifile: str, mask_pt: list):
    with h5py.File(ifile, "r") as f:

        jets = f["objects/jets/jet1_obs"][mask_pt,:3] #first three features

    print(f"jets.shape={jets.shape}")
    #(num_jets,num_jet_features)
    print(f"\n")

    return jets[:,0],jets[:,1],jets[:,2] #pt,eta,phi

#-------------------------------------------------------------------------------

# **ARRANGE DATA IN A DICTIONARY:**

def read_data(input,mask_pt):
    
    jets={}

    #rows -> jets ; columns -> particles
    jets['part_pt'],jets['part_eta'],jets['part_phi']=load_cnsts_rodem(input,mask_pt)

    #flat arrays
    jets['jet_pt'],jets['jet_eta'],jets['jet_phi']=load_jets_rodem(input,mask_pt)

    jets['jet_nparticles']=jets['part_pt'].count(axis=1)

    print(f"The smallest jet_pt is: {np.min(jets['jet_pt'])}")
    print(f"The biggest jet_pt is: {np.max(jets['jet_pt'])}")
    print(f"\n")
    
    return jets

#-------------------------------------------------------------------------------

# **CREATE A HISTOGRAM AND CALCULATE ITS COVARIANCE MATRIX:**
    
def hist_cov(R,bins_r,weights):
    
    # **MAKING A HISTOGRAM:**
    
    n_jets=len(R)
    
    R_middles = (bins_r[1:]+bins_r[:-1])/2.
    num_R_bins=len(R_middles)

    weights=np.ma.array(weights/(n_jets*np.sum(weights,axis=1).reshape(-1,1)),
                        dtype=np.float64)
        
    W=np.sum(weights,axis=1)
    
    hist=np.zeros([num_R_bins],dtype=np.float64)
    hist, _ = np.histogram(R.compressed(),
                           bins=bins_r,weights=weights.compressed())
        
    #---------------------------------------------------------------------------
    
    # **COMPUTE THE COVARIANCE MATRIX:**
    
    H=np.zeros([n_jets,num_R_bins],dtype=np.float64)
    H2=np.zeros([n_jets,num_R_bins],dtype=np.float64)

    for J in range(n_jets):

        H[J,:], _ = np.histogram(R[J].compressed(),
                            bins=bins_r,weights=weights[J].compressed())
        H2[J,:], _ = np.histogram(R[J].compressed(),
                            bins=bins_r,weights=weights[J].compressed()**2)

    a=-H[:,:,np.newaxis]*np.ones(num_R_bins)

    mask=np.eye(num_R_bins,dtype=bool)

    a[:,mask]+=np.repeat(W,num_R_bins).reshape(a.shape[0],a.shape[1])

    a=a/(n_jets*(W ** 2)[:, np.newaxis, np.newaxis])

    b=a
    
    cov=np.zeros([num_R_bins,num_R_bins],dtype=np.float64)
    cov=np.einsum('mik,mjk,mk->ij', a, b, H2)

    """
    for i in range(num_R_bins):

        for j in range(num_R_bins):

            for J in range(n_jets):

                for k in range(num_R_bins):

                    if k==i:
                        a=(W[J]-H[J,i])/(n_jets*W[J]**2)
                    else:
                        a=-H[J,i]/(n_jets*W[J]**2)

                    if k==j:
                        b=(W[J]-H[J,j])/(n_jets*W[J]**2)
                    else:
                        b=-H[J,j]/(n_jets*W[J]**2)

                    cov[i,j]+=a*b*H2[J,k]
    """
    
    #---------------------------------------------------------------------------
    
    return hist,cov

