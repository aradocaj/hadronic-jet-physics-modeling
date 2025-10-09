#!/usr/bin/env python
# coding: utf-8

import h5py #aspen,rodem
import vector #aspen
import awkward as ak #aspen

import logging
import numpy as np
from enum import StrEnum
from scipy.optimize import curve_fit
from scipy.special import eval_legendre
from inspect import signature
from multiprocessing.shared_memory import SharedMemory

def load_cnsts_aspen(ifile: str, mask_pt: list):
	"""Load constituents from an HDF5 file."""
	with h5py.File(ifile, "r") as f:

		PFCands = f["PFCands"][mask_pt,:,:4] #px,py,pz,E

	p4_cnsts=vector.zip({'px':PFCands[:,:,0],
		             'py':PFCands[:,:,1],
		             'pz':PFCands[:,:,2],
		             'energy':PFCands[:,:,3]})

	cnsts=np.zeros([PFCands.shape[0],PFCands.shape[1],3])
	cnsts[:,:,0]=ak.to_numpy(p4_cnsts.pt,allow_missing=True)
	cnsts[:,:,1]=ak.to_numpy(p4_cnsts.eta,allow_missing=True)
	cnsts[:,:,2]=ak.to_numpy(p4_cnsts.phi,allow_missing=True)

	mask = np.repeat(cnsts[:, :, 0] == 0, cnsts.shape[2])
	mask = mask.reshape(-1, cnsts.shape[1], cnsts.shape[2])
	cnsts = np.ma.masked_where(mask, cnsts)

	print(f"cnsts.shape={cnsts.shape}")
	print(f"\n",flush=True)

	return cnsts[:,:,0], cnsts[:,:,1], cnsts[:,:,2] #pt,eta,phi

def load_jets_aspen(ifile: str, mask_pt: list):
	"""Load jets from an HDF5 file."""
	with h5py.File(ifile, "r") as f:

		jets = f["jet_kinematics"][mask_pt,:3]

	print(f"jets.shape={jets.shape}")
	print(f"\n",flush=True)

	return jets[:,0], jets[:,1], jets[:,2] #pt,eta,phi

def load_cnsts_rodem(ifile: str, mask_pt: list):
	"""Load constituents from an HDF5 file."""
	with h5py.File(ifile, "r") as f:

		cnsts=f["objects/jets/jet1_cnsts"][mask_pt,:,:3]

	mask = np.repeat(cnsts[:, :, 0] == 0, cnsts.shape[2])
	mask = mask.reshape(-1, cnsts.shape[1], cnsts.shape[2])
	cnsts = np.ma.masked_where(mask, cnsts)

	print(f"cnsts.shape={cnsts.shape}")
	print(f"\n",flush=True)

	return cnsts[:,:,0], cnsts[:,:,1], cnsts[:,:,2] #pt,eta,phi

def load_jets_rodem(ifile: str, mask_pt: list):
	with h5py.File(ifile, "r") as f:

		jets = f["objects/jets/jet1_obs"][mask_pt,:3]

	print(f"jets.shape={jets.shape}")
	print(f"\n",flush=True)

	return jets[:,0], jets[:,1], jets[:,2] #pt,eta,phi

def mom_fit_lowpt(pt,a,b,pt_middles):

	pt=pt-np.min(pt_middles)
	pt=pt/(np.max(pt_middles)-np.min(pt_middles))
	return a+b*pt

def mom_fit_highpt(pt,a,b,c,d,pt_middles):

	pt=pt-np.min(pt_middles)
	pt=pt/(np.max(pt_middles)-np.min(pt_middles))
	return a+b*np.sqrt(pt)+c*pt+d*pt**2

def mom_fit_root(pt,A,alpha,pt_middles):

	p0=(np.min(pt_middles)+np.max(pt_middles))/2.

	return A*(p0/pt)**alpha

	#pt=pt-np.min(pt_middles)
	#pt=pt/(np.max(pt_middles)-np.min(pt_middles))
	#return a+b*pt

def hist_cov(idx,num_pt_bins,R,weights,bins_r,width,shm_hist_all_name,shm_cov_all_name,density:bool):

	logging.basicConfig(level=logging.INFO)
	logging.info(f"(density={density}) pT bin {idx}/{num_pt_bins} hist_cov function in progress..")

	R_middles = (bins_r[1:]+bins_r[:-1])/2.
	num_R_bins=len(R_middles)

	if density==True:
		denominator=R_middles

	elif density==False:
		denominator=np.ones(num_R_bins)
		width=1

	existing_shm_hist_all=SharedMemory(name=shm_hist_all_name)
	hist_all=np.ndarray((num_pt_bins*num_R_bins),dtype=np.float64,buffer=existing_shm_hist_all.buf)

	existing_shm_cov_all=SharedMemory(name=shm_cov_all_name)
	cov_all=np.ndarray((num_pt_bins*num_R_bins,num_pt_bins*num_R_bins),dtype=np.float64,buffer=existing_shm_cov_all.buf)

	index1=idx*num_R_bins
	index2=(idx+1)*num_R_bins

	n_jets=len(R)

	#this normalization does not influence the result
	#of the covariance matrix numerically (normalization factor cancels out),
	#but makes everything more stable:
	weights=weights/(n_jets*width*np.sum(weights,axis=1).reshape(-1,1))

	W=np.sum(weights,axis=1)

	#print(f"min of R={np.min(R)}")
	#print(f"max of R={np.max(R)}")

	#print(f"min of weights={np.min(weights)}")
	#print(f"max of weights={np.max(weights)}")

	#print(f"sum of weights={np.sum(weights)}")

	hist_all[index1:index2], _  = np.histogram(R.compressed(),bins=bins_r,weights=weights.compressed())

	#print(f"sum of hist={np.sum(h)}")

	hist_all[index1:index2]=hist_all[index1:index2]/denominator

	H=np.zeros([n_jets,num_R_bins])
	H2=np.zeros([n_jets,num_R_bins])

	for J in range(n_jets):

		H[J,:], _ = np.histogram(R[J].compressed(),bins=bins_r,weights=weights[J].compressed())
		H2[J,:], _ = np.histogram(R[J].compressed(),bins=bins_r,weights=weights[J].compressed()**2)

		H[J,:]=H[J,:]/denominator
		H2[J,:]=H2[J,:]/denominator**2

	# Compute covariance matrix:
	#-------------------------------------------------------------------------

	a=-H[:,:,np.newaxis]*denominator

	mask=np.eye(num_R_bins,dtype=bool)

	a[:,mask]+=np.repeat(W,num_R_bins).reshape(a.shape[0],a.shape[1])

	a=a/(width * n_jets * (W ** 2)[:, np.newaxis, np.newaxis])

	b=a

	cov_all[index1:index2,index1:index2]=np.einsum('mik,mjk,mk->ij', a, b, H2)

	"""
	for i in range(index1,index2):

		for j in range(index1,index2):

			for J in range(n_jets):

				for k in range(num_R_bins):

					if k==i-index1:
						a=(W[J]-H[J,i-index1]*denominator[k])/(n_jets*width*W[J]**2)
					else:
						a=-H[J,i-index1]*denominator[k]/(n_jets*width*W[J]**2)

					if k==j-index1:
						b=(W[J]-H[J,j-index1]*denominator[k])/(n_jets*width*W[J]**2)
					else:
						b=-H[J,j-index1]*denominator[k]/(n_jets*width*W[J]**2)

					cov_all[i,j]+=a*b*H2[J,k]
	"""
	#-------------------------------------------------------------------------

	if np.allclose(cov_all[index1:index2,index1:index2], cov_all[index1:index2,index1:index2].T)==False:
		logging.info(f"(density={density}) Error: Covariance matrix for pT bin {idx}/{num_pt_bins} is not symmetric")

	#print(f"sum_ij(R_i*R_j*Cov(h_i,h_j))={np.sum(cov_all[index1:index2,index1:index2]*np.outer(R_middles,R_middles))}")
	#print(f"surface area={width*np.sum(R_middles*hist_all[index1:index2])}")

	logging.info(f"(density={density}) pT bin {idx}/{num_pt_bins} hist_cov function done!")

def fun_pol(x,C,b1,b2,b4,b5,b6,b7,b8):
	x=2*x-1
	return C*np.exp(b1*eval_legendre(1,x)+b2*eval_legendre(2,x)+
			b4*eval_legendre(4,x)+b5*eval_legendre(5,x)+
			b6*eval_legendre(6,x)+b7*eval_legendre(7,x)+
			b8*eval_legendre(8,x))

def pt_bin_fit(idx,num_pt_bins,function,R_middles,h,sigma,p0,rs,shm_params_all_name,shm_params_err_all_name):

	logging.basicConfig(level=logging.INFO)
	logging.info(f"fitting of pT bin {idx}/{num_pt_bins} in progress..")

	num_params=len(signature(function).parameters)-1

	existing_shm_params_all=SharedMemory(name=shm_params_all_name)
	params_all=np.ndarray((num_pt_bins,num_params),dtype=np.float64,buffer=existing_shm_params_all.buf)

	existing_shm_params_err_all=SharedMemory(name=shm_params_err_all_name)
	params_err_all=np.ndarray((num_pt_bins,num_params),dtype=np.float64,buffer=existing_shm_params_err_all.buf)

	params_all[idx,:], cov = curve_fit(function, xdata=R_middles, ydata=h, sigma=sigma, p0=p0)

	params_err_all[idx,:] = np.sqrt(np.diag(cov))

	n_samples=1000
	param_samples=np.random.multivariate_normal(params_all[idx,:],cov,size=n_samples)

	y_samples=np.array([function(rs,*sample) for sample in param_samples])
	y_mean=np.mean(y_samples,axis=0)
	y_std=np.std(y_samples,axis=0)

	y_samples=np.array([function(R_middles,*sample) for sample in param_samples])
	y_check=np.mean(y_samples,axis=0)

	logging.info(f"Success! - pT bin {idx}/{num_pt_bins} fitted")

	return y_mean,y_std,y_check

def digit(x):
	str="{:.10e}".format(x)

	base,exp=str.split('e')
	exp=int(exp)

	return -exp

class options(StrEnum):
	const = "constant"
	lin   = "linear"
	sq    = "square"
	cub   = "cube"

def model(x,a,b,c,d):
	return a*x/x+b*x+c*x**2+d*x**3

def param_fit(param_function: options, pt_middles, params, params_err, pts):

	param_function = options(param_function)

	if param_function=="constant":
		fun=lambda x,a: model(x,a,0,0,0)

	if param_function=="linear":
		fun=lambda x,a,b: model(x,a,b,0,0)

	if param_function=="square":
		fun=lambda x,a,c: model(x,a,0,c,0)

	if param_function=="cube":
		fun=lambda x,a,d: model(x,a,0,0,d)

	pars,cov=curve_fit(fun,xdata=pt_middles,ydata=params,sigma=params_err)

	pars_err=np.sqrt(np.diag(cov))

	n_samples=1000
	pars_samples=np.random.multivariate_normal(pars,cov,size=n_samples)
	params_samples=np.array([fun(pts,*sample) for sample in pars_samples])
	mean_params=np.mean(params_samples,axis=0)
	std_params=np.std(params_samples,axis=0)

	params_samples=np.array([fun(pt_middles,*sample) for sample in pars_samples])
	mean_check=np.mean(params_samples,axis=0)

	if param_function=="constant":

		pars_err[0]=round(pars_err[0],digit(pars_err[0]))
		pars[0]=round(pars[0],digit(pars_err[0]))

		label=fr"$f(p_T)$={pars[0]}$\pm${pars_err[0]}"

	if param_function=="linear":

		label=fr"$f(p_T)$=a+b$\cdot\frac{{p_T-{np.min(pt_middles)}}}{{{np.max(pt_middles)-np.min(pt_middles)}}}$;"

		label+=f"\n"
		pars_err[0]=round(pars_err[0],digit(pars_err[0]))
		pars[0]=round(pars[0],digit(pars_err[0]))
		label+=fr"a={pars[0]}$\pm${pars_err[0]};"
		label+=f"\n"
		pars_err[1]=round(pars_err[1],digit(pars_err[1]))
		pars[1]=round(pars[1],digit(pars_err[1]))
		label+=fr"b={pars[1]}$\pm${pars_err[1]}"

	if param_function=="square":

		label=fr"$f(p_T)$=a+b$\cdot\frac{{p_T-{np.min(pt_middles)}}}{{{np.max(pt_middles)-np.min(pt_middles)}}}^2$;"

		label+=f"\n"
		pars_err[0]=round(pars_err[0],digit(pars_err[0]))
		pars[0]=round(pars[0],digit(pars_err[0]))
		label+=fr"a={pars[0]}$\pm${pars_err[0]};"
		label+=f"\n"
		pars_err[1]=round(pars_err[1],digit(pars_err[1]))
		pars[1]=round(pars[1],digit(pars_err[1]))
		label+=fr"b={pars[1]}$\pm${pars_err[1]}"

	if param_function=="cube":

		label=fr"$f(p_T)$=a+b$\cdot\frac{{p_T-{np.min(pt_middles)}}}{{{np.max(pt_middles)-np.min(pt_middles)}}}^3$;"

		label+=f"\n"
		pars_err[0]=round(pars_err[0],digit(pars_err[0]))
		pars[0]=round(pars[0],digit(pars_err[0]))
		label+=fr"a={pars[0]}$\pm${pars_err[0]};"
		label+=f"\n"
		pars_err[1]=round(pars_err[1],digit(pars_err[1]))
		pars[1]=round(pars[1],digit(pars_err[1]))
		label+=fr"b={pars[1]}$\pm${pars_err[1]}"

	return pars,mean_params,std_params,mean_check,label

def fun_2d_lowpt(x,c1,c2,b11,b12,b2,b41,b42,b51,b52,b61,b62,b71,b72,b81,b82,pt_middles):

	pt,r=x
	pt=pt-np.min(pt_middles)
	pt=pt/(np.max(pt_middles)-np.min(pt_middles))
	r=2*r-1

	C=c1+c2*pt
	b1=b11+b12*pt
	b2=b2
	b4=b41+b42*pt
	b5=b51+b52*pt
	b6=b61+b62*pt
	b7=b71+b72*pt
	b8=b81+b82*pt

	return C*np.exp(b1*eval_legendre(1,r)+b2*eval_legendre(2,r)+
			b4*eval_legendre(4,r)+b5*eval_legendre(5,r)+
			b6*eval_legendre(6,r)+b7*eval_legendre(7,r)+
			b8*eval_legendre(8,r))

def fun_2d_highpt(x,c1,c2,b11,b12,b21,b22,b41,b42,b5,b61,b62,b7,b8,pt_middles):

	pt,r=x
	pt=pt-np.min(pt_middles)
	pt=pt/(np.max(pt_middles)-np.min(pt_middles))
	r=2*r-1

	C=c1+c2*pt
	b1=b11+b12*pt
	b2=b21+b22*pt
	b4=b41+b42*pt
	b5=b5
	b6=b61+b62*pt
	b7=b7
	b8=b8

	return C*np.exp(b1*eval_legendre(1,r)+b2*eval_legendre(2,r)+
			b4*eval_legendre(4,r)+b5*eval_legendre(5,r)+
			b6*eval_legendre(6,r)+b7*eval_legendre(7,r)+
			b8*eval_legendre(8,r))

