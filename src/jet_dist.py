#!/usr/bin/env python
# coding: utf-8

# **IMPORTING PACKAGES:**

import sys
sys.path.append('./')
sys.path.append('./machine/')

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from mcmc import *
from parallel import *
from fit_prior import read_prior_par
from copy import deepcopy

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import vector
import awkward as ak
import matplotlib.cm as cm
from argparse import ArgumentParser
from inspect import signature
from functions import*
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from scipy.optimize import curve_fit
from functools import partial
from math import ceil

def main(choice,function,processes:int,ms:bool):

	if choice=="lowpt":

		input_qcd = "RunG_batch0.h5"
		n_jets = 400_000

		min_pt=300 #330
		max_pt=500
		width_pt=10

		R_max=0.75 #0.8
		width=0.01

	elif choice=="highpt":

		input_qcd = "QCDjj_pT_450_1200_train01.h5"
		n_jets = 400_000

		min_pt=450
		max_pt=1000
		width_pt=10

		R_max=0.9
		width=0.01

	else:

		print("Error: Invalid input.",flush=True)
		exit(1)

	#---------------------------------------------------------------------------------------------------------------------------

	# **FIGURING OUT pT BINS:**

	bins_pt=np.arange(min_pt,max_pt+width_pt,width_pt)
	pt_middles = (bins_pt[1:]+bins_pt[:-1])/2.
	num_pt_bins=len(pt_middles)

	print(f"num_pt_bins={num_pt_bins}")
	print(f"\n")

	#---------------------------------------------------------------------------------------------------------------------------

	# **FIGURING OUT R BINS:**

	bins_r=np.arange(0,R_max,width)
	R_middles = (bins_r[1:]+bins_r[:-1])/2.
	num_R_bins=len(R_middles)

	print(f"num_R_bins={num_R_bins}")
	print(f"\n",flush=True)

	#---------------------------------------------------------------------------------------------------------------------------

	if not os.path.exists("./Saved"):
		os.makedirs("./Saved")

	if not os.path.exists("./Plots"):
		os.makedirs("./Plots")

	num_jets_file=f"./Saved/num(jets)_{choice}_{n_jets}_{min_pt}_{max_pt}_{width_pt}.txt"
	moments_file=f"./Saved/moments_{choice}_{n_jets}_{min_pt}_{max_pt}_{width_pt}_{R_max}.txt"
	hist_prob_file=f"./Saved/hist_prob_{choice}_{n_jets}_{min_pt}_{max_pt}_{width_pt}_{R_max}_{width}.txt"
	cov_prob_file=f"./Saved/cov_prob_{choice}_{n_jets}_{min_pt}_{max_pt}_{width_pt}_{R_max}_{width}.txt"
	hist_dens_file=f"./Saved/hist_dens_{choice}_{n_jets}_{min_pt}_{max_pt}_{width_pt}_{R_max}_{width}.txt"
	cov_dens_file=f"./Saved/cov_dens_{choice}_{n_jets}_{min_pt}_{max_pt}_{width_pt}_{R_max}_{width}.txt"

	if (os.path.exists(num_jets_file) and os.path.exists(moments_file) and os.path.exists(hist_prob_file)
	and os.path.exists(cov_prob_file) and os.path.exists(hist_dens_file) and os.path.exists(cov_dens_file)):

		print("Loading number of jets per jet pt bins, moment for each jet pt bin, histograms, their covariance matrices...")
		print(f"\n",flush=True)

		#---------------------------------------------------------------------------------------------------------------------------

		# **LOADING NUMBER OF JETS PER JET pT BIN, MOMENT FOR EACH JET pT BIN, ALREADY EXISTING HISTOGRAMS, THEIR COVARIANCE MATRICES:**

		num_jets=np.loadtxt(num_jets_file,dtype=int)
		moments=np.loadtxt(moments_file)

		hist_prob_all=np.loadtxt(hist_prob_file)
		cov_prob_all=np.loadtxt(cov_prob_file)

		hist_dens_all=np.loadtxt(hist_dens_file)
		cov_dens_all=np.loadtxt(cov_dens_file)

		shm_hist_prob_all=None
		shm_cov_prob_all=None

		shm_hist_dens_all=None
		shm_cov_dens_all=None

	else:

		print(f"Reading from file: {input_qcd} ...")
		print(f"\n",flush=True)

		#---------------------------------------------------------------------------------------------------------------------------

		# **READING FROM A FILE:**

		jets={}

		if choice=="lowpt":

			with h5py.File(input_qcd, "r") as f:

				mask_pt=[index for index,value in enumerate(f["jet_kinematics"][:n_jets,0]) if ((value>=min_pt) and (value<=max_pt))]

			#rows -> jets ; columns -> particles
			jets['part_pt'],jets['part_eta'],jets['part_phi']=load_cnsts_aspen(input_qcd,mask_pt)

			#flat arrays
			jets['jet_pt'],jets['jet_eta'],jets['jet_phi']=load_jets_aspen(input_qcd,mask_pt)

		elif choice=="highpt":

			with h5py.File(input_qcd, "r") as f:
				mask_pt=[index for index,value in enumerate(f["objects/jets/jet1_obs"][:n_jets,0]) if ((value>=min_pt) and (value<=max_pt))]

			#rows -> jets ; columns -> particles
			jets['part_pt'],jets['part_eta'],jets['part_phi']=load_cnsts_rodem(input_qcd,mask_pt)

			#flat arrays
			jets['jet_pt'],jets['jet_eta'],jets['jet_phi']=load_jets_rodem(input_qcd,mask_pt)

		jets['jet_nparticles']=jets['part_pt'].count(axis=1)

		print(f"Smallest jet_pt is: {np.min(jets['jet_pt'])}")
		print(f"Biggest jet_pt is: {np.max(jets['jet_pt'])}",flush=True)

		#---------------------------------------------------------------------------------------------------------------------------

		# **R DISTRIBUTION OF JETS:**

		#phi angle differences between the constituents and the associated jet:
		#rows -> jets ; columns -> particles
		delta_phi=jets['jet_phi'].reshape(-1,1)-jets['part_phi']

		#because the phi angle difference should be inside [-np.pi,+np.pi]:
		delta_phi_new=np.ma.mod(delta_phi+np.pi,2*np.pi)-np.pi

		#eta angle differences between the constituents and the associated jet:
		#rows -> jets ; columns -> particles
		delta_eta=jets['jet_eta'].reshape(-1,1)-jets['part_eta']

		#rows -> jets ; columns -> particles
		R=np.ma.sqrt(delta_eta**2+delta_phi_new**2)

		print(f"number of particles loaded: \
		      {len(R.compressed())}")

		print(f"number of particles with R=0: \
		      {len(np.where(R.compressed()==0.0)[0])}")

		print(f"percentage of particles with R=0: \
		      {np.round(len(np.where(R.compressed()==0.0)[0])*100/len(R.compressed()),3)}%")

		print(f"number of particles with R>={R_max}: \
		      {len(np.where(R.compressed()>=R_max)[0])}")

		print(f"percentage of particles with R>={R_max}: \
		      {np.round(len(np.where(R.compressed()>=R_max)[0])*100/len(R.compressed()),3)}%")

		print(f"\n",flush=True)

		jets['part_pt']=np.ma.masked_where(R<0,jets['part_pt'])
		jets['part_pt']=np.ma.masked_where(R>=R_max,jets['part_pt'])

		R=np.ma.masked_where(R<0,R)
		R=np.ma.masked_where(R>=R_max,R)

		#---------------------------------------------------------------------------------------------------------------------------

		# **SORTING JETS IN JET pT BINS:**

		mask_binpt=[]
		for idx in range(num_pt_bins):

			mask_binpt.append([index for index,value in enumerate(jets['jet_pt']) if ((value>bins_pt[idx]) and (value<=bins_pt[idx+1]))])

		num_jets = np.array([len(sublist) for sublist in mask_binpt],dtype=int)

		print("Saving number of jets per jet pt bin..")
		np.savetxt(num_jets_file,num_jets,fmt="%d")

		print(f"\n",flush=True)

		#---------------------------------------------------------------------------------------------------------------------------

		# **CALCULATING R DISTRIBUTION MOMENTS FOR EACH JET pT BIN:**

		moments=np.zeros([num_pt_bins,2]) #(mean_value,standard_error) for each jet pt bin

		for idx in range(num_pt_bins):

			temp=np.sum(R[mask_binpt[idx],:]*jets['part_pt'][mask_binpt[idx],:],axis=1)
			temp/=np.sum(jets['part_pt'][mask_binpt[idx],:],axis=1)

			#mean value of moment for idx-th jet pt bin:
			moments[idx,0]=np.mean(temp)

			#standard error of moment for idx-th jet pt bin:
			moments[idx,1]=np.sqrt(np.sum((temp-moments[idx,0])**2)/(num_jets[idx]*(num_jets[idx]-1)))

		print("Saving R distribution moment for each jet pt bin..")
		np.savetxt(moments_file,moments,fmt="%s")

		print(f"\n",flush=True)

		#---------------------------------------------------------------------------------------------------------------------------

		# **MAKING OF R HISTOGRAM AND ITS COVARIANCE MATRIX FOR EACH pT BIN - MULTIPROCESSING:**

		#shared memory for histogram values - probability
		shm_hist_prob_all=SharedMemory(create=True,size=num_pt_bins*num_R_bins*8)
		hist_prob_all=np.ndarray((num_pt_bins*num_R_bins),dtype=np.float64,buffer=shm_hist_prob_all.buf)
		hist_prob_all[:]=0.0

		#shared memory for covariance matrices - probability
		shm_cov_prob_all=SharedMemory(create=True,size=((num_pt_bins*num_R_bins)**2*8))
		cov_prob_all=np.ndarray((num_pt_bins*num_R_bins,num_pt_bins*num_R_bins),dtype=np.float64,buffer=shm_cov_prob_all.buf)
		cov_prob_all[:,:]=0.0

		#shared memory for histogram values - probability density
		shm_hist_dens_all=SharedMemory(create=True,size=num_pt_bins*num_R_bins*8)
		hist_dens_all=np.ndarray((num_pt_bins*num_R_bins),dtype=np.float64,buffer=shm_hist_dens_all.buf)
		hist_dens_all[:]=0.0

		#shared memory for covariance matrices - probability density
		shm_cov_dens_all=SharedMemory(create=True,size=((num_pt_bins*num_R_bins)**2*8))
		cov_dens_all=np.ndarray((num_pt_bins*num_R_bins,num_pt_bins*num_R_bins),dtype=np.float64,buffer=shm_cov_dens_all.buf)
		cov_dens_all[:,:]=0.0

		#index1=idx*num_R_bins
		#index2=(idx+1)*num_R_bins
		#hist_()_all[index1:index2] -> histogram values of one jet pt bin
		#cov_()_all[index1:index2,index1:index2] -> covariance matrix between R bins of one jet pt bin

		density=False

		print(f"Start multiprocessing of hist_cov(density={density})..")
		print(f"\n",flush=True)

		with Pool(processes=processes) as pool:

			args=[(idx,num_pt_bins,R[mask_binpt[idx]],jets['part_pt'][mask_binpt[idx]],bins_r,width,shm_hist_prob_all.name,shm_cov_prob_all.name,density) for idx in range(num_pt_bins)]

			pool.starmap(hist_cov,args)

		print(f"Multiprocessing of hist_cov(density={density}) done!")

		print("Saving the results in .txt files..")
		np.savetxt(hist_prob_file,hist_prob_all,fmt="%s")
		np.savetxt(cov_prob_file,cov_prob_all,fmt="%s")

		print(f"\n",flush=True)

		density=True

		print(f"Start multiprocessing of hist_cov(density={density})..")
		print(f"\n",flush=True)

		with Pool(processes=processes) as pool:

			args=[(idx,num_pt_bins,R[mask_binpt[idx]],jets['part_pt'][mask_binpt[idx]],bins_r,width,shm_hist_dens_all.name,shm_cov_dens_all.name,density) for idx in range(num_pt_bins)]

			pool.starmap(hist_cov,args)

		print(f"Multiprocessing of hist_cov(density={density}) done!")

		print("Saving the results in .txt files..")
		np.savetxt(hist_dens_file,hist_dens_all,fmt="%s")
		np.savetxt(cov_dens_file,cov_dens_all,fmt="%s")

		print(f"\n",flush=True)

	#---------------------------------------------------------------------------------------------------------------------------

	# **CALCULATING CORRELATION MATRIX:**

	hist_prob_err_all=np.sqrt(np.diag(cov_prob_all))
	corr_prob_all=cov_prob_all/np.outer(hist_prob_err_all,hist_prob_err_all)

	#compute mean absolute correlation per R bin:
	corr_prob_per_bin_all=np.mean(np.abs(corr_prob_all),axis=1)

	#normalize to a colormap scale (0 to 1):
	cm_prob_all=np.zeros([num_pt_bins*num_R_bins])
	for idx in range(num_pt_bins):

		index1=idx*num_R_bins
		index2=(idx+1)*num_R_bins

		x=corr_prob_per_bin_all[index1:index2]
		minimum=np.min(corr_prob_per_bin_all[index1:index2])
		maximum=np.max(corr_prob_per_bin_all[index1:index2])
		cm_prob_all[index1:index2]=(x-minimum)/(maximum-minimum)

	hist_dens_err_all=np.sqrt(np.diag(cov_dens_all))
	corr_dens_all=cov_dens_all/np.outer(hist_dens_err_all,hist_dens_err_all)

	#compute mean absolute correlation per R bin:
	corr_dens_per_bin_all=np.mean(np.abs(corr_dens_all),axis=1)

	#normalize to a colormap scale (0 to 1):
	cm_dens_all=np.zeros([num_pt_bins*num_R_bins])
	for idx in range(num_pt_bins):

		index1=idx*num_R_bins
		index2=(idx+1)*num_R_bins

		x=corr_dens_per_bin_all[index1:index2]
		minimum=np.min(corr_dens_per_bin_all[index1:index2])
		maximum=np.max(corr_dens_per_bin_all[index1:index2])
		cm_dens_all[index1:index2]=(x-minimum)/(maximum-minimum)

	#index1=idx*num_R_bins
	#index2=(idx+1)*num_R_bins
	#corr_()_all[index1:index2,index1:index2] -> correlation matrix between R bins of one jet pt bin
	#corr_()_per_bin_all[index1:index2] -> mean absolute correlation per R bin of one jet pt bin
	#cm_()_all[index1:index2] -> normalized mean absolute correlation per R bin of one jet pt bin

	idx=1
	index1=idx*num_R_bins
	index2=(idx+1)*num_R_bins

	print(f"idx={idx}")
	print(f"num_R_bins={num_R_bins}")
	print("index1=idx*num_R_bins")
	print("index2=(idx+1)*num_R_bins")
	print(f"\n")

	print("cov_dens_all[index1:index2,index1:index2][num_R_bins-3:,num_R_bins-3:]:")
	print(cov_dens_all[index1:index2,index1:index2][num_R_bins-3:,num_R_bins-3:])
	print(f"\n")

	print("corr_dens_all[index1:index2,index1:index2][num_R_bins-3:,num_R_bins-3:]:")
	print(corr_dens_all[index1:index2,index1:index2][num_R_bins-3:,num_R_bins-3:])
	print(f"\n",flush=True)

	#---------------------------------------------------------------------------------------------------------------------------

	# **PLOTTING AND FITTING MOMENTS ACROSS JET pT BINS:**

	pts=np.linspace(np.min(pt_middles),np.max(pt_middles),100)

	frac=fr"\frac{{p_T-{np.min(pt_middles)}}}{{{np.max(pt_middles)-np.min(pt_middles)}}}"

	if choice=="lowpt":

		fun=partial(mom_fit_lowpt,pt_middles=pt_middles)

		label=fr"$f(p_T)$=a+b$\cdot$${frac}$"
		p=["a","b"]

	if choice=="highpt":

		fun=partial(mom_fit_highpt,pt_middles=pt_middles)

		label=fr"$f(p_T)$=a+b$\cdot$$\sqrt{{{frac}}}$+c$\cdot$${frac}$+d$\cdot$${frac}^2$"
		p=["a","b","c","d"]

	mom_params,cov=curve_fit(fun,xdata=pt_middles,ydata=moments[:,0],sigma=moments[:,1])
	mom_params_err=np.sqrt(np.diag(cov))

	for n,s in enumerate(p):
		label+=";"
		label+=f"\n"
		err_rounded=round(mom_params_err[n],digit(mom_params_err[n]))
		mean_rounded=round(mom_params[n],digit(err_rounded))
		label+=fr"{s}={mean_rounded}$\pm${err_rounded}"

	n_samples=1000
	mom_params_samples=np.random.multivariate_normal(mom_params,cov,size=n_samples)

	mom_samples=np.array([fun(pts,*sample) for sample in mom_params_samples])

	mom_mean=np.mean(mom_samples,axis=0)
	mom_std=np.std(mom_samples,axis=0)

	fig,ax=plt.subplots(figsize=(20,12))

	ax.errorbar(x=pt_middles,y=moments[:,0],yerr=moments[:,1],ls='none',capsize=2)
	ax.scatter(pt_middles,moments[:,0],s=4)

	ax.plot(pts,mom_mean,label=label,color='red')
	ax.fill_between(pts,mom_mean-mom_std,mom_mean+mom_std,color='red',alpha=0.3)

	ax.set_xlabel(r"$p_t$ of jet [GeV]")
	ax.set_ylabel(r"$M_1$")
	ax.legend(fontsize=22)

	fig.savefig(f"./Plots/moments_{choice}.jpg")

	#---------------------------------------------------------------------------------------------------------------------------

	# **PLOTTING PROBABILITY:**

	idx=7
	index1=idx*num_R_bins
	index2=(idx+1)*num_R_bins
	fig,ax=plt.subplots(figsize=(20,12))
	#color gradient from blue to red:
	colors = cm.coolwarm(cm_prob_all[index1:index2])
	#add colorbar to indicate correlation strength:
	sm = cm.ScalarMappable(cmap=cm.coolwarm)
	sm.set_array(corr_prob_per_bin_all[index1:index2])
	cbar = plt.colorbar(sm, ax=ax)
	cbar.set_label("Mean Absolute Correlation")

	ax.set_title(f"{num_jets[idx]} jets with pT in range {bins_pt[idx]}-{bins_pt[idx+1]} GeV")
	for j in range(num_R_bins):
		ax.errorbar(x=R_middles[j],y=hist_prob_all[index1+j],yerr=hist_prob_err_all[index1+j],ls='none',capsize=2,color=colors[j])
	ax.scatter(R_middles,hist_prob_all[index1:index2],s=4)

	#calculation of distribution moments:
	num_degs=5
	M=[np.sum(R_middles**deg*hist_prob_all[index1:index2]) for deg in range(num_degs)]
	M_err=[np.sqrt(np.sum((R_middles**deg*hist_prob_err_all[index1:index2])**2)) for deg in range(num_degs)]

	x=.71
	y=.99
	for deg in range(1,num_degs):

		err_rounded=round(M_err[deg],digit(M_err[deg]))
		mean_rounded=round(M[deg],digit(err_rounded))
		ax.text(x, y, fr"$M_{deg}$={mean_rounded}$\pm${err_rounded}", ha='left', va='top', transform=ax.transAxes)
		y=y-.04

	ax.set_yscale("log")
	ax.set_xlabel(r"$R$")
	ax.set_ylabel("$P - probability$")

	fig.savefig(f"./Plots/probability_{choice}.jpg")

	#---------------------------------------------------------------------------------------------------------------------------

	# **FITTING EACH JET pT BIN INDIVIDUALLY - JUST VARIANCES OR THE WHOLE COVARIANCE MATRIX - MULTIPROCESSING:**

	def fit_all_pt_bins(processes,args,param_function,pt_middles,shm_params_all_name,shm_params_err_all_name,pts):

		print(f"Start multiprocessing of pt_bin_fit..")
		print(f"\n",flush=True)

		with Pool(processes=processes) as pool:

			results=pool.starmap(pt_bin_fit,args)

		print(f"Multiprocessing of pt_bin_fit done!")
		print(f"\n",flush=True)

		y_mean_all=[] #y_mean_all[idx] -> mean(fit_function) of one jet pT bin - interpolation R points
		y_std_all=[] #y_std_all[idx] -> standard_deviation(fit_function) of one jet pt bin - interpolation R points
		y_check_all=[] #y_check_all[idx] -> mean(fit_function) of one jet pT bin - data R points

		#unpacking of the results from multiprocessing
		for y_mean,y_std,y_check in results: #iterating over jet pt bins

			y_mean_all.append(y_mean)
			y_std_all.append(y_std)
			y_check_all.append(y_check)

		num_pt_bins=len(pt_middles)
		num_params=len(param_function)

		existing_shm_params_all=SharedMemory(name=shm_params_all_name)
		params_all=np.ndarray((num_pt_bins,num_params),dtype=np.float64,buffer=existing_shm_params_all.buf)

		existing_shm_params_err_all=SharedMemory(name=shm_params_err_all_name)
		params_err_all=np.ndarray((num_pt_bins,num_params),dtype=np.float64,buffer=existing_shm_params_err_all.buf)

		#params_all[idx,:] -> parameters of the fit of one jet pt bin
		#params_err_all[idx,:] -> parameters' errors of the fit one jet pt bin

		nrows=ceil(num_params/2)
		ncolumns=2
		fig,ax=plt.subplots(nrows,ncolumns,figsize=(7*ncolumns,4*nrows))

		mean_check=np.zeros([num_pt_bins,num_params]) #mean_check[:,n] -> fitted 'param_function[n]' evaluated in pt_middles
		pars_all=[] #pars_all[n] -> parameters for the fit across jet pt bins of n-th parameter

		for n in range(num_params):

			pars,mean_params,std_params,mean_check[:,n],label=param_fit(param_function[n],pt_middles,params_all[:,n],params_err_all[:,n],pts)
			pars_all.append(pars)

			ax[int(n/2),n-2*int(n/2)].errorbar(pt_middles,params_all[:,n],params_err_all[:,n],capsize=2,ls='none')
			ax[int(n/2),n-2*int(n/2)].scatter(pt_middles,params_all[:,n],s=4)
			ax[int(n/2),n-2*int(n/2)].plot(pts,mean_params,label=label)
			ax[int(n/2),n-2*int(n/2)].fill_between(pts,mean_params-std_params,mean_params+std_params,alpha=0.3)
			ax[int(n/2),n-2*int(n/2)].set_xlabel(r"$p_T$")
			if n==0:
				ax[int(n/2),n-2*int(n/2)].set_ylabel(r"C")
			elif n>0 and n<3:
				ax[int(n/2),n-2*int(n/2)].set_ylabel(fr"$b_{n}$")
			elif n>=3:
		        	ax[int(n/2),n-2*int(n/2)].set_ylabel(fr"$b_{n+1}$")
			ax[int(n/2),n-2*int(n/2)].legend()

		plt.savefig(f"./Plots/param_fits_{choice}.jpg")

		return mean_check,pars_all

	#---------------------------------------------------------------------------------------------------------------------------

	# **THREE STEP PROCESS OF MODELLING:**

	#at the top level parameters are called 'params' -> 'function' parameters
	#at the bottom level parameters are called 'pars' -> 'param_function' parameters

	rs=np.linspace(0,R_max,100)

	num_params=len(signature(function).parameters)-1 #-1 comes from the fact that 'function' takes x which is not considered as parameter

	if choice=="lowpt":
		param_function=['linear','linear','constant','linear','linear','linear','linear','linear']

	elif choice=="highpt":
		param_function=['linear','linear','linear','linear','constant','linear','constant','constant']

	#shared memory for parameters of each jet pt bin
	shm_params_all=SharedMemory(create=True,size=num_pt_bins*num_params*8)

	#shared memory for parameters' errors of each jet pt bin
	shm_params_err_all=SharedMemory(create=True,size=num_pt_bins*num_params*8)

	#1ST STEP:
	#two step subprocess:
	#fit R distribution of each jet pt bin individually - just variances taken into account
	#and then fit n-th 'function' parameter across all jet bins using 'param_function[n]'
	args=[(idx,num_pt_bins,function,R_middles,hist_dens_all[idx*num_R_bins:(idx+1)*num_R_bins],
	np.sqrt(np.diag(cov_dens_all[idx*num_R_bins:(idx+1)*num_R_bins,idx*num_R_bins:(idx+1)*num_R_bins])),None,rs,
	shm_params_all.name,shm_params_err_all.name) for idx in range(num_pt_bins)]

	p0=fit_all_pt_bins(processes,args,param_function,pt_middles,shm_params_all.name,shm_params_err_all.name,pts)[0]

	#2ND STEP:
	#two step subprocess:
	#fit R distribution of each jet pt bin individually - whole covariance matrices taken into account
	# -> initial guess of n-th 'function' parameter in idx-th jet pt bin is its 'param_function[n]' evaluated in pt_middles[idx] from 1ST STEP
	#and then fit n-th 'function' parameter across all jet bins using 'param_function[n]'
	args=[(idx,num_pt_bins,function,R_middles,hist_dens_all[idx*num_R_bins:(idx+1)*num_R_bins],
	cov_dens_all[idx*num_R_bins:(idx+1)*num_R_bins,idx*num_R_bins:(idx+1)*num_R_bins],p0[idx],rs,
	shm_params_all.name,shm_params_err_all.name) for idx in range(num_pt_bins)]

	pars_all=fit_all_pt_bins(processes,args,param_function,pt_middles,shm_params_all.name,shm_params_err_all.name,pts)[1]

	#3RD STEP:
	#fit all the jet pt bins at once using one big covariance matrix
	#assume that the 2d fit function is 'function' whose n-th parameter depends on jet pt like 'param_function[n]'
	#note that the bottom level parameters 'pars_all[i]' and 'pars_all[j]' for i!=j were not correlated in two step subprocesses!!
	#all bottom level parameters will now be correlated
	# -> initial guess is going to be p0_final=[*pars_all[0],*pars_all[1],...,*pars_all[num_params-1]]

	p0_final=[]
	for n in range(num_params):
		p0_final.extend(pars_all[n])

	x1=np.repeat(pt_middles,num_R_bins)
	x2=np.tile(R_middles,num_pt_bins)
	x=np.zeros([2,len(x1)])
	x[0,:]=x1
	x[1,:]=x2

	if choice=="lowpt":

		fun2d=partial(fun_2d_lowpt,pt_middles=pt_middles)

	elif choice=="highpt":

		fun2d=partial(fun_2d_highpt,pt_middles=pt_middles)

	print("2D fitting taking place..")
	print(f"\n",flush=True)

	#pars_all will now be flat numpy array of all (now correlated) bottom level parameters
	pars_all, cov = curve_fit(fun2d,xdata=x,ydata=hist_dens_all,sigma=cov_dens_all,p0=p0_final,maxfev=30000)

	print("2D fitting successful!")
	print(f"\n")

	pars_err_all=np.sqrt(np.diag(cov))

	for n in range(len(pars_all)):
		err_rounded=round(pars_err_all[n],digit(pars_err_all[n]))
		mean_rounded=round(pars_all[n],digit(err_rounded))
		print(f"{mean_rounded}+-{err_rounded}")
		print("--------------------------------")

	print(f"\n",flush=True)

	#---------------------------------------------------------------------------------------------------------------------------

	# **SAMPLING THE INTERPOLATION FUNCTIONS:**

	if choice=="lowpt":

		num_params=len(signature(fun_2d_lowpt).parameters)-2

	elif choice=="highpt":

		num_params=len(signature(fun_2d_highpt).parameters)-2

	#interpolation points
	x1_pred=np.repeat(pt_middles,len(rs))
	x2_pred=np.tile(rs,num_pt_bins)
	x_pred=np.zeros([2,len(x1_pred)])
	x_pred[0,:]=x1_pred
	x_pred[1,:]=x2_pred

	#for calculating the loss
	x1_check=np.repeat(pt_middles,num_R_bins)
	x2_check=np.tile(R_middles,num_pt_bins)
	x_check=np.zeros([2,len(x1_check)])
	x_check[0,:]=x1_check
	x_check[1,:]=x2_check

	rho_mean=np.zeros([num_pt_bins,len(rs)])
	rho_std=np.zeros([num_pt_bins,len(rs)])

	rho_check=np.zeros([num_pt_bins,num_R_bins])

	n_samples=1000
	loss=0
	#sampling
	for idx in range(num_pt_bins):

		pars_samples=np.random.multivariate_normal(pars_all,cov,size=n_samples)

		#predictions of the model in the interpolation points
		rho_samples=np.array([fun2d(x_pred[:,idx*len(rs):(idx+1)*len(rs)],*sample) for sample in pars_samples])
		rho_mean[idx,:]=np.mean(rho_samples,axis=0)
		rho_std[idx,:]=np.std(rho_samples,axis=0)

		#predictions of the model in the data points
		rho_samples=np.array([fun2d(x_check[:,idx*num_R_bins:(idx+1)*num_R_bins],*sample) for sample in pars_samples])
		rho_check[idx,:]=np.mean(rho_samples,axis=0)

		#calculation of the loss
		hist_dens=hist_dens_all[idx*num_R_bins:(idx+1)*num_R_bins]
		hist_dens_err=hist_dens_err_all[idx*num_R_bins:(idx+1)*num_R_bins]
		loss+=np.sum((rho_check[idx,:]-hist_dens)**2/hist_dens_err**2)

	loss=loss/(num_pt_bins*num_R_bins-num_params)

	#---------------------------------------------------------------------------------------------------------------------------

	# **MACHINE SCIENTIST TRY:**

	if ms==True:

		density=True

		if density==True:

			string="dens"
		else:

			string="prob"

		nstep=3000
		msmodel_file=f"./Saved/msmodel_{string}_{choice}_{n_jets}_{min_pt}_{max_pt}_{width_pt}_{R_max}_{width}_{nstep}.txt"
		msval_file=f"./Saved/msval_{string}_{choice}_{n_jets}_{min_pt}_{max_pt}_{width_pt}_{R_max}_{width}_{nstep}.txt"

		x=np.zeros([len(x1),2])
		x[:,0]=x1
		x[:,1]=x2

		XLABS=['pt_middles','R']
		x=pd.DataFrame(data=x,columns=XLABS)

		if string=="prob":
			y=pd.Series(data=hist_prob_all)

		elif string=="dens":
			y=pd.Series(data=hist_dens_all)


		# Read the hyperparameters for the prior
		prior_par = read_prior_par('./machine/final_prior_param_sq.named_equations.nv13.np13.2016-09-01 17_05_57.196882.dat')

		if (os.path.exists(msmodel_file) and os.path.exists(msval_file)):

			print("Loading the machine scientist data..")
			print(f"\n",flush=True)

			with open(msmodel_file, 'r') as f:
				model_string = json.load(f)

			with open(msval_file, 'r') as f:
				model_parameters = json.load(f)

			# Instantiate a Tree from the desired string
			mdl_model = Tree(
				variables=XLABS,
				parameters=['a%d' % i for i in range(13)],
				x=x, y=y,
				prior_par=prior_par,
				from_string=model_string,
				)

			# Set the parameter values
			mdl_model.set_par_values(model_parameters)

		else:

			print("Machine scientist start..")
			print(f"\n",flush=True)

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
				if (i+1)%100==0:
					print(f"Step {i+1}/{nstep}")
					print(f"\n",flush=True)

			print("Machine scientist done.")
			print("Desc. length:\t", mdl)
			print(f"\n",flush=True)

			print("Saving the results in .txt files..")
			with open(msmodel_file, 'w') as f:
				json.dump(str(mdl_model), f)
			with open(msval_file, 'w') as f:
				json.dump(mdl_model.par_values['d0'], f)

		print("Best model:\t", mdl_model)
		print("Parameter values:", mdl_model.par_values)
		print(f"\n",flush=True)

		fig,ax=plt.subplots(figsize=(12,12))

		ax.scatter(mdl_model.predict(x), y)

		if string=="prob":
			ax.plot((hist_prob_all.min(),hist_prob_all.max()),(hist_prob_all.min(),hist_prob_all.max()))

		if string=="dens":
			ax.plot((hist_dens_all.min(),hist_dens_all.max()),(hist_dens_all.min(),hist_dens_all.max()))

		ax.set_xlabel('MDL model predictions', fontsize=14)
		ax.set_ylabel('Actual values', fontsize=14)
		fig.savefig(f"./Plots/ms_{string}_{choice}_{nstep}.jpg")

	#---------------------------------------------------------------------------------------------------------------------------

	# **PLOTTING OF THE FINAL MODEL:**

	idx=2
	index1=idx*num_R_bins
	index2=(idx+1)*num_R_bins
	fig,ax=plt.subplots(figsize=(20,12))
	#color gradient from blue to red:
	colors = cm.coolwarm(cm_dens_all[index1:index2])
	#add colorbar to indicate correlation strength:
	sm = cm.ScalarMappable(cmap=cm.coolwarm)
	sm.set_array(corr_dens_per_bin_all[index1:index2])
	cbar = plt.colorbar(sm, ax=ax)
	cbar.set_label("Mean Absolute Correlation")

	ax.set_title(f"{num_jets[idx]} jets with pT in range {bins_pt[idx]}-{bins_pt[idx+1]} GeV")
	for j in range(num_R_bins):
		ax.errorbar(x=R_middles[j],y=hist_dens_all[index1+j],yerr=hist_dens_err_all[index1+j],ls='none',capsize=2,color=colors[j])
	ax.scatter(R_middles,hist_dens_all[index1:index2],s=4)

	label=r"f(x,y)=$C\cdot$exp[$b_1$$\cdot$$L_1$(x)+$b_2$$\cdot$$L_2$(x)+$b_4$$\cdot$$L_4$(x)+$b_5$$\cdot$$L_5$(x)+$b_6$$\cdot$$L_6$(x)+$b_7$$\cdot$$L_7$(x)+$b_8$$\cdot$$L_8$(x)];"
	label+=f"\n"
	label+=r"  x=2$\cdot$R-1;"
	label+=f"\n"
	label+=fr"$y$=$\frac{{p_T-{np.min(pt_middles)}}}{{{np.max(pt_middles)}-{np.min(pt_middles)}}}$;"
	label+=f"\n"

	if choice=="lowpt":
		label+=r"C=$C_1$+$C_2$$\cdot$$y$;"
		label+=f"\n"
		label+=r"$b_1$=$b_{11}$+$b_{12}$$\cdot$$y$;"
		label+=f"\n"
		label+=r"$b_4$=$b_{41}$+$b_{42}$$\cdot$$y$;"
		label+=f"\n"
		label+=r"$b_5$=$b_{51}$+$b_{52}$$\cdot$$y$;"
		label+=f"\n"
		label+=r"$b_6$=$b_{61}$+$b_{62}$$\cdot$$y$;"
		label+=f"\n"
		label+=r"$b_7$=$b_{71}$+$b_{72}$$\cdot$$y$;"
		label+=f"\n"
		label+=r"$b_8$=$b_{81}$+$b_{82}$$\cdot$$y$;"

		p=[r"$C_1$",r"$C_2$",r"$b_{11}$",r"$b_{12}$",r"$b_2$",r"$b_{41}$",r"$b_{42}$",r"$b_{51}$",
			r"$b_{52}$",r"$b_{61}$",r"$b_{62}$",r"$b_{71}$",r"$b_{72}$",r"$b_{81}$",r"$b_{82}$"]

	if choice=="highpt":
		label+=r"C=$C_1$+$C_2$$\cdot$$y$;"
		label+=f"\n"
		label+=r"$b_1$=$b_{11}$+$b_{12}$$\cdot$$y$;"
		label+=f"\n"
		label+=r"$b_2$=$b_{21}$+$b_{22}$$\cdot$$y$;"
		label+=f"\n"
		label+=r"$b_4$=$b_{41}$+$b_{42}$$\cdot$$y$;"
		label+=f"\n"
		label+=r"$b_6$=$b_{61}$+$b_{62}$$\cdot$$y$;"

		p=[r"$C_1$",r"$C_2$",r"$b_{11}$",r"$b_{12}$",r"$b_2$",r"$b_{41}$",
			r"$b_{42}$",r"$b_5$",r"$b_{61}$",r"$b_{62}$",r"$b_7$",r"$b_8$"]


	label+=f"\n"
	label+=f"\n"

	for n,s in enumerate(p):
		err_rounded=round(pars_err_all[n],digit(pars_err_all[n]))
		mean_rounded=round(pars_all[n],digit(err_rounded))
		label+=fr"{s}={mean_rounded}$\pm${err_rounded};"
		label+=f"\n"

	label+=f"\n"
	label+=f"width of pT bins: {width_pt} GeV"
	label+=f"\n"
	label+=f"width of R bins: {width}"
	label+=f"\n"

	ax.plot(rs,rho_mean[idx,:],label=label,color='green')
	ax.fill_between(rs,rho_mean[idx,:]-rho_std[idx,:],rho_mean[idx,:]+rho_std[idx,:],alpha=0.3,color='green')

	ax.set_yscale("log")
	ax.set_xlabel(r"$R$")
	ax.set_ylabel(r"$\rho$$(R)$")

	lgd=ax.legend(ncol=2, bbox_to_anchor=(0.7,-0.1))
	fig.savefig(f"./Plots/FINAL_FIT_{choice}.jpg",bbox_extra_artists=(lgd,),bbox_inches='tight')

	#---------------------------------------------------------------------------------------------------------------------------

	# **CLOSING AND UNLINKING SHARED MEMORY:**

	if shm_hist_prob_all is not None:
		shm_hist_prob_all.close()  #detach from the shared memory
		shm_hist_prob_all.unlink()  #release the shared memory block

	if shm_cov_prob_all is not None:
		shm_cov_prob_all.close()  #detach from the shared memory
		shm_cov_prob_all.unlink()  #release the shared memory block

	if shm_hist_dens_all is not None:
		shm_hist_dens_all.close()  #detach from the shared memory
		shm_hist_dens_all.unlink()  #release the shared memory block

	if shm_cov_dens_all is not None:
		shm_cov_dens_all.close()  #detach from the shared memory
		shm_cov_dens_all.unlink()  #release the shared memory block

	shm_params_all.close()  #detach from the shared memory
	shm_params_all.unlink()  #release the shared memory block

	shm_params_err_all.close()  #detach from the shared memory
	shm_params_err_all.unlink()  #release the shared memory block

if __name__=="__main__":

	print("Start!")

	parser=ArgumentParser()
	parser.add_argument("choice", choices=["lowpt","highpt"],help="Choose either 'lowpt' or 'highpt'")
	args=parser.parse_args()

	choice = args.choice
	print(f"You chose: {choice}")
	print(f"\n",flush=True)

	print(f"Python version:{sys.version}") #3.12.3
	print(f"h5py version:{h5py.__version__}") #3.13.0
	print(f"Numpy version:{np.__version__}") #2.2.5
	print(f"Vector version:{vector.__version__}") #1.6.2
	print(f"Awkward version:{ak.__version__}") #2.8.1
	print(f"\n",flush=True)

	function=fun_pol
	processes=10

	ms=False #enable/disable the machine scientist try

	main(choice,function,processes,ms)

