import sys
sys.path.append('./')

import os
import scipy
import uproot
import vector
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from functions import*
from functools import partial
from scipy.optimize import curve_fit
from dataloader import read_file
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory

def main(processes:int):

	input = "WToQQ_070.root"

	min_pt=500
	max_pt=1000
	width_pt=10

	R_max=0.8
	width=0.01

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

	num_jets_file=f"./Saved/num(jets)_root_{min_pt}_{max_pt}_{width_pt}.txt"
	moments_file=f"./Saved/moments_root_{min_pt}_{max_pt}_{width_pt}_{R_max}.txt"
	hist_prob_file=f"./Saved/hist_prob_root_{min_pt}_{max_pt}_{width_pt}_{R_max}_{width}.txt"
	cov_prob_file=f"./Saved/cov_prob_root_{min_pt}_{max_pt}_{width_pt}_{R_max}_{width}.txt"
	jet_file="./Plots/jet.jpg"

	if (os.path.exists(num_jets_file) and os.path.exists(moments_file)
	and os.path.exists(hist_prob_file) and os.path.exists(cov_prob_file)
	and os.path.exists(jet_file)):

		print("Loading number of jets per jet pt bins, moment for each jet pt bin, histograms, their covariance matrices...")
		print(f"\n",flush=True)

		# **LOADING NUMBER OF JETS PER JET pT BIN, MOMENT FOR EACH JET pT BIN, ALREADY EXISTING HISTOGRAMS, THEIR COVARIANCE MATRICES:**

		num_jets=np.loadtxt(num_jets_file,dtype=int)
		moments=np.loadtxt(moments_file)

		hist_prob_all=np.loadtxt(hist_prob_file)
		cov_prob_all=np.loadtxt(cov_prob_file)

		shm_hist_prob_all=None
		shm_cov_prob_all=None

	else:

		print(f"Reading from file: {input} ...")
		print(f"\n",flush=True)

		#tree = uproot.open(input)['tree']
		#running tree.arrays() in Jupyter Notebook helps a lot

		#---------------------------------------------------------------------------------------------------------------------------

		# **LOADING THE DATA IN A DICTIONARY**:

		particle_features=['part_pt','part_eta','part_phi',
				   'part_deta','part_dphi','part_energy',
				   'part_isChargedHadron','part_isNeutralHadron',
				   'part_isElectron','part_isMuon','part_isPhoton']

		jet_features=['jet_pt','jet_eta','jet_phi']

		x_particles, x_jets = read_file(filepath=input,particle_features=particle_features,jet_features=jet_features)

		mask = x_particles[:, 0, :] == 0

		full_mask = np.repeat(mask[:, np.newaxis, :], x_particles.shape[1], axis=1)

		x_particles=np.ma.masked_where(full_mask,x_particles)

		jets={}

		for pf,label in enumerate(particle_features):
			jets[label]=x_particles[:,pf,:]

		for jf,label in enumerate(jet_features):
			jets[label]=x_jets[:,jf]

		jets['jet_nparticles']=jets['part_pt'].count(axis=1)

		#---------------------------------------------------------------------------------------------------------------------------

		# **PLOT OF THE JET WITH THE HIGHEST pT:**

		arg=np.argmax(jets['jet_pt'])
		print(f"index of the jet with the highest pT: {arg}")
		print(f"\n",flush=True)

		fig,ax=plt.subplots(figsize=(20,12))

		for i in range(jets['jet_nparticles'][arg]):
			if jets['part_isChargedHadron'][arg,i]==1:
				m='o'
				fcs='lightgreen'
			elif jets['part_isNeutralHadron'][arg,i]==1:
				m='o'
				fcs='none'
			elif jets['part_isElectron'][arg,i]==1:
				m='^'
				fcs='lightgreen'
			elif jets['part_isMuon'][arg,i]==1:
				m='v'
				fcs='lightgreen'
			elif jets['part_isPhoton'][arg,i]==1:
				m='p'
				fcs='none'
			ax.scatter(jets['part_deta'][arg,i],jets['part_dphi'][arg,i],color='lightgreen',marker=m,facecolors=fcs,s=jets['part_energy'][arg,i])

		ax.set_xlabel(r"$\Delta\eta$")
		ax.set_ylabel(r"$\Delta\phi$")
		fig.savefig("./Plots/jet.jpg")

		#---------------------------------------------------------------------------------------------------------------------------

		# **R DISTRIBUTION OF JETS:**

		#calculate delta_phi and delta_eta by hand just to check that:
		#just to check that abs(delta_phi)==abs(jets['part_dphi']) and abs(delta_eta)==abs(jets['part_deta'])

		#phi angle differences between the constituents and the associated jet:
		#rows -> jets ; columns -> particles
		delta_phi=jets['jet_phi'].reshape(-1,1)-jets['part_phi']
		#because the phi angle difference should be inside [-np.pi,+np.pi]:
		delta_phi=np.ma.mod(delta_phi+np.pi,2*np.pi)-np.pi #just to check that abs(delta)

		#eta angle differences between the constituents and the associated jet:
		#rows -> jets ; columns -> particles
		delta_eta=jets['jet_eta'].reshape(-1,1)-jets['part_eta']

		#rows -> jets ; columns -> particles
		R=np.ma.sqrt(delta_eta**2+delta_phi**2)

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

	#index1=idx*num_R_bins
	#index2=(idx+1)*num_R_bins
	#corr_prob_all[index1:index2,index1:index2] -> correlation matrix between R bins of one jet pt bin
	#corr_prob_per_bin_all[index1:index2] -> mean absolute correlation per R bin of one jet pt bin
	#cm_prob_all[index1:index2] -> normalized mean absolute correlation per R bin of one jet pt bin

	#---------------------------------------------------------------------------------------------------------------------------

	# **PLOTTING AND FITTING MOMENTS ACROSS JET pT BINS:**

	pts=np.linspace(np.min(pt_middles),np.max(pt_middles),100)

	fun=partial(mom_fit_root,pt_middles=pt_middles)

	label=fr"$f(p_T)$=A$\cdot\left(\frac{{p_0}}{{p_T}}\right)^\alpha$;"
	label+=f"\n"
	label+=fr"$p_0$={(np.min(pt_middles)+np.max(pt_middles))/2.} GeV"
	p=["A",r"$\alpha$"]

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

	fig,ax=plt.subplots(figsize=(21,12))

	ax.errorbar(x=pt_middles,y=moments[:,0],yerr=moments[:,1],ls='none',capsize=2)
	ax.scatter(pt_middles,moments[:,0],s=4)

	ax.plot(pts,mom_mean,label=label,color='red')
	ax.fill_between(pts,mom_mean-mom_std,mom_mean+mom_std,color='red',alpha=0.3)

	ax.set_xlabel(r"$p_t$ of jet [GeV]")
	ax.set_ylabel(r"$M_1$")

	ax.legend(fontsize=22)

	fig.savefig(f"./Plots/moments_root.jpg")

	#---------------------------------------------------------------------------------------------------------------------------

	# **PLOTTING PROBABILITY:**

	idx=2
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

	fig.savefig(f"./Plots/probability_root.jpg")

	#---------------------------------------------------------------------------------------------------------------------------

	# **CLOSING AND UNLINKING SHARED MEMORY:**

	if shm_hist_prob_all is not None:
		shm_hist_prob_all.close()  #detach from the shared memory
		shm_hist_prob_all.unlink()  #release the shared memory block

	if shm_cov_prob_all is not None:
		shm_cov_prob_all.close()  #detach from the shared memory
		shm_cov_prob_all.unlink()  #release the shared memory block


if __name__=="__main__":

	print("Python version:", sys.version) #3.12.3
	print("Uproot version:", uproot.__version__) #5.6.0
	print("Vector version:", vector.__version__) #1.6.1
	print(f"\n",flush=True)

	main(processes=10)
