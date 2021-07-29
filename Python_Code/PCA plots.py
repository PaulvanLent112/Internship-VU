# -*- coding: utf-8 -*-
"""
Created on Wed May 26 14:46:31 2021

@author: paulv
"""

import numpy as np

import pandas as pd
import sys
import matplotlib.pyplot as plt
sys.path.append('D:\Internship VU\Python\functions')
from scipy import stats
import functions.functions as VU
import networkx as nx
import infomap
import os

#%%

#Load fish data
fish= np.genfromtxt("forUVAanalysis/wildtype_midbrain/dff_200721fish10.txt",delimiter=" ")
fish=fish.T


fish=VU.trajectory_matrix(fish, 30)

#fish=stats.zscore(fish)

import pandas as pd
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
PCs = pca.fit_transform(fish)
var_exp= "Var: " +str(np.round(np.sum(pca.explained_variance_ratio_)*100,2))+"%"

#%%


#PCA plots for Results
cm=1/2.54
fig = plt.figure(figsize=(6*cm, 8*cm))
ax = fig.add_axes([0,0,1,1])
ax.text(0.75, 0.9,var_exp,
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=5, width=1, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=5, width=1, direction='in', right='on')
ax.scatter(PCs[:,0], PCs[:,1], c="#00b25d", s=20)

ax.set_xlabel('PC 1', labelpad=10)
ax.set_ylabel('PC 2', labelpad=10)
# Add legend to plot
ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
ax.set_title("K=30")
#Save figure
plt.savefig('Figures_Thesis/K_30_PCA.png', dpi=300, transparent=False, bbox_inches='tight')

#plt.scatter(PCs[:,0],PCs[:,1], c="black")
#plt.xlabel("PC 1")
#plt.ylabel("PC 2")
#plt.title("K = 0")

#plt.show()

#%%

mean_entropyrate= np.genfromtxt("Figures/Entropy rate decrease/Midbrain_wt/ER_mean_midbrain_wt_1904.txt",delimiter=",")[3,:]
std_entropyrate= np.genfromtxt("Figures/Entropy rate decrease/Midbrain_wt/ER_std_midbrain_wt_1904.txt",delimiter=",")[3,:]

#%%
K=np.arange(0,30,1)
cm=1/2.54
fig = plt.figure(figsize=(9*cm, 10*cm))
ax = fig.add_axes([0,0,1,1])
ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=5, width=1, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=5, width=1, direction='in', right='on')
ax.plot(K,mean_entropyrate, c="#00b25d")
ax.fill_between(range(len(mean_entropyrate)), mean_entropyrate-std_entropyrate, mean_entropyrate+std_entropyrate, alpha = 0.3)
ax.set_xlabel("K")
ax.set_ylabel("Entropy Rate (nat/s)")
plt.savefig('Figures_Thesis/Entropy_ratedecrease.png', dpi=300, transparent=False, bbox_inches='tight')

#%%
x= np.arange(0,25,0.025)
cm=1/2.54
fig = plt.figure(figsize=(12*cm, 16*cm))
ax = fig.add_axes([0,0,1,1])
ax.imshow(fish.T, cmap="gray", vmin=0, vmax=0.5)
ax.set_xlabel("Timesteps")
ax.set_ylabel("Neurons")
plt.savefig('Figures_Thesis/Neural_Population_Activity.png', dpi=300, transparent=False, bbox_inches='tight')



#%%
sys.path.append('D:\Internship VU\Python\PF_approx-master')
import clustering_methods as cl
import operator_calculations as op_calc  #This requires msmtools, and this 
#but is incompatible with python 3.8
import delay_embedding as embed

range_N=np.arange(10,200,50)
def determine_k(dataset, range_k, n_seed):
    """Way to determine the number of delay embeddings to include for constructing the transition matrix 
    later on, by firt partitioning the state space and then calculating the entropy rate decrease
    of the markov chain (TPM) when increasing k. # of partitions is a fairly robust  parameter, but should not
    be set too high, because of finite size effects. As the entropy is an estimation, this estimation
    will be done 10 times.
    
    Input:
        1. Dataset to perform this one
        2. range_K: range of K to measure entropy for
        3. n_seed: number of partitions to use (default=30)
        
        
    Output:
        1. list of length range_k with the entropy rates, 
    
    """
    range_Ks =  np.arange(0,range_k,1,dtype=int) #range of delays to study
    h_K=np.zeros((len(range_N),range_k))
    
    for i in range(len(range_N)):
        for k, K in enumerate(range_Ks):
            traj_matrix= embed.trajectory_matrix(dataset, K=K)
            labels= cl.kmeans_knn_partition(traj_matrix, n_seed[i])
            h= op_calc.get_entropy(labels)
            h_K[i,k]=h
        print(i)
    
    return h_K


ER_N_K=determine_k(fish, 10, range_N)



#%%
new_data=ER_N_K
K=np.arange(0,10,1)
cm=1/2.54
fig = plt.figure(figsize=(9*cm, 10*cm))
ax = fig.add_axes([0,0,1,1])
ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=5, width=1, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=5, width=1, direction='in', right='on')
ax.plot(K,new_data[0,:], c="#4053d3", label="N=10")
ax.plot(K,new_data[1,:], c="#ddb310", label="N=60")
ax.plot(K,new_data[2,:], c="#b51d14", label="N=110")
ax.plot(K,new_data[3,:], c="#00b25d",label="N=160")
ax.legend()
ax.set_xlabel("K")
ax.set_ylabel("Entropy rate (nat/s)")
plt.savefig('Figures_Thesis/Partitioning.png', dpi=300, transparent=False, bbox_inches='tight')