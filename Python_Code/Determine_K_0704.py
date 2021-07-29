# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:26:49 2021

@author: paulv
"""

import h5py
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import os
import scipy


import sys

sys.path.append('D:\Internship VU\Python\PF_approx-master')
import clustering_methods as cl
import operator_calculations as op_calc  #This requires msmtools, and this 
#but is incompatible with python 3.8
import delay_embedding as embed


fish=np.genfromtxt('200701_fish3_dff.txt', delimiter=' ')
fish=np.transpose(fish)
print(np.shape(fish))

#Compute predictability as a function of delay
## Basically a grid search
n_seed_range= np.arange(10,100,10)# Number of partitions to examine
range_Ks =  np.arange(1,25,3,dtype=int) #range of delays to study
h_K=np.zeros((len(range_Ks),len(n_seed_range)))
for k,K in enumerate(range_Ks):
    traj_matrix = embed.trajectory_matrix(fish,K=K-1)
    for ks,n_seeds in enumerate(n_seed_range):
        labels=cl.kmeans_knn_partition(traj_matrix,n_seeds)
        h = op_calc.get_entropy(labels)
        h_K[k,ks]=h
        


#This plot takes the last partition (e.g. 300), and shows the decrease in entropy for increasing K
plt.plot(range_Ks,h_K[:,-6])
plt.xlabel('K (frames)',fontsize=15)
plt.ylabel('h (nats/s)',fontsize=15)
plt.show()


#%%
import operator_calculations as op_calc 
def determine_k(dataset, range_k, n_seed=30):
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
        1. list of length range_k with the entropy rates 
    
    """
    range_Ks =  np.arange(0,range_k,1,dtype=int) #range of delays to study
    h_K=np.zeros((4,range_k))
    
    for i in range(4):
        for k, K in enumerate(range_Ks):
            traj_matrix= embed.trajectory_matrix(dataset, K=K)
            labels= cl.kmeans_knn_partition(traj_matrix, n_seed)
            h= op_calc.get_entropy(labels)
            h_K[i,k]=h
    
    return(h_K)


test=determine_k(fish, 10)


#%%


filenames=[]

x= "wildtype_forebrain"

for file in os.listdir("forUVAanalysis/%s"%x):
    if file.endswith(".txt"):
        temp=os.path.join("forUVAanalysis/wildtype_forebrain", file)
        filenames.append(temp)

def get_filenames(dir_brain_region):
    file_dir= "forUVAanalysis/%s"%dir_brain_region
    for file in os.listdir(file_dir):
        if file.endswith(".txt"):
           temp=os.path.join(file_dir, file) 
           filenames.append(temp)
    
    return filenames


filenames=get_filenames("wildtype_forebrain")