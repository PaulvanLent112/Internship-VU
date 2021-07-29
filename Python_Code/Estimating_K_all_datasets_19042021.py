# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:25:54 2021

@author: paulv
"""

import h5py
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import os
import scipy
import time

import sys

sys.path.append('D:\Internship VU\Python\PF_approx-master')
import clustering_methods as cl
import operator_calculations as op_calc  #This requires msmtools, and this 
#but is incompatible with python 3.8
import delay_embedding as embed



def get_filenames(dir_brain_region):
    filenames=[]
    file_dir= "forUVAanalysis/%s"%dir_brain_region
    for file in os.listdir(file_dir):
        if file.endswith(".txt"):
           temp=os.path.join(file_dir, file) 
           filenames.append(temp)
    
    return filenames





def load_file(filename):
    fish =np.genfromtxt(filename, delimiter=' ')
    fish=np.transpose(fish)
    return fish

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
        1. list of length range_k with the entropy rates, 
    
    """
    range_Ks =  np.arange(0,range_k,1,dtype=int) #range of delays to study
    h_K=np.zeros((10,range_k))
    
    for i in range(10):
        for k, K in enumerate(range_Ks):
            traj_matrix= embed.trajectory_matrix(dataset, K=K)
            labels= cl.kmeans_knn_partition(traj_matrix, n_seed)
            h= op_calc.get_entropy(labels)
            h_K[i,k]=h
    
    return(h_K)




def mean_std(h_K):
    """get the mean and standard deviation for each dataset for k
    Input: the dataframe from determine_k, where the rows are timesteps and columns
    individual neurons
    """
    mean= np.mean(h_K,0)
    std= np.std(h_K,0)
    
    r,c=np.shape(h_K)
    x=np.arange(0,c,1)
    
    cm = 1/2.54 
    plt.subplots(figsize=(10*cm, 10*cm))
    plt.plot(x,mean, c="m")
    plt.fill_between(range(len(mean)), mean-std, mean+std, alpha = 0.3)
    plt.xlabel("K")
    plt.ylabel("Entropy Rate (nat/s)")
    
    return mean, std
    











#%% 
if __name__=="__main__":
    a= time.time()
    filenames=get_filenames("het_forebrain")
    K=30
    mean_array= np.zeros((len(filenames),30))
    std_array= np.zeros((len(filenames),30))
    for i in range(len(filenames)):
        file= load_file(filenames[i])
        H_k= determine_k(file, 30)
        mean,std= mean_std(H_k)
        mean_array[i,]= mean
        std_array[i,]=std
    b=time.time()
    
    print(b-a)


#%%        
  
np.savetxt("ER_mean_forebrain_het_1904.txt", mean_array, delimiter=",")
np.savetxt("ER_std_forebrain_het_1904.txt", std_array, delimiter=",")
        
#%%
fish= np.genfromtxt("forUVAanalysis/wildtype_midbrain/dff_200721fish10.txt",delimiter=" ")
fish=fish.T


      