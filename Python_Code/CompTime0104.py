# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 12:00:10 2021

@author: paulv

The goal of this is to gain insight into how much time it will take to estimate parameter K (and perhaps N)
for larger subsets of time.


First part will be from 20-100 neurons, if that already increases exponentially we don't have to bother 
trying it for a 1000
"""

#%%

import time
import h5py
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import os


import sys

sys.path.append('D:\Internship VU\Python\PF_approx-master')
import clustering_methods as cl
import operator_calculations as op_calc  #This requires msmtools, and this 
#but is incompatible with python 3.8
import delay_embedding as embed

#%%   
#Load neurons 

a=time.clock()
fish = np.genfromtxt('200701_fish3_dff.txt', delimiter=' ')
fish=np.transpose(fish) #Rows are time, cols are neurons


sub= np.arange(20,200,20)


comp_Time=[]


#%%
for k, i in enumerate(sub):
    print(k)
    a=time.clock()
    time_fish=fish[:,0:sub[k]]
    print(np.shape(time_fish))
    #Compute predictability as a function of delay
    ## Basically a grid search
    n_seed_range= np.arange(30,100,30)# Number of partitions to examine
    range_Ks =  np.arange(2,30,2,dtype=int) #range of delays to study
    h_K=np.zeros((len(range_Ks),len(n_seed_range)))
    for k,K in enumerate(range_Ks):
        traj_matrix = embed.trajectory_matrix(time_fish,K=K-1)
        for ks,n_seeds in enumerate(n_seed_range):
            labels=cl.kmeans_knn_partition(traj_matrix,n_seeds)
            h = op_calc.get_entropy(labels)
            h_K[k,ks]=h
    b=time.clock()
    ctime=b-a
    comp_Time.append(ctime)
    
print(comp_Time)
    
#%%

plt.plot(sub ,comp_Time)
plt.xlabel('# Neurons',fontsize=15)
plt.ylabel('Computational time (in s)',fontsize=15)
plt.show()

















