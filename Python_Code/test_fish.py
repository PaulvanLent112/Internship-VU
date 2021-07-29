# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:01:22 2021

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



#%%
####### Import data, transpose (such that # rows is time, #cols is neurons)
fish=np.genfromtxt('200701_fish3_dff.txt', delimiter=' ')
fish=np.transpose(fish)
fish=fish[:,]
print(np.shape(fish))


#%%
#n_seeds=2
#labels = cl.kmeans_knn_partition(fish,n_seeds)

#Compute predictability as a function of delay
## Basically a grid search
n_seed_range= np.arange(30,100,10)# Number of partitions to examine
range_Ks =  np.arange(2,30,2,dtype=int) #range of delays to study
h_K=np.zeros((len(range_Ks),len(n_seed_range)))
for k,K in enumerate(range_Ks):
    traj_matrix = embed.trajectory_matrix(fish,K=K-1)
    for ks,n_seeds in enumerate(n_seed_range):
        labels=cl.kmeans_knn_partition(traj_matrix,n_seeds)
        h = op_calc.get_entropy(labels)
        h_K[k,ks]=h



#%%
#This plot shows the delay embeddings over time (entropy decreases as k is increased)
plt.plot(n_seed_range,h_K.T)
plt.xlabel('N',fontsize=15)
plt.ylabel('h (nats/s)',fontsize=15)
plt.show()

#This plot takes the last partition (e.g. 300), and shows the decrease in entropy for increasing K
plt.plot(range_Ks,h_K[:,-6])
plt.xlabel('K (frames)',fontsize=15)
plt.ylabel('h (nats/s)',fontsize=15)
plt.show()



#%%
labels, centers= cl.kmeans_knn_partition(fish, n_seeds=60, return_centers=True)



P = op_calc.transition_matrix(labels,16)
plt.imshow(P.todense(),vmax=.2)
plt.colorbar()
plt.show()


print(P)

eigvals,eigvecs = op_calc.sorted_spectrum(P,k=30)
plt.scatter(eigvals.real,eigvals.imag,s=1)
plt.show()

R = op_calc.get_reversible_transition_matrix(P)
plt.imshow(R.todense(),vmax=.2)
plt.colorbar()
plt.show()

eigvals,eigvecs = op_calc.sorted_spectrum(R,k=3)
print(eigvals)


X=eigvecs[:,1].real


#%%
print(labels)
print(np.shape(centers))


from scipy.sparse import csr_matrix
P_1=csr_matrix.toarray(P)

MM=np.linalg.matrix_power(P_1, 100)
print(MM[1,:])