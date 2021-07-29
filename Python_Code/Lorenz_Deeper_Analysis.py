# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 12:54:34 2021

@author: paulv
"""
#%%
#data format library
import h5py
#numpy
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import os
from scipy.integrate import odeint
#data format library

import sys
sys.path.append('../')
import clustering_methods as cl
import operator_calculations as op_calc
import delay_embedding as embed

#%%


def Lorenz(state,t,sigma,rho,beta):
    # unpack the state vector
    x,y,z = state
    # compute state derivatives
    xd = sigma * (y-x)
    yd = (rho-z)*x - y
    zd = x*y - beta*z
    # return the state derivatives
    return [xd, yd, zd]
dt = 0.01
frameRate=1/dt
T = 10000
state0 = np.array([-8, -8, 27])
t = np.linspace(0, T, int(T*frameRate))
sigma,rho,beta=10,28,8/3
tseries=np.array(odeint(Lorenz,state0,t,args=(sigma,rho,beta)),dtype=np.float64)[int(len(t)/2):]



#%%
n_seeds = 500

labels = ma.array(cl.kmeans_knn_partition(tseries,n_seeds),dtype=int)

#%%

delay = 15
P = op_calc.transition_matrix(labels,delay)

#%%
eigvals,eigvecs=op_calc.sorted_spectrum(P,k=3)

#%%
R = op_calc.get_reversible_transition_matrix(P)
eigvals,eigvecs=op_calc.sorted_spectrum(R,k=3)


#%%
X=eigvecs[:,2].imag


plt.figure(figsize=(10,7))
color_abs = np.max(np.abs(X))
plt.scatter(tseries[:,0],tseries[:,2],c=X[labels],cmap='hsv',s=1,vmin=-color_abs,vmax=color_abs)
plt.xticks(range(-20,21,5))
# plt.ticks()
# plt.savefig('Phi_2_Lorenz.png')
plt.show()

#%%
plt.plot(X[1:90])


#%%
#Infomap and its use on Antonio's transfer operator
sys.path.append('D:\Internship VU\Python\functions')
from scipy import stats
import functions.functions as VU
#Transition_Matrix=VU.load_file("lorenz.csv")
Infomap=VU.TPM_Infomap(P)


Infomap.run() 
#%%

label=Infomap.get_modules()
#%%
plt.figure(figsize=(10,7))
color_abs = np.max(np.abs(X))
plt.scatter(tseries[:,0],tseries[:,2],c=label.keys,cmap='hsv',s=1,vmin=-color_abs,vmax=color_abs)
plt.xticks(range(-20,21,5))
# plt.ticks()
# plt.savefig('Phi_2_Lorenz.png')
plt.show()

#%%

labs_over_time=np.array(list(label.values()))
labs_over_time=labs_over_time.reshape(-1,1)


plt.figure(figsize=(10, 8), dpi=80)
plt.imshow(labs_over_time[1:100], cmap="Dark2")


