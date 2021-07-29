# -*- coding: utf-8 -*-
"""
Created on Wed May 19 08:59:52 2021

@author: paulv
"""
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
from scipy import sparse
import sys
sys.path.append("PF_approx-master/")
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

cm=1/2.54
x=np.arange(170,250,1)
fig = plt.figure(figsize=(12*cm, 10*cm))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tseries[170:250,0], c="#b51d14")
ax.set_xlabel("time")
ax.set_ylabel("x-axis position")
ax.set_xticks([])
plt.savefig('Figures_Thesis/zoomed_intseries.png', dpi=300, transparent=False, bbox_inches='tight')

#%%
import pandas as pd
from sklearn.decomposition import PCA
r=embed.trajectory_matrix(tseries[1:10000,0:1],3)

pca = PCA(n_components=2)
PCs = pca.fit_transform(r)
var_exp= "Var: " +str(np.round(np.sum(pca.explained_variance_ratio_)*100,2))+"%"






#%%
cm=1/2.54
fig = plt.figure(figsize=(12*cm, 10*cm))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(PCs[:,0],PCs[:,1], c="#00b25d", alpha=0.3,s=3)
#ax.set_xlabel("")
ax.set_ylabel("PC 2")
ax.set_xlabel("PC 1")

plt.savefig('Figures_Thesis/reconstructed.png', dpi=300, transparent=False, bbox_inches='tight')


#%%
#Lorenz system
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cm=1/2.54
fig = plt.figure(figsize=(12*cm, 10*cm))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tseries[1:10000,0],tseries[1:10000,1],tseries[1:10000,2], c="#00b25d", alpha=0.3,s=3)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#ax.set_xlabel("")
ax.set_ylabel("y")

ax.set_xlabel("x")

plt.savefig('Figures_Thesis/lorenz.png', dpi=300, transparent=False, bbox_inches='tight')



#%%
sys.path.append('D:\Internship VU\Python\functions')
from scipy import stats
import functions.functions as VU
#%%
Transition_Matrix=VU.load_file("lorenz.csv")



#%%
evals, lvecs=VU.compute_spectrum(Transition_Matrix)


#%%

Infomap=VU.TPM_Infomap(Transition_Matrix)

#%%
Infomap.run() 


#%%
label=Infomap.get_modules()
#%%
P=sparse.csr_matrix(Transition_Matrix)

from sklearn.preprocessing import normalize

P = normalize(P, axis=1, norm='l1')

#%%

R=op_calc.get_reversible_transition_matrix(P)
evals, lvecs=op_calc.sorted_spectrum(P, k=3)
#%%

cm=1/2.54
fig = plt.figure(figsize=(12*cm, 10*cm))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(PCs[1:9995:,0],PCs[1:9995,1],c=list(label.values()),s=3)
#ax.set_xlabel("")
ax.set_ylabel("PC 2")
ax.set_xlabel("PC 1")


#%%

cm=1/2.54
fig = plt.figure(figsize=(12*cm, 10*cm))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(PCs[1:9995:,0],PCs[1:9995,1],c=lvecs.real[:,1],s=3)
#ax.set_xlabel("")
ax.set_ylabel("PC 2")
ax.set_xlabel("PC 1")


