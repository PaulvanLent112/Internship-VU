# -*- coding: utf-8 -*-
"""
Created on Mon May 31 13:02:05 2021

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
sys.path.append('D:\Internship VU\Python\PF_approx-master')
import clustering_methods as cl
import operator_calculations as op_calc  #This requires msmtools, and this 
#but is incompatible with python 3.8
import delay_embedding as embed

#%%

Transition_matrix=VU.load_file("data/"+"wt_midbrain_0721_fish10_TPM.csv")


#%%
evals, lvecs, evals_c, lvecs_c=VU.compute_spectrum(Transition_matrix)


#%%
#visualize transition matrix

sqrt_tm= np.sqrt(Transition_matrix)
x= np.arange(0,25,0.025)
cm=1/2.54
fig = plt.figure(figsize=(12*cm, 16*cm))
ax = fig.add_axes([0,0,1,1])
ax.imshow(sqrt_tm,cmap="viridis", vmin=0.01, vmax=np.max(sqrt_tm))
ax.set_xlabel("States")
ax.set_ylabel("States")
plt.savefig('Figures_Thesis/Neural_Population_Activity.png', dpi=300, transparent=False, bbox_inches='tight')

#%%
#The Stationary distribution 


cm=1/2.54
fig = plt.figure(figsize=(9*cm, 10*cm))
ax = fig.add_axes([0,0,1,1])
ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=5, width=1, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=5, width=1, direction='in', right='on')
ax.plot(lvecs[:,0]*100, c="#b51d14")
ax.set_xlabel("States")
ax.set_ylabel("Time spent in each state (in %)")
plt.savefig('Figures_Thesis/stat_dist.png', dpi=300, transparent=False, bbox_inches='tight')

#%%
# Complex eigenvalue plot for the largest complex number (reveals the oscillatory pattern)
sorted_complex_eigenvalues=np.argsort(evals_c)[::-1]
cm=1/2.54
fig = plt.figure(figsize=(9*cm, 10*cm))
ax = fig.add_axes([0,0,1,1])
ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=5, width=1, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=5, width=1, direction='in', right='on')
ax.plot(lvecs_c[:,8], c="#b51d14")
ax.set_xlabel("States")
ax.set_ylabel("complex largest ")
#plt.savefig('Figures_Thesis/second_eigenvector.png', dpi=300, transparent=False, bbox_inches='tight')


#%%

T= np.arange(0,75,1.5)
cm=1/2.54
fig = plt.figure(figsize=(9*cm, 10*cm))
ax = fig.add_axes([-1,-1,1,1])
ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=5, width=1, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=5, width=1, direction='in', right='on')
ax.scatter(evals,evals_c, c="#b51d14", s=10)
ax.set_xlabel("real $\lambda$")
ax.set_ylabel("complex $\lambda$")
ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
#plt.savefig('Figures_Thesis/eigenvalues.png', dpi=300, transparent=False, bbox_inches='tight')

#%%

T= np.arange(0,75,1.5)
cm=1/2.54
fig = plt.figure(figsize=(9*cm, 10*cm))
ax = fig.add_axes([-1,-1,1,1])
ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=5, width=1, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=5, width=1, direction='in', right='on')
ax.scatter(lvecs[:,8],lvecs_c[:,8], c="#b51d14", s=10)
ax.set_xlabel("real $\lambda$")
ax.set_ylabel("complex $\lambda$")
ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
#plt.savefig('Figures_Thesis/eigenvalues.png', dpi=300, transparent=False, bbox_inches='tight')



#%%
np.savetxt("lorenz_delayembeddedx_axis.csv", r,delimiter=",")


