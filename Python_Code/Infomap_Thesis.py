# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 14:31:46 2021

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

Tran_Mat= VU.load_file("data/wt_midbrain_0721_fish10_TPM.csv")
Infomap= VU.TPM_Infomap(Tran_Mat)
Infomap.run()

ind_set= np.arange(0,969,1)
labels=Infomap.get_modules()
labels=list(labels.values())

#%%
#This is the partitioning using infomap, colorcoded
partitions= np.zeros((200, len(labels)))
for i in range(200):
    partitions[i,:]=labels



cm=1/2.54
fig = plt.figure(figsize=(36*cm, 16*cm))
ax = fig.add_axes([0,0,1,1])
ax.imshow(partitions.T,cmap="Dark2")
ax.set_xlabel("")
ax.set_ylabel("Timesteps")
ax.set_xticks([])
#plt.savefig('Figures_Thesis/Infomap_coarsegraining.png', dpi=300, transparent=False, bbox_inches='tight')
        
#%%
#we are interested whether the partitioning captures similar patterns in the neural population
#The state partition are therefore further partitioned into states that are subsequently following in time, and 
#between these states we want to calculate the correlation between these states        
state_dictionary=VU.coherent_subsequent_states(labels)

fish10=np.genfromtxt("forUVAanalysis/wildtype_midbrain/dff_200721fish10.txt",delimiter=" ")
K=30  
fish10=fish10.T

#%%
                
NPS_per_partition, state_name= VU.fish_correlation(state_dictionary, fish10, K)
Correlation_between_partitions=np.corrcoef(NPS_per_partition)
        
        
#%%
#Alternative: calculate correlation between partitions


def subsetting_states(fish,infomap_labels,label_of_interest, K):
    indices= np.array(np.where(infomap_labels==label_of_interest))
    indices= indices+K-1
    fish=fish[indices,:]
    average_state= np.mean(fish,1).T
    return average_state

state=subsetting_states(fish10, labels, 1, K)

#%%

state_ID=[]
for i in range(len(state_name)):
    state_ID.append("p " + str(state_name[i][0])+","+str(state_name[i][1]))




#Visualize correlations
cm=1/2.54
fig = plt.figure(figsize=(36*cm, 16*cm))
ax = fig.add_axes([0,0,1,1])
plt.imshow(Correlation_between_partitions,cmap="seismic")
plt.colorbar()
ax.set_xticks(np.arange(len(state_ID)))
ax.set_yticks(np.arange(len(state_ID)))
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
# Loop over data dimensions and create text annotations.
ax.set_xticklabels(state_ID)
ax.set_yticklabels(state_ID)
plt.savefig('Figures_Thesis/Correlation_betweenpartitions.png', dpi=300, transparent=False, bbox_inches='tight')       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        