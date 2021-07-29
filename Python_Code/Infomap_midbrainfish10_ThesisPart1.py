# -*- coding: utf-8 -*-
"""
Created on Mon May 24 15:19:36 2021

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



Transition_Matrix= VU.load_file("data/"+"wt_midbrain_0721_fish10_TPM.csv")
P=Transition_Matrix
Infomap= VU.TPM_Infomap(P)
    #x.to_csv("test.txt",sep="\t", index=False)
Infomap.run()

#%%
Infomap_modules= list(Infomap.get_modules().values())

from matplotlib.pyplot import figure
labels=np.zeros((200,len(Infomap_modules)))
for i in range(200):
    labels[i,:]=Infomap_modules


#Coarse grained representation of the dynamics 
figure(figsize=(10, 8), dpi=80)
plt.imshow(labels.T, cmap="Dark2")
plt.gca().set_xticks([])
plt.ylabel("Timesteps")



#%%
#Load fish data
fish= np.genfromtxt("forUVAanalysis/wildtype_midbrain/dff_200721fish10.txt",delimiter=" ")
fish=fish.T
fish=stats.zscore(fish,0)
K=1000-len(Infomap_modules)-1

#%%
#Subsetting the states and average over the partition of interest (sometimes done manually for recurrent states)

# On/of states neurons
labels= np.array(Infomap_modules)

subsetted_states=np.array(np.where(labels==7))[0,:]
subsetted_states=subsetted_states+K

subset_fish=fish[subsetted_states,:]
mean=np.mean(subset_fish,0)

#%%
#Now we look at the spatial organization of neurons and their activity pattern
coord=np.load("cellCoordinates/com_200721_fish10.npy")

plt.scatter(coord[:,0],coord[:,1], c=mean,cmap="magma")
plt.clim(0,2)
plt.colorbar()
plt.xlabel("x position")
plt.ylabel("y position")
plt.show()


plt.scatter(coord[:,1],coord[:,2], c=mean,cmap="magma")
plt.clim(0,2)
plt.colorbar()
plt.xlabel("y position")
plt.ylabel("z postion")
plt.show()

