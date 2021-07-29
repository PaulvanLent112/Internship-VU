# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 10:31:39 2021

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
filenames=os.listdir("data/")

nr_of_modules=[]
relative_codelength=[]
for i in range(len(filenames)):
    Transition_Matrix= VU.load_file("data/"+filenames[i])
    P=Transition_Matrix
    Infomap= VU.TPM_Infomap(P)
    #x.to_csv("test.txt",sep="\t", index=False)
    Infomap.run()

    print(Infomap.tree)
    print(Infomap.num_top_modules)
    print(f"Found {Infomap.num_top_modules} modules with codelength: {Infomap.codelength}")
    print("Result")
    print("\n#node module")

    nr_of_modules.append(Infomap.num_top_modules)
    relative_codelength.append(Infomap.relative_codelength_savings)
    print(Infomap.relative_codelength_savings)

    modules=[]
    for node in Infomap.tree:
        if node.is_leaf:
            temp= [node.node_id, node.module_id]
            modules.append(temp)

    modules= np.array(modules)
    modules=modules[np.argsort(modules[:,0])]

    partition_map=VU.colorcoding_partitions(modules[:,1])

    # A plot of the partition map
    plt.figure(figsize=(10,10))
    plt.imshow(partition_map, cmap='tab20')
    plt.show()


#%%
G=nx.DiGraph(Transition_Matrix)
keys=np.unique(modules[:,1])
print(keys)
colors=["b","g","r","c","m","y","b","w"]
mylabs=dict(zip(keys,colors))

col=[]
for i in range(len(modules[:,1])):
    temp=mylabs.get(modules[i,1])
    col.append(temp)
#%%
#VU.plot_states_graph(G,col)

#%%
fish= np.genfromtxt("forUVAanalysis/wildtype_midbrain/dff_200721fish10.txt",delimiter=" ")
fish=fish.T
fish=stats.zscore(fish)
K=1000-np.shape(modules)[0]





#%%

counts, rs_counts_mean, rs_counts_std, my_neurons=VU.count_on_neurons(fish, modules[:,1],34, K,3)
plt.plot(counts,"bo")
plt.fill_between(range(len(counts)), 0, rs_counts_mean+3*rs_counts_std,color="red", alpha = 0.3)
plt.xlabel("neurons")
plt.ylabel("probability")


#%% Now look at the 3d structure, do we see some spatial organization
coord=np.load("cellCoordinates/com_200721_fish10.npy")

#%%

n_modules=np.unique(modules[:,1])

for i in range(len(n_modules)):
    counts, rs_counts_mean, rs_counts_std, my_neurons=VU.count_on_neurons(fish, modules[:,1],i+1, K,3)
    color_code=np.zeros(np.shape(coord)[0])
    color_code[my_neurons]=1
    VU.color_coding_neurons(coord, color_code)
 
    
#%%

# Relative codelength savings
data= ["dm_forebrain","dm_forebrain","dm_forebrain","dm_midbrain",
       "dm_midbrain","dm_midbrain","dm_midbrain","dm_midbrain",
       "wt_forebrain","wt_forebrain","wt_forebrain","wt_forebrain",
       "wt_forebrain","wt_midbrain","wt_midbrain","wt_midbrain",
       "wt_midbrain","wt_midbrain"]
plt.plot(data, relative_codelength, "bo")



#%%
fig=plt.figure(frameon=False, figsize=(12,6))
im1= plt.imshow(fish, cmap='binary',vmin=0, vmax=0.5)


my_lab=np.ones((3082,1000))
for i in range(K,1000):

    my_lab[:,i]=modules[i-K,1]
my_lab[:,0:K]=0
im2=plt.imshow(my_lab.T,cmap="Paired", alpha=0.4)


plt.xlabel("Neurons")
plt.ylabel("Timesteps")
plt.show()

#%%
from matplotlib.pyplot import figure
labels=np.zeros((200,len(modules[:,1])))
for i in range(np.shape(labels)[0]):
    labels[i,:]=modules[:,1]

figure(figsize=(10, 8), dpi=80)
plt.imshow(labels.T, cmap="Dark2")
plt.gca().set_xticks([])


#%%

def subsetting_states(fish,infomap_labels,label_of_interest, K):
    indices= np.array(np.where(infomap_labels==label_of_interest))
    indices= indices+K-1
    fish=fish[indices,:]
    average_state= np.mean(fish,1).T
    return average_state



color_code= subsetting_states(fish, modules[:,1], 8,K)
color_code= color_code.reshape(color_code.shape[0])

cm=1/2.54
fig = plt.figure(figsize=(12*cm, 12*cm))
ax = fig.add_axes([0,0,1,1])
plt.scatter(coord[:,0],coord[:,1],c=color_code,cmap="seismic")
plt.clim(0,2)
plt.xlabel("x-coordinate")
plt.ylabel("y-coordinate")
cbar=plt.colorbar()
plt.savefig('Figures_Thesis/fluorescence8.png', dpi=300, transparent=False, bbox_inches='tight')  


#%%
#plot of relative codelength savings
dm_midbrain_ind= np.arange(3,8,1)
wt_midbrain_ind= np.arange(13,18,1)
wt_forebrain_ind=np.arange(8,13,1)
dm_forebrain_ind=np.arange(0,3,1)
cm=1/2.54

relative_codelength= np.array(relative_codelength)

x=np.array([1,2,3,4])

#plt.boxplot(relative_codelength[wt_midbrain_ind],0,'',widths=0.3,patch_artist=True,
 #           boxprops=dict(facecolor="#ddb310", color="black"),
  #          capprops=dict(color="black"),
   #         medianprops=dict(color="black"))

#plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['forebrain wildtype', 'forebrain CSF1R-mutant'], loc='upper right')
#plt.xticks(x,x)

plt.boxplot(relative_codelength[wt_midbrain_ind])
plt.boxplot(relative_codelength[dm_midbrain_ind],positions=)
plt.xticks([1, 2, 3], ['wt_midbrain', 'tue', 'wed'])

#%%
print(stats.ttest_ind(relative_codelength[wt_forebrain_ind],relative_codelength[wt_midbrain_ind]))
