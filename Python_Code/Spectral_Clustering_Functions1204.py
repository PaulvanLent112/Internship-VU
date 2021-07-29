# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:09:28 2021

@author: paulv
"""


from scipy import sparse
from scipy import linalg
from scipy import stats
import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sb
import umap

import msmtools.estimation as msm_estimation
import msmtools.analysis as msm_analysis
import numpy.ma as ma
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import diags,identity,coo_matrix

#https://www.kdnuggets.com/2020/05/getting-started-spectral-clustering.html 
#https://towardsdatascience.com/spectral-graph-clustering-and-optimal-number-of-clusters-estimation-32704189afbe

        
def generate_graph_laplacian(A):
    """Make a graph laplacian from a transition probability matrix"""

    #Create symmetric matrix
    n=np.shape(A)[0]
    #A=0.5* (A+ A.T)
            
    #D is just the identity matrix (because sum(P)=1)
    Degree=np.sum(A,1)
    D=np.diag(Degree)
    
    #Laplacian matrix
    L=D-A
    return L



def compute_spectrum2(P):
    
    """This one takes the left eigenvalue"""
    evals, lvecs= linalg.eig(P,right=False, left=True)
    evals1= np.real(evals)
    lvecs1= np.real(lvecs)
    evals_c=np.imag(evals)
    lvecs_c= np.imag(lvecs)
    
    lvecs1 = lvecs1/lvecs1.sum(axis=0, keepdims=True)
    
    return evals1, lvecs1, evals_c, lvecs_c
    

def trajectory_matrix(X,K):
    '''
    
    Build a trajectory matrix (Code taken from Antonio Costa)
    X: N x dim data
    K: the number of delays
    out: (N-K)x(dim*K) dimensional
    '''
    if ma.count_masked(X)>0:
        traj_matrix = ma.zeros((X.shape[0],X.shape[1]*(K+1)))
        traj_matrix[int(np.floor(K/2)):-int(np.ceil(K/2)+1)] = ma.vstack([ma.hstack(np.flip(X[t:t+K+1,:],axis=0)) for t in range(len(X)-K-1)])
        traj_matrix[np.any(traj_matrix.mask,axis=1)]=ma.masked
        traj_matrix[traj_matrix==0]=ma.masked
        return traj_matrix
    else:
        return np.vstack([np.hstack(np.flip(X[t:t+K+1,:],axis=0)) for t in range(len(X)-K-1)])
    
def kmeans_clustering(proj_df, k):
    """ This functions clusters the states i into groups chosen by k
    based on the original transition probability matrix projected on the first
    X eigenvectors
    
    input: 
        - the projected dataframe 
        - k: number of clusters
        
    output: 
        labels: the kmeans labels for the delay embedded data we consider
    """
    k_means= k_means = KMeans(random_state=25, n_clusters=k)
    k_means.fit(proj_df)
    labels= k_means.predict(proj_df)
    
    return labels
    

def stationary_distribution(P):
    '''
    Corresponds to taking the first left eigenvector
    '''
    probs = msm_analysis.statdist(P)
    return probs


def get_reversible_transition_matrix2(P, stat_dist):
    probs = stat_dist
    P_hat = diags(1/probs)*P.transpose()*diags(probs)
    R=(P+P_hat)/2
    return R

def get_reversible_transition_matrix(P):
    probs = stationary_distribution(P)
    P_hat = diags(1/probs)*P.transpose()*diags(probs)
    R=(P+P_hat)/2
    return R
#%%
  
if __name__=="__main__":
    transition_probability_matrix= np.genfromtxt("data/wt_midbrain_0721_fish10_TPM.csv",delimiter=",")
    
    eigenvals, eigenvcts, c1,c2=compute_spectrum2(transition_probability_matrix)
 

#%%   
    stat_dist= eigenvcts[:,0]
 #%%


   
    symm_tpm= get_reversible_transition_matrix2(transition_probability_matrix, stat_dist)
    
#%%
    eigenvals2, eigenvcts2, eigenvals_c2, eigenvcts_c2=compute_spectrum2(symm_tpm)
    
    states=np.arange(0,len(stat_dist),1)
    plt.scatter(states, stat_dist, c=eigenvcts[:,1], cmap="inferno")
    
    sorted_eigenvals= np.sort(eigenvals2)
    
    #neurons= np.load("com_200701_fish3.npy")
    fish=np.genfromtxt('forUVAanalysis/wildtype_midbrain/dff_200721fish10.txt', delimiter=' ')
    
    fish=np.transpose(fish)
    fish= trajectory_matrix(fish,30)
    #graph_laplacian, A= generate_graph_laplacian(TPM)
    #eigenvals, eigenvcts= compute_spectrum_graph_laplacian(np.transpose(A))
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(fish)
    
    norm_eig=stats.zscore(eigenvcts[:,1])
    plt.scatter(principalComponents[:,0], principalComponents[:,1], c=norm_eig, cmap="inferno")
    
    




#%%
T= np.array([[0.2,0.3,0.4,0.1],
             [0.1,0,0.5,0.4],
             [0,0.2,0.1,0.7],
             [0.5,0,0.5,0]])
print(stationary_distribution(T))

stat= np.linalg.matrix_power(T, 1000)[1,:]
print(get_reversible_transition_matrix(T))
NT=get_reversible_transition_matrix(T)
NT2= get_reversible_transition_matrix2(T, stat)
eigenvals1, eigenvcts1, eigenvals_c1, eigenvcts_c1=compute_spectrum2(NT)
eigenvals2, eigenvcts2, eigenvals_c2, eigenvcts_c2=compute_spectrum2(NT2)

#%%
array_sort= np.sort(abs(eigenvals2))
reverse= np.sort(array_sort)[::-1]

