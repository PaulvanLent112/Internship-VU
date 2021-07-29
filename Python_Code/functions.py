# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 14:15:46 2021

@author: paulv


Functions used for the data analysis of the Zebrafish project
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import scipy
from mpl_toolkits import mplot3d
import sys
sys.path.append('D:\Internship VU\Python\PF_approx-master')
#but is incompatible with python 3.8
from scipy import linalg
from scipy.sparse import diags,identity,coo_matrix
from sklearn.cluster import KMeans
import infomap
import networkx as nx
from scipy import stats
import clustering_methods as cl
import operator_calculations as op_calc  #This requires msmtools, and this 
#but is incompatible with python 3.8
import delay_embedding as embed


def load_file(filename):
    """Loading of a file """
    file =np.genfromtxt(filename, delimiter=',')
    return file


def compute_spectrum(P):
    """Calculates eigenvalues and eigenvectors of the transition probability matrix"""
    evals, lvecs= linalg.eig(P,right=False, left=True)

    lvecs = lvecs/lvecs.sum(axis=0, keepdims=True)
    
    return evals, lvecs


def get_highly_probable_states(stat_dist, threshold):
    """Given the threshold retrieves the indices for likely states"""
    states=[]
    for i in range(len(stat_dist)):
        if stat_dist[i]>threshold:
            states.append(i)
    return states


def get_on_neurons(Fish,state):
    """given the chosen state, what neurons are on and off for the labeling
    input: fish (zscored), state before delay embedding
    """
    state_snap= Fish[state,:]
    
    label=[]
    for i in range(len(state_snap)):
        if state_snap[i]>2:
            label.append(1)
        else: 
            label.append(0)
    return label


def color_coding_neurons(Coordinates, Color_Code):
    """Generates a 3D plot colorcoded by the detected neuron assemblies
    Input: 
        
    3D coordinates: a numpy array of X by 3 corresponding to x,y,z coordinates
    Color code vector: assembly identities (length X). E.g. 
    [0,0,0,0,1,2,1,1,2,2,0]. Numpy array
    
    Output: 3D plot"""
    
    keys=np.unique(Color_Code)
    values=["grey","red","y","k","r","c","m"]
    dictionary = dict(zip(keys, values))
    print(dictionary)
    
    col=[]
    for i in range(0,np.shape(Color_Code)[0]):
        col.append(dictionary.get(Color_Code[i]))
    
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Coordinates[:,0],Coordinates[:,1],Coordinates[:,2], c=col, s=10)
    # plot the point (2,3,4) on the figure
    plt.show()
    
    
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
    
    
def generate_graph_laplacian(A):
    """Make a graph laplacian from a transition probability matrix"""

    #Create symmetric matrix
    #A=0.5* (A+ A.T)
            
    #D is just the identity matrix (because sum(P)=1)
    Degree=np.sum(A,1)
    D=np.diag(Degree)
    
    #Laplacian matrix
    L=D-A
    return L

def get_entropy(TPM, stationary_dist):
    '''
    Compute entropy rate of the symbolic sequence
    labels: masked array of ints
    returns entropy rate (code taken from Antonio Costa)
    '''
    
    #get dtrajs to deal with possible nans
    P = scipy.sparse.csr_matrix(TPM)
    probs = stationary_dist
    logP = P.copy()
    logP.data = np.log(logP.data)
    return (-diags(probs).dot(P.multiply(logP))).sum()



def kmeans_clustering(proj_df, k):
    """ This functions clusters the states i into groups chosen by k
    based on the original transition probability matrix projected on the first
    X eigenvectors (Code taken from Antonio Costa)
    
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


    
def TPM_Infomap(P):
    """Function for making an infomap based on the transition probability_matrix
    Output: link_list
    """
    im =infomap.Infomap("--directed")
    for i in range(np.shape(P)[0]):
        for k in range(np.shape(P)[1]):
            if P[i,k]==0:
                continue
            else: 
                im.add_link(i,k,P[i,k])

        
    return im

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

def colorcoding_partitions(partition_labels):
    """Creates a N*N matrix of the matrix partitioned based on the infomap labels
    Input: partition_label: labels of the states from the InfoMap algorithm  (or any other labeling of states based on a clustering result)
    """
    mat=np.zeros((len(partition_labels),len(partition_labels)))
    for i in range(len(partition_labels)):
        for j in range(len(partition_labels)):
            if partition_labels[i]==partition_labels[j]:
                mat[i,j]=partition_labels[i]
            else:
                mat[i,j]=0
    return mat
    
    

def plot_states_graph(G, color_labels):
    """Plot all states in a graph from networkx, this function will take some time.
    Inspired by: https://orbifold.net/default/community-detection-using-networkx/
    Input: Networkx graph G, coloring: the labels from the InfoMap """
    pos = nx.spring_layout(G, k=0.1)
    plt.rcParams.update({'figure.figsize': (7, 7)})
    nx.draw_networkx(
        G, 
        pos=pos, 
        node_size=20, 
        node_color=color_labels ,
        arrowsize=0.001,
        edge_color="#C0C0C0", 
        alpha=0.3, 
        with_labels=False)
    plt.gca().set_facecolor("white")    
    
    
    
def count_on_neurons(fish, labeling, label_nr, K, std):
    """This functions gives the count of how many times a neuron is on given a certain clustering
    Input: calcium dataset (fish), labeling (Infomap labels), label_nr (integer), K (delay steps),
    std is the number of standard deviations a probability has to be from for identifying neurons that are
    significantly more on than expected from the data"""
    z_fish=stats.zscore(fish)
    r,c = np.shape(z_fish)
    b_fish=np.zeros((r,c))
    for i in range(r):
        for k in range(c):
            if z_fish[i,k]<2:
                b_fish[i,k]=0
            else:
                b_fish[i,k]=1
    
    #Count the probability of a neuron being on (for the whole b_fish)
    

    #This subsets given the labeling. Adds K.
    original_label=[]
    for i in range(len(labeling)):
        if labeling[i]== label_nr:
            original_label.append(i+K)
    print(len(original_label))
    
    #Count the probability of a neuron being on (for the whole b_fish) 
    #Of equal group sizes for appropriate comparison
    counts_full=[]
    for i in range(1000):
        indices=np.random.randint(0,1000,len(original_label))
        random_fish=b_fish[indices,:]
        rs_counts= np.sum(random_fish,0)/ len(original_label)
        counts_full.append(rs_counts)
    counts_full= np.array(counts_full)
    rs_counts_mean= np.mean(counts_full,0)
    rs_counts_std= np.std(counts_full,0)

    #With these labels, subset b_fish
    b_fish= b_fish[original_label,:]
    counts= np.sum(b_fish,0)
    for i in range(len(counts)):
        counts[i]= counts[i]/len(original_label)
    
    my_neurons=[]
    for i in range(len(counts)):
        if counts[i]> rs_counts_mean[i]+(3*rs_counts_std[i]):
            my_neurons.append(i)
    
    
    return counts, rs_counts_mean, rs_counts_std, my_neurons


def groupSequence(lst):
    "stole this from: https://www.geeksforgeeks.org/python-find-groups-of-strictly-increasing-numbers-in-a-list/"
    res = [[lst[0]]]
  
    for i in range(1, len(lst)):
        if lst[i-1]+1 == lst[i]:
            res[-1].append(lst[i])
  
        else:
            res.append([lst[i]])
    return res
      




def coherent_subsequent_states(Infomap_labels):
    """Given a labeling, we first partition them to their corresponding set (so if it is module 1
    , then index 1,2,3,4,8,9,10. Then in this dictionary, we identify states that are subsequent
    Input: labels 
    Output: {1:[[1,2,3],[8,9,10,11]], 2: etc.}"""
    unique_labels= np.unique(Infomap_labels)
    dictionary= {}
    for i in range(len(unique_labels)):
        label_index=[]
        for j in range(len(Infomap_labels)):
            if unique_labels[i]==Infomap_labels[j]:
                label_index.append(j)
        subsequent=groupSequence(label_index)
        
        dictionary[i]=subsequent
        
    return dictionary
        
        
def fish_correlation(state_dictionary, fish, K):
    """Given the state_dictionary, which subsets partitions into subsequent states, these states are independently subset
    and the mean fluorescence is calculated. 
    input: state_dictionary, dataset calcium fluorescence over time
    output: correlation_matrix"""
    neural_population_state=[]
    partition_place=[]
    for i in range(len(state_dictionary)):
        partition_states=state_dictionary[i]
        for j in range(len(partition_states)):
            temp= partition_states[j]
            my_state=[x+K for x in temp] #adds K to the state_ID
            nps=np.mean(fish[my_state,:],0)
            neural_population_state.append(nps)
            partition_place.append([i,j])
    neural_population_state=np.array(neural_population_state)
   # neural_population_corr= np.corrcoef(neural_population_state)
    return neural_population_state, partition_place
            



