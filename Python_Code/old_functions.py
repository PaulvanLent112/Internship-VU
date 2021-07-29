# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:05:30 2021

@author: paulv
"""

def spatial_intersection(X, state_1, state_2):
    """given a random sampling, what is the overlap between chosen neurons
    - input: X: number of neurons to draw the two states from
     state_1: number of on neurons, state 1
     state_2: number of on neurons, state 2
    
    
    """
    intersections=[]
    for i in range(1000):
        neurons= list(np.arange(0,X))
        a= random.sample(neurons, 25)
        b= random.sample(neurons, 30)
        c=len(list(set(a) & set(b)))
        intersections.append(c)
    
    intersections=np.array(intersections)
    mean= np.mean(intersections)
    std= np.std(intersections)
    
    return intersections, mean, std

def intersecting_neurons_counter(two_neurons):
    """ counts the number of neurons that are on in both states of the above function"""
    r,c= np.shape(two_neurons)
    counter=[]
    on_neurons_s1=[]
    on_neurons_s2=[]
    
    for i in range(r):
        temp=0
        temp2=0
        temp3=0
        for j in range(c):
            if two_neurons[i,j]==3:
                temp+=1
            elif two_neurons[i,j]==2:
                temp2+=1
            elif two_neurons[i,j]==1:
                temp3+=1
                
        counter.append(temp)
        on_neurons_s1.append(temp2)
        on_neurons_s2.append(temp3)
    
    return counter, on_neurons_s1, on_neurons_s2
            
def stupid_function(states, states2):
    """Just a try out to see if partitions of the state space
    are spatially separated (when it comes to the neurons that are on)
    """
    minimum_row=min(np.shape(states)[0], np.shape(states2)[0])
    states= states[0:minimum_row,]
    states2= states2[0:minimum_row]
    
    #now we colorcode: if state_partition 1 and state_partition 2 is off--> 0
    # if state_partition1 is on and the other of 1, vice versa 2
    # if both are one 3
    r,c= np.shape(states)
    for i in range(r):
        for j in range(c):
            if states[i,j]==0 and states2[i,j]==0:
                states[i,j]=0
            elif states[i,j]==0 and states2[i,j]==1:
                states[i,j]=1
            elif states[i,j]==1 and states2[i,j]==0:
                states[i,j]=2
            elif states[i,j]==1 and states2[i,j]==1:
                states[i,j]=3
    
    return states
                
def color_coding_neurons(Coordinates, Color_Code):
    """Generates a 3D plot colorcoded by the detected neuron assemblies
    Input: 
        
    3D coordinates: a numpy array of X by 3 corresponding to x,y,z coordinates
    Color code vector: assembly identities (length X). E.g. 
    [0,0,0,0,1,2,1,1,2,2,0]. Numpy array
    
    Output: 3D plot"""
    
    keys=np.unique(Color_Code)
    values=["c","m","g","k","r","y","b"]
    dictionary = dict(zip(keys, values))
    col=[]
    for i in range(0,np.shape(Color_Code)[0]):
        col.append(dictionary.get(Color_Code[i]))
    
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Coordinates[:,0],Coordinates[:,1],Coordinates[:,2], c=col, s=4)
    # plot the point (2,3,4) on the figure
    plt.show()
    
def kmeans_knn_partition(tseries,n_seeds,batchsize=None,return_centers=False):
    if batchsize==None:
        batchsize = n_seeds*5
    if ma.count_masked(tseries)>0:
        labels = ma.zeros(tseries.shape[0],dtype=int)
        labels.mask = np.any(tseries.mask,axis=1)
        kmeans = MiniBatchKMeans(batch_size=batchsize,n_clusters=n_seeds).fit(ma.compress_rows(tseries))
        labels[~np.any(tseries.mask,axis=1)] = kmeans.labels_
    else:
        kmeans = MiniBatchKMeans(batch_size=batchsize,n_clusters=n_seeds).fit(tseries)
        labels=kmeans.labels_
    if return_centers:
        return labels,kmeans.cluster_centers_
    return labels


def Neurons_On(raw_data,labels, clustern, X):
    """ Given the labels, the old dataframe is partitioned based on the partitioning of states
    to see whether there is a local relationship between neurons
    
    Input: labels from the kmeans algorithm on the states
    clustern: cluster to consider for the on neurons
    raw_data: the raw fluorescence activity time trace of the neurons
    """
    

    temp=[]
    for i in range(0,len(labels)):
        if labels[i]==clustern:
            temp.append(i+X) #X is the delay embedding, shifting the indices 
            #by that number
    
    #Subset the original dataframe and binarize for later colorcoding
    z_score_data= stats.zscore(raw_data, 0)
    z_score_data= z_score_data[temp,:]
    
    #If std of the z-score is higher than 2, we say that neuron is on, otherwise
    #it is 0
    l,n= np.shape(z_score_data)
    for i in range(l):
        for j in range(n):
            if z_score_data[i,j]>2:
                z_score_data[i,j]=1
            else:
                z_score_data[i,j]=0
    return z_score_data
      


def on_neurons(labeln,label, fish, delayembedding):
    """Looks at the number of on neurons for each cluster
    Input: label number to consider, the labels, the fish data set, and the number of timesteps that we have delay embedded"""
    temp=[]
    for i in range(np.shape(label)[0]):
        if label[i]==labeln:
            temp.append(i+delayembedding)
    
    fish= stats.zscore(fish, 0)
    temp= temp
    fish= fish[temp,:]
    
    r,c = np.shape(fish)
    
    for i in range(r):
        for k in range(c):
            if fish[i,k] <2:
                fish[i,k]= 0
            elif fish[i,k]>2:
                fish[i,k]=1
                
    on_neurons= np.sum(fish,1)

    return on_neurons