# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 15:52:05 2021

@author: paulv
"""

import os
import h5py
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import os
import scipy
import time

import sys

sys.path.append('D:\Internship VU\Python\PF_approx-master')



#but is incompatible with python 3.8
import delay_embedding as embed

from scipy import sparse
from scipy import linalg
from scipy import stats
from scipy.sparse import diags,identity,coo_matrix



def load_file(filename):
    TPM =np.genfromtxt(filename, delimiter=',')
    return TPM

def compute_spectrum2(P):
    
    """This one takes the left eigenvalue"""
    evals, lvecs= linalg.eig(P,right=False, left=True)
    evals1= np.real(evals)
    lvecs1= np.real(lvecs)
    evals_c=np.imag(evals)
    lvecs_c= np.imag(lvecs)
    
    lvecs1 = lvecs1/lvecs1.sum(axis=0, keepdims=True)
    
    return evals1, lvecs1, evals_c, lvecs_c



def get_entropy(TPM, stationary_dist):
    '''
    Compute entropy rate of the symbolic sequence
    labels: masked array of ints
    returns entropy rate
    '''
    
    #get dtrajs to deal with possible nans
    P = scipy.sparse.csr_matrix(TPM)
    probs = stationary_dist
    logP = P.copy()
    logP.data = np.log(logP.data)
    return (-diags(probs).dot(P.multiply(logP))).sum()



def t_characteristic(eigenval):
    t_char=1.5/(1-eigenval)
    return t_char



#%%

if __name__=="__main__":
    second_largest=[]
    entropies=[]
    spectrum_comparison= np.zeros((18,5))
    complex_spectrum_comparison=np.zeros((18,5))
    filenames=os.listdir("data/")
    for i in range(len(filenames)):
        print(filenames[i])
        TPM= load_file("data/"+filenames[i])
        evals1, lvecs1, evals_c, lvecs_c=compute_spectrum2(TPM)
        sorted_eval=np.sort(evals1)[::-1]
        
        sort_cval=np.argsort(evals1)[::-1]
        second_largest.append(sorted_eval[1])
        
        "Select stationary vector"
        stationary_ind=np.argmax(evals1)
        stationary_distribution= lvecs1[:,stationary_ind]
        entropies.append(get_entropy(TPM, stationary_distribution))
        
        #Spectrum comparison
        SC= sorted_eval[0:5]
        c_SC= evals_c[sort_cval[0:5]]
        complex_spectrum_comparison[i,:]=c_SC
        spectrum_comparison[i,:]=SC

#%%

data= ["dm_forebrain","dm_forebrain","dm_forebrain","dm_midbrain",
       "dm_midbrain","dm_midbrain","dm_midbrain","dm_midbrain",
       "wt_forebrain","wt_forebrain","wt_forebrain","wt_forebrain",
       "wt_forebrain","wt_midbrain","wt_midbrain","wt_midbrain",
       "wt_midbrain","wt_midbrain"]




#%%
spectrum_comparison=spectrum_comparison[:,1:]
r,c=np.shape(spectrum_comparison)

t_relaxation=np.zeros((r,c))
for i in range(r):
    for j in range(c):
        t_relaxation[i,j]=t_characteristic(spectrum_comparison[i,j])


#%%
dm_midbrain_ind= np.arange(3,8,1)
wt_midbrain_ind= np.arange(13,18,1)
wt_forebrain_ind=np.arange(8,13,1)
dm_forebrain_ind=np.arange(0,3,1)



wt_midbrain=t_relaxation[wt_midbrain_ind,:]
dm_midbrain=t_relaxation[dm_midbrain_ind,:]
wt_forebrain=t_relaxation[wt_forebrain_ind,:]
dm_forebrain= t_relaxation[dm_forebrain_ind,:]


dm_brain_ind=[0,1,2,13,14,15,16,17]
wt_brain= spectrum_comparison[3:14,:]
dm_brain=spectrum_comparison[dm_brain_ind,:]
#print(stats.ttest_ind(wt_midbrain[:,3], dm_midbrain[:,3]))





    

#%%
x = np.array([2,3,4,5])

cm = 1/2.54
plt.figure(figsize=(16*cm, 12*cm))
bp1=plt.boxplot(wt_forebrain,0,'',positions=x-0.2,widths=0.3,patch_artist=True,
            boxprops=dict(facecolor="#ddb310", color="black"),
            capprops=dict(color="black"),
            medianprops=dict(color="black"))

bp2=plt.boxplot(dm_forebrain,0,'',positions=x+0.2,widths=0.3, patch_artist=True,
            boxprops=dict(facecolor="#b51d14", color="black"),
            capprops=dict(color="black"),
            medianprops=dict(color="black"))
plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['forebrain wildtype', 'forebrain CSF1R-mutant'], loc='upper right')
plt.xticks(x,x)

plt.plot([2.2,2.2,1.8 , 1.8], [1.5, 1.2, 1.2, 1.5], linewidth=1, color='k')
plt.text(1.9,1.3,s="n.s.")
plt.plot([3.2,3.2,2.8 , 2.8], [1.5, 1.2,1.2 , 1.5], linewidth=1, color='k')
plt.text(2.95,1.3,s="n.s.")
plt.plot([4.2,4.2,3.8 , 3.8], [1.5, 1.2, 1.2, 1.5], linewidth=1, color='k')
plt.text(3.95,1.3,s="n.s.")
plt.plot([5.2,5.2,4.8 , 4.8], [1.5, 1.2, 1.2, 1.5], linewidth=1, color='k')
plt.text(4.95,1.3,s="n.s.")
plt.xlabel("eigenvalue $\lambda$")
plt.ylabel("$t_{rel}$")
#plt.show()
plt.savefig('Figures_Thesis/forebrain_dm_wt.png', dpi=300, transparent=False, bbox_inches='tight')  





