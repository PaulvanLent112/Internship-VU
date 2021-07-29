# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:36:44 2021

@author: paulv
"""

import numpy as np
import random

neurons=list(np.arange(0,1000))
a= random.sample(neurons, 100)
b= random.sample(neurons, 100)




intersections=[]

for i in range(1000):
    neurons=list(np.arange(0,1000))
    a= random.sample(neurons, 25)
    b= random.sample(neurons, 30)
    c=len(list(set(a) & set(b)))
    intersections.append(c)
    
    
intersections=np.array(intersections)
print(np.mean(intersections))
print(np.std(intersections))
    





