#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:12:09 2022

@author: arshdeep
"""

#%%
import numpy as np
import matplotlib.pyplot as plt

#%%


C1=np.load('/home/arshdeep/Pruning/DCASE_2021_NETWORK/DCASE2021_pruning/Similarity_distribution_analysis/C1_dist.npy')
C2=np.load('/home/arshdeep/Pruning/DCASE_2021_NETWORK/DCASE2021_pruning/Similarity_distribution_analysis/C2_dist.npy')
C3=np.load('/home/arshdeep/Pruning/DCASE_2021_NETWORK/DCASE2021_pruning/Similarity_distribution_analysis/C3_dist.npy')



#%%

# C1= (C1- np.mean(C1))/np.std(C1)
# C1= (C2- np.mean(C2))/np.std(C2)
# C3= (C3- np.mean(C3))/np.std(C3)
import seaborn as sns

plt.figure(figsize=(7,4))
# sns.set_style('white')


sns.distplot(C1,color='red',norm_hist=True,kde=False,label="C1 ($\mu$=0.55,$\sigma$=0.12)")
sns.distplot(C2,color='green',norm_hist=True,kde=False,label="C2 ($\mu$=0.36,$\sigma$=0.20)")
sns.distplot(C3,color='blue',norm_hist=True,kde=False,label="C3 ($\mu$=0.39,$\sigma$=0.085)")


plt.xlabel('Cosine distance between closest filter pairs',fontsize=16)
plt.xticks(fontsize=16)
plt.ylim(0,10)
plt.xlim(0,1)
plt.ylabel('Frequency',fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=14)

#%%


