#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 13:26:56 2022

@author: arshdeep
# """ #the code contains training complexity plot and random, l1 and proposed method accuracy.

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import matplotlib.cm as cm
#%%

# acc= [45.21,44.14,44.21, 43.27,41.78,38.46,16.23]

acc= [44.32,44.21, 43.27,41.78,38.46,16.23]
acc_random=[44.45,43.97,41.95,39.90,35,10]
acc_L1=[42.98,43.074,41.95,40.416,36.83,9.67]
# FLOPs= [0,0,25,50,75,90,100]
# FLOPs= [0,0,17,34,90,150,200]#200,200,150,90,34,17,0
FLOPs=[200,150,90,34,17,0]
#%%
# plt.figure(figsize=(18,8))
fig,ax = plt.subplots(figsize=(12,6.5))
# make a plot
# X= [200,200,150,90,34,17,0] #TRAINING TIME FOR 30 EPOCHS
X=[100,75,50,25,10,0]# NUMBER OF FLOPSSAVED
ax.plot( X, acc, color="red", marker="o", label='pruned weight initilization')
ax.plot( X, acc_random, linestyle='--',color="black", marker="o",label='ranodm initialization')
ax.plot( X, acc_L1, linestyle='dotted',color="g", marker="o",label='l1_norm')
# set x-axis label
ax.set_xlabel("Training data (%) during fine-tuning",fontsize=24)
# set y-axis label
ax.set_ylabel("Accuracy (%)",color="red",fontsize=24)
# ax.legend(loc='right',ncol=2)
plt.ylim(0,46)
plt.grid()

plt.xticks(fontsize=22)
plt.yticks([0,10,15,20,25,30,35,40,45],fontsize=22,color='red')
ax2=ax.twinx()
ax2.set_xticks(X)
# make a plot with different y-axis using second axis object
# ax2.tick_params(axis='x', labelsize=40 )
ax2.plot(X, FLOPs,color="blue",marker="o")
# plt.setp(ax2.get_xticklabels(), fontsize=40)
ax2.set_ylabel("Training time (mins)",color="blue",fontsize=22)

# ax2.set_ylabel("Reduced FLOPs (%)",color="blue",fontsize=18)
plt.yticks([0,17,34,90,150,200],fontsize=22,color='blue')
plt.grid()
plt.show()
# plt.legend()
# ax.xticks(X,size=14)
# plt.yticks([],size=14)
# save the plot as a file
# fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',
#             format='jpeg',
#             dpi=100,
#             bbox_inches='tight')

#%%





#%% DCASE dev, eva, ESC-50

# plt.figure(figsize=(12,5))

# acc_l1= [43.95,42.58,41.84,39.59,37.67]
# acc_ours=[44.14,44.54,42.79,41.44,37.30]

# plt.plot(acc_l1,c='green',marker='*',markerfacecolor='b',markersize=14,label='l_1-norm')
# plt.plot(acc_ours,c='red',marker='p',markerfacecolor='r',markersize=14,label='Proposed')

# plt.axhline(y=48.59, color='g', linestyle='dotted',linewidth=3)
# plt.text(1.5, 49, 'Baseline', bbox=dict(facecolor='red', alpha=0.1),fontsize=17)
# # plt.legend(loc='best', bbox_to_anchor=(0.2, 1),
#            # fancybox=False, shadow=False, ncol=2,fontsize=20.0)
# plt.legend(ncol=2,fontsize=18,loc='best')


# plt.yticks(fontsize=20)
# plt.xticks(np.arange(0,5),[100,75,50,25,10],fontsize=20)
# # plt.ylim(25,52) #50%---40-52; 25...45---52
# # plt.xlim(0,102)

# plt.xlabel(' Training data (%)',fontsize=21)
# plt.ylabel(' Accuracy (%)',fontsize=22)
# plt.grid()


#%%




# # Generate dummy data into a dataframe
# plt.figure(figsize=(12,5))


# L1_norm=[43.95,43.074,41.95,40.416,36.83,9.67]
# ours=[44.14,44.21,43.27,41.782,38.46,16.23]
# ran_acc=[45.12,43.97,41.95,39.90,35,10]
# L1_std= [0,0.71,0.65,0.71,1.90]
# Ours_std= [0,0.42,0.38,0.87,0.70]
# ran_std=[2.06,0.74,1,1.20,1.10]
# index = np.arange(6)


# # plt.errorbar(index,L1_norm,yerr=L1_std)

# # plt.errorbar(index,L1_norm,yerr=L1_std,c='cyan',marker='o',markerfacecolor='black',markersize=8,label='l_1-norm',uplims=True, lolims=True)
# # plt.errorbar(index,ours,yerr=Ours_std,c='red',marker='p',markerfacecolor='black',markersize=8,label='Proposed',uplims=True, lolims=True)
# # plt.errorbar(index,ran_acc,yerr=ran_std,c='green',marker='+',markerfacecolor='black',markersize=8,label='random',uplims=True, lolims=True)



# plt.plot(index,L1_norm,c='cyan',linestyle='--',marker='o',markerfacecolor='black',markersize=8,label='l_1-norm')
# plt.plot(index,ours,c='red',marker='p',markerfacecolor='black',markersize=8,label='Proposed')
# plt.plot(index,ran_acc,c='green',linestyle='dotted',marker='*',markerfacecolor='black',markersize=8,label='random')

# plt.axhline(y=48.59, color='magenta', linestyle='dotted',linewidth=3)
# plt.text(1.5, 49, 'Baseline', bbox=dict(facecolor='red', alpha=0.1),fontsize=17)
# # plt.legend(loc='best', bbox_to_anchor=(0.2, 1),
#            # fancybox=False, shadow=False, ncol=2,fontsize=20.0)
# plt.legend(ncol=2,fontsize=18,loc='best')


# plt.yticks(fontsize=20)
# plt.xticks(np.arange(0,6),[100,75,50,25,10,0],fontsize=20)
# plt.grid()


# plt.xlabel(' Training data during fine-tuning (%)',fontsize=21)
# plt.ylabel(' Accuracy (%)',fontsize=22)

