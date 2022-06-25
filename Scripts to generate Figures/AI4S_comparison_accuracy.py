#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 00:32:06 2022

@author: arshdeep
"""

import numpy as np
import matplotlib.pyplot as plt
   
L1= [47.976, 46.12,45.524, 44.86,43.76,42.98]
CS = [48.056, 46.67,46.54,46.41,44.96,44.32]
 
L1_std= [0.40, 0.30,0.49,0.44,0.34,0.54] 
CS_std= [0.37,0.67,0.67,0.40,0.35,0.13]


param=[42056,34461,36256,31496,27021,24056]
red_parm= (46246-np.array(param))*100/46246



n=6
r = np.arange(n)
width = 0.25

plt.figure(figsize=(14,5))  
  
plt.bar(r, L1, yerr=L1_std,color='g', label='$l_1$-norm',align='center', width = width, capsize=10,edgecolor = 'black')
        #label='Women')
        
        
plt.plot(r,L1, '-o', color='g',linestyle='--')        
plt.bar(r + width, CS, yerr=CS_std, color = 'r', width = width, align='center',edgecolor = 'black',capsize=10, label='Proposed')
 
plt.plot(r+width,CS, '-o', color='red',linestyle='--')  
# plt.xlabel("Various layers (with reduced parameters after pruning)",fontsize=23)
plt.ylabel("Accuracy (%)",fontsize=23)
  
# plt.grid(linestyle='--')
# plt.xticks(r + width/2,['C1 (9%)','C3','C2','C1+C2','C2+C3','C1+C2+C3'],fontsize=23)
plt.xticks([])
plt.yticks(fontsize=23)
# plt.legend(ncol=2,fontsize=20,loc=6)
# plt.legend(loc='best', bbox_to_anchor=(1, 1),fancybox=False, shadow=False, ncol=2,fontsize=23.0)
plt.ylim(40,50)  
plt.show()

# align='center', alpha=0.5, ecolor='black', capsize=10