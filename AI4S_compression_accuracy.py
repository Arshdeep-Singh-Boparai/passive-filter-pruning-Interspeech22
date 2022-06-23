# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os

#%% DCASE dev, eva, ESC-50
fig = plt.figure(figsize=(10,4))



#x1 = [46246,39542,27390,30422,30474,27906,14270,22614]



#x1=np.array(x1)/462.46

#y1 = [48.58,48.85,47.51,47.422,48.07,48.05,42.75,46.70]

y_75=[48.59,47.74,47.61,47.24,46.29,44.78,44.14]
x_75=[46246,42056,34461,36256,31496,27021,24056]

# y_50=[ 64.69,62.31 ,63.98 ,63.58 ,65.65 ,63.22,63.66,64.30]
# x_50=[ 55.361,55.15, 54.18, 27.83, 54.05, 27.63, 26.95, 26.82]

# y_25=[64.69,64.34,65.65, 64.77 ,63.62 ,63.30 ,63.82 ,64.01]
# x_25=[55.361,55.25, 54.73, 41.45, 54.64, 41.34, 40.89, 40.81]




y1=np.array(y_75)
x1=np.array(x_75)
x1=x1*100/46246



#25%      
plt.scatter(x1[0], y1[0], c ="red", linewidths = 2, marker ="s",edgecolor ="green", s = 200,label='Baseline')
plt.scatter(x1[1], y1[1], c ="pink", linewidths = 2, marker =".", edgecolor ="green", s = 300,label='C1 (1.8%)')
plt.scatter(x1[2], y1[2], c ="yellow", linewidths = 2, marker ="p", edgecolor ="green", s = 200, label='C2 (27.35%)')
plt.scatter(x1[3], y1[3], c ="g", linewidths = 2, marker ="^", edgecolor ="green", s = 200,label='C3 (2.2%)')
plt.scatter(x1[4], y1[4], c ="b", linewidths = 2, marker ="*", edgecolor ="green", s =200,label='C1,C2 (29.06%)')
# plt.scatter(x1[5], y1[5], c ="black", linewidths = 2, marker ="o", edgecolor ="green", s = 200,label='L1+L2+L5+L6 (26%)')
plt.scatter(x1[5], y1[5], c ="brown", linewidths = 2, marker ="X", edgecolor ="green", s = 200,label='C2,C3 (29.54%)')
plt.scatter(x1[6], y1[6], c ="brown", linewidths = 2, marker ="+", edgecolor ="green", s = 200,label='C1,C2,C3 (31.24%)')

# # # #            linewidths = 2,
# #            marker ="^", 
# #            edgecolor ="red", 
# #            s = 200)

# #50% 16.91 33.42 29.70 46.99 46.62 59.74 73.53

# plt.scatter(x1[0], y1[0], c ="red", linewidths = 2, marker ="s",edgecolor ="green", s = 200,label='Baseline')
# plt.scatter(x1[1], y1[1], c ="pink", linewidths = 2, marker =".", edgecolor ="green", s = 300,label='C1,C2 (17%)')
# plt.scatter(x1[2], y1[2], c ="yellow", linewidths = 2, marker ="p", edgecolor ="green", s = 200, label='C3,C4 (33%)')
# plt.scatter(x1[3], y1[3], c ="g", linewidths = 2, marker ="^", edgecolor ="green", s = 200,label='C5,C6 (30%)')
# plt.scatter(x1[4], y1[4], c ="b", linewidths = 2, marker ="*", edgecolor ="green", s =200,label='C1 to C4 (47%)')
# # plt.scatter(x1[5], y1[5], c ="black", linewidths = 2, marker ="o", edgecolor ="green", s = 200,label='L1+L2+L5+L6 (47%)')
# plt.scatter(x1[6], y1[6], c ="brown", linewidths = 2, marker ="X", edgecolor ="green", s = 200,label='C3 to C6 (60%)')
# plt.scatter(x1[7], y1[7], c ="brown", linewidths = 2, marker ="+", edgecolor ="green", s = 200,label='C1 to C6 (74%)')




#75% 22.87 45.11 39.55 60.47 62.56 77.14 92.50
# plt.scatter(x1[0], y1[0], c ="red", linewidths = 2, marker ="s",edgecolor ="green", s = 200,label='Baseline')
# plt.scatter(x1[1], y1[1], c ="pink", linewidths = 2, marker =".", edgecolor ="green", s = 300,label='C1,C2 (23%)')
# plt.scatter(x1[2], y1[2], c ="yellow", linewidths = 2, marker ="p", edgecolor ="green", s = 200, label='C3,C4 (45%)')
# plt.scatter(x1[3], y1[3], c ="g", linewidths = 2, marker ="^", edgecolor ="green", s = 200,label='C5,C6 (40%)')
# plt.scatter(x1[4], y1[4], c ="b", linewidths = 2, marker ="*", edgecolor ="green", s =200,label='C1 to C4 (61%)')
# # plt.scatter(x1[5], y1[5], c ="black", linewidths = 2, marker ="o", edgecolor ="green", s = 200,label='L1+L2+L5+L6 (63%)')
# plt.scatter(x1[6], y1[6], c ="brown", linewidths = 2, marker ="X", edgecolor ="green", s = 200,label='C3 to C6 (77%)')
# plt.scatter(x1[7], y1[7], c ="brown", linewidths = 2, marker ="+", edgecolor ="green", s = 200,label='C1 to C6 (93%)')




plt.plot([-5,102],[48.59,48.59],'r--')

plt.plot([100,100],[0,100],'r--')


plt.legend(loc='upper center', bbox_to_anchor=(1.26, 1.1),
           fancybox=False, shadow=False, ncol=1,fontsize=20.0)
# plt.legend(ncol=2,fontsize=18,loc='best')


plt.yticks([42,44,46,48,50],fontsize=22)
plt.xticks(fontsize=22)
plt.ylim(42,50) #50%---40-52; 25...45---52
plt.xlim(0,102)

plt.xlabel(' Number of parameters (%)',fontsize=21)
plt.ylabel(' Accuracy (%)',fontsize=24)
plt.grid()	


#%% data




#%% L1 vs PCS Plot
# fig = plt.figure(figsize=(7,5))

# L1=np.load('/home/arshdeep/PCS_pruning/DCASE2021_Pruning/16_16_16/l1/val_acc.npy')
# PCS=np.load('/home/arshdeep/PCS_pruning/DCASE2021_Pruning/16_16_16/pcs/val_acc.npy')

# plt.plot(L1[0:100]*100,c='r',marker='D',label='L1')
# plt.plot(PCS[0:100]*100,c='b',marker='*',label='PCS')
# #plt.legend()
# plt.yticks(fontsize=20)
# plt.xticks(fontsize=20)

# plt.xlabel(' Iterations',fontsize=21)
# plt.ylabel(' Accuracy (%)',fontsize=22)
# #plt.grid()	

#%%

