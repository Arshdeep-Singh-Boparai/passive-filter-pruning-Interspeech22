# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os

#%% DCASE dev, eva, ESC-50
fig = plt.figure(figsize=(10,4))


#y_75=[48.59,48.65,47.61,47.24,46.29,44.78,44.14]
x_75=[46246,42056,34461,36256,31496,27021,24056]
y_75=[48.58,48.056,46.54,46.67,46.41,44.96,44.32]
y_WF=[48.59,34.53,27.16,18.29,24.09,15.02,16.24]# without fine-tuning



y1=np.array(y_75)
y2=np.array(y_WF)
x1=np.array(x_75)
x1=x1*100/46246


s=300
l=0
#25%      
plt.scatter(x1[0], y1[0], c ="red", linewidths = l, marker ="s", s = s,label='Baseline')
plt.scatter(x1[1], y1[1], c ="y",linewidths = l, marker ="o", s = s,label='C1 (1.8%)')
plt.scatter(x1[2], y1[2], c ="y", linewidths = l, marker ="p", s = s, label='C2 (27.35%)')
plt.scatter(x1[3], y1[3], c ="y", linewidths = l, marker ="^", edgecolor ="green", s = s,label='C3 (2.2%)')
plt.scatter(x1[4], y1[4], c ="y", linewidths = l, marker ="*", edgecolor ="green", s =s,label='C1 + C2 (29.06%)')
# plt.scatter(x1[5], y1[5], c ="black", linewidths = 2, marker ="o", edgecolor ="green", s = 200,label='L1+L2+L5+L6 (26%)')
plt.scatter(x1[5], y1[5], c ="y", linewidths = l, marker ="X", edgecolor ="green", s = s,label='C2 + C3 (29.54%)')
plt.scatter(x1[6], y1[6], c ="y", linewidths = l, marker ="+", edgecolor ="green", s = s,label='C1 + C2 + C3 (31.24%)')


c=[0,0.37,0.67,0.67,0.40,0.35,0.13]

plt.errorbar(x1,y1,yerr=c,fmt='.',elinewidth=10,ecolor='g', barsabove=True)

plt.scatter(x1[1], y2[1], c ="b", linewidths = l, marker ="o", edgecolor ="b", s = s)
plt.scatter(x1[2], y2[2], c ="b", linewidths = l, marker ="p", edgecolor ="b", s = s)
plt.scatter(x1[3], y2[3], c ="b", linewidths = l, marker ="^", edgecolor ="b", s = s)
plt.scatter(x1[4], y2[4], c ="b", linewidths = l, marker ="*", edgecolor ="b", s =s)
# plt.scatter(x1[5], y1[5], c ="black", linewidths = 2, marker ="o", edgecolor ="green", s = 200,label='L1+L2+L5+L6 (26%)')
plt.scatter(x1[5], y2[5], c ="b", linewidths = l, marker ="X", edgecolor ="b", s = s)
plt.scatter(x1[6], y2[6], c ="b", linewidths = l, marker ="+", edgecolor ="b", s = s)






plt.plot([-5,102],[48.59,48.59],'r--')

plt.plot([100,100],[0,100],'r--')

plt.grid()

plt.yticks(fontsize=22)

plt.xticks(fontsize=22)
plt.ylim(0,55) #50%---40-52; 25...45---52
plt.xlim(0,102)

plt.xlabel(' Number of parameters (%)',fontsize=21)
plt.ylabel(' Accuracy (%)',fontsize=24)

