# This is a code to calculate important filters in a given intermediate layer of CNN.
# Input:  A set of filters or trained model.
# Output: Important filters. (imp_list)


# load baseline model.

model= load_model('~/DCASE2021/baseline_model.h5py')
W_dcas=model.get_weights() # extract pre-trained layer wise weights.


# Rank-1 approximation of a given data.

def rank1_apporx(data):
    u,w,v= np.linalg.svd(data)
    M = np.matmul(np.reshape(u[:,0],(-1,1)),np.reshape(v[0,:],(1,-1)))
    M_prototype = M[:,0]/np.linalg.norm(M[:,0],2)
    return M_prototype


#%% layer-wise important filter index computation. Please note that the indexes {0,6,12} in "W_dcas" represent first, second and third convolutional layer in DCASE 2021 TASK1A baseline network.


Z= W_dcas[12] # choose 0,6,12 

a,b,c,d=np.shape(Z) # "d" is number of filters, "c" is number of channels, "a" and "b" represent length of the filter.

A= np.reshape(Z,(-1,c,d)) # reshape filters


#%% Approximate each filter using rank-1 approximation

N = np.zeros((a*b,d)) 

for i in range(d):
    data= A[:,:,i]
    N[:,i]=rank1_apporx(data)

#%% pair-wise similarity calculation in filter space approximated by Rank-1 method.
W= np.zeros((d,d))

for i in range(d):
    for j in range(d):
        W[i,j] = W[i,j] + distance.cosine(N[:,i],N[:,j])


#%% store closest filter to a given filter and store each pair with the closest distance. 
Q=[]
S=[]
for i in range(np.shape(W)[0]):
    n=np.argsort(W[i,:])[1]
    Q.append([i,n,W[i,n]])  # store closest pairs with their distance.
    S.append(W[i,n])   # store closest distance for each filter (ordered pairwise distance)
    
        
#%%
Q_sort=[]
q=list(np.argsort(S)) # save the indexes of filters with closest pairwise distance.
        
for i in q: 
    Q_sort.append(Q[i]) # sort closest filter pairs.
    
    
#%% select important filter indexes.

imp_list=[]
red_list=[]

for i in range(np.shape(W)[0]): 
    index_imp = Q_sort[i][0]
    index_red = Q_sort[i][1]
    if index_imp not in red_list:
        imp_list.append(index_imp)
        red_list.append(index_red)



