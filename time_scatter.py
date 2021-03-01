import numpy as np
import matplotlib.pyplot as plt
import networkx  as nx
from scipy import fftpack
from scipy import signal

data = np.loadtxt('Ach_spike.txt')
W = np.loadtxt('strangr_1_2.txt')
data_array = np.loadtxt('Ach_array.txt')
plt.figure()
plt.scatter(data[:,0],data[:,1],cmap='viridis',linewidth=0.5,color="k",marker='.',s=9,alpha=0.5)

spike_onetime = []
spike_totall = []
#print(len(np.nonzero(data_array[int(50/0.0125 +123),:])[0]))
# print(data_array[int(50/0.0125 +123):int(50/0.0125 +128),:])
# print(len(np.nonzero(data_array[int(50/0.0125 +123):int(50/0.0125 +123)+ 1600 ,:])[0]) )
index = np.where(data_array.any(axis=1))[0]

difference = index[1:]- index[0:-1]
difference = np.array(difference)
space = np.argwhere(difference>100)
print(space)
final = index[np.vstack((space,[[len(index)-1]]))]
start = index[np.vstack(([[0]],space+1))]
print(final)
print(start)
print(len(start)+1)

#print(index[space[0,0]-1],index[space[0,0]],index[space[0,0]+1])

mid = index[np.rint((space + np.vstack(([[0]],np.delete(space,-1,0)+1)))/2).astype('int')]
average_fire_time = np.average(final[5:30]-start[5:30])
average_wating_time = np.average(start[6:31]-final[5:30])
total = final[-1]-start[0]
print('fire,wait，total,time',average_fire_time,average_wating_time,total,len(start)+1)

plt.vlines(final[-1]*0.0125,ymax=80,ymin=0,color = 'r',alpha = 0.5)
plt.vlines(start[0]*0.0125,ymax=80,ymin=0,color = 'b',alpha = 0.5)
plt.show()
spike_group = {}
spike_group_arry = np.zeros(shape=(len(mid),int(40/0.0125),data_array.shape[1]))
for i in range(len(mid)):
    spike_group[i] = data_array[int(mid[i,0]-int(20/0.0125)):int(mid[i,0]+int(20/0.0125)),:]
    spike_group_arry[i] = data_array[int(mid[i,0]-int(20/0.0125)):int(mid[i,0]+int(20/0.0125)),:]



    for j in range(spike_group[i].shape[1]):

        spike_group[i][:,j] = np.convolve(spike_group[i][:,j],np.ones(160)/160,'same')

        spike_group[i][:,j] = (spike_group[i][:,j] - np.mean(spike_group[i][:,j])) / (np.std(spike_group[i][:,j]) * len(spike_group[i][:,j]))


print('spikegrouparray',np.where(np.where(spike_group_arry == 1)[2] == 2))
print(np.where(spike_group_arry == 1)[2])
print(np.where(spike_group_arry == 1)[1][np.where(np.where(spike_group_arry == 1)[2] == 2)])


# fig, ax = plt.subplots(nrows=data_array.shape[1], ncols=1, figsize=(12,6))
# for i in range(data_array.shape[1]):
#     ax[i].set_xlabel('time')
#     ax[i].set_ylabel('')
#     spiketime =np.convolve( np.sum(spike_group_arry[:,:,i]),np.ones(80)/80,'same')
#     ax[i].plot(np.arange(spike_group_arry.shape[0]),spiketime)








def cxcorr(a,v):
    return np.corrcoef(a,v)[0,1]
cor = np.zeros(shape=(len(spike_group.keys()),len(spike_group.keys()),spike_group[0].shape[1]))
for i in spike_group.keys():
    for h in spike_group.keys():
        for k in range(spike_group[i].shape[1]):
            cor[i,h,k] = cxcorr(spike_group[i][:,k],spike_group[h][:,k])

print(cor[:,:,0])

import seaborn as sns
plt.figure()
sns.set()
yticklabels =yticks = np.linspace(0,10,1)/50
C = cor[:,:,69]
#C = (C - np.mean(C))/np.var(C)
ax = sns.heatmap(C, annot=False,center=0.75,cmap='YlGnBu',vmin=0.5,vmax=1)
ax.set_ylim(50, 0)
ax.set_xlim(0,50)
plt.title('correlation',fontsize='large',fontweight='bold')


plt.figure()
plt.plot(np.arange(spike_group[0].shape[0]),spike_group[0][:,2],color = 'b')
plt.plot(np.arange(spike_group[2].shape[0]),spike_group[2][:,2],color = 'r')
#print('*******',np.corrcoef(spike_group[0][:,2],spike_group[5][:,2]))

# plt.show()






#
# for i in range(int(50/0.0125-5),data_array.shape[0]-int(2000/0.125)):
#
#     if (len(np.nonzero(data_array[i+1:i+ 1600,:])[0]) == 0 and len(np.nonzero(data_array[i,:])[0]) == 0 ) or (len(np.nonzero(data_array[i- 1600:i,:])[0]) == 0 and len(np.nonzero(data_array[i,:])[0]) == 0):
#         print('a')
#         continue
#
#     elif len(np.nonzero(data_array[i- 1600:i ,:])[0]) == 0 and len(np.nonzero(data_array[i+1:i+ 1600,:])[0]) != 0  and  len(np.nonzero(data_array[i ,:])[0]) != 0:
#         print('2')
#
#         spike_onetime = []
#         spike_onetime.append(list(data_array[i,:]))
#     elif len(np.nonzero(data_array[i- 1600:i ,:])[0]) != 0 and len(np.nonzero(data_array[i+1:i+ 1600,:])[0]) == 0  and  len(np.nonzero(data_array[i ,:])[0]) != 0:
#         print('3')
#         spike_onetime.append(list(data_array[i,:]))
#         spike_totall.append(np.array(spike_onetime))
#
#     else:
#         print('4')
#         spike_onetime.append(list(data_array[i,:]))








data_1 =np.array(data[:,0])
data_l =  [(i and j) for i, j in zip(data_1>1985, data_1<1995)]
order = (data[data_l,1])
print(np.sort(order))

data_2 = data[:,1]
data_3 = data[:,1]
plt.figure()

W = (W - np.mean(W))/np.var(W)

bx = sns.heatmap(W, annot=False,center=0.5,cmap='YlGnBu',vmin=0,vmax=1)
bx.set_ylim(85, 0)
bx.set_xlim(0,85)

weight = []
weight_ori = []
weight_1 = np.zeros_like(W)
for i in range(len(data_2)):
    #print(data_2[i],np.argwhere(order==data_2[i])[0,0])

    data_2[i] = np.argwhere(order==data_2[i])[0,0]

for i in range(len(order)):


    for h in range(len(order)):
        weight_ori.append(tuple([i,h,W[i,h]]))


        weight.append(tuple([i,h,W[int(order[i]),int(order[h])]]))
        weight_1[int(i),int(h)]=W[int(order[i]),int(order[h])]



print(data[:,1].shape)
print(data[:,0].shape)
print(data_2.shape)
print(weight_1)

#fig, ax = plt.subplots(nrows=data_array.shape[1], ncols=1, figsize=(12,6))
fig, ax = plt.subplots(nrows=10, ncols=1)

for i in range(0,80,8):
    ax[int(i/8)].set_xlabel('time')
    ax[int(i/8)].set_ylabel(str(i),color = 'r')
    ax[int(i/8)].set_ylim(0,5.0)

    spiketime = spike_group_arry[:,:,int(order[i])]
    print('QWERTY',spiketime.shape)
    #spiketime =np.convolve( np.sum(spike_group_arry[:,int(order[i]),:]),np.ones(80)/80,'same')
    ax[int(i/8)].bar(np.arange(spiketime.shape[1])[1200:2500],np.sum(spiketime,axis=0)[1200:2500],width=1,color = 'b', edgecolor='b')


data = np.loadtxt('normal_3_1_spike.txt')
plt.figure()
G = nx.DiGraph()
G.add_nodes_from(np.arange(0,80))
G.add_weighted_edges_from(weight)
nx.draw(G, with_labels=True)

plt.figure()

#weight_1 = (weight_1 - np.mean(weight_1))/np.var(weight_1)

cx = sns.heatmap(weight_1, annot=False,center=0.5,cmap='YlGnBu',vmin=0,vmax=1)
cx.set_ylim(85, 0)
cx.set_xlim(0,85)


plt.figure()
#plt.plot(np.arange(len(data[:,0])),data[:,0])
plt.scatter(data[:,0],data_3,cmap='viridis',linewidth=0.5,marker='.',s=9,alpha=0.5,c='R')
plt.scatter(data[:,0],data[:,1],cmap='viridis',linewidth=0.5,marker='.',s=9,alpha=0.5,c='B')

plt.show()