import reverberation_network as re
import numpy as np
import matplotlib.pyplot as plt
n = 80
a = np.random.random(n)
Type = np.ones(shape=n)
Type[a<0.1] = 0
w = np.zeros((n,n),dtype=np.float32)

p_reccur = np.random.uniform(0,1,(n,n))
row, col = np.diag_indices_from(p_reccur)
p_reccur[row,col] = np.zeros(shape=n)

print(p_reccur>0.8)

#w[p_reccur>0.8] =

w[p_reccur>0.8] =np.random.normal(3.41, 3, size=w[p_reccur > 0.8].shape)

w = np.abs(w)

# w[4:42,0] = 2
# w[42::,1] = 2
# w[4:42,2] = np.arange(1,8.6,0.2)
# w[42::,3] = np.arange(1,8.6,0.2)
#w[5,0]=1.5
# w = np.abs(w)
w[0:15,0:15] = w[0:15,0:15]*3.5
w[0:15,16:80] = w[0:15,16:80]*1.5
w[16:80,0:15] = w[16:80,0:15]*0.8
w[16:80,16:80] = w[16:80,16:80]*0.8
#w[5,0] = 10
# w[4,0] = 2.2
# w[:,4] = 2

print(w)

T = 3000
dt = 0.05
w = w.T

# n = 100
# Type = np.ones(shape=n)
# w = np.zeros(shape=(n,n))
#
# w[0,:]=np.arange(0,0.2,0.002)
# w = w.T*10
# network = re.network(neuron_number=n,type=np.ones(shape=n),w=w,dt=dt,external_W=None)

network = re.network(neuron_number=n,type=Type,w=w,dt=dt,external_W=None)

step =int( T/dt)
voltage = []
Ca = []
X = []
Y = []
Z = []
S = []
Ge = []
sayrate = []
satge = []
mK = []
background_input = np.arange(80,240,2)
for i in range(step):


    if  i == 100/dt :
        print('50***')
        network.background_input = np.zeros(shape=n)
        network.background_input[0] = 1000
    # elif i == 200/dt:
    #     network.background_input = np.zeros(shape=n)
    #     network.background_input[2:4] = 1000

        #network.background_input = background_input
    # elif i % 150 == 5:
    #     network.background_input = np.array([0/dt,150/dt,0/dt,0/dt])

    else:
        network.background_input= np.zeros(shape=n)


    network.update()
    #print(network.Y[:,0])
    V = network.V
    X.append(network.X[5,0])
    Y.append(network.Y[5,0])
    Z.append(network.Z[5,0])
    S.append(network.S[5,0])
    mK.append(network.mK)
    Ge.append(np.sum(network.ge[5,:]))

    sayrate.append(network.asynrate[5])
    satge.append(np.sum(network.asynge[5,:]))


    Ca.append(network.CaConcen)
    voltage.append(V)

    #print(network.dV)
voltage = np.array(voltage)
spike = network.spike_record_scatter
spike = np.array(spike)
mK = np.array(mK)
Ca = np.array(Ca)
np.savetxt('reverbrtarion_spike.txt',spike)
say = np.array(network.asynrate)
print(say.shape)
print(spike)
print(spike.shape)
plt.figure()
plt.plot(np.arange(T/dt)*dt,voltage, alpha = 0.3)
plt.figure()
plt.plot(np.arange(T/dt)*dt,mK , alpha = 0.3)
plt.title('mk')
plt.figure()
plt.plot(np.arange(T/dt)*dt,voltage[:,0], alpha = 0.3)
plt.title('V0')
plt.figure()
plt.plot(np.arange(T/dt)*dt,Ca[:,0], alpha = 0.3)
plt.title('Ca_5')
plt.figure()
plt.title('sayrate')
plt.plot(np.arange(T/dt)*dt,sayrate, alpha = 0.3)
plt.figure()
plt.title('GE')
plt.plot(np.arange(T/dt)*dt,Ge, alpha = 0.3,color='blue')
plt.plot(np.arange(T/dt)*dt,satge, alpha = 0.3,color='red')
plt.figure()
plt.plot(np.arange(T/dt)*dt,X, alpha = 0.3,color= 'blue')
plt.plot(np.arange(T/dt)*dt,Y, alpha = 0.3, color= 'red')
plt.plot(np.arange(T/dt)*dt,Z, alpha = 0.3, color= 'black')
plt.plot(np.arange(T/dt)*dt,S, alpha = 0.3, color= 'yellow')
plt.legend(['X','Y','Z','S'])
plt.figure()
plt.title('satge')
plt.plot(np.arange(T/dt)*dt,satge, alpha = 0.3)


#plt.plot(np.arange(49)*0.1,np.max(voltage[:,1::],axis=0), alpha = 0.3)
#plt.axvline(50/dt,color = 'r',alpha = 0.1)
#plt.legend(['neuron1','neuron2'])
#plt.legend(['neuron1','neuron2','neuron3','neuron4'])
#plt.legend(['input1mA','input2mA','input3mA'])
#plt.legend(['fired neuron','w=1','w=2','w=2.5'])
# plt.figure()
# plt.plot(np.arange(38)*0.2+1,np.max(voltage[150::,4:42],axis=0), alpha = 0.3,color = 'r')
# plt.plot(np.arange(38)*0.2+1,np.max(voltage[150::,42::],axis=0), alpha = 0.3,color = 'b')
# plt.title('epsp')
# plt.xlabel('W')
# plt.ylabel('Max_Voltage')
# plt.legend(['without_Synchronous','with_Synchronous'])
# plt.figure()
# plt.scatter(spike[:,0],spike[:,1],cmap='viridis',linewidth=0.5,color="k",marker='.',s=9,alpha=0.5)
# plt.figure()
# total = []
# for key in network.firing_time_raster:
#     times = len(network.firing_time_raster[key])
#     print(times)
#     total.append(times)
# total = np.array(total)
# plt.plot(np.arange(n)*2+80,total/10)
# plt.xlabel('I(uA)')
# plt.ylabel('firing_rate (times/s)')

plt.show()
print(network.firing_time_raster)

print(np.sum(w))
#print(network.cortical_matrix)
# import seaborn as sns
# plt.figure()
# sns.set()
# #ax = sns.heatmap(R)
# yticklabels =yticks = np.linspace(0,10,1)/20
# ax = sns.heatmap(w, annot=True,center=0.5,cmap='YlGnBu',vmin=0,vmax=1)
# ax.set_ylim(5.0, 0)
# plt.title('befor',fontsize='large',fontweight='bold')
#
# plt.figure()
# sns.set()
# #ax = sns.heatmap(R)
# yticklabels =yticks = np.linspace(-10,10,1)/10
# ax = sns.heatmap(network.cortical_matrix, annot=True,center=0.5,cmap='YlGnBu',vmin=0,vmax=1)
# ax.set_ylim(5.0, 0)
#
# plt.title('after',fontsize='large',fontweight='bold')
#
# plt.show()