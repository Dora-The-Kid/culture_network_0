import reverberation_network as re
import numpy as np
import matplotlib.pyplot as plt
n = 100
Type = np.ones(shape=n)

possion_rate= np.arange(0,1,0.01)
w = np.zeros(shape=(n,n))
# w[0,2]=350
w_in = np.zeros(shape=(n,n))
row, col = np.diag_indices_from(w_in)
w_in[row,col] = 3
print(w_in)

T = 10000
dt = 0.025

network = re.network(neuron_number=n,type=Type,w=w,dt=dt,external_W=None)
network.sike_train_input_flag =True
network.input_w = w_in
network.g_in = np.zeros(shape=(n,n))


step =int( T/dt)
voltage = []
Ca = []
X = []
Y = []
Z = []
S = []
Ge = []
In = []
sayrate = []
satge = []
mK = []
#background_input = np.arange(80,240,2)
test = np.arange(0,1,0.01)


for i in range(step):

    b = np.random.poisson(possion_rate*dt,None)
    if i*dt >80 :

        network.input_spike = b
    else:
        network.input_spike = np.zeros(n)
    #print(network.input_spike)
    # if i*dt == 50:
    #     network.background_input = np.zeros(shape=n)
    #     network.background_input[0] =100/ dt
    # else:
    #     network.background_input = np.zeros(shape=n)
    #network.input_spike = 0
    #print(network.input_spike)

    # if  i % 50/dt == 0:
    #     print('50***')
    #     network.background_input = np.zeros(shape=n)
    #     network.background_input[0] = 100/dt
    # elif i == 200/dt:
    #     network.background_input = np.zeros(shape=n)
    #     network.background_input[2:4] = 1000

        #network.background_input = background_input
    # elif i % 150 == 5:
    #     network.background_input = np.array([0/dt,150/dt,0/dt,0/dt])




    network.update()
    #print(network.Y[:,0])
    V = network.V
    X.append(network.X[5,0])
    Y.append(network.Y[5,0])
    Z.append(network.Z[5,0])
    S.append(network.S[5,0])
    mK.append(network.mK)
    Ge.append(np.sum(network.ge[5,:]))
    In.append(np.sum(network.increment[5,:]))

    sayrate.append(network.asynrate[5])
    satge.append(np.sum(network.asynge[5,:]))


    Ca.append(network.CaConcen)
    voltage.append(V)
    #print(V)

    #print(network.dV)
voltage = np.array(voltage)
spike = network.spike_record_scatter
spike = np.array(spike)

mK = np.array(mK)
Ca = np.array(Ca)

plt.figure()
total = []
for key in network.firing_time_raster:
    times = len(network.firing_time_raster[key])
    print(times)
    total.append(times)
total = np.array(total)
plt.plot(np.arange(n)/100,total/10000)
plt.xlabel('poisson lamda')
plt.ylabel('firing_rate (kHz)')

plt.figure()
plt.title('voltage')
plt.ylabel('V/mv')
plt.xlabel('time/ms')
plt.plot(np.arange(T/dt)*dt,voltage, alpha = 0.3)

plt.show()