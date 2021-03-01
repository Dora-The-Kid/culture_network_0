import reverberation_network as re
import numpy as np
def run():
    import matplotlib.pyplot as plt
    n = 80
    a = np.random.random(n)
    Type = np.ones(shape=n)
    #Type[a<0.1] = 0
    w = np.zeros((n,n),dtype=np.float32)

    p_reccur = np.random.uniform(0,1,(n,n))
    row, col = np.diag_indices_from(p_reccur)
    p_reccur[row,col] = np.zeros(shape=n)

    print(p_reccur>0.8)

    #w[p_reccur>0.8] =

    w[p_reccur>0.8] =np.random.normal(4, 2, size=w[p_reccur > 0.8].shape)

    w = np.abs(w)
    ratio=4000/np.sum(w)
    w = w*ratio



    w[0:15,0:15] = w[0:15,0:15]*3.5
    w[0:15,16:80] = w[0:15,16:80]*1.5
    w[16:80,0:15] = w[16:80,0:15]*0.8
    w[16:80,16:80] = w[16:80,16:80]*0.8



    T =6500
    dt = 0.0125

    w = np.loadtxt('Ach_1.txt')
    # print(np.sum(w))
    ratio=6100/np.sum(w)
    w = w*ratio

    network = re.network(neuron_number=n,type=Type,w=w,dt=dt,external_W=None)

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
    spike_array = []
    background_input = np.arange(80,240,2)


    for i in range(step):
        #print(i)


        #if  50.025/dt >i > 50/dt :
        if  i == 50 / dt :
            print('50***')
            network.background_input = np.zeros(shape=n)
            network.background_input[0] = 100/dt
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
        In.append(np.sum(network.increment[5,:]))

        sayrate.append(network.asynrate[5])
        satge.append(np.sum(network.asynge[5,:]))


        Ca.append(network.CaConcen)
        voltage.append(V)
        spike_array.append(network.spike_train_output)

    print(np.sum(network.cortical_matrix))
    voltage = np.array(voltage)
    spike = network.spike_record_scatter
    spike = np.array(spike)
    spike_array = np.array(spike_array)
    np.savetxt('Ach_array.txt', spike_array)
    mK = np.array(mK)
    Ca = np.array(Ca)
    # np.savetxt('reverbrtarion_spike_6000ms_2_stdp.txt',spike)
    np.savetxt('Ach_spike.txt', spike)
    say = np.array(network.asynrate)
    return spike,spike_array,voltage