import numpy as np
import matplotlib.pyplot as plt
step = 1
V = np.loadtxt('lzhi_voltage.txt') #shape = (time,neuron_number)
print(V.shape)
V= np.array([V[i,:] for i in range(0,V.shape[0],step)])
spike = np.loadtxt('lzhi_spike_matrix.txt')
spike= np.array([spike[i,:] for i in range(0,spike.shape[0],step)])
neuron_number = V.shape[1]
print(neuron_number)
T = np.arange(V.shape[0])
#
#
# plt.figure()
# plt.title('neuron response')
# plt.ylabel('V')
# plt.xlabel('Time (msec)')
#
# plt.plot(range(V.shape[0]),V,alpha = 0.5)
# #plt.scatter(steps*dt, neurons, s=3,)
# #plt.legend(["Neuron1","Neuron2"])
# plt.legend(["Neuron1","Neuron2",'Neuron3','Neuron4','Neuron5'])
# plt.figure()
# plt.title('neuron response')
# plt.ylabel('V')
# plt.xlabel('Time (msec)')
#
# plt.plot(range(V1.shape[0]),V1,alpha = 0.5)
# #plt.scatter(steps*dt, neurons, s=3,)
# #plt.legend(["Neuron1","Neuron2"])
# plt.legend(["Neuron1","Neuron2",'Neuron3','Neuron4','Neuron5'])
# plt.show()

#shape = [time,number_of_neurons]
def regression(array1,array2,k=3,l=3,refactory=None):


    '''

    :param array1: voltage of neuron i
    :param array2: spike of all neuron
    :param k: voltage regression order
    :param l: spike regression order
    :return:
    '''
    a = np.zeros((1+k*1+l*(array2.shape[1]),1+k*1+l*(array2.shape[1])))


    b = np.zeros((1+k*1+l*(array2.shape[1]),1))



    for t in range(array1.shape[0]-k):

        V_j_k = array1[t:t+k]

        S_j_l = array2[t:t+l,:].flatten()


        V_j_k = np.array(np.concatenate((np.array([1]), V_j_k)))

        S_j_l = np.array(S_j_l)

        X_j = np.concatenate((V_j_k,S_j_l)).reshape((-1,1)) #column vector

        b_t = array1[t+k]*X_j #array[t+k] is a scalar

        a_t = np.dot(X_j,X_j.T)

        if t+k in refactory:

            continue
        else:

            a = a+ a_t
            b = b + b_t


    a = a/(array1.shape[0]-k)
    print(a.shape)


    b = b/(array1.shape[0]-k)

    #ai = np.dot(np.linalg.inv(a),b)
    if a.shape[0] == 1:
        ai = b/a
    else:

        ai = np.dot(np.linalg.inv(a), b)

    return ai.T


W = []
for i in range(0,neuron_number-1):

    #prepare data we use
    spike_time = np.argwhere(spike[:,i]==1)

    spike_and_refactory = np.array([np.arange(x-1,x+11) for x in spike_time]).flatten()

    mask = spike_and_refactory <= T.max() #delete number Greater than time
    spike_and_refactory = spike_and_refactory[mask]
    #V_x_T = np.setdiff1d(T,spike_and_refactory).tolist()
    V_x_T = T.tolist()

    V_x = np.array(V[V_x_T,i])
    S_x = spike[V_x_T,:]
    S_x = np.delete(S_x, i, axis=1) # delete spike of neuron i
    #print(S_x.shape)
    w = regression(array1=V_x,array2=S_x,refactory = spike_and_refactory)
    W.append(w.flatten())

W = np.array(W)
print('finish')
print(W)
np.savetxt('LIF_W_2.txt',W)



