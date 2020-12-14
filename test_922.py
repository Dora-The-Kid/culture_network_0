import brian2 as brain
import numpy as np

# a = np.loadtxt('LIF_W_2.txt')
# '''
# b = np.loadtxt("LIF_W_1.txt")
# c = np.loadtxt('LIF_W_0.txt')
# a = (a+b+c)/3
# '''
# print(a.shape)
# a = a[:,2::]
#
# mask = np.ones_like(a)
# mask[:,3::] = 0
# print(mask)
# #a = np.where(mask,a/0.32,-a/(-0.15))
# print(a)
#
time = 20000

V = np.random.random(size=time)
S =np.random.randint(0,5,(time,5))
aplha = np.array([0,0.06,0.2,0.8,0.1])
Y = np.zeros(shape=V.shape)
for i in range(time):
    a = 0.4*V[i] + np.sum(np.multiply(aplha,S[i,:]))
    Y[i ] = a

print(Y[0:5])
print(0.4*V[0:5]+np.sum(aplha*S[0:5,:],axis=1))

np.savetxt('V.txt',V)
np.savetxt('S.txt',S)
np.savetxt('Y.txt',Y)



def regression(array1,array2,array3,k=1,l=1,refactory=[]):


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

        b_t = array3[t]*X_j #array[t+k] is a scalar

        a_t = np.dot(X_j,X_j.T)

        if t+k in refactory:

            continue
        else:

            a = a+ a_t
            b = b + b_t


    a = a/(array1.shape[0]-k)


    b = b/(array1.shape[0]-k)

    #ai = np.dot(np.linalg.inv(a),b)

    ai = np.dot(np.linalg.inv(a), b)

    return ai.T

a = regression(V,S,Y)
print(a)