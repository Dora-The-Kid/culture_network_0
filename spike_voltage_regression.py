import numpy as np
import loader
V = np.loadtxt('voltage.txt')
spike = np.loadtxt('spike_matrix.txt')
X = np.zeros((V.shape[0],69,70))
for t in range(V.shape[0]):
    spike_t =spike[t,:]
    zero_mat = np.zeros((V.shape[1],V.shape[1]))
    s = (spike_t+zero_mat)

    x_it =V[t,:]
    #x_it = np.expand_dims(x_it,axis=0)
    x_it = np.column_stack((x_it,s))
    X[t,:,:] = x_it

del x_it

n_i = V.shape[0]
A = np.zeros((69,70))
for i in range(V.shape[1]):
    a = np.zeros((1))
    b = np.zeros((70))
    for t in range(V.shape[0]-1):
        if t%100 == 0:
            continue
        else:
            x = X[t, i, :].reshape((-1, 1))


            b_t = V[t+1,i]*x
            a_t = np.dot(x,x.T)

            a = a+a_t
            b = b+b_t
    a = a/V.shape[0]


    b = b/V.shape[0]
    a_i = b/a
    A[i,: ] = a_i


np.savetxt('spike_regression_ai.txt',A)
print(A.shape)