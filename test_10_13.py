import matplotlib.pyplot as plt
import numpy as np
# a = [
#     [-0.6,-1,-10],
#     [0.004,-0.02,0],
#     [0,0,-0.1]
# ]
# e_vals,e_vecs = np.linalg.eig(a)
# print(e_vals)
# print(e_vecs)
#
# t  = np.arange(0,10,0.001)
# v = -(-24.36)*np.exp(-0.59*t)+0.868*(-2.5)*np.exp(-0.026*t)-0.99*22.22*np.exp(-0.1*t)
# #v = 0.045*22.22*np.exp(-0.1*t)
# #v = 0.006*(-24.39)*np.exp(-0.59*t)-0.497*(-2.5)*np.exp(-0.21*t)-0.049*16.6*np.exp(-0.1*t)-14
# plt.figure()
# plt.plot(t,v)
# plt.show()
#
# for i in  range(10):
#     a = np.random.poisson(lam=[0.5, 0.1],size=[2,2]).T
#     print(a)
# a = np.ones(shape=(60,60))
#
# b = np.arange(0,60)
#
# c = b.reshape(-1,1)
# print(c)
# print(b*a)
# d = c*a
# print(d)
# print(d[0,:])
import math
import matplotlib.pyplot as plt
V_x = np.arange(-60,30)
F =  (1+np.tanh((V_x- 2) /(30)))/2
plt.figure()
plt.plot(V_x,F)
plt.show()