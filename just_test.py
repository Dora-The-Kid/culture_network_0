import numpy as np
b = []
data_array = np.loadtxt('spike_array_normal_1.txt')
a = data_array[500,:]
#b.append(a)

c = list(a)
b.append(c)
b.append(c)
b.append(c)
d = np.array(b)
print(d.shape)