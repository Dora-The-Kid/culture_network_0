import matplotlib.pyplot as plt
import numpy as np
tau_s =10e3
tau_r =61
tau_d =10
tau_l =1250

def XYZS(steps,X,Y,Z,S):
    dx =S/tau_s +Z/tau_r
    dy =-Y/tau_d
    dz =-Z/tau_r-Z/tau_l+Y/tau_d
    ds = -S/tau_s+Z/tau_l
    return X+dx*steps,Y+dy*steps,Z+dz*steps,S+ds*steps
X = 0.6867477271574788
Y = 0.24469999136126802
Z = 0.0049187558951177355
S = 0.06363352558583248
x = []
y = []
z = []
s = []

for i in range(40000):
    x.append(X)
    y.append(Y)
    z.append(Z)
    s.append(S)
    X,Y,Z,S = XYZS(0.025,X,Y,Z,S)


real_x = np.loadtxt('X_1.txt').tolist()
real_y = np.loadtxt('Y_1.txt').tolist()
real_z = np.loadtxt('Z_1.txt').tolist()
real_s = np.loadtxt('S_1.txt').tolist()
print(real_x[50120],real_y[50120],real_z[50120],real_s[50120])
part_x = real_x[50120:90120]

plt.figure()
plt.plot(np.arange(40000)*0.025,x,c='R',alpha = 0.5)

plt.plot(np.arange(40000)*0.025,part_x,c = 'B',alpha = 0.5)
plt.legend(['ODE','simulate'])

plt.show()
