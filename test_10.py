import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns

W_simulation = [
    [0,0,0,0,-0.9],
    [1.7,0,0.5,0,-1.6],
    [0,2,0,0,0],
    [0,0,0.8,0,0],
    [1.2,0,0,-2,0]
]
W_simulation = np.array(W_simulation)

W_regression = np.loadtxt('LIF_W_2.txt')
W_regression = W_regression[:,3::]
W_1 = W_regression[:,0:5]
W_2 = W_regression[:,5:10]
W_regression = np.maximum(W_1,W_2)
W_1[np.abs(W_2)>np.abs(W_1)] = W_2[np.abs(W_2)>np.abs(W_1)]
W_regression = W_1
print(W_regression)
R_bar = np.mean(W_regression)

R_min, R_max = W_regression.min(), W_regression.max()
R = 2*W_regression/(R_max-R_min)
S_min, S_max = W_simulation.min(), W_simulation.max()
S = 2*W_simulation/(S_max-S_min)
plt.figure()
sns.set()
#ax = sns.heatmap(R)
yticklabels =yticks = np.linspace(-10,10,1)/10
ax = sns.heatmap(R, annot=True,center=0,cmap='YlGnBu',vmin=-1,vmax=1)
ax.set_ylim(6.0, 0)
plt.title('Regression_W',fontsize='large',fontweight='bold')

plt.figure()
sns.set()
yticklabels =yticks = np.linspace(-10,10,1)/10
ax = sns.heatmap(S, annot=True,center=0,cmap='YlGnBu',vmin=-1,vmax=1)
ax.set_ylim(6.0, 0)
plt.title('Simulation_W',fontsize='large',fontweight='bold')
plt.show()


# def f(x, y):
#     return R[x,y]
#
# def d(x, y):
#     return S[x,y]
# x = np.array([0,1,2,3,4])
# x = np.tile(x,[5,1])
# print(x)
# y = x.T
# print(y)
# z = f(x,y)
# k = d(x,y)
# X, Y = np.meshgrid(x, y)
# Z = f(X,Y)
# print(f(4,2))
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# #ax.contour3D(X, Y, Z, 50, cmap='viridis')
# print(x.ravel(),y.ravel())
# ax.scatter(x.ravel(), y.ravel(), z, cmap='viridis', linewidth=0.5)
# ax.scatter(x.ravel(), y.ravel(), k, cmap='viridis', linewidth=0.5)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.legend(["Regression","Simulation"])
# plt.show()
# W_regression = np.loadtxt('LIF_W_2.txt')
# W_regression = W_regression[:,3::]
# W_1 = W_regression[:,0:5]
# W_2 = W_regression[:,6:10]
# W = np.maximum(W_1,W_2)