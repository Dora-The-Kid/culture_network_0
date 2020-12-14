import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


W_regression = np.loadtxt('regression_synapse_graph.txt')
W_simulation = np.loadtxt('synape_graph.txt')
#W_regression = W_regression[:,1::]
neuro = np.loadtxt('neuron_graph.txt')
neuro.reshape((1,-1))
W_regression = W_regression*neuro
print(W_regression.shape)

R_min, R_max = W_regression.min(), W_regression.max()
R = (W_regression-R_min)/(R_max-R_min)
S_min, S_max = W_simulation.min(), W_simulation.max()
S = (W_simulation-S_min)/(S_max-S_min)
sns.set()
ax = sns.heatmap(R)
plt.show()

sns.set()
ax = sns.heatmap(S)
plt.show()

def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))
print(np.corrcoef(W_simulation,W_regression))