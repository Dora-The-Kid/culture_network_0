import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


def small_world_network(n,k,p):

    W = np.zeros(shape=(n,n))
    G = nx.watts_strogatz_graph(n,k,p)
    ps = nx.circular_layout(G)
    for u, v in G.edges:
        G.add_edge(u, v, weight=np.random.normal(2.5, 1))
        

    for (x,y,w) in G.edges(data=True):
        W[x,y] = w['weight']
        print(w['weight'])
    print(W)
    W[0:15, 0:15] = W[0:15, 0:15] * 3.5
    W[0:15, 16:80] = W[0:15, 16:80] * 1.5
    W[16:80, 0:15] = W[16:80, 0:15] * 0.8
    W[16:80, 16:80] = W[16:80, 16:80] * 0.8
    plt.figure()
    nx.draw(G, ps, with_labels=False, node_size=30)
    plt.savefig("W_heat_map.png")
    plt.figure()
    W_n = (W-np.average(W))/np.std(W)
    ax = sns.heatmap(W_n, annot=False, center=1, cmap='YlGnBu', vmin=0, vmax=4)
    ax.set_ylim(61, 0)
    ax.set_xlim(0,61)
    plt.savefig("network_fig.png")
    #plt.show()
    return W



if __name__ == '__main__':
    small_world_network(60,7,0.2)