import numpy as np
import matplotlib.pyplot as plt
import torch

def default_pars(**kwargs):
  pars = {}

  # typical neuron parameters#
  pars['V_th'] = -55.     # spike threshold [mV]
  pars['V_reset'] = -75.  # reset potential [mV]
  pars['tau_m'] = 10.     # membrane time constant [ms]
  pars['g_L'] = 10.       # leak conductance [nS]
  pars['V_init'] = -75.   # initial potential [mV]
  pars['E_L'] = -75.      # leak reversal potential [mV]
  pars['tref'] = 2.       # refractory time (ms)

  # simulation parameters #
  pars['T'] = 400.  # Total duration of simulation [ms]
  pars['dt'] = .1   # Simulation time step [ms]

  # external parameters if any #
  for k in kwargs:
    pars[k] = kwargs[k]

  pars['range_t'] = np.arange(0, pars['T'], pars['dt'])  # Vector of discretized time points [ms]

  return pars


pars = default_pars()
#
# A class representing a population of simple neurons
#
class SimpleNeurons(object):
    def __init__(self,neuron_number,membrance_recovery=None,reset_potential=None,recovery_speed=None ,v_init=None,recovery_boost=None):
        # The number of neurons
        self.n = neuron_number
        self.rec_spikes = []

        if v_init is None:
            self.v_init = -75
        else:
            self.v_init = v_init

        # Scale of the membrane recovery (lower values lead to slow recovery)
        if membrance_recovery is None:
            self.A = np.full((self.n),0.02,dtype=np.float32)
        else:
            self.A = membrance_recovery

            # Sensitivity of recovery towards membrane potential (higher values lead to higher firing rate)
        if recovery_speed is None:
            self.B = np.full((self.n), 0.2, dtype=np.float32)
        else:
            self.B = recovery_speed

        # Membrane recovery 'boost' after a spike
        if recovery_boost is None:
            self.D = np.full((self.n), 8.0, dtype=np.float32)
        else:
            self.D = recovery_boost



            # Membrane voltage reset value
        if reset_potential is None:
            self.C = np.full((self.n), -65.0, dtype=np.float32)
        else:
            self.C = reset_potential

        # Spiking threshold
        self.SPIKING_THRESHOLD = 35.0
        # Resting potential
        self.RESTING_POTENTIAL = -70.0

         # Initialize voltage and current
    def initialize(self):
        #memberane potential
        self.v = np.full(shape=(self.n),fill_value= self.v_init)

        #memberane recovery
        self.u = np.full(shape=(self.n), fill_value=self.B * self.C)

        self.I = np.zeros(shape=(self.n),dtype=np.float32)

        self.dt = None

    def get_reset_neurons(self):


        #find which neuros have reach the threshold
        has_fired_neuro = np.greater_equal(self.v,self.SPIKING_THRESHOLD)
        self.rec_spikes.append(has_fired_neuro.tolist)

        #reset membrane to reset potential(self.C)
        v_reset_neuro = np.where(has_fired_neuro,self.C,self.v)

        # Membrane recovery is increased by D
        u_reset_op = np.where(has_fired_neuro, np.add(self.u, self.D), self.u)

        return has_fired_neuro,v_reset_neuro,u_reset_op
    def get_input(self,has_fired,v):

        return np.add(self.I,0)

    def update(self,has_fired,v_reset,u_reset,i):
        # Evaluate membrane potential increment for the considered time interval
        # dv = 0 if the neuron fired, dv = 0.04v*v + 5v + 140 + I -u otherwise
        #print(i)

        dv = np.where(has_fired,
                      np.zeros(shape=(self.n)),
                      np.subtract((np.multiply(np.square(v_reset),0.04)+np.multiply(v_reset,5.0)+np.full(shape=(self.n),fill_value=140.0)+i),
                                  self.u)
                      )
        du = np.where(has_fired,np.zeros(shape=(self.n)),
                      np.multiply(self.A,np.subtract(np.multiply(self.B,v_reset),u_reset)))


        # Increment membrane potential, and clamp it to the spiking threshold
        # v += dv * dt
        v =  self.v = np.minimum(np.full(shape=[self.n],fill_value=self.SPIKING_THRESHOLD),
                                        np.add(v_reset,np.multiply(dv,self.dt)))
        u = self.u = np.add(u_reset,np.multiply(du,self.dt))

        return v,u
    def det_response_ops(self):
        has_fired, v_reset, u_reset = self.get_reset_neurons()
        i = self.get_input(has_fired,self.v)
        v,u = self.update(has_fired,v_reset,u_reset,i)
        return v,u

class SimpleSynapticNeurons(SimpleNeurons):
    def __init__(self,neuron_number,m,membrance_recovery=None,reset_potential=None,recovery_speed=None ,v_init=None,recovery_boost=None,W_in=None):
        # Additional model parameters
        n = neuron_number
        self.m = m
        self.input = 0
        self.tau = 10.0
        if W_in is None:
            self.W_in = np.full((n, m), 0.07, dtype=np.float32)
        else:
            self.W_in = W_in
        # The reason this one is different is to allow broadcasting when subtracting v
        self.E_in = np.zeros((m), dtype=np.float32)

        # Call the parent contructor
        # This will call the methods we have overidden when building the graph
        super(SimpleSynapticNeurons, self).__init__(n, membrance_recovery,reset_potential,recovery_speed ,v_init,recovery_boost=recovery_boost)
    def initialize(self):
        # Get parent grah variables and placeholders
        super(SimpleSynapticNeurons, self).initialize()
        self.g_in = np.zeros(shape=(self.m),dtype=np.float32)
        self.syn_has_spiked = None

    def get_input(self,has_fired,v):
        #update synaptic conductance
        g_in_update = np.where(self.syn_has_spiked,
                               np.add(self.g_in,np.ones(shape=self.g_in.shape)),
                               np.subtract(self.g_in,np.multiply(self.dt,np.divide(self.g_in,self.tau)))
                               )
        # update g_in
        g_in = self.g_in = g_in_update
        i = np.subtract(np.einsum('ij,j->i',self.W_in,np.multiply(g_in,self.E_in)),
                        np.multiply(np.einsum('ij,j->i',self.W_in,g_in),v))

        self.input = i

        return i

#
# A class representing a population of simple neurons with synaptic inputs
#
class SimpleSynapticRecurrentNeurons(SimpleSynapticNeurons):
    def __init__(self,neuron_number,m,membrance_recovery=None,reset_potential=None,recovery_speed=None ,v_init=None,recovery_boost=None,W_in=None,W=None, E=None):
        # Additional model parameters
        self.W = W
        self.E = E
        super(SimpleSynapticRecurrentNeurons, self).__init__(neuron_number=neuron_number, m=m, membrance_recovery=membrance_recovery,reset_potential=reset_potential,recovery_speed=recovery_speed ,recovery_boost=recovery_boost,v_init=v_init,W_in=W_in)

    def initialize(self):
        super(SimpleSynapticRecurrentNeurons, self).initialize()
        # Recurrent synapse conductance dynamics (increases on each synapse spike)
        self.g = np.zeros(shape=(self.n),dtype=np.float32)

    def get_input(self,has_fired,v):
        # First, update recurrent conductance dynamics:
        g_update = np.where(has_fired,
                            np.add(self.g,np.ones(shape=self.g.shape)),
                            np.subtract(self.g,np.multiply(self.dt,np.divide(self.g,self.tau))))
        g = self.g = g_update
        # I_rec = Σ wjgj(Ej -v(t))
        i_rec = np.einsum('mn,n->m',self.W,np.multiply(g,np.subtract(self.E,v)))

        #Get synaptic input
        i_in = super(SimpleSynapticRecurrentNeurons,self).get_input(has_fired,v)

        i = i_rec + i_in
        self.input = i
        return i




T = 900
dt = 0.1

n=2
steps = range(int(T/dt))
neuron_map = [1,0]
#neuron_map = [1,1,1,1,1]
neuron_map = np.array(neuron_map,dtype=bool)
a = np.full((n), 0.1, dtype=np.float32)
a[neuron_map] = 0.02


d = np.full((n), 2.0, dtype=np.float32)
d[neuron_map] = 8.0

#e = np.zeros((n), dtype=np.float32)
e = np.full((n),-80,dtype=np.float32)
e[neuron_map] = 0
print('e')
print(e)

w = [
    [0,0,0,0,-0.9],
    [1.7,0,0.5,0,-1.6],
    [0,2,0,0,0],
    [0,0,0.8,0,0],
    [1.2,0,0,-2,0]
]

W_in = [
    [1,0,0,0,0],
    [0,1,0,0,0],
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1]
]

W_in =  [
    [0,0],
    [0,5],

]


w = [
    [0,2.5],
    [0,0],
]
w = np.array(w)
W_in = np.array(W_in)

neuron_map = np.array(neuron_map)
w = np.array(w)*0.01
W_in = np.array(W_in)*0.02

#lif = SimpleSynapticRecurrentNeurons(W=w,neuron_map=neuron_map,n=5,m=5,dt=dt)
lif = SimpleSynapticRecurrentNeurons(neuron_number=n,m=n,recovery_speed=None,v_init=-65,reset_potential=None,membrance_recovery=a,recovery_boost=d,W_in=W_in,W=w,E=e)
lif.initialize()
lif.W_in = W_in
lif.W = w
lif.neuron_map = neuron_map

voltage = []
spike_timme = []

for steps in range(int(T/dt)):
    #
    # if steps == 1:
    #     lif.external_input = np.array([1,1])
    # elif steps == 100:
    #     lif.external_input = np.array([0,0,0,0,0])
    # else:
    #    lif.external_input = np.array([0, 0, 0, 0, 0])

    mask = np.random.uniform(0,1,size=lif.n)
    b = np.zeros(shape=lif.m)
    if steps >900:
        b[mask<0.02] = 1





    lif.syn_has_spiked = b
    # if steps ==2700:
    #     lif.syn_has_spiked = np.full(lif.m,fill_value=1)
    # else:
    #     lif.syn_has_spiked = np.zeros(lif.m)
    #
    lif.dt = dt
    #print(neuron.I)
    i_in = lif.input
    v,u = lif.det_response_ops()

    voltage.append(v)
print(lif.dt)
voltage = np.array(voltage)
np.savetxt('lzhi_voltage.txt',voltage)
spike = np.argwhere(voltage == 35)
spike_matrix = (voltage == 35)
np.savetxt('lzhi_spike_matrix.txt',spike_matrix,delimiter=' ')

print(spike)
steps,neurons = spike.T
steps = np.arange(int(T/dt))

plt.figure()
plt.title('neuron response')
plt.ylabel('V')
plt.xlabel('Time (msec)')

plt.plot(steps*dt,voltage,alpha = 1)
plt.axvline(2782*0.1,alpha = 0.2)
#plt.scatter(steps*dt, neurons, s=3,)
#plt.legend(["Neuron1","Neuron2"])
plt.legend(["Neuron1","Neuron2",'Neuron3','Neuron4','Neuron5'])
plt.show()

# # Number of neurons
# n = 69
# # Number of synapses
# m = 1
# # Synapses firing rate
# frate = 0.004
#
# # Generate a random distribution for our neurons
# p_neurons = np.random.uniform(0, 1, (n))
# neuron_graph = np.full((n),1)
# neuron_graph[p_neurons<0.3] = -1
# neuron_graph = np.loadtxt('neuron_graph.txt')
# np.savetxt('neuron_graph.txt',neuron_graph,delimiter=' ')
# # Assign neuron parameters based on the probability
# a = np.full((n), 0.02, dtype=np.float32)
# a[p_neurons < 0.3] = 0.1
# d = np.full((n), 8.0, dtype=np.float32)
# d[p_neurons < 0.3] = 2.0
#
# # Randomly connect 10% of the neurons to the input synapses
# p_syn = np.random.uniform(0,1,(n,m))
# #print(p_syn)
# w_in = np.zeros((n,m), dtype=np.float32)
#
# #print(w_in)
#
# #Distribute recurrent connections
# w = np.zeros((n,n),dtype=np.float32)
# p_reccur = np.random.uniform(0,1,(n,n))
# #print(p_reccur)
# #w[p_reccur>0.1] = np.random.gamma(shape=4, scale=0.02 ,size=(w[p_reccur>0.1].shape))
# w[p_reccur>0.87] =np.random.gamma(2, 0.026, size=w[p_reccur > 0.87].shape)
# # Identify inhibitory to excitatory connections (receiving end is in row)
# inh_2_exc = np.ix_(p_neurons >= 0.3, p_neurons < 0.3)
# # Increase the strength of these connections
# w[ inh_2_exc ] = 2* w[ inh_2_exc]
# #load W
# w = np.loadtxt("synape_graph.txt")
# np.savetxt('synape_graph.txt',w,delimiter=' ')
# # E_in = -85mv
# e = np.zeros((n), dtype=np.float32)
# e[p_neurons<0.3] = -85.0
#
# I_in = []
# vtotal =[]
# T = 100000
# dt = 0.5
# steps = range(int(T/dt))
# v_out = np.zeros((int(T/dt),n))
# neuron = SimpleSynapticRecurrentNeurons(neuron_number=n,m=m,recovery_speed=None,v_init=None,reset_potential=None,membrance_recovery=a,recovery_boost=d,W_in=w_in,W=w,E=e)
# neuron.initialize()
# pre_spike_time = []
# for step in steps:
#     t = step * dt
#     #print(t)
#     if t%100==0 :
#         index = np.arange(n)
#         index = np.argwhere(p_neurons>=0.3)
#         index = np.squeeze(index,axis=1)
#         index = np.random.choice(index)
#         neuron.w_in = np.zeros((n, m), dtype=np.float32)
#         neuron.W_in[index,m-1]=np.random.uniform(0.14,0.25,1)
#         print(index)
#         #r = np.random.uniform(0,1,(m))
#         r = np.array([1])
#         # A synapse has spiked when r is lower than the spiking rate
#         #p_syn_spike = r < frate * dt
#         p_syn_spike = r
#
#
#     else:
#         # No synapse activity during that period
#         p_syn_spike = np.zeros((m), dtype=bool)
#     neuron.syn_has_spiked = p_syn_spike
#     neuron.dt = dt
#     #print(neuron.I)
#     i_in = neuron.input
#     v,u = neuron.det_response_ops()
#     I_in.append((t,i_in))
#     v_out[step, :] = v
#     vtotal.append(v)
#
# matrix = neuron.W
#
# print(matrix.shape)
# node = range(n)
#
# import node_graph
# graph = node_graph.Graph_Matrix(node,matrix)
# node_graph.draw_undircted_graph(graph)
#
#
#
#
# '''
# plt.rcParams["figure.figsize"] =(12,6)
# # Draw the input current and the membrane potential
# plt.figure()
# plt.title('Input current')
# plt.ylabel('Current (mA)')
# plt.xlabel('Time (msec)')
# plt.plot(*zip(*I_in))
# plt.figure()
# plt.title('Neuron response')
# plt.ylabel('Membrane Potential (mV)')
# plt.xlabel('Time (msec)')
# plt.plot(*zip(*v_out))
# plt.show()
#
# plt.rcParams["figure.figsize"] =(12,6)
# '''
#
# # Split between inhibitory and excitatory
# #inh_v_out = np.where(p_neurons < 0.2, v_out, 0)
# #exc_v_out = np.where(p_neurons >= 0.2, v_out, 0)
# # Identify spikes
# #inh_spikes = np.argwhere(inh_v_out == 35.0)
# #exc_spikes = np.argwhere(exc_v_out == 35.0)
# # Display spikes over time
# vtotal = np.array(vtotal)
# #spiked = np.argwhere(v_out == 35.0)
# np.savetxt('voltage.txt',v_out)
# inh_v_out = np.where(p_neurons < 0.3, v_out, 0)
# exc_v_out = np.where(p_neurons >= 0.3, v_out, 0)
# # Identify spikes
# inh_spikes = np.argwhere(inh_v_out == 35.0)
# exc_spikes = np.argwhere(exc_v_out == 35.0)
# spike_matrix = (v_out == 35.0)
# np.savetxt('spike_matrix.txt',spike_matrix,delimiter=' ')
#
# plt.figure()
# plt.axis([0, T, 0, n])
# plt.title('spikes')
# plt.ylabel('Neurons')
# plt.xlabel('Time (msec)')
#
#
# #plt.fill_between(steps[200:201],0,69,facecolor='red', alpha=0.5)
# plt.vlines(range(0,T,100), 0,69,color="red")
#
# #steps,neurons =spiked.T
# #plt.scatter(steps*dt,neurons,s=3)
# # Plot inhibitory spikes
# steps, neurons = inh_spikes.T
# plt.scatter(steps*dt, neurons, s=3)
# # Plot excitatory spikes
# steps, neurons = exc_spikes.T
# plt.scatter(steps*dt, neurons, s=3)
# # Plot inhibitory spikes
# #steps, neurons = inh_spikes.T
# #plt.scatter(steps*dt, neurons, s=3)
# # Plot excitatory spikes
# #steps, neurons = exc_spikes.T
# #plt.scatter(steps*dt, neurons, s=3)
# plt.figure()
# plt.plot(range(int(T/dt)),vtotal[:,1])
# plt.show()






