import numpy as np
import matplotlib.pyplot as plt

def default_pars(**kwargs):
  pars = {}

  ### typical neuron parameters###
  pars['V_th'] = -55.    # spike threshold [mV]
  pars['V_reset'] = -75. # reset potential [mV]
  pars['tau_m'] = 10.    # membrane time constant [ms]
  pars['g_L'] = 10.      # leak conductance [nS]
  pars['V_init'] = -65.  # initial potential [mV]
  pars['E_L'] = -75.     # leak reversal potential [mV]
  pars['tau_ref'] = 2.      # refractory time (ms)

  ### simulation parameters ###
  pars['T'] = 400.  # Total duration of simulation [ms]
  pars['dt'] = .1   # Simulation time step [ms]

  ### external parameters if any ###
  for k in kwargs:
    pars[k] = kwargs[k]

  pars['range_t'] = np.arange(0, pars['T'], pars['dt'])  # Vector of discretized time points [ms]

  return pars
class LIFNeuron(object):
    def __init__(self,W,neuron_map,neuron_number,dt=0.1,external_input= None,m=0):
        # Membrane reset potential
        self.n = neuron_number
        self.V = np.full(shape=(neuron_number),fill_value=0)
        self.external_input = external_input
        self.VE = 14/3 # [mV] excitatory reversal potential
        self.VI = -2/3 # [mV] inhibitory reversal potential
        self.E_L = 0
        self.dt = dt
        self.tau_ref = 2 #absolute refractory period (ms)



        self.dt = dt
        self.W_in = None

        self.gL = np.full(shape=(neuron_number),fill_value=0.1)      # leak conductance mS


        self.SPIKING_THRESHOLD = 1
        self.W = W

        # to make update g more convenient

        self.neuron_map = neuron_map
        self.negetive_neuron_map = np.ones_like(neuron_map)-neuron_map

        E_neuron = np.argwhere(self.neuron_map>0)
        self.g_j_map = np.zeros(shape=(neuron_number,neuron_number))
        self.g_j_map[:,E_neuron] = 1

        self.g_j_map_t =1-self.g_j_map
        self.g_j_map = self.g_j_map.astype(bool)
        self.g_j_map_t = self.g_j_map_t.astype(bool)



        self.sigma_d = np.where(self.g_j_map,2,5)
        self.sigma_r = np.where(self.g_j_map,0.5,0.8)


        self.Ex_H_j = np.full(shape=(neuron_number,m),fill_value=0)
        self.Ex_g_j = np.full(shape=(neuron_number,m),fill_value=0)

        self.dV_external = None





    def initialize(self):
        self.tau_rfp = np.zeros(shape=(self.n))



        self.g_j = np.full(shape=(self.g_j_map.shape),fill_value=0) #N*N g_ij means conductance for synapse from neuron j to neuron i
        self.H_j = np.full(shape=(self.g_j_map.shape),fill_value=0)



    # Evaluate input current
    def get_input_op(self,fired_neuron_index):
        # First, update recurrent conductance dynamics:
        self.alternative_way_to_cal_g(fired_neuron_index)
        #print('g_j')
        #print(self.g_j)
        #self.H_j[self.tau_rfp > 0, np.argwhere(self.neuron_map > 0)] = 0
        #self.g_j[self.tau_rfp > 0, np.argwhere(self.neuron_map > 0)] = 0
        #print('g_j')
        #print(self.g_j)
        #self.Ex_H_j[self.tau_rfp > 0, :] = 0
        #self.Ex_g_j[self.tau_rfp > 0, :] = 0


        G_j = np.multiply(self.W,self.g_j)
        print('G_j')
        print(G_j)

        GE = np.where(self.g_j_map,G_j,0)
        #print('GE——1')
        #print(GE)
        GE = np.sum(GE,axis=1)
        #print('GE——2')
        #print(GE)
        GI = np.where(self.g_j_map_t,G_j,0)

        GI = np.sum(GI,axis=1)






        # dv_j = Σ wj(g_j/gl)(v(t)-E)
        self.dV_recurent = (GE*(self.V-self.VE)+GI*(self.V-self.VI))
        #print('tau_rfp')
        #print(self.tau_rfp)



        G_in = self.external_g(self.external_input)*self.W_in
        #print(G_in)




        #G_in = np.where(self.tau_rfp>0,0,G_in)

        self.dV_external = np.dot(G_in.T,self.V-self.VE)
        #print('dV_external')
        #print(self.dV_external)
        #print('dV_recurent')
        #print(self.dV_recurent)
        #print(self.dV_external)
        self.dV_output = self.dV_external+self.dV_recurent






        return self.dV_output


    # reset
    def get_reset_neurons(self):
        has_fired_neuro = np.greater_equal(self.V, self.SPIKING_THRESHOLD)
        #print(has_fired_neuro)
        #print('has_fired_neuro')
        #print(has_fired_neuro)

        has_fired_neuro_index = np.argwhere(has_fired_neuro)


        v_reset_neuro = np.argwhere(self.tau_rfp > 0)


        self.tau_rfp = np.where(has_fired_neuro,self.tau_ref,self.tau_rfp)
        self.tau_rfp = self.tau_rfp -self.dt

        return has_fired_neuro,has_fired_neuro_index,v_reset_neuro

    # update the synaptic conductance
    def alternative_way_to_cal_g(self,has_fired):
        has_fired_g = np.full(shape=self.g_j_map.shape,fill_value=0)
        has_fired_g[:,has_fired]=1
        #print(has_fired_g)



        #dH = -(self.dt/self.sigma_r)*self.H_j + has_fired_g/self.dt
        self.H_j =self.H_j*np.exp(-self.dt/self.sigma_r) + has_fired_g
        #self.H_j = self.H_j + dH

        dg = -(self.dt/self.sigma_d)*self.g_j + self.H_j*dt
        self.g_j = self.g_j +dg


        #print('dg')
        #print(dg)



    def external_g(self,has_fired):
        has_fired_g = np.zeros_like(self.W_in)
        has_fired = np.argwhere(has_fired)
        has_fired_g[:,has_fired] = 1

        #has_fired_g[has_fired,:] = 0
        #print(has_fired_g)
        #(self.Ex_H_j)

        #dH = -(self.dt/0.5) * self.Ex_H_j + has_fired_g/self.dt

        #self.Ex_H_j = self.Ex_H_j + dH
        self.Ex_H_j = self.Ex_H_j * np.exp(-self.dt / 0.5) + has_fired_g
       #print(self.Ex_H_j)self.dt*2
        dg = -(self.dt/2) * self.Ex_g_j + self.Ex_H_j*dt

        #print(dg)
        self.Ex_g_j = self.Ex_g_j + dg

        return  self.Ex_g_j

    # Update part
    def update(self):
        has_fired_neuro, has_fired_neuro_index, v_reset_neuro = self.get_reset_neurons()

        dV_output = self.get_input_op(has_fired_neuro_index)
        #print('has_fired_neuro_index')
        #print(has_fired_neuro_index)
        dV_output= np.where(self.tau_rfp > 0,
                          0,
                          dV_output)

        dv = -self.dt*self.gL* (self.V - self.E_L) - dV_output*self.dt
        print('dv_out')
        print(dV_output)
        self.V =np.where(has_fired_neuro,
                         0,
                         np.minimum(self.V+dv,np.full(shape=self.n,fill_value=self.SPIKING_THRESHOLD)))






def Poisson_generator(pars, rate, n, myseed=False):
  """
  Generates poisson trains

  Args:
    pars       : parameter dictionary
    rate       : noise amplitute [Hz]
    n          : number of Poisson trains
    myseed     : random seed. int or boolean

  Returns:
    pre_spike_train : spike train matrix, ith row represents whether
                      there is a spike in ith spike train over time
                      (1 if spike, 0 otherwise)
  """

  # Retrieve simulation parameters
  dt, range_t = pars['dt'], pars['range_t']
  Lt = range_t.size

  # set random seed
  if myseed:
      np.random.seed(seed=myseed)
  else:
      np.random.seed()

  # generate uniformly distributed random variables
  u_rand = np.random.rand(n, Lt)

  # generate Poisson train
  poisson_train = 1. * (u_rand < rate * (dt / 1000.))

  return poisson_train

T = 20000
dt = 0.2


steps = range(int(T/dt))
neuron_map = [1,1,1,0,0]
#neuron_map = [1,0]
w = [
    [0,0,0,0,0.9],
    [1.7,0,0.5,0,1.6],
    [0,2,0,0,0],
    [0,0,0.8,0,0],
    [1.2,0,0,2,0]
]
# w =np.zeros(shape=(5,5))
#
# w =[
#     [0,2],
#     [0,0]
# ]

W_in = [
    [1,0,0,0,0],
    [0,1,0,0,0],
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1]
]

# W_in = [
#     [0,0],
#     [0,1]
# ]
neuron_map = np.array(neuron_map)
w = np.array(w)*0.01
W_in = np.array(W_in)*0.02
lif = LIFNeuron(W=w,neuron_map=neuron_map,neuron_number=5,m=5,dt=dt)
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
    b = np.zeros(shape=lif.n)
    b[mask<0.2] = 1


    lif.external_input = b
    #lif.external_input = np.zeros(shape=(5))
    print('ex_put')
    print(lif.external_input)
    lif.update()
    #print('V')
    #print(lif.V)
    voltage.append(lif.V)
print(lif.dt)
voltage = np.array(voltage)
np.savetxt('LIF_voltage.txt',voltage)
spike = np.argwhere(voltage == 1)
spike_matrix = (voltage == 1)
np.savetxt('LIF_spike_matrix.txt',spike_matrix,delimiter=' ')

print(spike)
steps,neurons = spike.T
steps = np.arange(int(T/dt))
#voltage = voltage[:,[3,4]]
plt.figure()
plt.title('neuron response')
plt.ylabel('V')
plt.xlabel('Time (msec)')

plt.plot(steps*dt,voltage,alpha = 0.5)
#plt.scatter(steps*dt, neurons, s=3,)
#plt.legend(["Neuron1","Neuron2"])
plt.legend(["Neuron1","Neuron2",'Neuron3','Neuron4','Neuron5'])
plt.show()
#print(lif.W)


'''
plt.figure()
plt.axis([0, T, 0, 5])
plt.title('spikes')
plt.ylabel('Neurons')
plt.xlabel('Time (msec)')

#plt.fill_between(steps[200:201],0,69,facecolor='red', alpha=0.5)
#plt.vlines(range(0,T,100), 0,69,color="red")

#steps,neurons =spiked.T
#plt.scatter(steps*dt,neurons,s=3)
# Plot inhibitory spikes
steps, neurons = inh_spikes.T
plt.scatter(steps*dt, neurons, s=3)
# Plot excitatory spikes
steps, neurons = exc_spikes.T
plt.scatter(steps*dt, neurons, s=3)

plt.show()

'''









