import numpy as np
import math


class network(object):
    def __init__(self,neuron_number,type,dt,w,external_W):
        '''

        :param neuron_number:
        :param type: 1 for Excitatory 0 for Inhibitory
        :param state: 1 for active 0 for refractory
        '''
        self.spike_record_scatter = []
        self.short_term_plasticity_use_flag = True
        self.xacc = 1.0e-8
        self.learing_rate = 1
        self.n = neuron_number
        self.Maxnum_search = 100
        self.type = np.array(type)
        self.state = np.full(shape=self.n,fill_value=0)
        self.GMAX = 10
        #cortical_matrix[index_neuron][synapse_index]
        self.cortical_matrix = np.array(w)
        self.STDP_RULE ='LOG_RULE'
        self.C = 1#1
        #shape = [output,input]
        self.X = np.full(shape=(self.n,self.n),fill_value=1,dtype=np.float64)
        self.Y = np.full(shape=(self.n,self.n),fill_value=0,dtype=np.float64)
        self.Z = np.full(shape=(self.n,self.n),fill_value=0,dtype=np.float64)
        self.S = np.full(shape=(self.n,self.n),fill_value=0,dtype=np.float64)
        self.x_plus = np.full(shape=(self.n,self.n),fill_value=1,dtype=np.float64)
        self.x_minus = np.full(shape=(self.n,self.n),fill_value=1,dtype=np.float64)
        self.dt = dt
        self.TAU_PLUS = 20.0
        self.TAU_MINUS =  20.0

        self.Vot_Threshold = 0
        self.cnt_bad = 0
        self.time = 0

        self.GMAX = 0.002

        self.APLUS = 0.024

        self.AMINUS = 0.016

        self.eta_max = 1

        self.firing_time_raster = {x: [] for x in range(self.n)}
        self.Vot_Excitatory =0
        self.Vot_Inhibitory =-60
        self.ge = np.full(shape=(self.n,self.n),fill_value=0,dtype=np.float64)
        self.gE = np.full(shape=self.n,fill_value=0,dtype=np.float64)
        self.gI = np.full(shape=self.n,fill_value=0,dtype=np.float64)
        self.CaConcen_output = 2e3
        self.CaConcen = np.full(shape=self.n,fill_value=0,dtype=np.float64)
        self.background_input = np.full(shape=self.n,fill_value=3000,dtype=np.float64)
        self.tau_d =10
        self.tau_s =10e3
        self.tau_r =61
        self.tau_1 =1250
        self.g_l =2
        self.g_Ca =4.4

        self.g_K =8#8

        self.V = np.full(shape=self.n,fill_value=-40,dtype=np.float64)
        self.V_Leakage =-60
        self.V_Ca = 120
        self.V_K =-84
        self.dV =np.full(shape=self.n,fill_value=0,dtype=np.float64)
        self.V1 =-1.2
        self.V2 =18
        self.V3 =2
        self.V4 =30
        self.mK = np.full(shape=self.n,fill_value=0,dtype=np.float64)
        self.alpha = 2.5
        self.beta = 1.25e-3
        self.gamma_ca = 6.5e-3
        self.theta = 0.04
        self.xi = 0.0095#0.00625
        self.kr = 0.2951
        self.ka = 0.1
        self.Ip = 0.002e-3
        self.applied_current = 3.5
        self.u_porssion_forse = 0.19#0.25
        self.order = 4
        self.possion_rate = 0

        self.spike_train_output = np.zeros(shape=self.n)


        #test

        # self.g_Ca = 1.1
        # self.g_K = 2
        # self.g_l = 0.5
        # self.V_Ca = 100
        # self.V_K = -70
        # self.C = 1
        # self.V_Leakage = -65
        # self.V1 = -1
        # self.V2 = 15
        # self.V3 = 0
        # self.V4 = 30
        # self.theta = 0.2
        # self.tau_d = 10
        # self.tau_r = 300
        # self.tau_s = 10e3
        # self.ka = 0.1
        # self.xi = 1e-2
        # self.kr = 0.4
        # self.Ip = 0.11e-3
        # self.gamma_ca = 40e-3
        # self.CaConcen_output = 2e3
        # self.beta = 5e-3
        # self.tau_1 = 5e3
        # self.alpha = 2.5
        # self.CaConcen = np.full(shape=self.n, fill_value=0.06, dtype=np.float64)


        #
        c0 = self.tau_r * self.tau_s + self.tau_1 * self.tau_s - self.tau_r * self.tau_1
        c1 = self.tau_r * self.tau_d + self.tau_1 * self.tau_d - self.tau_r * self.tau_1
        c2 = self.tau_r * self.tau_d + self.tau_1 * self.tau_d - self.tau_s * self.tau_1
        self.coeff01_0 = self.tau_r * self.tau_s * self.tau_s / ((self.tau_d - self.tau_s) * c0)
        self.coeff01_1 = -self.tau_d * c2 / ((self.tau_d - self.tau_s) * c1)
        self.coeff01_2 = -self.tau_r * (self.tau_r - self.tau_s) * self.tau_1 * self.tau_1 / (c0 * c1)
        self.coeff02_0 = -self.tau_r * self.tau_s / c0
        self.coeff02_1 = (self.tau_r - self.tau_s) * self.tau_1 / c0
        self.coeff21_0 = self.tau_r * self.tau_1 / c1
        self.coeff31_0 = self.tau_s * self.tau_1 * self.tau_r * self.tau_r / (c0 * c1)
        self.coeff31_1 = self.tau_d * self.tau_s * self.tau_r / ((self.tau_d - self.tau_s) * c1)
        self.coeff31_2 = -self.tau_s * self.tau_s * self.tau_r / ((self.tau_d - self.tau_s) * c0)
        self.coeff32_0 = self.tau_r * self.tau_s / c0
        self.external_W = external_W
        self.external_spike = None

        self.asynrate = None
        self.asynge =np.full(shape=(self.n,self.n),fill_value=0,dtype=np.float64)
        self.increment = None

        #for outside in_put spike train
        self.input_w = None
        self.input_spike = None
        self.g_in = None
        self.input_X = 0.8
        self.sike_train_input_flag = False

    def spike_train_input(self):
        increment = self.u_porssion_forse * np.tanh(self.alpha * self.input_X)*self.input_spike
        self.g_in += -self.g_in /self.tau_d *self.dt
        self.g_in += increment*self.input_w








    def force_input(self):

        asynchronous_release = []
        #np.random.seed(123)

        self.asynrate = (self.eta_max *np.power(self.CaConcen, self.order) / (math.pow(self.ka, self.order) + np.power(self.CaConcen, self.order)))
        self.possion_rate = possion_rate = 0.4*self.asynrate*self.dt



        for i in range(len(possion_rate)):

            n = np.random.poisson(lam=self.possion_rate[i],size=self.n)
            spike_array = []
            #spike_array = n
            #spike_array = np.random.random(n) * self.dt
            for j in range(len(n)):
                spiking_time =n[j]*1

                #spiking_time = np.sum(np.random.rand(n[j]))

                spike_array.append(spiking_time)

            asynchronous_release.append( spike_array)
        #asynchronous_release = np.array(asynchronous_release)
        #print(asynchronous_release)
        #asynchronous_release = 1



        asynchronous_release = np.array(asynchronous_release).T
        #print(asynchronous_release.shape)
        #print(asynchronous_release)
        # print('X')
        # print(self.X[0,:])
        # print(self.X[:,5])
        # print(self.X[:,0])
        self.increment = increment = self.xi*asynchronous_release*np.tanh(self.alpha*self.X)
        #print(np.nonzero(increment))
        # print('increment')
        # print(increment[0,:])
        # print(increment[:,0])
        # print(increment[:,5])



        self.X -= increment
        self.Y += increment
        self.ge += increment*self.cortical_matrix.T
        self.asynge += increment*self.cortical_matrix.T
        #print(self.asynge)
        #self.ge[:,self.type==1] += increment * self.cortical_matrix[:,self.type==1]

        #self.gE_asynch += increment * self.cortical_matrix


    def clip_weight(self,w):
        if w<0 :
            w = 0
        if w > self.GMAX :
            w = self.GMAX
        return w

    def stdp(self,index):

        connection_out = np.argwhere(self.cortical_matrix[:,index]>0)

        connection_in = np.argwhere(self.cortical_matrix[index,:]>0)

        for i in range(self.n):

            if i in connection_out and self.type[i] == 1 and i!=index and len(self.firing_time_raster[i]): # update cortical strength only when there is connection
                # w[i][firingneuron_index]
                # w[firingneuron_index][i]
                # start from the beginning of postsynaptic_train to get the first time that is larger than t1
                if self.STDP_RULE == 'ADD':
                    self.cortical_matrix[i,index] -= 0.002
                if self.STDP_RULE == 'LOG_RULE':
                    self.cortical_matrix[i,index] -= self.learing_rate*self.AMINUS*self.GMAX*self.x_minus[i,index]*math.exp(-(self.firing_time_raster[index][-1]-self.firing_time_raster[i][-1])/self.TAU_MINUS)
                    self.clip_weight(math.fabs(self.cortical_matrix[i,index]))

            if i in connection_in and self.type[i] == 1 and i != index and len(self.firing_time_raster[i]):
                # w[i][firingneuron_index]
                if i in connection_in and i != index:
                    if self.STDP_RULE == 'ADD':
                        self.cortical_matrix[index, i] += 0.002
                    if self.STDP_RULE == 'LOG_RULE':
                        self.cortical_matrix[ index, i ] += self.learing_rate *self.APLUS*self.GMAX* self.x_plus[i,index] * math.exp(-(self.firing_time_raster[index][ -1] -self.firing_time_raster[i][ -1]) / self.TAU_PLUS)
                        self.clip_weight(math.fabs(self.cortical_matrix[index, i]))

        for i in range(self.n):
            if self.firing_time_raster[index][-1] != 0 and len(self.firing_time_raster[i]):

                self.x_plus[i,index] = self.x_plus[i,index] * math.exp(-(self.firing_time_raster[index][ -1] -self.firing_time_raster[i][ -1])/self.TAU_PLUS) + 1
                self.x_minus[i,index] = self.x_minus[i,index]*math.exp(-(self.firing_time_raster[index][-1]-self.firing_time_raster[i][-1])/self.TAU_MINUS) + 1

            else:
                self.x_plus[i,index] = 1
                self.x_minus[i,index] = 1

    def hermit(self,a,b,va,vb,dva,dvb,x):
        f1 = va * (2 * x + b - 3 * a) * (x - b) * (x - b) / (b - a) / (b - a) / (b - a)
        f2 = vb * (3 * b - 2 * x - a) * (x - a) * (x - a) / (b - a) / (b - a) / (b - a)
        f3 = dva * (x - a) * (x - b) * (x - b) / (b - a) / (b - a)
        f4 = dvb * (x - a) * (x - a) * (x - b) / (b - a) / (b - a)

        return  (f1 + f2 + f3 + f4 - self.Vot_Threshold)

    def root_search(self,x1,x2,fx1,fx2,dfx1,dfx2,xacc):
        f = self.hermit(x1,x2,fx1,fx2,dfx1,dfx2,x1)
        fmid = self.hermit(x1,x2,fx1,fx2,dfx1,dfx2,x2)
        if f * fmid > 0 :
            return x1
        tempx1 = x1
        tempx2 = x2
        for j in range(self.Maxnum_search):
            dx = tempx2 - tempx1
            xmid = tempx1 + dx / 2
            fmid = self.hermit(x1,x2,fx1,fx2,dfx1,dfx2,xmid)
            if fmid <=0:
                tempx1 = xmid
            else:
                tempx2 = xmid
        if np.abs(fmid)<xacc :
            root = xmid
            return root

        self.cnt_bad +=1
        if self.cnt_bad >  0.001*self.n*self.time :
            print('There are over 1 warning per second per neuron (too many bisections in root searching')
        return xmid

    def spikeing_time(self,Ta,Tb,spike_index,next_v,next_mK,next_gE,next_CaConcen):
        va = self.V[spike_index]
        dva,_,_,_,_ = self.voltage_dt(self.V,self.mK,self.gE,self.CaConcen,self.asynge)
        dva = dva[spike_index]
        vb = next_v[spike_index]
        dvb,_,_,_,_ = self.voltage_dt(next_v,next_mK,next_gE,next_CaConcen,self.asynge)
        dvb = dvb[spike_index]


        for i in range(len(spike_index)):
            fire_time = self.root_search(Ta,Tb,va[i],vb[i],dva[i],dvb[i],xacc=self.xacc)
            self.firing_time_raster[spike_index[i]].append(fire_time)
            self.spike_record_scatter.append([fire_time,spike_index[i]])

    def get_fired_neuron(self,V_pre,V_post):
        spike_neuron = np.array(V_pre < self.Vot_Threshold) * np.array( V_post > self.Vot_Threshold)

        spike_index = np.where(spike_neuron == True)[0]



        return  spike_neuron,spike_index

    def get_fire_finished_neuron(self,V_pre,V_post):
        spike_finished_neuron = np.array(V_pre > self.Vot_Threshold) * np.array(V_post < self.Vot_Threshold)
        spike_finished_neuron_index = np.where(spike_finished_neuron == True)[0]

        return  spike_finished_neuron,spike_finished_neuron_index

    def short_term_plasticity_use(self):
        #renew X, Y, Z and S
        e0 = np.exp(-self.dt/self.tau_s)
        e1 = np.exp(-self.dt/self.tau_d)
        e2 = np.exp(-self.dt*(1/self.tau_r+1/self.tau_1))


        X_copy,Y_copy,Z_cpoy,S_copy = self.X,self.Y,self.Z,self.S

        self.X  = X_copy  + (e0*self.coeff01_0 + e1*self.coeff01_1+e2*self.coeff01_2 + 1)*Y_copy  + (e0*self.coeff02_0 + e2*self.coeff02_1 + 1)*Z_cpoy  + (-e0 +1)*S_copy
        self.Y  = e1*Y_copy
        self.Z  = (e1-e2)*self.coeff21_0 * Y_copy  + e2*Z_cpoy
        self.S  = (e2*self.coeff31_0 + e1*self.coeff31_1+e0*self.coeff31_2)*Y_copy  + (e0-e2)*self.coeff32_0*Z_cpoy  + e0*S_copy

        # e0 = np.exp(-self.dt/self.tau_s)
        # e1 = np.exp(-self.dt/self.tau_d)
        # e2 = np.exp(-self.dt*(1/self.tau_r+1/self.tau_1))
        #
        #
        # X_copy,Y_copy,Z_cpoy,S_copy = self.X,self.Y,self.Z,self.S
        #
        # self.X[:,index] = X_copy[:,index] + (e0*self.coeff01_0 + e1*self.coeff01_1+e2*self.coeff01_2 + 1)*Y_copy[:,index] + (e0*self.coeff02_0 + e2*self.coeff02_1 + 1)*Z_cpoy[:,index] + (-e0 +1)*S_copy[:,index]
        # self.Y[:,index] = e1*Y_copy[:,index]
        # self.Z[:,index] = (e1-e2)*self.coeff21_0 * Y_copy[:,index] + e2*Z_cpoy[:,index]
        # self.S[:,index] = (e2*self.coeff31_0 + e1*self.coeff31_1+e0*self.coeff31_2)*Y_copy[:,index] + (e0-e2)*self.coeff32_0*Z_cpoy[:,index] + e0*S_copy[:,index]

    def calcium(self,firing_index):


        self.CaConcen[firing_index] += self.gamma_ca*np.log(self.CaConcen_output/self.CaConcen[firing_index])

    def force_cortic(self,index_spiking_neuron):

        increment = self.u_porssion_forse * np.tanh(self.alpha*self.X[:,index_spiking_neuron])
        # print('X')
        # print(self.X[:,index_spiking_neuron])
        # print('increment')
        # print(increment)

        self.X[:,index_spiking_neuron] -= increment
        self.Y[:,index_spiking_neuron] += increment
        #self.ge[:, index_spiking_neuron] += increment
        self.ge[:,index_spiking_neuron] += increment*self.cortical_matrix[index_spiking_neuron,:]
        # print('X')
        # print(self.X)
        # print('increment')
        # print(increment)






    def voltage_dt(self,V_x,mK,ge,CaConcen,asynge):
        #M = 0.5*(1+np.tanh((self.V-self.V1)/self.V2))
        V_matrix = np.zeros(shape=(self.n,self.n))
        V_matrix[:,self.type==1]= self.Vot_Excitatory
        V_matrix[:,self.type==0]= self.Vot_Inhibitory




        #M = 1/(1+np.exp(-2*(V_x-self.V1)/self.V2))
        M = (1+np.tanh((V_x-self.V1)/self.V2))/2


        I_ion = self.g_Ca * M*(V_x-self.V_Ca) + self.g_K*mK*(V_x-self.V_K)
        # print('V_x')
        # print(V_x)

        if self.external_W != None:
            #self.external_input()
            print('111')
            #self.background_input = self.external_W*self.external_ge*(self.V-self.Vot_Excitatory)

            #print('back_ground')
            #print(self.background_input)

        if self.sike_train_input_flag ==True:
            self.background_input = -np.sum(self.g_in*(self.V - self.Vot_Excitatory),axis=1)




        #print(self.ge*self.cortical_matrix*self.Y*(self.V-V_matrix))

        #gE = np.sum(ge[:,self.type==1]*self.cortical_matrix[:,self.type ==1]*self.Y[:,self.type ==1],axis=0)
        #gI = np.sum(ge[:,self.type==0]*self.cortical_matrix[:,self.type ==0]*self.Y[:,self.type ==0],axis=0)
        #print(self.ge*self.cortical_matrix*self.Y*(self.V-V_matrix))
        #print(np.sum(self.ge*self.cortical_matrix*self.Y*(self.V-V_matrix),axis=1))



        dV =(( -self.g_l*(V_x- self.V_Leakage)- I_ion - np.sum(self.ge*(self.V.reshape(-1,1)-V_matrix),axis=1) + self.applied_current + self.background_input) )/ self.C

        dV = dV*self.dt
        # print('*********')
        # print(dV[0])
        # print(-self.g_l*(V_x- self.V_Leakage)[0])
        # print(- I_ion[0])
        # print(mK)



        #Potassium current

        #mK_infty = 1 / (1 + np.exp(-2*(V_x- self.V3) / self.V4))
        mK_infty = (1+np.tanh((V_x- self.V3) /(self.V4)))/2
        # print('m')
        # print(M)
        # print('mk_infty')
        # print(mK_infty)
        # print( 0.5*(1+np.tanh((V_x- self.V3) /2*(self.V4))))
        #tau_mK = 2/(np.exp(-(V_x-self.V3)/(2*self.V4)) + np.exp((V_x-self.V3)/(2*self.V4)))
        tau_mK =1/ np.cosh((V_x-self.V3)/(2*self.V4))




        dmK = self.theta*((mK_infty- mK)/tau_mK)*self.dt
        #print( np.nonzero(dmK))
        #dmK = np.zeros(shape=self.n)
        #dmK = 0
        #print(dmK)
        #print(dmK)


        dge = -ge/self.tau_d*self.dt

        dCaConcen = (-self.beta*np.square(CaConcen)/(self.kr*self.kr + np.square(CaConcen)) + self.Ip)*self.dt
        #print(CaConcen)
        dasynch = -asynge/self.tau_d*self.dt



        return dV,dmK,dge,dCaConcen,dasynch




    def runge_kutta4_vec(self,X,mK,gE,CaConcen,asynge):

        dv1,dmK1,dgE1,dCaConcen1,dasynge1 = self.voltage_dt(X,mK,gE,CaConcen,asynge)
        dv2,dmK2,dgE2,dCaConcen2,dasynge2 = self.voltage_dt(X+0.5*dv1,mK+0.5*dmK1,gE+0.5*dgE1,CaConcen+0.5*dCaConcen1,asynge+0.5*dasynge1)
        dv3,dmK3,dgE3,dCaConcen3,dasynge3 = self.voltage_dt(X+0.5*dv2,mK+0.5*dmK2,gE+0.5*dgE2,CaConcen+0.5*dCaConcen2,asynge+0.5*dasynge2)
        dv4,dmK4,dgE4,dCaConcen4,dasynge4 = self.voltage_dt(X+dv3,mK+dmK3,gE+dgE3,CaConcen+dCaConcen3,asynge+dasynge3)
        dv = (dv1+2*dv2+2*dv3+dv4)/6

        dmK = (dmK1+2*dmK2+2*dmK3+dmK4)/6

        dgE = (dgE1+2*dgE2+2*dgE3+dgE4)/6
        dCaConcen = (dCaConcen1 + 2 * dCaConcen2 + 2 * dCaConcen3 + dCaConcen4) / 6
        dasynge = (dasynge1+2*dasynge2+2*dasynge3+dasynge4)/6




        return dv,dmK,dgE,dCaConcen,dasynge




    def update(self):


        self.force_input()
        if self.external_W != None:
            self.external_input()

        dV, dmK ,  dge ,dCaConcen ,dasynge= self.runge_kutta4_vec(self.V,self.mK,self.ge,self.CaConcen,self.asynge)
        self.dV = dV
        nextV = self.V+dV
        nextmK = self.mK+dmK
        nextge = self.ge + dge
        nextCaConcen = self.CaConcen + dCaConcen
        nextasynge = self.asynge+dasynge





        spike_neuron,spike_index = self.get_fired_neuron(self.V,nextV)

        spike_finished_neuron , spike_finished_index = self.get_fire_finished_neuron(self.V,nextV)

        self.short_term_plasticity_use()
        self.V = nextV
        self.mK = nextmK
        self.time += self.dt
        self.ge = nextge
        self.CaConcen = nextCaConcen
        self.asynge = nextasynge

        self.spike_train_output = np.zeros(shape=self.n)

        if self.sike_train_input_flag ==True:
            self.spike_train_input()


        if len(spike_index):
            self.spike_train_output[spike_index] = 1
            self.spikeing_time(self.time, self.time + self.dt, spike_index, nextV, nextmK, nextge, nextCaConcen)
            print(spike_index)
            print('666')
            for i in spike_index:
                #self.stdp(i)
                self.calcium(i)
                self.force_cortic(i)


