import torch
from torch import nn
import numpy as np

import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import loader
import matplotlib.pyplot as plt


class spike(nn.Module):
    def __init__(self, time, start_part,neuron_number,):
        super().__init__()
        self.time = time
        print(self.time)
        self.start = torch.tensor(start_part)
        self.neuron_number = neuron_number
    def condition(self,time):
        time <self.time
    def body(self,time,x,spike):

        time+=1
        spike[time,:] = -spike[time-1,:]+torch.diagonal(torch.mm(spike[time,:], x))
        return time,spike
    def forward(self,x):
        time = 200
        spike = torch.zeros(size=(self.time,self.neuron_number))
        spike = spike[200::,:]
        spike = torch.cat([self.start,spike],dim=0)

        while self.condition(time):

            time,spike = self.body(time,x,spike)

        print(spike.size())

        return spike[200::,:]

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self,pred,truth):
        return torch.sum((pred-truth)**2)


class MyTrainData(data.Dataset):
    def __init__(self):
        self.video_path = '/data/FrameFeature/Penn/'
        self.video_file = '/data/FrameFeature/Penn_train.txt'
        fp = open(self.video_file,'r')
        lines = fp.readlines()
        fp.close()
        self.video_name = []
        for line in lines:
            self.video_name.append(line.strip().split(' ')[0])
    def __len__(self):
        return len(self.video_name)
    def __getitem__(self, index):
        data = loader.loader()

        return data



def train(model, myloss,  epoch):
    model.train()
    train_data = np.random.random(size=(69,69))
    train_data = torch.tensor(train_data,requires_grad=True)
    truth,shape = loader.loader('D:\\document\\体外网络发放数据\\a195e390b707a65cf3f319dbadbbc75f_6b245c0909b1a21072dd559d4203e15b_8.txt')
    truth = truth[200::,:]
    truth = torch.tensor(truth,dtype=torch.float32)
    optimizer = optim.SGD([train_data,], lr=0.001)
    for epoch_id in range(epoch):
        train_data.cuda()
        optimizer.zero_grad()
        output = model(train_data)
        output = torch.unsqueeze(output,dim=0)
        truth = torch.unsqueeze(truth,dim=0)

        loss = myloss(output, truth)
        loss.backward()
        optimizer.step()
        if epoch_id % 1 == 0:
            print('Train Epoch: {} \tloss: {:.6f}'.format(
                epoch,  loss.data.cpu().numpy()[0]))

'''
if __name__=='__main__':
    # main()
    data,shape = loader.loader('D:\\document\\体外网络发放数据\\a195e390b707a65cf3f319dbadbbc75f_6b245c0909b1a21072dd559d4203e15b_8.txt')
    time = shape[0]
    neuron_number = shape[1]
    start = data[0:200,:]
    model = spike(time=time,neuron_number=neuron_number,start_part=start).cuda()
    myloss = MyLoss()

    train(model,myloss,5000)
'''
def spike(x):

    time_range = 2247
    truth, shape = loader.loader(
        'D:\\document\\体外网络发放数据\\a195e390b707a65cf3f319dbadbbc75f_6b245c0909b1a21072dd559d4203e15b_8.txt')
    ture = truth[200::, :]

    start = torch.tensor(truth[0:200, :],dtype=torch.float32)
    ture = torch.tensor(ture, dtype=torch.float32)
    spike = torch.zeros(size=(shape[0]-200,shape[1]),dtype=torch.float32)
    spike = torch.cat([start,spike],dim=0)
    result = torch.empty((0,69))

    for time in range(200,time_range):
        spike_F = spike[time-1,:]
        #spike_F = truth[time-1,:]
        spike_F = torch.tensor(spike_F,dtype=torch.float32)
        spike_t = -spike_F +spike_F@ x.t() + 0.01*torch.randn((69))
        spike_t = spike_t.unsqueeze(0)
        spike_t = torch.clamp(spike_t,min=0.0)


        result = torch.cat([result,spike_t],dim=0)
        #spike[time, :] = -spike[time-1, :] +spike[time-1, :]@ x.t()





    return torch.sum((result-ture)**2),result

if __name__ =='__main__':
    train_data = 2*np.random.random(size=(69,69))-1
    train_data = torch.tensor(train_data,requires_grad=True,dtype=torch.float32)
    optimizer = torch.optim.SGD([train_data],lr=0.001,momentum=0)
    for step in range(1000):

        pre,result = spike(train_data)
        optimizer.zero_grad()
        pre.backward()
        optimizer.step()
        if step % 50 == 0:
            print ('step:{} , value = {}'.format(step, pre))
    truth, shape = loader.loader(
        'D:\\document\\体外网络发放数据\\a195e390b707a65cf3f319dbadbbc75f_6b245c0909b1a21072dd559d4203e15b_8.txt')
    ture = truth[200::, :]
    plt.plot(range(200,2247), np.mean(ture, 1), 'ro', ls='-', label='Original data')
    plt.plot(range(200,2247), np.mean(result.detach().numpy() , 1), 'b', ls='-', label='Fitting line')
    plt.show()






