import torch as t
from torch import nn
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import RMSprop
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10
from pylab import plt
from loader import loader
import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.utils.data
from torch.nn import functional as F

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple
from Conv1D_same import Conv1d
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
writer = SummaryWriter('./runs/exp1')

d_steps = 5
g_steps = 1

def extract(v):
    return v.data.storage().tolist()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def conv2d_same_padding(input, weight, bias=None, stride=[1], padding=1, dilation=[1], groups=1):
    # 函数中padding参数可以无视，实际实现的是padding=same的效果
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    #effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)

class Config:
    lr = 0.00005
    nc = 3  # chanel of img
    ngf = 64  # generate channel
    ndf = 64  # discriminative channel
    beta1 = 0.5
    batch_size = 32
    max_epoch = 50  # =1 when debug
    workers = 2
    gpu = True  # use gpu or not
    clamp_num = 0.01  # WGAN clip gradient
    nz = 262144  # noise dimension


opt = Config()



data,shape = loader("D:\\document\\体外网络发放数据\\a195e390b707a65cf3f319dbadbbc75f_6b245c0909b1a21072dd559d4203e15b_8.txt")#(time,Neuro)
time = shape[0]
time_fomer = data[0:200,:]
#time = time - 200
size = shape[1]
print('size:')
print(size)
turth = data[200::,:]
turth1 = turth[np.newaxis,:]
turesize = turth.shape

print(turth.shape)



class firing_model(nn.Module):
    def __init__(self, time,start_part,size):
        super().__init__()
        self.time = time

        self.startpart =start_part
        self.size = size
        self.ReLu = nn.LeakyReLU()

    def run_simulatr(self,pre_spike,w,i):
        spike = -pre_spike[i] + F.tanh(torch.sum(torch.mul(w[:,i] , pre_spike)))
        #print(spike)
        return spike





    def forward(self,x):
        spike = np.zeros(shape)
        spike[0:200,:] = time_fomer
        spike= torch.tensor(spike,dtype=torch.float32)

        #print('----------')
        #print(spike.size())
        for step in range(201,self.time):
            for neuron in range(69):
                spike[step,neuron] = self.run_simulatr(spike[step-1,:],x,neuron)

        spike = spike[200::,:]
        #print(spike.size())



        return spike

'''
class covtransposed2dsame(nn.Module):

    def __init__(self, input,output,kernalsize):
        super().__init__()
        self.time = output / input
        self.upsample = F.upsample(input,self.time,mode='nearest', align_corners=None)
        self.cov2d = conv2d_same_padding(input=output,weight=kernalsize,bias=None, stride=1,stridedilation=1, groups=1)

    def forward(self,x):
        x = self.upsample(x)
        return self.cov2d(x)
'''
def covtransposed2dsame(input,time,weight):
    input = F.upsample(input=input,size=time ,mode='nearest', align_corners=None)
    input = conv2d_same_padding(input,stride=[1],weight=weight,bias=None)
    return input
def conv_block(in_channels, out_channels): # 卷积层一套
    blk = nn.Sequential(nn.BatchNorm2d(in_channels),
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    return blk

class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels # 计算输出通道数

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 在通道维上将输入和输出连结
        return X

class DCGAN_G(nn.Module):

    def __init__(self):
        super().__init__()

        self.ReLu = nn.ReLU(True)
        self.input = nn.Linear(2048,opt.ngf*8*5*5)
        self.BatchNorm2d1 = nn.BatchNorm2d(512,affine=True)
        self.BatchNorm2d2 = nn.BatchNorm2d(256)
        self.BatchNorm2d3 = nn.BatchNorm2d(128)
        self.BatchNorm2d4 = nn.BatchNorm2d(64)
        self.tah = nn.Tanh()
        self.dense2 = nn.Linear(80*80,69*69)

        self.firingrate = firing_model(time=time,size=size,start_part=time_fomer)

        '''
        self.netg = nn.Sequential(
            nn.ConvTranspose2d((opt.nz,opt.ngf * 8,1, 4,4), stride=2, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.ngf, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 1, 4, 1, 1, bias=False),

            nn.Tanh(),


        )
        self.spike = firing_model(time,start,size)
        '''

    def forward(self,x):
        #print("--------------------")
        #print('generator')
        x = self.input(x)
        x = torch.reshape(x,(-1,opt.ngf*8,5,5))
        #print(x.size())

        x = covtransposed2dsame(x,10, weight=torch.rand(512,512,4,4))
        #print(x.size())
        x = self.BatchNorm2d1(x)
        x = self.ReLu(x)

        x = covtransposed2dsame(x,20, torch.rand(256,512,4,4))
        x = self.BatchNorm2d2(x)
        x = self.ReLu(x)

        x = covtransposed2dsame(x,40, torch.rand(128,256,4,4))
        x = self.BatchNorm2d3(x)
        x = self.ReLu(x)

        x = covtransposed2dsame(x, 80,  torch.rand(64,128,  4, 4))
        x = self.BatchNorm2d4(x)
        x = self.ReLu(x)

        x = conv2d_same_padding(
            x,torch.rand(1,64,4,4),stride=[1]
        )
        #print(x.size())
        x = self.tah(x)
        x = x.view(80*80)
        x = self.dense2(x)
        x = torch.reshape(x,(69,69))
        #print(x.size())

        x = self.firingrate(x)
        x = torch.unsqueeze(x, 0)


        return x

class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    # 修改这里的实现函数
    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)


class DCDIS(nn.Module):
    def __init__(self):
        super().__init__()

        self.LeakyReLu = nn.LeakyReLU(0.2, inplace=True)
        self.BatchNorm2d1 = nn.BatchNorm1d(512)
        self.BatchNorm2d2 = nn.BatchNorm1d(256)
        self.BatchNorm2d3 = nn.BatchNorm1d(128)
        self.Cov1d1 = Conv1d(size,256,5,2,True)
        self.Cov1d2 = Conv1d(256,512,5,2,True)
        self.Cov1d3 = Conv1d(512, 1024, 5, 2, True)
        self.Cov1d4 = Conv1d(1024, 2048, 5, 2, True)
        #self.Cov1d5 = Conv1d(2048, 512, 5, 1, True)
        '''
        self.net = nn.Sequential(
    conv2d_same_padding(1, (opt.ndf, 1, 4, 4),stride=2, bias=False),
    nn.LeakyReLU(0.2, inplace=True),

    conv2d_same_padding(opt.ndf, (opt.ndf * 2, 1, 4,4 ),stride=2, bias=False),
    nn.BatchNorm2d(opt.ndf * 2),
    nn.LeakyReLU(0.2, inplace=True),

    conv2d_same_padding(opt.ndf * 2, (opt.ndf * 4, 1,4,4 ),stride=2, bias=False),
    nn.BatchNorm2d(opt.ndf * 4),
    nn.LeakyReLU(0.2, inplace=True),

    conv2d_same_padding(opt.ndf * 4, (opt.ndf * 8, 1,4, 4),stride=2, bias=False),
    nn.BatchNorm2d(opt.ndf * 8),
    nn.LeakyReLU(0.2, inplace=True),

    conv2d_same_padding(opt.ndf * 8,( 1, 1,4, 4,),stride=1, bias=False),
    # Modification 1: remove sigmoid
    # nn.Sigmoid()
    )
    '''


    def forward(self, x):
        #print("----------")
        #print('Discriminator')
        x = self.Cov1d1(x)
        x = self.LeakyReLu(x)

        x = self.Cov1d2(x)
        x = self.BatchNorm2d1(x)
        x = self.LeakyReLu(x)

        x = self.Cov1d3(x)
        x = self.BatchNorm2d2(x)
        x = self.LeakyReLu(x)

        x = self.Cov1d4(x)
        x = self.BatchNorm2d3(x)
        x = self.LeakyReLu(x)


        x = torch.reshape(x,[-1,int(128*(time-200+1))])

        x = F.normalize(x,p=2,dim=-1)

        #print(x.size())


        return x

(name, preprocess, d_input_func) = ("Raw data", lambda data: data, lambda x: x)

def gen_noise():
  return lambda : torch.tensor(np.zeros(opt.nz))  # Uniform-dist data into generator, _NOT_ Gaussian


ncritic = 5

class FirstDataset(Dataset):#需要继承data.Dataset
    def __init__(self):


        # 1. 初始化文件路径或文件名列表。
        #也就是在这个模块里，我们所做的工作就是初始化该类的一些基本参数。
        pass
    def __getitem__(self, index):
        # TODO
        turth = torch.tensor(turth)
        return turth

        pass
    def __len__(self):
        return 1

def WGAN_fit(num_epochs):
    netG = DCGAN_G()
    #netG.cuda()
    netD = DCDIS()
    #netD.cuda()
    truth = DataLoader(FirstDataset(),batch_size=1)



    #turth,shape = loader('D:\\document\\体外网络发放数据\\a195e390b707a65cf3f319dbadbbc75f_6b245c0909b1a21072dd559d4203e15b_8.txt')#(time.Neron)
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    #netD.load_state_dict(torch.load('D:\\Renyi\\culture_network\GAN\\netD.pkl'))
    #netG.load_state_dict(torch.load('D:\\Renyi\\culture_network\GAN\\netG.pkl'))

    for epoch in range(num_epochs):
        for d_index in range(d_steps):

            # 1. Train D on real+fake
            netD.zero_grad()
            #  1A: Train D on real
            d_real_data = Variable(torch.tensor(turth1))

            d_real_decision = netD(d_real_data)
            d_real_error = torch.mean(d_real_decision)
            #d_real_error.backward()  # compute/store gradients, but don't change params
            #  1B: Train D on fake
            fake = np.random.random(2048)
            fake = fake[np.newaxis,:]
            fake = torch.tensor(fake,dtype=torch.float32)
            d_gen_input = Variable(fake)
            d_fake_data = netG(d_gen_input).detach()  # detach to avoid training G on these labels

            d_fake_decision = netD(preprocess(d_fake_data))
            # d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1)))  # zeros = fake
            d_fake_error = torch.mean(d_fake_decision)
            #d_fake_error.backward()
            d_error = d_fake_error-d_real_error
            d_error.backward()
            optimizerD.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
            # Weight Clipping
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)
        for g_index in range(g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            netG.zero_grad()
            #gen_input = Variable(gen_noise())
            fake1 = np.random.random(2048)
            fake1 = fake[np.newaxis, :]
            fake1 = torch.tensor(fake, dtype=torch.float32)
            g_fake_data = netG(fake1)
            dg_fake_decision = netD(preprocess(g_fake_data))
            # g_error = criterion(dg_fake_decision, Variable(torch.ones(1)))  # we want to fool, so pretend it's all genuine
            g_error = torch.mean(dg_fake_decision)
            g_error.backward()
            optimizerG.step()  # Only optimizes G's parameters
        writer.add_scalar('d_error', extract(d_error)[0], epoch)
        #writer.add_scalar('d_fake_error', extract(d_fake_error)[0], epoch)
        writer.add_scalar('g_error',extract(g_error)[0], epoch)
        if epoch % 5 == 0:
            print("%s: D: %s/%s G: %s" % (epoch,
                                                                extract(d_real_error)[0],
                                                                extract(d_fake_error)[0],
                                                                extract(g_error)[0],
                                                                ))

            torch.save(netG.state_dict(), 'D:\\Renyi\\culture_network\GAN\\netG.pkl')
            torch.save(netD.state_dict(), 'D:\\Renyi\\culture_network\GAN\\netD.pkl')


    torch.save(netG.state_dict(), 'D:\\Renyi\\culture_network\GAN\\netG.pkl')
    torch.save(netD.state_dict(), 'D:\\Renyi\\culture_network\GAN\\netD.pkl')
    writer.close()


    generatedata = extract(d_fake_data)
    plt.plot(range(len(generatedata.detach().numpy())), np.mean(generatedata.detach().numpy(), 1), 'ro', ls='-', label='Original data')
    plt.plot(range(len(turth1)), np.mean(turth1, 1), 'b', ls='-', label='Fitting line')
    plt.show()


def main():
    WGAN_fit(100)



if __name__ == '__main__':
    main()









