from torch import nn
import torch
import numpy as np
from torch.nn import functional as F

class Conv1d(nn.Module):
    """
    inputs: tensor of shape (batch size, num channels, height, width)

    returns: tensor of shape (batch size, num channels, height, width)
    """
    def __init__(self,in_channels,out_channel,kernal_size,stride,bias):
        super().__init__()
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.kernal_size = kernal_size
        self.stride = stride
        self.bias = bias


        fan_in = in_channels* kernal_size
        fan_out = out_channel * kernal_size / stride
        filters_stdev = np.sqrt(2. / (fan_in + fan_out))

        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')
        self.kernal_size =  uniform(filters_stdev,(self.kernal_size, in_channels, out_channel))
        #print(self.kernal_size.shape)
        self.Cov1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channel, kernel_size=kernal_size, stride=stride, padding=kernal_size // 2, bias=bias)


    def forward(self,x):

        x = torch.tensor(x,dtype=torch.float32)
        x = x.permute(0,2,1)# name='NCHW_to_NHWC'
        x = self.Cov1d(x)
        #print("----------------")
        #print(x.size())
        x = x.permute(0,2,1)#'NHWC_to_NCHW'
        #x = torch.squeeze(x)
        return x