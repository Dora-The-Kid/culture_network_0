import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import loader
neuron_filter = np.loadtxt('neuron_graph.txt')
print(neuron_filter)
#neuron_filter = torch.tensor(neuron_filter,dtype=torch.float32)
PATH = 'D:\\Renyi\\culture_network\\stimulation_data_liner_regression_v.pkl'
truth, shape = loader.loader(
        'spike_matrix.txt')
print(shape)
x1 = truth[0:-2, :]
x = truth[1:-1,:]
x = torch.tensor(x,dtype=torch.float32)
x1 = torch.tensor(x1,dtype=torch.float32)
x_1,xishape = loader.loader('voltage.txt')
#x1 = torch.tensor(x_1[0:-1,:],dtype=torch.float32)

xtotal = torch.cat([x,x1],dim=1)
print(xtotal.size())
y = truth[2::,:]
print(y.shape)
y = torch.tensor(y,dtype=torch.float32)

class fit(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(69*2,69)

        self.relu = nn.ReLU(True)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #out = torch.mul(x,neuron_filter)
        out = self.linear(x)
        #out1 = self.liner1(x1)

        out = self.relu(out)

        #out = torch.add(out1,out)
        return out
'''
if torch.cuda.is_available():
    model = fit().cuda()
else:
    model = fit()
'''
model = fit()


model.load_state_dict(torch.load(PATH))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 7000
for epoch in range(num_epochs):
    '''
    if torch.cuda.is_available():
        inputs = Variable(x).cuda()
        target = Variable(y).cuda()
    else:
        inputs = Variable(x)
        target = Variable(y)
    '''
    inputs = Variable(x)
    inputs1 = Variable(x1)
    target = Variable(y)

    # 向前传播
    out = model(xtotal)
    loss = criterion(out, target)

    # 向后传播
    optimizer.zero_grad()  # 注意每次迭代都需要清零
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print('Epoch[{}/{}], loss:{:.6f}'.format(epoch + 1, num_epochs, loss))
predict = model(xtotal)

params = list(model.named_parameters())
print(params.__len__())
#print(params[0])

w = params[0]
w = w[1].detach().numpy()[:,0:69]
w = w*neuron_filter
np.savetxt('regression_synapse_graph.txt',w)


torch.save(model.state_dict(), PATH)

#plt.plot(range(len(y.detach().numpy())), np.mean(y.detach().numpy(),1), 'ro',ls='-', label='Original data')
#plt.plot(range(len(y.detach().numpy())), np.mean(predict.detach().numpy(),1), 'b', ls='-', label='Fitting line')
#plt.show()
plt.figure()
for i in range(69):

    plt.plot(range(len(y.detach().numpy())),y.detach().numpy()[:,i])

plt.figure()
for i in range(69):

    plt.plot(range(len(y.detach().numpy())), predict.detach().numpy()[:,i])

plt.show()
