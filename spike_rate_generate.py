import numpy as np
import matplotlib.pyplot as plt
PATH = 'voltage.txt'
data = np.loadtxt(PATH)
time_window = 10
print(data.shape)
data = data.reshape(-1,time_window,69)
print(data.shape)
data = data.mean(axis=1)
print(np.array(data).shape)
np.savetxt('voltage_5ms.txt',data)
axis = np.arange(20000)
axis = axis *5

plt.figure()
for i in range(data.shape[1]):
    plt.plot(axis,data[:,i],alpha = 0.5)
plt.plot(axis,np.mean(data[:,:],1),color = 'r',label = "average")
plt.ylabel('voltage(mV)')
plt.xlabel('time(ms)')
plt.legend(bbox_to_anchor=(1,1),#图例边界框起始位置
                 loc="upper right",#图例的位置
                 ncol=1,#列数
                 mode="expend",#当值设置为“expend”时，图例会水平扩展至整个坐标轴区域
                 borderaxespad=0)#坐标轴和图例边界之间的间距
plt.show()