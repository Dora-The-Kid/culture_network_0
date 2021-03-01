import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math
import pandas as pd
data_1 = np.array([260925,296296,299490])
data_2 = np.array([356846,338160,292290])
data = [
    [260925,296296,299490],
    [356846,338160,292290]
]
data = np.array(data)
df = pd.DataFrame(data.T,columns=['ctrl','epsc_down'])
a =stats.levene(data_1, data_2)
b = stats.ttest_ind(data_1,data_2,equal_var=True)
plt.figure()
f = df.boxplot(sym = 'o',            #异常点形状
               vert = True,          # 是否垂直
               whis=1.5,             # IQR
               patch_artist = True,  # 上下四分位框是否填充
               meanline = False,showmeans = True,  # 是否有均值线及其形状
               showbox = True,   # 是否显示箱线
               showfliers = True,  #是否显示异常值
               notch = False,    # 中间箱体是否缺口
               return_type='dict')  # 返回类型为字典

plt.title('duration')
plt.show()
print(a,b)
def plot_sig(xstart,xend,ystart,yend,sig):
    for i in range(len(xstart)):
        x = np.ones((2))*xstart[i]
        y = np.arange(ystart[i],yend[i],yend[i]-ystart[i]-0.1)
        plt.plot(x,y,label="$y$",color="black",linewidth=1)

        x = np.arange(xstart[i],xend[i]+0.1,xend[i]-xstart[i])
        y = yend[i]+0*x
        plt.plot(x,y,label="$y$",color="black",linewidth=1)

        x0 = (xstart[i]+xend[i])/2
        y0=yend[i]
        plt.annotate(r'%s'%sig, xy=(x0, y0), xycoords='data', xytext=(-15, +1),
                     textcoords='offset points', fontsize=16,color="red")
        x = np.ones((2))*xend[i]
        y = np.arange(ystart[i],yend[i],yend[i]-ystart[i]-0.1)
        plt.plot(x,y,label="$y$",color="black",linewidth=1)
        plt.ylim(0,math.ceil(max(yend)+4))             #使用plt.ylim设置y坐标轴范围
    #     plt.xlim(math.floor(xstart)-1,math.ceil(xend)+1)
        #plt.xlabel("随便画画")         #用plt.xlabel设置x坐标轴名称
        '''设置图例位置'''
        #plt.grid(True)
    plt.show()
plot_sig([0.42,1.42],[1.42,2.42],[30,20],[30.8,20.8],'***')