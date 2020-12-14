import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def loader(path):
    data = np.loadtxt(path)
    return data,data.shape



def vs_V(data):
    '''

    :param data:
    :return:
    '''
    plt.figure()
    plt.xlabel('time (s)')
    plt.ylabel('$V_m$ (V)')
    t = data[0]
    v = data[1]
    plt.plot(t, v, 'k.')

def vs_spike(data):
    x = data[0]
    y = data[1]
    


class folder_reader:
    def __init__(self,train_root_path):
        self.train_root_path = train_root_path
        self.txt_adress = None

    def load_file_name(self,path):
        file_name_list = []
        with open(path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                    pass
                file_name_list.append(lines)
                pass
        print(file_name_list)
        return file_name_list

    def write_name_list(self, name_list, file_name):
        self.txt_adress = str(self.train_root_path + file_name)
        f = open(self.train_root_path + file_name, 'w')
        for i in range(len(name_list)):
            f.write(name_list[i] + "\n")
        f.close()

    def read_line(self):
        fh = open(self.txt_adress)
        imgs = []
        for line in fh.readlines():
            imgs.append(line.strip('\n'))

