import tensorflow as tf
import loader
def spike(x):
    time_range = 2247
    truth, shape = loader.loader(
        'D:\\document\\体外网络发放数据\\a195e390b707a65cf3f319dbadbbc75f_6b245c0909b1a21072dd559d4203e15b_8.txt')
    ture = truth[200::, :]

    start =tf.tensor(truth[0:200, :],dtype=tf.float32)
    ture = tf.tensor(ture, dtype=tf.float32)

    spike = tf.zeros(size=(shape[0]-200,shape[1]),dtype=tf.float32)
    spike = tf.concat([start,spike],axis=0)
    print(spike.size())
    def condition(self,time):
        time <time_range
