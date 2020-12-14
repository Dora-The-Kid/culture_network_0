import tensorflow as tf
import numpy as np
import os
import PIL
import tensorflow.keras.layers as layers
import time
import argparse
import loader

def date_generat_test():
    data,shape = loader.loader('D:\\document\\体外网络发放数据\\a195e390b707a65cf3f319dbadbbc75f_6b245c0909b1a21072dd559d4203e15b_8.txt')
    neuro_number = shape[1]

class SPIKE(tf.keras.layers.Layer):
    def __init__(self, input,time,neuro,shape):
        super(SPIKE, self).__init__()
        self.input = input
        self.time = time
        self.neuro = neuro
        self.shape = shape

    def condition(self,time):
        return tf.less(time, self.time)
    def body(self,time,spike):
        spike[time] =
        return time + 1


    def build(self, input_shape):
        self.spik = tf.Variable(tf.zeros(dtype=tf.float32, shape=[self.shape]),
                             dtype=tf.float32
                             )





def generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model