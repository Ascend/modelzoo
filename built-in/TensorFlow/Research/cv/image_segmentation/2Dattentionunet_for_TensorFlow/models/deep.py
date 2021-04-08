import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os

def relu(inputs):
    return tf.nn.relu(inputs)

def prelu(inputs):
    return PReLU(inputs)

def elu(inputs):
    return tf.nn.elu(inputs)

def leaky(inputs):
    return tf.nn.leaky_relu(inputs)

def PReLU(_tensor):
    with tf.variable_scope(None, default_name="prelu"):
        alphas = tf.get_variable('alpha', _tensor.get_shape()[-1],
                            initializer=tf.constant_initializer(0.01),
                            dtype=tf.float32)
        pos = tf.nn.relu(_tensor)
        neg = alphas * (_tensor - abs(_tensor)) 
        return pos + neg



def ConvBlock(inputs, n_filters0 = 64, n_filters1 = 32,  kernel_size0=[3, 3], kernel_size1=[2, 2],strides=[1, 1]):

    net = slim.conv2d(inputs, n_filters0, kernel_size=kernel_size0, stride=strides)
    net = elu(net)
    net = slim.batch_norm(net, fused=True)
    net = slim.conv2d(net, n_filters1, kernel_size=kernel_size1, stride=strides)
    net = slim.batch_norm(net, fused=True)
    net = elu(net)
    #net = slim.batch_norm(net, fused=True)
    
    #net = tf.nn.relu(net)
    #net = slim.batch_norm(net, fused=True)
    
    return net


def convBlock(inputs, n_filters, kernel_size=[3, 3]):
   
    net = slim.conv2d(inputs, n_filters, kernel_size=kernel_size)
    #net = slim.batch_norm(net, fused=True)
    #net = PReLU(net)
    net = elu(net)
    #net = slim.batch_norm(net)
    return net



def batch_relu(net):
    net = slim.batch_norm(net, fused=True)
    net = elu(net)
    #net = slim.batch_norm(net, fused=True)
    #net = slim.batch_norm(net)
    return net

def Upsampling(inputs, scale=2):
    net = tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * scale, tf.shape(inputs)[2] * scale])
    return net

def build_deep(inputs, num_classes, gpu):
    #os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    #Downsampling Path

    net = ConvBlock(inputs, 64, 64, kernel_size1=[3, 3])
    #net = encoderBlock(inputs, 64, n = 2)
    #net = slim.batch_norm(net)
    #net = tf.nn.relu(net)
    #down1 = slim.conv2d(net, 32, kernel_size=[2, 2], stride=[1, 1], activation_fn=None)
    #net = batch_relu(net)
    skip0 = slim.conv2d(net, 32, kernel_size=[2, 2], activation_fn=None)
    skip0 = batch_relu(skip0)
    net = slim.pool(skip0, [2, 2], stride=[2, 2], pooling_type='MAX')
    #net = batch_relu(net)
    skip = net

    net = ConvBlock(net, kernel_size1=[3, 3])
    #net = encoderBlock(inputs, 64)
    #net = tf.add(skip0, net, name='add1')
    skip1 = tf.concat([net, skip], axis=3)
    net = slim.pool(skip1, [2, 2], stride=[2, 2], pooling_type='MAX')
    #net = batch_relu(net)
    add1 = net
    
    #256
    net = ConvBlock(net)
    #net = tf.add(add1, net)
    skip2 = tf.concat([net, add1], axis=3)
    net = slim.pool(skip2, [2, 2], stride=[2, 2], pooling_type='MAX')
    #net = batch_relu(net)
    add2 = net
    
    #128
    net = ConvBlock(net)
    #net = tf.add(add2, net)
    skip3 = tf.concat([net, add2], axis=3)
    net = slim.pool(skip3, [2, 2], stride=[2, 2], pooling_type='MAX')
    #net = batch_relu(net)
    add3 = net
  
    #64
    net = ConvBlock(net)
    #net = tf.add(add3, net)
    skip4 = tf.concat([net, add3], axis=3)
    net = slim.pool(skip4, [2, 2], stride=[2, 2], pooling_type='MAX')
    #net = batch_relu(net)
    add4 = net
    
    #32
    net = ConvBlock(net)
    #net = tf.add(add4, net)
    skip5 = tf.concat([net, add4], axis=3)
    net = slim.pool(skip5, [2, 2], stride=[2, 2], pooling_type='MAX')
    #net = batch_relu(net)
    add5 = net
    #net = tf.nn.dropout(net, keep_prob)
    
    #16
    net = ConvBlock(net)
    #net = tf.add(add5, net)
    skip6 = tf.concat([net, add5], axis=3)
    net = slim.pool(skip6, [2, 2], stride=[2, 2], pooling_type='MAX')
    #net = batch_relu(net)
    #net = tf.nn.dropout(net, keep_prob)
    #add6 = net
    


    #Up Sampling Path

    up1 = Upsampling(net)
    net = tf.concat([up1, skip6], axis=3)
    net = ConvBlock(net, kernel_size1=[3, 3])
    #net = tf.add(add7, net)
    net = tf.concat([net, up1], axis=3)
    #net = batch_relu(net)
    #net = Upsampling(net)
    #add8 = net

    up2 = Upsampling(net)
    net = tf.concat([up2, skip5], axis=3)
    net = ConvBlock(net, kernel_size1=[3, 3])
    #net = tf.add(add8, net)
    net = tf.concat([net, up2], axis=3)
    #net = Upsampling(net)
    #add9 = net

    up3 = Upsampling(net)
    net = tf.concat([up3, skip4], axis=3)
    net = ConvBlock(net, kernel_size1=[3, 3])
    #net = tf.add(add9, net)
    net =tf.concat([net, up3], axis=3)
    #net = Upsampling(net)
    #add10 = net

    up4 = Upsampling(net)
    net = tf.concat([up4,skip3], axis=3)
    net = ConvBlock(net, kernel_size1=[3, 3])
    #net = tf.add(add10, net)
    net = tf.concat([net, up4], axis=3)
    #net = Upsampling(net)
    #add11 = net

    up5 = Upsampling(net)
    net = tf.concat([up5, skip2], axis=3)
    net = ConvBlock(net, kernel_size1=[3, 3])
    #net = tf.add(add11, net)
    net = tf.concat([net, up5], axis=3)
    #net = Upsampling(net)
    #add12 = net

    up6 = Upsampling(net)
    net = tf.concat([up6, skip1], axis=3)
    net = ConvBlock(net, kernel_size1=[3, 3])
    net = tf.concat([net, up6], axis=3)
    #net = batch_relu(net)

    net = Upsampling(net)

    print(tf.shape(net))
    #net = Upsampling(net)
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None)
    return net




