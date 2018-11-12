from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np
import tensorflow.contrib as tc


def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])

def inverted_bottleneck(input, up_sample_rate, channels, subsample, normalizer, bn_params, iter):
  with tf.variable_scope('inverted_bottleneck{}_{}_{}'.format(iter, up_sample_rate, subsample)):
    stride = 2 if subsample else 1
    #conv 1x1 with relu6
    output = tc.layers.conv2d(input, up_sample_rate*input.get_shape().as_list()[-1], 1, activation_fn=tf.nn.relu6, normalizer_fn=normalizer, normalizer_params=bn_params)
    #conv 3x3 with relu6
    output = tc.layers.separable_conv2d(output, None, 3, 1, stride=stride, activation_fn=tf.nn.relu6, normalizer_fn=normalizer, normalizer_params=bn_params)
    #conv 1x1 without activation function
    output = tc.layers.conv2d(output, channels, 1, activation_fn=None, normalizer_fn=normalizer, normalizer_params=bn_params)

    #now fuse main stream with residual
    if input.get_shape().as_list()[-1] == channels:
      output = tf.add(input,output)
    return output

def crop(x1,x2):
    with tf.name_scope("crop_and_concat"):
        x1_shape = shape(x1)
        #print("shape 1 ", x1_shape)
        x2_shape = shape(x2)
        #print("shape 2 ", x2_shape)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return x1_crop

def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob, name):
  #current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
  current = slim.batch_norm(current,activation_fn=None)
  current2 = tf.nn.elu(current)
  #current = conv2d(current, in_features, out_features, kernel_size)
  current3 = slim.conv2d(current2, out_features, [kernel_size, kernel_size], scope=name)
  current4 = tf.nn.dropout(current3, keep_prob)

  return current4

def Convolution(current, out_features, kernel_size, name):
  current2 = tf.nn.elu(current)
  current_out = slim.conv2d(current2, out_features, [kernel_size, kernel_size], scope=name)

  return current_out


def Denseblock(input, layers, Din_features, growth, is_training, keep_prob, name):
  #print("BEFORE ANY DENSE BLOCK ")
  #print(layers)
  current = input
  features = Din_features
  for idx in range(layers):
    name2 = name + str(idx)
    #print(name2)
    tmp = batch_activ_conv(current, features, growth, 3, is_training, keep_prob, name2)
    #print(current)
    #print(tmp)
    current =  tf.concat((current, tmp), 3)
    features += growth
  return current

def TransitionUp_elu(input, filters, strideN,  name):
  #print("Transition UP")
  current = input
  #print(name)
  #output_shape = [1, 75, 75, 128]
  upconv = slim.conv2d_transpose(current, filters, [3, 3], stride=strideN,  scope=name)
  upconv = tf.nn.relu(upconv)
  #print(upconv)
  return upconv

def Concat_layers(conv1, conv2, nm='test'):
    #Concat values
    if(nm=='upconv2'):
      pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
      conv1 = tf.pad(conv1, pattern)
    if(nm=='upconv3'):
      pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
      conv1 = tf.pad(conv1, pattern)
    if(nm=='upconvV3'):
      pattern = [[0, 0], [1, 0], [1, 0], [0, 0]]
      conv1 = tf.pad(conv1, pattern)

    fused = tf.concat([conv1, conv2],3)
    return  fused
