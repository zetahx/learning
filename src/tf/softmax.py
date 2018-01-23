# -*- coding:utf-8 -*-
#'''
# Created on 2017年12月11日
# 
# @author: user
# '''
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# """A very simple MNIST classifier.
# 
# See extensive documentation at
# https://www.tensorflow.org/get_started/mnist/beginners
# """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import numpy as np

#添加一层网络
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def t_main():
    # 1.训练的数据
    # Make up some real data 
    x_data = np.linspace(-1,1,300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise
    
    # 2.定义节点准备接收数据
    # define placeholder for inputs to network  
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])
    
    # 3.定义神经层：隐藏层和预测层
    # add hidden layer 输入值是 xs，在隐藏层有 10 个神经元   
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
    # add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
    prediction = add_layer(l1, 10, 1, activation_function=None)
    
    # 4.定义 loss 表达式
    # the error between prediciton and real data    
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                        reduction_indices=[1]))
    
    # 5.选择 optimizer 使 loss 达到最小                   
    # 这一行定义了用什么方式去减少 loss，学习率是 0.1       
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    
    
    # important step 对所有变量进行初始化
    init = tf.initialize_all_variables()
    sess = tf.Session()
    # 上面定义的都没有运算，直到 sess.run 才会开始运算
    sess.run(init)
    
    # 迭代 1000 次学习，sess.run optimizer
    for i in range(1000):
        # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
        # to see the step improvement
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))





def main(_):
    FLAGS = 'MNIST_data/'
    # Import data
    mnist = input_data.read_data_sets(FLAGS, one_hot=True)
    print (mnist.test.images)
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b
    
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    
    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)