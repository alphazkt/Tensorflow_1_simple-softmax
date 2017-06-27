# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 19:40:27 2017

@author: Alphatao
#最简单softmax
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#读取数据
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#定义参数
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32,[None, 784])
    y_ = tf.placeholder(tf.float32,[None, 10])

with tf.name_scope('layer'):
    with tf.name_scope('weights'):    
        W = tf.Variable(tf.zeros([784,10]))
        tf.summary.histogram("weight",W)
    with tf.name_scope('bias'):
        b= tf.Variable(tf.zeros([10]))
        tf.summary.histogram("bias",b)
    #设置softmax y=x*W+b 以及实际y_
    y = tf.nn.softmax(tf.matmul(x,W)+b)
    tf.summary.histogram("output",y)

sess = tf.Session()

#误差cross
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
    tf.summary.scalar('Loss',cross_entropy)

#梯度下降训练
with tf.name_scope('train'): 
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

merged = tf.summary.merge_all() 
writer = tf.summary.FileWriter("logs/",sess.graph)  
#初始化
init = tf.global_variables_initializer()
sess.run(init)


#循环训练 每次选取100
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
    if i%50==0:
        result = sess.run(merged,feed_dict={x:batch_xs,y_:batch_ys}) 
        writer.add_summary(result,i) 

#正确率计算
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print (sess.run(accuracy,feed_dict={x:mnist.test.images, y_:mnist.test.labels}))