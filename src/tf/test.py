'''
Created on 2018年1月19日

@author: user
'''
import tensorflow as tf
label =tf.Variable([-1,2,3,4,5,6,7,8,9,-2])
onehot_labels = tf.one_hot(label, depth=10)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(onehot_labels))