#!/usr/bin/python

'''
Show the most used activation functions in Network
'''

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 200)

#1. struct
#following are popular activation functions
y_relu = tf.nn.relu(x)
y_sigmod = tf.nn.sigmoid(x)
y_tanh = tf.nn.tanh(x)
y_softplus = tf.nn.softplus(x)

#2. session
sess = tf.Session()
y_relu, y_sigmod, y_tanh, y_softplus =sess.run([y_relu, y_sigmod, y_tanh, y_softplus])

# plot these activation functions
plt.figure(1, figsize=(8,6))

plt.subplot(221)
plt.plot(x, y_relu, c ='red', label = 'y_relu')
plt.ylim((-1, 5))
plt.legend(loc = 'best')

plt.subplot(222)                             #subplot创建单个子图
plt.plot(x, y_sigmod, c ='b', label = 'y_sigmod')
plt.ylim((-1, 5))
plt.legend(loc = 'best')

plt.subplot(223)
plt.plot(x, y_tanh, c ='b', label = 'y_tanh')
plt.ylim((-1, 5))
plt.legend(loc = 'best')

plt.subplot(224)
plt.plot(x, y_softplus, c ='c', label = 'y_softplus')
plt.ylim((-1, 5))
plt.legend(loc = 'best')

plt.show()