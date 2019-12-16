#Juypter 案例

import tensorflow as tf
import numpy as np

#使用numpy生成100个随机点
x_data=np.random.rand(100)
y_data=x_data*0.1+0.2      #类似于平面坐标系上的一条直线
#上面的可以看做样本

#构造一个线性模型
b=tf.Variable(0.)   #k和b中变量改成其他值也可
k=tf.Variable(0.)
y=k*x_data+b

#二次代价函数
loss=tf.reduce_mean(tf.square(y_data-y))  #tf.reduce_mean求平均值
#定义一个梯度下降法来进行训练的优化器
optimizer=tf.train.GradientDescentOptimizer(0.2)    #括号中0.2是学习率
#最小化代价函数
train=optimizer.minimize(loss)

#初始化变量
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20==0:   #每20次打印
            print(step,sess.run([k,b]))