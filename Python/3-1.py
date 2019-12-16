#Juypter 非线性回归
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt   #python中一个画图的工具包

#使用numpy生成200个随机点
#在-0.5到0.5范围生成均匀分布的200个点
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis] #np.newaxis使增加一个维度，一维变二维
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise

#定义两个placeholder
x=tf.placeholder(tf.float32,[None,1])  #None行不确定,有一列
y=tf.placeholder(tf.float32,[None,1])

#构建神经网络中间层 10个  所以构架是1 - 10 - 1
Weight_L1=tf.Variable(tf.random_normal([1,10])) #1行10列
biase_L1=tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1=tf.matmul(x,Weight_L1)+biase_L1   #tf.matmul用于两矩阵的相乘
L1=tf.nn.tanh(Wx_plus_b_L1)

#定义神经网络输出层 中间10个神经元，输出1个，故是从[1,10]到[10,1]
Weight_L2=tf.Variable(tf.random_normal([10,1]))
biase_L2=tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2=tf.matmul(L1,Weight_L2)+biase_L2
prediction=tf.nn.tanh(Wx_plus_b_L2)

#二次代价函数
loss=tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)   #通过minimize最小化loss的值

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
        
    #获得预测值
    prediction_value=sess.run(prediction,feed_dict={x:x_data})
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()