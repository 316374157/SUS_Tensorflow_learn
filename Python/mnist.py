from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import math

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 10


#输入与占位符
#inference()： 输入：输入图片，隐藏层1神经元个数，隐藏层2神经元个数;输出：神经网络输出
#tf.name_scope 主要结合 tf.Variable() 来使用，方便参数命名管理。
def inference(images,hidden1_units,hidden2_units):
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS,hidden1_units],
            stddev=1.0/math.sqrt(float(IMAGE_PIXELS))),name = 'weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),name = 'biases')

        hidden1 = tf.nn.relu(tf.matmul(images,weights)+biases)

    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units,hidden2_units],
            stddev=1.0/math.sqrt(float(hidden1_units))),name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),name = 'biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1,weights)+biases)

    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([hidden2_units,NUM_CLASSES],
            stddev=1.0/math.sqrt(float(hidden2_units))),name = 'weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),name = 'biases')
        logits = tf.matmul(hidden2,weights) + biases

    return logits

#loss():通过添加所需的op来构建图;输入参数 logits:网络输出，为float类型，[batch_size,NUM_CLASSES]
#labels:目标标签，为int32类型，[batch_size]
#输出：损失，float类型
    #tf.losses.sparse_softmax_cross_entropy(
    #labels,形状为[d_0, d_1, ..., d_{r-1}]的Tensor(其中,r是labels和结果的秩),并且有dtype int32或int64.
    #logits,形状为[d_0, d_1, ..., d_{r-1}, num_classes],并且是dtype float32或float64的未缩放的日志概率.
    #weights=1.0, loss的系数.这必须是标量或可广播的labels(即相同的秩,每个维度是1或者是相同的).
    #scope=None, 计算loss时执行的操作范围.
    #loss_collection=tf.GraphKeys.LOSSES, 将添加loss的集合.
    #reduction=Reduction.SUM_BY_NONZERO_WEIGHTS) 适用于loss的减少类型.
    #返回值  与logits具有相同类型的加权损失Tensor.如果reduction是NONE,它的形状与labels相同；否则,它是标量
def loss(logits,labels):
    labels = tf.to_int64(labels)
    return tf.losses.sparse_softmax_cross_entropy(labels= labels,logits= logits)

#training():  输入：损失，学习速率 输出：训练op  训练方法为梯度下降。
#tf.summary.scalar()：TensorBoard可以将训练过程中的各种绘制数据展示出来，包括标量，图片，音频，计算图分布，直方图和嵌入式向量。
def training(loss,learning_rete):
    tf.summary.scalar('loss',loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rete)
    global_step = tf.Variable(0,name = 'global_step',trainable = False)
    train_op = optimizer.minimize(loss,global_step = global_step)
    return train_op


#tf.nn.in_top_k(prediction, target, K):prediction就是表示你预测的结果，大小就是预测样本的数量乘以输出的维度，类型是tf.float32等。
# target就是实际样本类别的标签,大小就是样本数量的个数。K表示每个样本的预测结果的前K个最大的数里面是否含有target中的值,一般都是取1。
def evaluation(logits,labels):
    correct = tf.nn.in_top_k(logits,labels,1)
    return tf.reduce_sum(tf.cast(correct,tf.int32))