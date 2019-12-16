#1.导入模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
#import mnist 导入自己写的

FLAGS = None

#在开始训练之前我们需要先用tf.placeholder占位符帮手写体数据集占一个位置,这里我们的标签的维度是[batch_size, 1]
def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32,shape=(batch_size,mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32,shape=(batch_size))
    return images_placeholder,labels_placeholder





#做完这些基本的操作之后我们就可以对其进行迭代训练了，但是在迭代训练之前我们得处理一下我们的输入数据，
# 之前我们一直都是用tf.placeholder来假装我们有输入数据，现在我们需要将真实数据传入进来：
#目的：在训练时对应次数自动填充字典
#输入：数据源data_set，来自input_data.read_data_sets
 #    图片holder，images_pl,来自placeholder_inputs()
 #    标签holder,labels_pl,来自placeholder_inputs()
#输出：喂养字典feed_dict.

def fill_feed_dict(data_set,images_pl,labels_pl):
    images_feed,labels_feed = data_set.next_batch(FLAGS.batch_size,FLAGS.fake_data)
    feed_dict = {images_pl:images_feed,labels_pl:labels_feed}
    return feed_dict

#接下来我们定义一个评估模型的函数:每循环1000次或结束进行一次评估。
#输入：sess: 模型训练所使用的Session
 #   eval_correct: 预测正确的样本数量
 #   images_placeholder: images placeholder.
 #   labels_placeholder: labels placeholder.
 #   data_set: 图片和标签数据，来自input_data.read_data_sets().
#输出：打印测试结果。
def do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_set):
    true_count = 0
    steps_per_epoh = data_set.num_examples//FLAGS.batch_size
    num_examples = steps_per_epoh * FLAGS.batch_size
    for step in range(steps_per_epoh):
        feed_dict = fill_feed_dict(data_set,images_placeholder,labels_placeholder) 
        true_count += sess.run(eval_correct,feed_dict = feed_dict)
    precision = float(true_count)/num_examples
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f'
        %(num_examples,true_count,precision))




#run_training():在开始训练之前，我们需要去读取数据
#input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
#train_dir——文件夹的文件夹的位置;fake_data——是否使用假数据，默认为False
#one_hot——是否把标签转为一维向量，默认为False，由于没有采用one-hot编码，那么这里的返回值就是图片数字的下标，
# 也就是图片数字到底是几。是一个单纯的数字，而不是一个十维的向量（[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]）。
def run_training():
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir,FLAGS.fake_data)
    with tf.Graph().as_default():   #这个命令表明所有已经构建的操作都要与默认的tf.Graph全局实例关联起来。Graph包含一组 Operation对象，表示计算单位
        images_placeholder,labels_placeholder = placeholder_inputs(FLAGS.batch_size)
        #我们就假装我们有了输入的图片和其对应的标签(其实是没有的，我们需要之后再将其传进来)，接下来我们构建图的输出公式
        logits = mnist.inference(images_placeholder,FLAGS.hidden1,FLAGS.hidden2)
        #选择损失和训练参数:
        loss = mnist.loss(logits,labels_placeholder)
        train_op = mnist.training(loss,FLAGS.learning_rate)
        #评估模型结果指标:
        eval_correct = mnist.evaluation(logits,labels_placeholder)
        #接下来我们把图运行过程中发生的事情(产生的数据记录下来):
        summary = tf.summary.merge_all()
        #初始化变量:
        init = tf.global_variables_initializer()
        #建立一个保存训练中间数据的存档点:
        saver = tf.train.Saver()
        #建立会话:
        sess = tf.Session()
        #创建一个记事本写入器:
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir,sess.graph)
        #之后再初始化变量:
        sess.run(init)

        #再接着就可以对其进行迭代训练
        for step in range(FLAGS.max_steps):
            start_time = time.time()
            feed_dict = fill_feed_dict(data_sets.train,images_placeholder,labels_placeholder)
            _,loss_value = sess.run([train_op,loss],feed_dict = feed_dict)  

        #sess.run()会返回一个有两个元素的元组。其中每一个Tensor对象，对应了返回的元组中的numpy数组，而这些数组中包含了当前这步训练中对应Tensor的值。
        # 由于train_op并不会产生输出，其在返回的元祖中的对应元素就是None，所以会被抛弃。但是，如果模型在训练中出现偏差，loss Tensor的值可能会变成NaN，
        # 所以我们要获取它的值，并记录下来。我们也希望记录一下程序运行的时间
            duration = time.time()-start_time
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)'%(step,loss_value,duration))
                summary_str = sess.run(summary,feed_dict = feed_dict)
                summary_writer.add_summary(summary_str,step)
                summary_writer.flush()
        #在每次运行summary时，都会往事件文件中写入最新的即时数据，函数的输出会传入事件文件读写器（writer）的add_summary()函数。
            if (step + 1) %1000 == 0 or (step+1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir,'model.ckpt')
                saver.save(sess,checkpoint_file,global_step = step)
                #对模型进行评估。do_eval函数会被调用三次，分别使用训练数据集、验证数据集合测试数据集。
                print('Training Data Eval:')
                do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.train)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.test)
def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=2000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/tmp/tensorflow/mnist/logs/fully_connected_feed',
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)