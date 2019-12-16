#Juypter Fetch and Feed
import tensorflow as tf

#Fetch
input1=tf.constant(3.0)
input2=tf.constant(2.0)
input3=tf.constant(5.0)

add=tf.add(input2,input3)
mu1=tf.multiply(input1,add)

with tf.Session() as sess:
    result=sess.run([mu1,add])    #同时运行两个op
    print(result)

#Feed
#创建占位符
input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)
output=tf.multiply(input1,input2)

with tf.Session() as sess:
    #feed的数据以字典的形式传入
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))