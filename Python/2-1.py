#Jupyter 创建、启动图

import tensorflow as tf
#创建一个常量op  (一行两列)
m1=tf.constant([[3,3]])
#创建一个常量op  (两行一列的矩阵)
m2=tf.constant([[2],[3]])
#创建一个矩阵乘法op,把m1和m2传入
product=tf.matmul(m1,m2)
#这样的输出结果是Tensor
print(product)               #Shift+Enter运行快捷键,输入.之后可以按Tab键有提示

#定义一个会话,启动默认的图
sess=tf.Session()              #用"with tf.Session() as sess:"下面再操作就可以不用写sess.close()的操作
#调用sess的run方法来执行矩阵乘法op
#run(product)触发了图中3个op
result=sess.run(product)
print(result)
sess.close()