from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# MNIST数据存放的路径
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32,shape=[None,784])
y_ = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# 创建Session
sess = tf.InteractiveSession()
saver = tf.train.Saver()

#权重初始化
#为了创建这个模型，我们需要创建大量的权重和偏置项。这个模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度。
# 由于我们使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题。
# 为了不在建立模型的时候反复做初始化操作，我们定义两个函数用于初始化。

#tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
#从截断的正态分布中输出随机值。 shape表示生成张量的维度，mean是均值，stddev是标准差,seed,一个整数，当设置之后，每次生成的随机数都一样
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#tf.constant(value,dtype=None,shape=None,name=None) 
#创建一个常量tensor，按照给出value来赋值，可以用shape来指定其形状。value可以是一个数，也可以是一个list。 如果是一个数，那么这个常量中所有值的按该数来赋值。 
#如果是list,那么len(value)一定要小于等于shape展开后的长度。赋值时，先将value中的值逐个存入。不够的部分，则全部存入value的最后一个值。
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


#卷积和池化
#在这个实例里，我们会一直使用vanilla版本。我们的卷积使用1步长，0边距的模板，保证输出和输入是同一个大小。我们的池化用简单传统的2x2大小的模板做max pooling

#tf.nn.conv2d (input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
#input:输入的要做卷积的图片，要求为一个张量，shape为 [ batch, in_height, in_weight, in_channel ]，其中batch为图片的数量，in_height 为图片高度，
# in_weight 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3
#filter:卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，其中 filter_height 为卷积核高度，
# filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量
#strides:卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
#padding:string类型，值为"SAME"和 "VALID"，表示的是卷积的形式，是否考虑边界。"SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
#use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#max pooling是CNN当中的最大值池化操作，其实用法和卷积很类似,tf.nn.max_pool(value, ksize, strides, padding, name=None)
#第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
#第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
#第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
#第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'
#返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


#第一层卷积
#它由一个卷积接一个max pooling完成。卷积在每个5x5的patch中算出32个特征。卷积的权重张量形状是[5, 5, 1, 32]，
# 前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。 而对于每一个输出通道都有一个对应的偏置量。

w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

#为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数
#tf.reshape(tensor,shape,name=None)函数的作用是将tensor变换为参数shape形式，其中的shape为一个列表形式，特殊的是列表可以实现逆序的遍历，
# 即list(-1).-1所代表的含义是我们不用亲自去指定这一维的大小，函数会自动进行计算，但是列表中只能存在一个-1。（如果存在多个-1，就是一个存在多解的方程） 
x_image = tf.reshape(x,[-1,28,28,1])

#然后我们x_image与权重张量进行卷积，添加偏差，应用ReLU函数，最后应用最大池。我们把x_image和权值向量进行卷积，加上偏置项，
# 然后应用ReLU激活函数，最后进行max pooling。
#tf.nn.relu(features, name = None) 这个函数的作用是计算激活函数 relu，即 max(features, 0)。将大于0的保持不变，小于0的数置为0。
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积
#为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个5x5的的贴片会得到64个特征。
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#密集连接层
#现在，图片尺寸减小到7x7的，我们加入一个有1024个神经元的全连接层，用于处理整个图片。我们把池化层输出的张量重塑成一些向量，乘上权重矩阵，加上偏置，然后对其使用RELU。
w_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

#退出
#为了减少过拟合，我们在输出层之前加入辍学。用我们一个placeholder来代表一个神经元型态的输出在降中保持不变的概率。这样我们可以在训练过程中启用辍学，
# 在测试过程中关闭辍学。 TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元型态的输出外，还会自动处理神经元输出值的比例。所以用差的时候可以不用考虑规模。
keep_prob = tf.placeholder(tf.float32)
#def dropout(x, keep_prob, noise_shape=None, seed=None, name=None)x，你自己的训练、测试数据等,keep_prob，dropout概率
#Dropout就是在不同的训练过程中随机扔掉一部分神经元。也就是让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算
#简单来说，就是使输入tensor中某些元素变为0，其它没变0的元素变为原来的1/keep_prob大小
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#输出层
w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

#训练和评估模型

cross_entropy = -tf.reduce_sum(y_* tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(10001):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {x:batch[0],y_:batch[1],keep_prob:1.0})
        print("训练了 %d,概率是%g"%(i,train_accuracy))
    train_step.run(feed_dict = {x:batch[0],y_:batch[1],keep_prob:0.5})


saver.save(sess,"Model/CNN1_model.ckpt")
print("训练结束" )