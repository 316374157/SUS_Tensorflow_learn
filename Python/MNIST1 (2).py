from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#softmax回归
#evidence=积分求和(W(i,j)X(j)+b(i)) W代表权重，b代表偏置量，然后转换成概率
#y=softmax(evidence) 然后可以转换成关于x的函数
#softmax(x)=normalize(exp(x)) 最后则可以写成最终式子 y=softmax(wx+b)

#tf.placeholder(
#    dtype,数据类型。常用的是tf.float32,tf.float64等数值类型
#    shape=None, 数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
#    name=None 名称
#)
#所以placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。
# 等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
x = tf.placeholder(tf.float32,[None,784])

#一个Variable代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中。它们可以用于计算输入值，也可以在计算中被修改。
#tf.Variable( initial_value=None,张量,Variable 类的初始值，这个变量必须指定 shape 信息，否则后面 validate_shape 需设为 False
#    trainable=True,    Bool,是否把变量添加到collection GraphKeys.TRAINABLE_VARIABLES 中(collection 是一种全局存储，不受变量名生存空间影响，一处保存，到处可取)
#    collections=None,      Graph collections,全局存储，默认是 GraphKeys.GLOBAL_VARIABLES
#    validate_shape=True,   Bool,是否允许被未知维度的 initial_value 初始化
#    caching_device=None,   string,指明哪个 device 用来缓存变量
#    name=None,             string,变量名
#    variable_def=None,
#    dtype=None,            dtype,如果被设置，初始化的值就会按照这个类型初始化
#    expected_shape=None,   TensorShape,要是设置了，那么初始的值会是这种维度
#    import_scope=None)
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#tf.matmul（）将矩阵a乘以矩阵b，生成a * b。
#tf.matmul(a,一个类型为 float16, float32, float64, int32, complex64, complex128 且张量秩 > 1 的张量
# b,  一个类型跟张量a相同的张量。
# transpose_a=False,  如果为真, a则在进行乘法计算前进行转置
# transpose_b=False,  如果为真, b则在进行乘法计算前进行转置。
# adjoint_a=False,    如果为真, a则在进行乘法计算前进行共轭和转置。
# adjoint_b=False,    如果为真, b则在进行乘法计算前进行共轭和转置。
# a_is_sparse=False,  如果为真, a会被处理为稀疏矩阵。
# b_is_sparse=False,  如果为真, b会被处理为稀疏矩阵。
# name=None) 
#返回值： 一个跟张量a和张量b类型一样的张量且最内部矩阵是a和b中的相应矩阵的乘积。
y = tf.nn.softmax(tf.matmul(x,W)+b)

#为了训练的模型，我们首先需要定义一个指标来评估这个模型是好的。其实，在机器学习，我们通常定义指标来表示一个模型是坏的，
# 这个指标称为成本（cost）或损失（loss），然后尽量最小化这个指标。一个非常常见的，非常漂亮的成本函数是“交叉熵”,它的定义如下：
# H(y) = -积分和(y(撇)*log(y))

y_ = tf.placeholder(tf.float32,[None,10])

#reduce_sum应该理解为压缩求和，用于降维
#reduce_sum ( 
#    input_tensor , 按照axis中已经给定的维度来减少的
#    axis = None ,  要减小的尺寸.如果为None(默认),则缩小所有尺寸.必须在范围[-rank(input_tensor), rank(input_tensor))内.
#    keep_dims = False , 如果keep_dims为true,则减小的维度将保留为长度1.
#    name = None , 
#    reduction_indices = None )
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#我们要求TensorFlow用梯度下降算法以0.01的学习速率最小化交叉熵。梯度下降算法是一个简单的学习过程，TensorFlow只需将每个变量一点点地往使成本不断降低的方向移动
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化变量
init = tf.global_variables_initializer()

sess = tf.Session()
saver = tf.train.Saver()
sess.run(init)

#训练模型,该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(50)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

#评估模型
#首先让我们找出那些预测正确的标签。tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
# 由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，
# 而 tf.argmax(y_,1) 代表正确的标签，我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

#计算概率,tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
#tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换，cast(x, dtype, name=None),第一个参数:待转换的数据(张量),第二个参数:目标数据类型,
# 这行代码会给我们一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。
# 例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#最后，我们计算所学习到的模型在测试数据集上面的正确率
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))


saver.save(sess,"Model/MNIST1_model.ckpt")
print("保存成功")