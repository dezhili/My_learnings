#  Introduction:

'''
Tensorflow :
graph : 使用图来表示计算任务
Session: 在被称之为会话的上下文(context)中执行图, 为了进行计算，图必须在会话里被启动。
         会话将图的 op 分发到诸如CPU GPU之类的设备，同时提供执行op的方法。
tensor: 使用tensor 表示数据, 每个Tensor是一个类型化的多维数组，numpy ndarray对象
        图像集 表示为一个四维浮点数数组，这四个维度分别是[batch, height, width, channels]
Variable : 通过变量 维护状态
op: Tensorflow 是一个编程系统，使用图来表示计算任务， 图中的节点被称为op(operation)
    一个op 获得0 或 多个 Tensor ，执行计算，产生0个或多个Tensor 

'''

import tensorflow as tf 
import numpy as np 

'''
intro_exm: 生成一些三维数据，然后用一个平面拟合它
'''
# x_data = np.float32(np.random.rand(2, 100))
# y_data = np.dot([0.100, 0.200], x_data) + 0.300

# W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
# b = tf.Variable(tf.zeros([1]))
# y = tf.matmul(W, x_data) + b

# loss = tf.reduce_mean(tf.square(y - y_data))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)

# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)

# for step in range(200):
#     sess.run(train)
#     if step % 10 == 0:
#         print(step, sess.run(loss), sess.run(W), sess.run(b))


'''
1. 计算图:

Tensorflow 程序通常被组织成一个构建阶段，和一个执行阶段。在构建阶段，op的执行步骤被描述成
    一个图。 在执行阶段，使用会话执行执行图中的op

构建图: 创建源 op，源op不需要任何输入，例如 常量(Constant)
       Tensorflow Python库有一个默认图(default graph), op构造器可以为其增加节点
在一个会话中启动图: 构造阶段完成后， 才能启动图。启动图的第一步是创建一个Session对象，
        如果无任何创建参数，会话构造器将启动默认图
'''
matrix1 = tf.constant([[3., 3.]]) # 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点, 加到默认图中.
matrix2 = tf.constant([[2.], [2.]]) # 创建另外一个常量 op, 产生一个 2x1 矩阵.
product = tf.matmul(matrix1, matrix2) # 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入

# 启动默认图
with tf.Session() as sess:
    result = sess.run(product)# 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的
    print(result) # 返回值 'result' 是一个 numpy `ndarray` 对象. [[12.]]


'''
2. Tensor
TensorFlow 程序使用 tensor 数据结构来代表所有的数据, 计算图中, 操作间传递的数据都是 tensor. 你可以
把 TensorFlow tensor 看作是一个 n 维的数组或列表. 

张量的阶、形状、数据类型:
阶:
    在TensorFlow系统中，张量的维数来被描述为阶.一个二阶张量就是我们平常所说的矩阵，一阶张量可以认为是一个向量.
    对于一个二阶张量你可以用语句t[i, j]来访问其中的任何元素.而对于三阶张量你可以用't[i, j, k]'来访问其中的任何元素.
    v = [1.1, 2.2, 3.3]- 一阶   m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]- 二阶
    t = [[[2], [4], [6]], [[8], [10], [12]], [[14], [16], [18]]] -三阶张量
形状:
    [D0] 一个1维张量的形式[5]   [D0, D1] 一个2维张量的形式[3, 4] 
数据类型:
    除了维度，Tensors有一个数据类型属性.你可以为一个张量指定下列数据类型中的任意一个类型
    tf.float32  tf.float64 tf.int64 tf.int32 tf.string  tf.bool
'''



'''
3. 变量: 创建、初始化、保存和加载 tf.Variable类 tf.train.Saver类
    当训练模型时， 用变量来存储和更新参数。 变量包含张量(Tensor)存放于内存的缓冲区
    建模时它们需要被明确地初始化，模型训练后它们必须被存储到磁盘。
    这些变量的值可在之后模型训练和分析是被加载。

创建:
    当创建一个变量时，你将一个张量作为初始值传入构造函数Variable()。
    TensorFlow提供了一系列操作符来初始化张量，初始值是常量或是随机值。
    weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name='weights')
    biases = tf.Variable(tf.zeros([200]), name='biases')

初始化:
    变量的初始化必须在模型的其它操作运行之前先明确地完成。
    最简单的方法就是添加一个给所有变量初始化的操作，并在使用模型之前首先运行那个操作
    tf.global_variables_initializer() 对变量做初始化。在完全构建好模型并加载之后再运行

保存和加载
    tf.train.Saver 对象。 构造器给graph的所有变量，或是定义在列表里的变量，添加save和restore ops
    saver对象提供了方法来运行这些ops，定义检查点文件的读写路径
保存变量:
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    # Later, launch the model, initialize the variables, do some work, save the
    # variables to disk.
    with tf.Session() as sess:
        sess.run(init_op)
        # Do some work with the model.

        # Save the variables to disk.
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print "Model saved in file: ", save_path
恢复变量:
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, "/tmp/model.ckpt")
        print "Model restored."
        # Do some work with the model
'''
state = tf.Variable(0, name='counter')
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


'''
4. Fetch:
    为了取回操作的输出内容, 可以在使用 Session 对象的 run() 调用 执行图时, 传入一些 tensor, 
    这些 tensor 会帮助你取回结果.
'''
input1 = tf.constant(3.0) 
input2 = tf.constant(2.0) 
input3 = tf.constant(5.0) 
intermed = tf.add(input2, input3) 
mul = tf.mul(input1, intermed)
with tf.Session() as sess: 
    result = sess.run([mul, intermed]) 
    print (result)



'''
Feed:
'''
input1 = tf.placeholder(dtype = tf.float32) 
input2 = tf.placeholder(dtype = tf.float32) 
output = tf.mul(input1, input2)
with tf.Session() as sess:
    result = sess.run(output, feed_dict={input1:[7.], input2:[2.]})
    print(result)
