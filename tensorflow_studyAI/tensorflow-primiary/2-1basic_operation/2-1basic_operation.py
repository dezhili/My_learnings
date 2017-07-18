import tensorflow as tf 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 使用Tensorflow输出Hello

# 创建一个常量操作(Constant op)
# 这个op 会被作为一个节点(node)添加到默认计算图上
# 该构造函数返回的值就是常量节点(Constant op)的输出
hello = tf.constant('Hello, Tensorflow!')

# 启动Tensorflow会话
sess = tf.Session()

# 运行hello节点
print(sess.run(hello))

# 基本常量操作
# 该构造函数返回的值就是常量节点(Constant op)的输出
a = tf.constant(2)
b = tf.constant(3)

# 启动默认的计算图
with tf.Session() as sess:
    print("a=2, b=3")
    print("常量节点相加: %i"% sess.run(a+b))
    print("常量节点相乘: %i"% sess.run(a*b))


# 使用变量(variable)作为计算图的输入
# tf Graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print("变量相加: %i"% sess.run(add, feed_dict={a:2, b:3}))

# 矩阵相乘
# 创建一个 Constant op, 产生 1x2 matrix
# 该op会作为一个节点被加入到默认的计算图
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])

# 创建一个 Matmul op 以 "matrix1", 'matrix2'作为输入
product = tf.matmul(matrix1, matrix2)

# 'product'表达了 matmul op 的输出。 这表明我们想要 fetch back matmul op 的输出
# op 需要的所有输入都会由session自动运行，某些过程可以自动并行执行
# 调用'run(product)' 就会引起计算图上三个节点的执行: 2个constant 和一个 matmul
# 'product'op 的输出会返回到'result': 一个numpy 'ndarray' 对象
with tf.Session() as sess:
    result = sess.run(product)
    print("矩阵相乘的结果: ", result)

# 保存计算图
writer = tf.summary.FileWriter(logdir='logs', graph=tf.get_default_graph())
writer.flush() 

