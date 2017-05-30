'''
变量: 创建、初始化、保存和加载
当你在训练模型时，用变量来存储和更新参数。变量是内存中用于存放张量(tensor)的缓存区。它们必须被明确的
初始化，并能在训练过程中和结束后被保存到磁盘上。随后可以恢复保存的值来训练和分析模型
tf.Variable  tf.train.Saver
'''


#创建

# 当创建一个变量时，将一个张量作为初始值传入Variable()构造函数，TensorFlow提供了一系列操作符来初始
#  化张量初始值是常量或随机值
#所有的这些操作符都需要指定张量的shape。变量的shape通常是固定的，但有高级的机制来重新调整其行列数

import tensorflow as tf
# Create two variables(调用tf.Variable()添加一些操作Op 到graph)
# 一个 Variable 操作存放变量的值。 一个初始化op将变量设置为初始值。这事实上是一个 tf.assign 操作.

weights = tf.Variable(tf.random_normal([700, 200], stddev=0.35), name='weights')
biases = tf.Variable(tf.zeros([200]), name='biases')  # 返回值是python的tf.Variable类的一个实例





# 初始化(在完全构建好模型并加载之后再运行)

# 添加一个给所有变量初始化的操作，并在使用模型之前首先运行那个操作
# 或可以从检查点文件中重新获取变量值

# Add an op to initialize the variables
init_op = tf.global_variables_initializer()

# Later, when launching the model
with tf.Session() as sess:
    sess.run(init_op)
    '''
    Use the model
    '''
# 由另一个变量初始化
# 用其它变量的值初始化一个新的变量时，使用其它变量的 initialized_value() 属性。
# 你可以直接把已初始化的值作为新变量的初始值，或者把它当做tensor计算得到一个值赋予新变量。
w2 = tf.Variable(weights.initialized_value(), name='w2')
w_twice = tf.Variable(weights.initialized_value()*0.2, name='w_twice')



# 保存和加载

# 最简单的保存和恢复模型的方法是使用 tf.train.Saver对象。构造器给graph的所有变量，
# 或是定义在列表内的变量，添加 save和restore ops，saver对象提供方法来运行这些ops，
# 定义checkpoint文件的读写文件。

# checkpoint文件: 变量存储在二进制文件里，主要包含从变量名到tensor值的映射关系。
#                 当你创建一个Saver对象时，可以选择性的为checkpoint文件中的变量挑选
#                 变量名。默认情况下，赋以将每个变量Variable.name属性的值

# 保存变量
# tf.train.Saver() 传建一个Saver来管理模型中的所有变量
v1 = tf.Variable(..., name='v1')
v2 = tf.Variable(..., name='v2')

init_op = tf.global_variables_initializer()    # Add an op to initialize the variables.
saver = tf.train.Saver()                       # Add ops to save and restore all the variables.

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  ..
  # Save the variables to disk.
  save_path = saver.save(sess, './model.ckpt')
  print("Model saved in file: ", save_path)

# 恢复变量(用同一个Saver对象来恢复变量。 当从文件中恢复变量时，不需要事先做初始化)
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print "Model restored."
  # Do some work with the model
  ...




# 选择存储和恢复哪些变量

# 有时候在检查点文件中明确定义变量的名称很有用。举个例子，
# 你也许已经训练得到了一个模型，其中有个变量命名为"weights"，你想把它的值恢复到一个新的变量"params"中。

# 有时候仅保存和恢复模型的一部分变量很有用。再举个例子，你也许训练得到了一个5层神经网络，
# 

# 可以通过给tf.train.Saver()构造函数传入Python字典，很容易地定义需要保持的变量及对应名称：
# 键对应使用的名称，值对应被管理的变量。

# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to save and restore only 'v2' using the name "my_v2"
saver = tf.train.Saver({"my_v2": v2})
# Use the saver object normally after that.
...