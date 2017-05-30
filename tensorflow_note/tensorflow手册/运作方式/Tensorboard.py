'''
数据序列化
tensorboard 通过读取tensorflow的events文件来运行。 事件文件包括会在tensorflow运行中涉及到的主要数据
下面是tensorb中 Summary data 的大体生命周期:
    首先，创建想汇总数据的图， 再选择想在哪个节点进行 summary 操作
        比如，假设正在训练一个卷及神经网络。你可能希望记录learning rate的变化，以及目标函数的变化。
        通过向节点 附加 summary.scalar 操作来分别输出学习速度和期望误差
        或希望显示一个特殊层激活的分布，或梯度权重的分布。附加 summary.histogram 收集权重变量和梯度输出

    我们刚才创建的这些节点(summary nodes)都围绕着图像，没有任何操作依赖于它们的结果。因此，为了生成
        汇总信息，我们需要运行所有节点。使用tf.summary.merge_all()合并为一个操作  --> 将所有数据生成一个
        序列化的 Summary protobuf 对象。 
    最后，为了将汇总数据写入磁盘，需要将汇总的protobuf对象 传递给 tf.train.Summarywriter

'''
import tensorflow as tf 

merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./logs', sess.graph)

total_step = 0
while training:
    total_step += 1
    sess.run(training_op)
    if total_step % 100 ==0:
        summary_str = sess.run(merged_summary_op)
        summary_writer.add_summary(summary_str, total_step)


# tensorboard --logdir=/path/to/log-directory



'''
图表可视化
name_scope node
'''
with tf.name_scope('hidden')as scope:
    a = tf.constant(5, name='alpha')    #hidden/alpha
    W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights') #hidden/weights
    b = tf.Variable(tf.zeros([1]), name='biases')  #hidden/biases