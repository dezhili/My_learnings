'''
使用文件: mnist.py  fully_connected_feed.py

准备数据:
data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)
data_sets.train - 55000
data_sets.validation - 5000
data_sets.test  -10000
输入与占位符: placeholder_inputs()





构建图表
在为数据创建占位符之后，就可以运行 mnist.py 文件，经过三阶段的模式函数操作： 
    inference() ， loss() ，和 training() 。图表就构建完成了。

推理(inference): inference() 函数会尽可能地构建图表，做到返回包含了预测结果（output prediction）的Tensor
    logits = tf.matmul(hidden2, weights) + biases   返回包含了输出结果的 logits Tensor

损失(loss): loss() 函数通过添加所需的损失操作，进一步构建图表
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean') 返回包含了损失值的Tensor

训练(training): 添加了通过梯度下降（gradient descent）将损失最小化所需的操作
    tf.summary.scalar('loss', loss) 后者在与 SummaryWriter 配合使用时，
            可以向事件文件（events file）中生成汇总值（summary values）。
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False) 
    train_op = optimizer.minimize(loss, global_step=global_step) 
            返回包含了训练操作（training op）输出结果的Tensor





训练模型:
图表:
    with tf.Graph().as_default(): 表明所有已经构建的操作都要与默认的 tf.Graph 全局实例关联起来
    
会话:
    with tf.Session() as sess:
        init = tf.global_variable_initializer()
        sess.run(init)

训练循环:
    for step in xrange(max_steps): 
        sess.run(train_op)

    向图表提供反馈:
        feed_dict = fill_feed_dict(data_sets.train,
                                 images_placeholder,
                                 labels_placeholder)

    检查状态:
        _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)
        if step % 100 == 0:
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

    状态可视化(tensorboard):   
        为了释放TensorBoard所使用的事件文件（events file），所有的即时数据（在这里只有一个）
        都要在图表构建阶段合并至一个操作（op）中
            summary = tf.summary.merge_all()
        在创建好会话（session）之后，可以实例化一个 tf.summary.FileWriter ，用于写入包含了
        图表本身和即时数据具体值的事件文件
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        最后，每次运行 summary 时，都会往事件文件中写入最新的即时数据，函数的输出会传入事件文件
        读写器（writer）的 add_summary() 函数
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)

    保存检查点(checkpoint):
        为了得到可以用来后续恢复模型以进一步训练或评估的检查点文件（checkpoint file），
        我们实例化一个 tf.train.Saver 。
            saver = tf.train.Saver()
            checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=step)
        我们以后就可以使用 saver.restore() 方法，重载模型的参数，继续训练。
            saver.restore(sess, FLAGS.train_dir)






评估模型
     do_eval 函数会被调用三次，分别使用训练数据集、验证数据集合测试数据集。

'''