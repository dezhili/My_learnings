# 一、使用ReLU作为CNN激活函数，成功解决了Sigmoid在网络较深时的梯度弥撒问题。 
# 二、训练时使用Dropout随机忽略一部分神经元，以避免模型过拟合。（AlexNet中主要在最后几个全连接层使用Dropout） 
# 三、AlexNet全部使用最大池化，避免平均池化的模糊化效果。并且AlexNet提出让步长比池化核的尺寸小，
#     这样池化层的输出之间会有重叠和覆盖，提升了特征的丰富性。 
# 四、提出LRN层，对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，
#     提升了的模型的泛化能力。 
# 五、使用CUDA加速深度卷积网络的训练。 
# 六、数据增强。随机地从256*256的原始图像中截取224*224大小的区域（以及水平翻转的镜像），
#     相当于增加了（256-224）*（256-224）*2=2048倍的数据量。数据增强后可以大大减轻过拟合，提升泛化能力。
#     进行预测时，则是取图片的四个角加中间共五个位置，并进行左右翻转，一共获得10张图片，
#     对他们进行预测并对10次结果求平均值。另外，对图像的RGB数据进行PCA处理，
#     并对主成分做一个标准差为0.1的高斯扰动（具有高斯分布的随机扰动），增加一些噪声，
#     这个Trick可以让错误率再下降1%。 
# 备注：AlexNet需要训练参数的层是卷积层和全连接层（不包括池化层和LRN）。
#         卷积层有用的地方就是可以通过较小的参数量提取有效的特征。


from datetime import datetime
import math
import time
import tensorflow as tf 

batch_size = 32
num_batches = 100

# 显示网络每一层结构的函数 print_activations, 展示每一个卷积层或池化层输出tensor的尺寸。
# Args: tensor  Return: t.op.name t.get_shape.as_list()
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


# 网络架构
def inference(images):
    parameters = []

    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64],
                                    dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                    trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]
    # lrn1
    # 考虑到LRN层效果不明显，而且会让forward和backwood的速度大大下降所以没有添加
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], 
                                    padding='VALID', name='pool1')
    print_activations(pool1)


    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192],
                                    dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                                    trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv2)

    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], 
                                    padding='VALID', name='pool2')
    print_activations(pool2)


    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                    dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                                    trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)


    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                    dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                    trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)


    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                    dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                    trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                    padding='VALID', name='pool5')
    print_activations(pool5)

    return pool5, parameters


# 评估AlexNet每轮计算时间的函数 time_tensorflow_run (session, target, info_string)
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i>= num_steps_burn_in:
            if not i%10 :
                print('%s: step %d, duration=%.3f'%(datetime.now(), i-num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration

    # 在循环结束后，计算每轮迭代的平均耗时 mm 和 标准差 sd
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec/batch'% (datetime.now(), info_string, num_batches, mn, sd))


# 主函数 
def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],
                                        dtype=tf.float32, stddev=1e-1))
        pool5, parameters = inference(images)
        
        init = tf.global_variables_initializer()
        sess =  tf.Session()
        sess.run(init) 

        # Run the forward benchmark.
        time_tensorflow_run(sess, pool5, "Forward")
        
        # Add a simple objective so we can calculate the backward pass.设置优化目标loss
        objective = tf.nn.l2_loss(pool5)
        # Compute the gradient with respect to all the parameters.根据梯度更新参数
        grad = tf.gradients(objective, parameters)
        # Run the backward benchmark.
        time_tensorflow_run(sess, grad, "Forward-backward")


run_benchmark()