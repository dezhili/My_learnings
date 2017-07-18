"""Builds the MNIST network.
Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
import math
import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10
# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
# batch_size = 50
# hidden1_units = 20
# hidden2_units = 15
# learning_rate = 0.1

# images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
# labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))


# 构建学习器模型的前向预测过程(从输入到输出的计算图路径)
def inference(images, hidden1_units, hidden2_units):
    """
    Args:
        images: Images placeholder, from inputs().
        hidden1_units: Size of the first hidden layer.
        hidden2_units: Size of the second hidden layer.

    Returns:
        softmax_linear: Output tensor with the computed logits.
    """
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))), name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))), name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))), name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits

# 根据logits 和 labels计算输出层 loss
def loss(logits, labels):
    """
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].

    Returns:
        loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

# 为损失模型添加训练节点(需要产生和应用梯度的节点)
def training(loss, learning_rate):
    """
      Creates a summarizer to track the loss over time in TensorBoard.
      Creates an optimizer and applies the gradients to all trainable variables.
      The Op returned by this function is what must be passed to the
      `sess.run()` call to cause the model to train.

      Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.

      Returns:
        train_op: The Op for training.
    """
    with tf.name_scope('scalar_summaries'):
        # 为保存loss的值添加一个标量汇总(scalar summary)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
    # 根据给定的学习率创建梯度下降优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # 创建一个变量来跟踪global step
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

# 评估模型输出的logits在预测类标签方面的质量
def evaluation(logits, labels):
    """
      Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the range [0, NUM_CLASSES).

      Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


# logits = inference(images=images_placeholder, hidden1_units=hidden1_units, hidden2_units=hidden2_units)
# batch_loss = loss(logits=logits, labels=labels_placeholder)
# train_on_batch = training(loss=batch_loss, learning_rate=learning_rate)
# correct_counts = evaluation(logits=logits, labels=labels_placeholder)

# 将模型写入计算图
# writer = tf.summary.FileWriter('F:\Anaconda\logs', tf.get_default_graph())
# writer.close()