'''
CNN model for cifar-10
2conv + 2pool + 2lrn(局部响应归一化层) + 3fc
from 2 classes to 10 classes


In the future
AlexNet VGG GoogleNet ResNet(from scrach自己从头干)
Run on AWS
write useable code
classification > classification+localization(定位) > instances segmentation(分割)

'''
import os
import os.path
import math

import numpy as np 
import tensorflow as tf 

import cifar_10_input_data


BATCH_SIZE = 128
learning_rate = 0.05
MAX_STEP = 1000


def inference(images):
	'''
	Args:
		images: 4D tensor [batch_size, img_width, img_height, img_channel]
	Notes:
		In each conv, the kernel size is:
		[kernel_size, kernel_size, number of input channels, number of output channels]

	'''

	#conv1, [3,3,3,96], the first two dimensions are the patch size
	with tf.variable_scope('conv1') as scope:
		weights = tf.get_variable('weights',
								  shape=[3, 3, 3, 96],
								  dtype= tf.float32,
								  initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32))
		biases = tf.get_variable('biases',
								 shape=[96],
								 dtype=tf.float32,
								 initializer = tf.constant_initializer(0.0))
		conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(pre_activation, name=scope.name)

	#pool1 and norm1
	with tf.variable_scope('pooling_lrn') as scope:
		pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
								padding='SAME', name='pooling')
		norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
							beta=0.75, name='norm1')



	#conv2, [3,3,96,64], the first two dimensions are the patch size
	with tf.variable_scope('conv1') as scope:
		weights = tf.get_variable('weights',
								  shape=[3, 3, 96, 64],
								  dtype= tf.float32,
								  initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32))
		biases = tf.get_variable('biases',
								 shape=[64],
								 dtype=tf.float32,
								 initializer = tf.constant_initializer(0.1))
		conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(pre_activation, name='conv2')

	#pool2 and norm2
	with tf.variable_scope('pooling2_lrn') as scope:
		norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
							beta=0.75, name='norm2')
		pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
								padding='SAME', name='pooling2')



	#fc1
	with tf.variable_scope('local3') as scope:
		reshape = tf.reshape(pool2, shape=[BATCH_SIZE, -1])
		dim = reshape.get_shape()[1].value
		weights = tf.get_variable('weights',
								   shape=[dim,384],
								   dtype=tf.float32,
								   initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
		biases = tf.get_variable('biases',
								  shape=[384],
								  dtype=tf.float32,
								  initializer = tf.constant_initializer(0.1))
		local3 = tf.nn.relu(tf.matmul(reshape,weights) + biases, name=scope.name)

	#fc2
	with tf.variable_scope('local4') as scope:
		weights = tf.get_variable('weights',
								   shape=[384,192],
								   dtype=tf.float32,
								   initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
		biases = tf.get_variable('biases',
								  shape=[192],
								  dtype=tf.float32,
								  initializer = tf.constant_initializer(0.1))
		local4 = tf.nn.relu(tf.matmul(local3,weights) + biases, name='local4')

	# softmax
	with tf.variable_scope('softmax_linear') as scope:
		weights = tf.get_variable('softmax_linear',
								   shape=[192, 10],
								   dtype = tf.float32,
								   initializer = tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
		biases = tf.get_variable('biases',
								  shape=[10],
								  dtype = tf.float32,
								  initializer = tf.constant_initializer(0.1))
		softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
	return softmax_linear



#%% 
def losses(logits, labels):
	with tf.variable_scope('loss') as scope:
		labels = tf.cast(labels, tf.int64)
		# 不需要进行 one-hot encoding
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
						logits=logits, labels=labels, name='entropy_per_example')
		loss = tf.reduce_mean(cross_entropy, name='loss')
		tf.summary.scalar(scope.name+'/loss', loss)
	return loss


#%% Train the model on the training data
def train():
	my_global_step = tf.Variable(0, name='global_step', trainable = False)

	data_dir = 'F:\\tensorflow学习\\site-packages\\cifar_10_data\\cifar-10-batches-bin\\'
	log_dir = 'F:\\tensorflow学习\\site-packages\\cifar_10_logs\\'

	images, labels = cifar_10_input_data.read_cifar10(data_dir=data_dir,
													  is_train = True,
													  batch_size = BATCH_SIZE,
													  shuffle = True)
	logits = inference(images)

	loss = losses(logits, labels)

	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	trian_op = optimizer.minimize(loss, global_step=my_global_step)

	saver = tf.trian.Saver(tf.global_variables())
	summary_op = tf.summary.merge_all()

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	coord = tf.trian.Coordinator()
	threads = tf.trian.start_queue_runners(sess=sess, coord=coord)

	summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

	try:
		for step in np.arange(MAX_STEP):
			if coord.should_stop():
				break
			_, loss_value - sess.run([train_op, loss])

			if step % 50 ==0:
				print('step: %d, loss: %.4f'%(step, loss_value))
			if step % 100 ==0:
				summary_str = sess.run(summary_op)
				summary_writer.add_summary(summary_str, step)

			if step % 200 ==0 or (step+1)==MAX_STEP:
				checkpoint_path = os.path.join(log_dir, 'model.ckpt')
				saver.save(sess, checkpoint_path, global_step=step)

	except tf.errors.OutOfRangeError:
		print('Done training -- epoch limit reached')
	finally:
		coord.request_stop()

	coord.join(threads)
	sess.close()



#%% Test the model on the test data
def evaluate():
	with tf.Graph().as_default():
		log_dir = 'F:\\tensorflow学习\\site-packages\\cifar_10_logs\\'
		test_dir = 'F:\\tensorflow学习\\cifar_10_data\\cifar-10-batches-bin\\'
		n_test = 10000

		# reading test data
		images, labels = cifar_10_input_data.read_cifar10(data_dir=test_dir,
														  is_train=False,
														  batch_size=BATCH_SIZE,
														  shuffle=False)
		logits = inference(images)
		top_k_op = tf.nn.in_top_k(logits, labels, 1)
		saver = tf.trian.Saver(tf.global_variables())

		with tf.Session() as sess:
			print("Reading checkpoints...")
			ckpt = tf.train.get_checkpoint_state(log_dir)
			if ckpt and ckpt.model_checkpoint_path:
				global_step = ckpt.model_checkpoint_path.split('\\')[-1].split('-')[-1]
				saver.restore(sess, ckpy.model_checkpoint_path)
				print('Loading success, global_step is %s'% global_step)
			else:
				print('No checkpoint file found')
				return

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)
			try:
				num_iter = int(math.ceil(n_test / BATCH_SIZE))
				true_count = 0
				total_sample_count = num_iter * BATCH_SIZE
				step = 0

				while step < num_iter and not coord.should_stop():
					predictions = sess.run([top_k_op])
					true_count += np.sum(predictions)
					step += 1
					precision = true_count / total_sample_count
				print('precision = %.3f'% precision)
			except Exception as e:
				coord.request_stop(e)
			finally:
				coord.request_stop()
				coord.join(threads)


