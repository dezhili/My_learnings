#  省内存 placeholder才是王道

import tensorflow as tf 

def use_placeholder():
	graph = tf.Graph()
	with graph.as_default():
		value1 = tf.placeholder(dtype=tf.float64)
		value2 = tf.Variable([3,4],dtype=tf.float64)
		mul = value1*value2
		print(type(value2))
		print(type(value1))
		print(type(mul))

	with tf.Session(graph = graph) as mySess:
		init = tf.global_variables_initializer()
		mySess.run(init)

		#想象一下这个数据是从远程加载加进来，文件 网络
		#假装10GB

		value = load_from_remote()
		for partialValue in load_partial(value,2):
			runResult = mySess.run(mul,feed_dict={value1:partialValue})
			print('乘法(value1,value2)= ',runResult)


def load_from_remote():
	return [-x for x in range(1000)]


#自定义的 Iterator
#yield generator function
def load_partial(value,step):
	index = 0
	while index < len(value):
		yield value[index:index+step]
		index+=step
	return

if __name__ == '__main__':
	use_placeholder()