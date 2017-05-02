from scipy.io import loadmat as load 
import matplotlib.pyplot as plt
import numpy as np  


def reformat(samples,labels):
	#  改变原始数据的形状
	#     0      1      2       3           3       0      1      2
	#  (图片高，图片宽，通道数，图片数) --> (图片数，图片高，图片宽，通道数)
	samples  = np.transpose(samples,(3,0,1,2)).astype(np.float32)

	#  labels 变成 one-hot encoding [2]->[0,0,1,0,0,0,0,0,0,0]
	#  digit 0 代表 10  [10]->[1,0,0,0,0,0,0,0,0,0]
	labels = np.array([x[0] for x in labels])
	one_hot_labels = []
	for num in labels:
		one_hot = [0.0]*10
		if num==10:
			one_hot[0] = 1.0
		else:
			one_hot[num] = 1.0
		one_hot_labels.append(one_hot)
	labels = np.array(one_hot_labels).astype(np.float32)
	return samples,labels


def normalize(samples):
	'''
	灰度化:将三色通道 变为 单色通道   省内存+加快训练速度  (R+G+B)/3
	归一化:将图片从 0-255 变为 -1.0-1.0  loss function gd....
	@samples: numpy array
	'''
	a = np.add.reduce(samples,keepdims=True,axis=3)  #(图片数，图片高，图片宽，通道数)
	a = a/3
	return a/128 - 1


def distribution(labels,name):
	#  查看一下每个label的分布，并画个直方图 0-9
	# keys: 1,2,3,4,...,10
	count={}
	for label in labels:
		key = 0 if label[0]==10 else label[0]
		if key in count:
			count[key]+=1
		else:
			count[key] = 1

	x = []
	y = []

	for k,v in count.items():
		x.append(k)
		y.append(v)

	#直方图显示
	y_pos = np.arange(len(x))
	plt.bar(y_pos,y,align='center',alpha=0.5)
	plt.xticks(y_pos,x)
	plt.ylabel('Count')
	plt.title(name+' Label Distribution')
	plt.show()





#(32, 32, 3, 73257) --> (73257,32,32,3)
def inspect(dataset,labels,i):
	#显示样本图片
	print(labels[i])
	plt.imshow(dataset[i])
	plt.show()




traindata = load('train_32x32.mat')
testdata = load('test_32x32.mat')
# extradata = load('extra_32x32.mat')

# print('Train Data Samples Shape:',traindata['X'].shape)  #(32, 32, 3, 73257)
# print('Train Data Labels Shape:',traindata['y'].shape)

# print('Test Data Samples Shape:',testdata['X'].shape)
# print('Test Data Labels Shape:',testdata['y'].shape)

# print('Extra Data Samples Shape:',extradata['X'].shape)
# print('Extra Data Labels Shape:',extradata['y'].shape)


train_samples = traindata['X']
train_labels = traindata['y']
test_samples = testdata['X']
test_labels = testdata['y']
# print(train_labels[1])
# print(train_labels[0])
# print(train_labels[1][0])


# extra_samples = extradata['X']
# extra_labels = extradata['y']


n_train_samples, _train_labels = reformat(train_samples, train_labels)
n_test_samples, _test_labels = reformat(test_samples, test_labels)
_train_samples = normalize(n_train_samples)
_test_samples = normalize(n_test_samples)


num_labels = 10
image_size = 32
num_channels = 1

# _train_samples,_train_labels = reformat(train_samples,train_labels)
# _test_samples,_test_labels = reformat(test_samples,test_labels)

if __name__ =='__main__':

	# inspect(_train_samples,_train_labels,0)
	# _train_samples = normalize(_train_samples)
	# inspect(_train_samples,_train_labels,0)
	distribution(train_labels,'Train Labels ')
	distribution(test_labels,'Test Labels ')
