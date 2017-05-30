'''
作为课程介绍的第一个方法，我们来实现一个Nearest Neighbor分类器。
虽然这个分类器和卷积神经网络没有任何关系，实际中也极少使用，但通过实现它，
可以让读者对于解决图像分类问题的方法有个基本的认识。

CIFAR-10  60000(32x32)==10 classes  --> 50000(train) + 10000(test)
'''

'''
Nearest Neighbor :
那么具体如何比较两张图片呢？在本例中，就是比较32x32x3的像素块。
最简单的方法就是逐个像素比较，最后将差异值全部加起来。换句话说，
就是将两张图片先转化为两个向量I_1和I_2，然后计算他们的L1距离： 求和


以图片中的一个颜色通道为例来进行说明。两张图片使用L1距离来进行比较。
逐个像素求差值，然后将所有差值加起来得到一个数值。如果两张图片一模一样，
那么L1距离为0，但是如果两张图片很是不同，那L1值将会非常大。
'''


'''
首先，我们将CIFAR-10的数据加载到内存中，并分成4个数组：训练数据和标签，测试数据和标签。
在下面的代码中，Xtr（大小是50000x32x32x3）存有训练集中所有的图像，
Ytr是对应的长度为50000的1维数组，存有图像对应的分类标签（从0到9）：
'''

import numpy as np 

def load_CIFAR10(path):


Xtr, Ytr, Xte, Yte = load_CIFAR10('F:\\tensorflow学习\\site-packages\\cifar_10_data\\cifar-10-batches-bin')

print('Xte: %d' %(len(Xte)))
print('Yte: %d' %(len(Yte)))
Xtr = np.asarray(Xtr)
Ytr = np.asarray(Ytr)
Xte = np.asarray(Xte)
Yte = np.asarray(Yte)

# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32*32*3)  # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32*32*3)  # Xte_rows becomes 10000 x 3072


nn = NearestNeighbor()  # Create a Nearest Neighbor classfier class
nn.train(Xtr_rows, Ytr)

Yte_predict = nn.predict(Xte_rows)
print('accuracy: %f'%(np.mean(Yte_predict == Yte)))


'''
下面就是使用L1 距离的Nearest Neighbor 分类器
'''
class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        ''' X is NxD where each row is an example. Y is 1-dim of size N '''
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]

        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in xrange(num_test):
            distances = np.sum(np.abs(self.Xtr-X[i, :]), axis=1)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]

        return Ypred




