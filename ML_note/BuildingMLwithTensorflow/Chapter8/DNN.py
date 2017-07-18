'''
Deep Neural Networks
Deep network architectures through time
'''

'''
LeNet5 5 conv1 pool1 conv2 pool2 fc1 fc2 softmax

AlexNet 8 11x11conv,(48+48),/4,pool/2  -> 5x5conv,(128+128),pool/2 -> 3x3conv,(192+192)
         ->3x3conv,(192+192) ->3x3conv,(128+128),pool/2 -> fc,4096 ->fc,4096 -> fc,1000
        Data Augmentation  Dropout  Relu Local Response Normalization  Overlapping pooling 多GPU并行

VGG -19  3x3conv, 64 -> 3x3conv,64,pool/2 -> 3x3conv,128 -> 3x3conv,128,pool/2 -> 3x3conv,256
        ->3x3 conv,256 ->...

GoogLeNet - 22  Inception, Network in Network 

ResNet - 152  残差网络，这个网络的提出本质上还是要解决层次比较深的时候无法训练的问题。
借鉴了Highway Network思想的网络相当于旁边专门开个通道使得输入可以直达输出，而优化的目标由原来的
拟合输出H(x)变成输出和输入的差H(x)-x, 其中H(x)是某一层原始的期望映射输出， x是输入
'''