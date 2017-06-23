'''
Convolutional neural networks are part of many of the most advanced models currently being
employed. 
1. Getting an idea of how convolution functions and convolutional networks work and the 
   main operation types used in building them
2. Applying convolution operations to image data and learning some of the preprocessing 
   techniques applied to images to improve the accuracy of the methods
3. Classifying digits of the MNIST dataset using a simple setup of CNN
4. Classifying real images of the CIFAR dataset, with a CNN model applied to color images

'''


'''
Getting started with convolution
Flip the signal  Shift it  Multiply it  Integrate the resulting curve


Kernels and convolutions 
When applying the concept of convolution in the discrete domain,kernels are used quite
frequently .
The convolution operation consists of multilying the corresponding pixels with the kernel,
one pixel at a time, and summing the values for the purpose of assigning that value to the 
central pixel.



Interpretation of the convolution operations
The convolution kernels highlight or hide patterns. Depending on the trained(or in the
example, manually set)parameters, we can begin to discover parameters,such as orientation
and edges in different dimensions. We may also cover some unwanted details or outliers by 



Applying convolution in Tensorflow
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu, data_format, name=None)
input:[batch, in_height, in_width, in_channels]
filter:[filter_height, filter_width, in_channels, out_channels]
strides ...



Subsampling operation - pooling
The idea is to apply a kernel(of varying dimensions) and extract one of the elements covered
by the kernel , the max_pool and avg_pool being a few of the most well known

Properties of subsampling layers
to reduce the quantity and complexity of information while retaining the most important information
elements. They build a compact representation of the underlying information

Invariance property
By sliding the filter across the image,we translate the detected features to more significant 
image parts,eventually reaching a 1-pixel image, with the feature represented by that pixel value.
Conversely, this property could also produce the model to lose the locality of feature detection.

Subsampling layers implementation performance 

Applying pool operations in TensorFlow
tf.nn.max_pool(value, ksize, strides, padding, data_format, name)
value: [batch length, height, width, channels]
ksize: strides data_format ordering, padding




Improving efficiency - dropout operation
This operation reduces the value of some randomly selected weights to zero,making null the subsequent layers

Applying the dropout operaiton in TTensorlfow
tf.nn.dropout(x, keep_prob, noise_shape, seed, name)

'''