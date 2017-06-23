'''
Sample code - applying convolution to a grayscale image
In this sample code, we will read a grayscale image in the GIF format, which will generate
a three-channel tensor but with the same RGB values per pixel. We will then transform the tensor
into a real grayscale matrix, apply a kernel, and retrieve the results in an output image in the JPEG format

'''

import tensorflow as tf 

# Generate the filename queue, and read the gif files contents
filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once('./data/test.gif'))
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
image = tf.image.decode_gif(value)

# Define the kernel parameters
kernel = tf.constant([
                      [[[-1.]],[[-1.]],[[-1.]]],
                      [[[-1.]],[[8.]],[[-1.]]],
                      [[[-1.]],[[-1.]],[[-1.]]]
                     ])

# Define the train coordinator
coord = tf.train.Coordinator()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
threads = tf.train.start_queue_runners(coord=coord)

# Get first image
image_tensor = tf.image.rgb_to_grayscale(sess.run([image])[0])
# apply convolution, preserving the image size
imagen_convoluted_tensor = tf.nn.conv2d(tf.cast(image_tensor, tf.float32),kernel, [1,1,1,1], 'SAME')
# Prepare to save the convolution option
file=open('blur2.png', 'wb+')
#Cast to uint8 (0..255), previous scalation, because the convolution could alter the scale of the final image
out=tf.image.encode_png(tf.reshape(tf.cast(imagen_convoluted_tensor/tf.reduce_max(imagen_convoluted_tensor)*255.,tf.uint8), tf.shape(imagen_convoluted_tensor.eval()[0]).eval()))
file.write(out.eval())
file.close()
coord.request_stop()
coord.join(threads)
    


