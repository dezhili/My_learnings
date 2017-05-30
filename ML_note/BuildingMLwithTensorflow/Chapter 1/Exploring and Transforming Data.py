# Data flow graph   Graph  Operations  Tensors
# Multidimensional data arrays

# Tensor--
# a tensor is just a typed, multidimensional array,with additional operations,
#   modeled in the tensor object
# --Tensor properties - ranks,shapes,and types

import tensorflow as tf 
# tens1 = tf.constant([[[1,2],[2,3]],[[3,4], [5,6]]])
# sess = tf.InteractiveSession()
# print(sess.run(tens1)[1, 1, 0])
# print(sess.run(tens1).shape)

# --Creating new tensors

import numpy as np 
# x = tf.constant(np.random.rand(32).astype(np.float32))
# y = tf.constant([1, 2, 3])
# print(sess.run(x).dtype)
# print(sess.run(y))

# print(x.eval())

# x_data = np.array([[1., 2., 3.], 
#                    [3., 4., 5.]])
# x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
# print(sess.run(x_data))




#Variables
# b = tf.Variable(tf.zeros([1000]))

# Saving data flow graphs
# Data flow graphs are written using Google's protocol buffers(for serializing structured data)
# g = tf.Graph()
# with g.as_default():
#     sess = tf.Session()
#     W_m = tf.Variable(tf.zeros([10, 5]))
#     x_v = tf.placeholder(tf.float32, [None, 10])
#     result = tf.matmul(x_v, W_m)
#     print(g.as_graph_def())


# Running our programs - Sessions


# Basic tensor methods
# Simple matrix operations
# Reduction(an operation that applies an operation across one of the tensor's dimensions)
sess = tf.InteractiveSession()
# x = tf.constant([[1, 2, 3],
#                  [3, 2, 1],
#                  [-1, -2, -3]])
# print(tf.reduce_prod(x, reduction_indices=1).eval())
# print(tf.reduce_mean(x, reduction_indices=0).eval())



# Tensor segmentation
# seg_ids = tf.constant([0, 1, 1, 2, 2])
# tens1 = tf.constant([[1, 2, 3, 4],
#                      [2, 3, 4, 5],
#                      [3, 4, 5, 6],
#                      [4, 5, 6, 7],
#                      [5, 6, 7, 8]])
# print(tf.segment_sum(tens1, seg_ids).eval())


# Sequences
# x = tf.constant([[2, 5, 4, -5],
#                  [3, 4, 0, 6],
#                  [4, 3, 6, 3],
#                  [1, 6, 7, 4]])
# print(x.eval())
# listx = tf.constant([1, 2, 3, 4, 5, 6, 7, 8])
# listy = tf.constant([4, 5, 8, 9])

# boolx = tf.constant([[True,False], [False,True]])
# print(tf.argmin(x, axis=1).eval())

# print(tf.argmax(x, 0).eval())
# print(tf.setdiff1d(listx, listy)[0].eval())
# print(tf.where(boolx).eval())
# print(tf.unique(listx)[0].eval())



# # Tensor shape transformations
# print(tf.expand_dims(x, 1).eval())
# print(tf.size(x).eval())
# print(tf.rank(x).eval())


# Tensor slicing and joining
# print(tf.slice(x, [1, 1], [2, 2]).eval())
# print(tf.tile([1, 2], [3]).eval())
# print(tf.pad(x, [[0, 1], [2, 1]]).eval())
# t_array =tf.constant([1, 2, 3, 4, 9, 8, 7, 6])
# t_array2=tf.constant([2, 3, 4, 5, 6, 7, 8, 9])
# print(tf.concat([t_array, t_array2], axis=0).eval())
# print(tf.stack([t_array, t_array2]).eval())




'''
Dataflow structure and results visualization - Tensorboard

How TensorBoard works--
    All the tensors and ops of a graph can be set to write information to logs
    Every computation graph we build, Tensorflow has a real-time logging mechanism for,
      in order to save almost all the information that a model possesses. 

    To save all the required informatin, Tensorflow API uses data output objects,
      called Summaries. 
    These Summaries write results into Tensorflow event files,which gather all the 
    required data generated during a Session's run. 

Adding Summary nodes
tf.summary.FileWriter()--The command will create a SummaryWriter and an event file,in the
  path of the parameter. This event file will contain Event type protocol buffers constructed
  when you call one of the following functions: add_summary() 

'''

'''
First, create the Tensorflow graph that you'd like to collect summary data from and decide
    which nodes you would like to annotate with summary operations. To generate summaries, we
    need to run all of these summary nodes.Managing them manually would be tedious,so use 
    tf.summary.merge_all() to combine them into a single op that generates all the summary data. 
Then, you can just run the merged summary op,which will generate a serialized Summary protobuf 
    object with all of your summary data at a given step.
Finally, to write this summary data to disk, pass the Summary protobuf to a tf.summary.FileWriter()

Now that you've modified your graph and have a SummaryWriter,you're ready to start running your
    network! if you want, you could run the merged summary op every single step, and record a ton 
    of training data. Instead, consider running the merged summary op every n steps
'''





'''
Reading information from disk

Tabulated formats -CSV
    First,we must create a filename queue object with the list of files we'll be using,
    then create a TextLineReader, With this line reader, the remaining operation will be 
    to decode the CSV columns, and save it on tensors. If we want to mix homogeneous data together,
    the stack method will work
'''
# import tensorflow as tf 
# sess = tf.Session()
# filename_queue = tf.train.string_input_producer(['./iris.csv'], shuffle=True)
# reader = tf.TextLineReader(skip_header_lines=1)
# key, value = reader.read(filename_queue)
# record_defaults = [[0.], [0.], [0.], [0.], [""]]
# # convert csv records to tensors. Each column maps to one tensor
# col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)
# features = tf.stack([col1, col2, col3, col4])

# sess.run(tf.global_variables_initializer())
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(coord=coord, sess=sess)

# for iteration in range(0, 5):
#     example = sess.run([features])
#     print(example)
#     coord.request_stop()
#     coord.join(threads)



'''
Reading image data
The accepted image formats will be JPG and PNG and the internal representation will
be uint8 tensors, one rank two tensor for each image channel
'''

# import tensorflow as tf
# sess = tf.Session()

# filename_queue = tf.train.string_input_producer(['./timg.jpg'])
# reader = tf.WholeFileReader()
# key, value = reader.read(filename_queue)
# image = tf.image.decode_jpeg(value)
# flipImageUpDown = tf.image.encode_jpeg(tf.image.flip_up_down(image))
# flipImageLeftRight = tf.image.encode_jpeg(tf.image.flip_left_right(image))

# sess.run(tf.global_variables_initializer())
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(coord=coord, sess = sess)
# example = sess.run(flipImageLeftRight)
# print(example)

# file = open('flippedUpDown.jpg', 'wb+')
# file.write(sess.run(flipImageUpDown))
# file.close()

# file = open('flippedLeftRight.jpg', 'wb+')
# file.write(sess.run(flipImageLeftRight))
# file.close()


'''
Reading from the standard Tensorflow format
    You can write a little program that gets your data, stuffs it in an example protocol buffer,
    serializes the protocol buffer to a string,and then writes the string to a TFRecords file 
    using tf.python_io.TFRecordWriter class

To read a file of TFRecords, use tf.TFRecordReader with the tf.parse_single_example
    decoder(decodes the example protocol buffers into tensors)
'''


