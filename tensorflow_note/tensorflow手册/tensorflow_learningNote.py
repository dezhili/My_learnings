'''
Reading data
3 main methods of getting data into a TensorFlow program
Feeding: placeholder  , provides the data when running each step
Reading from files: an input pipeline reads the data from files
Preloaded data: a constant or variable in the Tensorflow graph holds all the data
'''

Reading from files:
# a typical pipeline for reading records from files
1. The list of filenames
2. (Optional) filename shuffle
3. (Optional) epoch limit
4. Filename queue
5. A Reader for the file format
6. A decoder for a record read by the Reader
7. (Optional) preprocessing
8. Example queue

tf.train.match_filenames_once()
tf.train.string_input_producer() # creates a FIFO queue for holding the filenames util the reader needs them
                                 # also has options for shuffling and setting a maximum number of epochs
# A queue runner adds the whole list of filenames to the queue once for each epoch
# The queue runner works in a thread separate from the reader that pulls filenames from the queue, 
# so the shuffling and enqueuing process does not block the reader

File formats:
# select the reader that matches your input file format and pass the filename queue to the reader's read  method

1. CSV files(tf.TextLineReader() tf.decode_csv())
filename_queue = tf.train.string_input_producer(['file0.csv', 'file1.csv'])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.concat(0, [col1, col2, col3, col4])

with tf.Session() as sess:
    # start populating the filename queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(1200):
        # Retrieve a single instance
        example, label = sess.run([features, col5])
    coord.request_stop()
    coord.join(threads)

2. Fixed length records(tf.FixedLengthRecordReader() tf.decode_raw())
# to read binary files in which each record is a fixed number of bytes


3. Standard TensorFlow format (a TFRecords file)(tf.python_io.TFRecordWriter)
# write a little program that gets your data, stuffs it in an Example protocol buffer,
# serializes the protocol buffer to a string, and then writes the string to a TFRecords file

# To read a file of TFRecords , tf.TFRecordReader tf.parse_single_example(decoder)


4. preprocessing(normalization , picking a random slice, adding noise, distortions)

5. Batching(tf.train.shuffle_batch()) :
# At the end of the pipeline we use another queue to batch together examples for training,
# evaluation, or inference. For this we use a queue that randomizes the order of examples

def read_my_file_format(filename_queue):
    reader = tf.SomeReader()
    key, record_string = reader.read(filename_queue)
    example, label = tf.some_decoder(record_string)
    processed_example = some_processing(example)
    return processed_example, label
def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    example, label = read_my_file_format(filename_queue)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch

Creating threads to prefetch using QueueRunner objects
# many of the tf.train() listed above add QueueRunner to your graph. These require 
# you call tf.train.start_queue_runners() before running any training or inference steps
# This will start threads that run the input pipeline , fulling the example queue so that
# the dequeue to get the examples will succeed.will
# tf.train.Coordinator() to cleanly shut down these threads 

# start input enqueue threads
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
try:
    while not coord.should_stop():
        sess.run(train_op)
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # when done , ask the threads to stop
    coord.request_stop()
# wait for threads to finish
coord.join(threads)
sess.close()



