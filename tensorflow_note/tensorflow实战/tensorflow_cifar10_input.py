import tensorflow as tf 
import os
import tensorflow.python.platform
from tensorflow.python.platform import gfile

IMAGE_SIZE = 24
NUM_CLASSES = 10
NUM_TRAINING = 50000
NUM_TESTING = 10000

def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    label_bytes = 1
    result.HEIGHT = 32
    result.WIDTH = 32
    result.DEPTH = 3
    image_bytes = result.HEIGHT * resutl.WIDTH * result.DEPTH
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLenthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.unit8)
    result.label = tf.cast(tf.slice(record_bytes,[0],[label_bytes]), tf.int32)

    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.slice(record_bytes,[label_bytes],[image_bytes]), [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.unit8image = tf.transpose(depth_major, [1, 2, 0])
    return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):
    """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
    num_preprocess_threads = 16
    images, label_batch = tf.train.shuffle_batch([image, label],
                                                batch_size = batch_size,
                                                num_threads = num_preprocess_threads,
                                                capacity = min_queue_examples + 3 * batch,
                                                min_after_dequeue = min_queue_examples)
    tf.summary.image('images', images)
    return images, tf.reshape(label_batch, [batch_size])


# data augmentation + input
def distorted_inputs(data_dir, batch_size):
    '''
    Args:
        data_dir: Path to the CIFAR-10 data directory.
        batch_size: Number of images per batch.
    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    '''
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1,6)]
    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file :' + f)
    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.unit8image, tf.float32)

    # Image processing(data augmentation)
    Height = IMAGE_SIZE
    Width = IMAGE_SIZE
    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.image.random_crop(reshaped_image, [height, width])
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                   max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                                 lower=0.2, upper=1.8)
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_TRAINING * min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
             'This will take a few minutes.' % min_queue_examples)
    return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size)




def inputs(eval_data, data_dir, batch_size):
    """Construct input for CIFAR evaluation using the Reader ops.
    Args:
        eval_data: bool, indicating if one should use the train or eval data set.
        data_dir: Path to the CIFAR-10 data directory.
        batch_size: Number of images per batch.
    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    for f in filenames:
        if not gfile.Exists(f):
          raise ValueError('Failed to find file: ' + f)
    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)
    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                             width, height)
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(resized_image)
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                               min_fraction_of_examples_in_queue)
    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                             min_queue_examples, batch_size)

