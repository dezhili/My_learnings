'''
制作自己的TFRecord数据集 读取 显示 
tf.python_io.TFRecordWriter()
tf.train.Example()    tf.train.Features()   tf.trian.Feature() 
'''
import tensorflow as tf 
import os
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np 

'''
tfrecord, 这是一种将图像数据和标签放在一起的二进制文件， 能更好的利用内存，在tensorflow中快速
的复制 移动 读取 存储
tfrecord 会根据你选择输入文件的类，自动给每一类打上同样的标签 
如在本类中， 只有0 1 两类
'''

'''
tf.train.Example 协议内存块包含了Features字段， 通过feature将图片的二进制数据和label进行统一封装
然后将example协议内存块转换为字符串， tf.pythin_io.TFRecordWriter 写入到TFRecords文件中
'''

cwd = 'C:\\Users\\lenovo\\Desktop\\My_learnings\\tensorflow_note\\tensorflow_TFRecord\\'
# classes = {'dog', 'cat'}  # 人为设定 2 类
# writer = tf.python_io.TFRecordWriter("cats_dogs_train.tfrecords")  # 要生成的文件目录

# for index, name in enumerate(classes):
#     class_path = cwd + name + '\\'
#     for img_name in os.listdir(class_path):
#         img_path = class_path + img_name  # 每一个图片的地址
#         img = Image.open(img_path)
#         img = img.resize((128, 128))
#         img_raw = img.tobytes()           # 将图片转化为二进制格式

#         example = tf.train.Example(features=tf.train.Features(feature={
#             'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
#             'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
#             }))                           # example对象对label和image数据进行封装
#         writer.write(example.SerializeToString())  # 序列化为字符串

# writer.close()

# img.show()



'''
读取TFRecord文件, 将该文件读入到数据流中
'''
# def read_and_decode(filename):
#     filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列

#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
#     features = tf.parse_single_example(serialized_example, 
#                                         features={
#                                             'label':tf.FixedLenFeature([], tf.int64),
#                                             'img_raw':tf.FixedLenFeature([], tf.string)
#                                         })  # 将image数据和label取出来
#     img = tf.decode_raw(features['img_raw'], tf.uint8)
#     img = tf.reshape(img, [128, 128, 3])
#     img = tf.cast(img, tf.float32) * (1./255) - 0.5
#     label = tf.cast(features['label'], tf.int32)
#     print('image is ', img)
#     print('label is ', label)
#     return img, label

# read_and_decode('C:\\Users\\lenovo\\Desktop\\My_learnings\\tensorflow_note\\tensorflow_TFRecord\\')



'''
显示tfrecord 格式的图片
在session会话中， 将tfrecord的图片从流中读取出来，再保存。
'''

def plot_images(images, labels):
    '''plot one batch size
    '''
    plt.imshow(images)
    plt.show()


filename_queue = tf.train.string_input_producer(['cats_dogs_train.tfrecords'])  # 生成一个queue队列

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
features = tf.parse_single_example(serialized_example, 
                                    features={
                                        'label':tf.FixedLenFeature([], tf.int64),
                                        'img_raw':tf.FixedLenFeature([], tf.string)
                                    })  # 将image数据和label取出来
img = tf.decode_raw(features['img_raw'], tf.uint8)
img = tf.reshape(img, [128, 128, 3])
label = tf.cast(features['label'], tf.int32)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(20):
        image, l = sess.run([img, label])  # 在会话中取出image and label
        # print(example,l)
        plot_images(image, l)
        image = Image.fromarray(image, 'RGB')

        image.save(cwd+str(i)+'_'+'Label_'+str(l)+'.jpg')
        print(image,l)
    coord.request_stop()
    coord.join(threads)
