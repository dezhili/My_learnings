'''
1. 创建一个VGGNet-16并加载已经训练好的参数
'''
import numpy as np 
import tensorflow as tf 
import scipy.io
import utils

import os
import time

'''
因为加载的模型是mat形式，用scipy.io 读取后是numpy的形式，需要进行转换
首先我们要知道需要提取的参数有卷积+relu层和池化层的参数
这一个cell的程序都是模型转换的部分
'''
# 权重提取函数，返回权重的值
def _weight(vgg_layers, layer, expected_layer_name):
    W = vgg_layers[0][layer][0][0][2][0][0]
    b = vgg_layers[0][layer][0][0][2][0][1]
    layer_name = vgg_layers[0][layer][0][0][0][0]
    assert layer_name == expected_layer_name
    return W,b.reshape(b.size)

#  提取卷积层参数，即filter的权重
def _conv2d_relu(vgg_layers, prev_layer, layer, layer_name):
    '''
      函数目的是返回这一层使用的filter的权重和bias
      输入：
         vgg_layers: VGGNet的所有层
         prev_layer: 前一层的输出tensor
         layer: 当前层的index，这个是由使用的VGG模型决定的
         layer_name: 当前层使用的名字，这个用于指定变量空间
     输出：
         relu的结果
    '''
    with tf.variable_scope(layer_name) as scope:
        W, b = _weight(vgg_layers, layer, layer_name)
        W = tf.constant(W, name='weights')
        b = tf.constant(b, name='bias')
        conv2d = tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv2d + b)

def _avgpool(prev_layer):
    """
    实现平均池化层
    Input:
        prev_layer: 前一层的输出

    Output:
        平均池化结果
    Hint for choosing strides and kszie: choose what you feel appropriate
    """
    return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                          padding='SAME', name='avg_pool_')


def load_vgg(path, input_image):
    '''
    函数用于转换VGG为TensorFlow，用一个dict来保存模型。
    想要更好的理解这一部分需要了解.mat文件的内容结构和VGGNet-16的结构
    需要注意matlab里是从1开始的而Python是从0开始的，所以用matlab打开VGG的时候一下关于层数之类的数量可能会差1
    '''
    vgg = scipy.io.loadmat(path)#读取文件
    vgg_layers = vgg['layers'] #读取文件中layer下的的值
    
    graph = {} 
    graph['conv1_1']  = _conv2d_relu(vgg_layers, input_image, 0, 'conv1_1')#第一部分的卷积1
    graph['conv1_2']  = _conv2d_relu(vgg_layers, graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(vgg_layers, graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(vgg_layers, graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(vgg_layers, graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(vgg_layers, graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2d_relu(vgg_layers, graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(vgg_layers, graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(vgg_layers, graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(vgg_layers, graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(vgg_layers, graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(vgg_layers, graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(vgg_layers, graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(vgg_layers, graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(vgg_layers, graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(vgg_layers, graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    return graph




'''
2. 上面的三个函数都是为了能从现有的模型中提取出参数，下面就到了正式实现的时候了，
首先需要常量的赋值和损失函数的计算
计算content_loss
计算style_loss
计算总的损失
'''
STYLE = 'pattern'
CONTENT = 'deadpool'
STYLE_IMAGE = 'styles/' + STYLE + '.jpg'
CONTENT_IMAGE = 'content/' + CONTENT + '.jpg'

IMAGE_HEIGHT = 250 #图像尺寸
IMAGE_WIDTH = 333

NOISE_RATIO = 0.6 # 生成噪声图像时用的

CONTENT_WEIGHT = 0.3 #content和style的权重，可以随意调整
STYLE_WEIGHT = 1

#style的一些参考层，和每一层的权重，层数越深对style的影响越大，这个是可以随意调整的
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
W = [0.5, 1.0, 1.5, 3.0, 4.0] 
#content的参考层，可以随意调整
CONTENT_LAYER = 'conv3_2'

learning_rate = 5#学习率
STEPS = 600 #step的次数

# MEAN_PIXELS这个和我们用的VGG模型有关，它在训练的时候是减去均值训练的，所以这里我们也是需要在训练的时候减去均值
# 不过这样肯定会影响构造出的图像的效果
MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'



# 开始计算loss
#首先是content_loss
def _create_content_loss(p, f):
    '''
    计算内容损失函数
    输入：
        p和f分别为每一层输出的激活值
        f为生成图片的特征表示，p是content图片的特征表示
    输出：
        content_loss
    '''
    #这里有一点注意的是，loss的计算方法和paper中的计算不同，paper中损失函数的收敛速度过慢了，
    # 所以把1/2编程1/(4s),s为p的维度的乘积
    return tf.reduce_sum((f - p)**2)/(4.0 * p.size)

def _gram_matrix(F, N, M):
    '''
    计算gram矩阵
    输入：
        F为图片在某一层通过某个filter后的激活值(第一个维度为输入图片数量等于1)
        N为特征map的第四个维度（filter数量）
        M为每个filter的维度乘积
    输出：
        gram矩阵的值
    '''
    F = tf.reshape(F,(M,N))
    return tf.matmul(tf.transpose(F), F)
def _single_style_loss(a, g):
    """ 计算某一层的style损失
    Inputs:
        a 真实图片的特征表示
        g 生产图片的特征表示
    Output:
        某一层的style损失

    Hint: 1. you'll have to use the function _gram_matrix()
        2. we'll use the same coefficient for style loss as in the paper
        3. a and g are feature representation, not gram matrices
    """
    N = a.shape[3]
    M = a.shape[1]*a.shape[2]
    A = _gram_matrix(a, N, M)
    G = _gram_matrix(g, N, M)
    return tf.reduce_sum((G-A)**2)/((2.0*N*M)**2)

def _style_loss(A, model):
    '''
    计算总的style损失
    输入：
        A 真实图片的在各指定层的特征表示
        model 生成图片在各层的生产结果（把所有层的结果都放进来了）
    输出：
        各层的损失和
    '''
    num_layer = len(STYLE_LAYERS)
    E = [_single_style_loss(A[i],model[STYLE_LAYERS[i]]) for i in range(num_layer)]
    return sum([W[i]*E[i] for i in range(num_layer)])

def _creat_loss(model, input_image, content_image, style_image):
    '''
    计算总的损失函数值,这里还是有点绕的，需要对Session理解的够透彻,重复用了图的某一个部分
    输入：
        model：VGG模型
        输入的三个原图像
    输出：
        总的损失和
    '''
    with tf.variable_scope('loss') as scope:
        with tf.Session() as sess:
            #这个sess用于计算content image在某层的输出结果
            sess.run(input_image.assign(content_image))#赋值操作
            p = sess.run(model[CONTENT_LAYER])#计算content image 在给定层的输出值
        content_loss = _create_content_loss(p, model[CONTENT_LAYER])
        #同理计算style_loss
        with tf.Session() as sess:
            #这个sess用于计算style image在某几层的输出结果
            sess.run(input_image.assign(style_image))#赋值操作
            p = sess.run([model[layers] for layers in STYLE_LAYERS])#这里注意一下
        style_loss = _style_loss(p, model)
        
        #计算总的损失
        total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss
    return content_loss, style_loss, total_loss


# 在定义完损失函数后，需要定义summary的函数，用在Tensorboard上可视化
def _create_summary(model):
    with tf.name_scope('summaries'):
        tf.summary.scalar('content_loss', model['content_loss'])
        tf.summary.scalar('style_loss', model['style_loss'])
        tf.summary.scalar('total_loss', model['total_loss'])
        tf.summary.histogram('histogram_content_loss', model['content_loss'])
        tf.summary.histogram('histogram_style_loss', model['style_loss'])
        tf.summary.histogram('histogram_total_loss', model['total_loss'])
        return tf.summary.merge_all()





# 搭建图的函数流程，调用上面的函数构成
# 定义输入变量,这里把输入也变为了一个变量节点，可以进行传播求导
with tf.variable_scope('input') as scope:
    #注意这里是变量，这个就有点类似于用变量表达Placeholder的感觉，因为后面都在给input_image赋值
    input_image = tf.Variable(np.zeros([1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),dtype=tf.float32)
    
#读取图像和VGG模型(注意这些现在还都是在构造图)
model = load_vgg(VGG_MODEL, input_image)#构造模型
model['global_step'] = tf.Variable(0, dtype=tf.int32,trainable=False, name='global_step' )#这个为了用于观察过程中的图片生产效果

#对输入的图像进行一些处理，如尺度变换
content_image = utils.get_resized_image(CONTENT_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
content_image = content_image - MEAN_PIXELS
style_image = utils.get_resized_image(STYLE_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
style_image = style_image - MEAN_PIXELS


#定义计算损失值
model['content_loss'], model['style_loss'], model['total_loss'] = _creat_loss(model, input_image, content_image, style_image)
#设计优化函数
model['optimizer'] = tf.train.AdagradOptimizer(learning_rate).minimize(model['total_loss'])
#设计summary用于TensorBoard展示
model['summary_op'] = _create_summary(model)




# 运行图，得到最后生成的图片
# 构造一个噪声图像,这样比直接用白噪声快一点
#initial_image = utils.generate_noise_image(content_image, IMAGE_HEIGHT, IMAGE_WIDTH, NOISE_RATIO)
initial_image =  np.random.normal(0,0.1,size =(1,IMAGE_HEIGHT, IMAGE_WIDTH,3))

with tf.Session() as sess:
    #初始化变量创建保存器和summary的writer
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('result/',sess.graph)
    skip_step = 1
    
    #构造检查点
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        
    sess.run(input_image.assign(initial_image)) #这里要明白为什么要这样做
    initial_step = model['global_step'].eval()#得到全局变量的值
    
    start_time = time.time()#用于计算运行时间
    for index in range(initial_step, STEPS):
        #从慢到快的记录数据
        if index >= 5 and index < 20:
            skip_step = 10
        elif index >= 20:
            skip_step = 20
        sess.run(model['optimizer'])# 计算优化方程
        #下面是用于获取图像、记录检查点和打印展示信息
        if (index + 1) % skip_step == 0:
            gen_image, total_loss, summary = sess.run([input_image, model['total_loss'],model['summary_op']])
            gen_image = gen_image + MEAN_PIXELS #还原图像
            writer.add_summary(summary, global_step=index)
            print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
            print('   Loss: {:5.1f}'.format(total_loss))
            print('   Time: {}'.format(time.time() - start_time))
            #计算时间
            start_time = time.time()
            filename = 'outputs/%d.png'%(index)
            utils.save_image(filename, gen_image)
            if (index + 1) % 20 == 0:
                saver.save(sess,'checkpoints/style_transfer', index)