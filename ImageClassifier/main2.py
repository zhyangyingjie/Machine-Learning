import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time
import os.path

max_steps = 30000  # 最大迭代轮数
batch_size = 128  # 批大小
data_dir = 'cifar-10-batches-bin'  # 数据所在路径

# 初始化weight函数，通过wl参数控制L2正则化大小
def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        # L2正则化可用tf.contrib.layers.l2_regularizer(lambda)(w)实现，自带正则化参数
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


cifar10.maybe_download_and_extract()
# 此处的cifar10_input.distorted_inputs()和cifar10_input.inputs()函数
# 都是TensorFlow的操作operation，需要在会话中run来实际运行
# distorted_inputs()函数对数据进行了数据增强
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                            batch_size=batch_size)
print(images_train)
# 裁剪图片正中间的24*24大小的区块并进行数据标准化操作
images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                data_dir=data_dir,
                                                batch_size=batch_size)

# 定义placeholder
# 注意此处输入尺寸的第一个值应该是batch_size而不是None
image_holder = tf.placeholder(tf.float32, [None, 24, 24, 3], name='input_x')
label_holder = tf.placeholder(tf.int32, [None] ,name='input_y')

# 卷积层1，不对权重进行正则化 5x5 3通道 64个卷积核数量
weight1 = variable_with_weight_loss([5, 5, 3, 64], stddev=5e-2, wl=0.0)  # 0.05
kernel1 = tf.nn.conv2d(image_holder, weight1,
                       strides=[1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75) # 4 ？
#LRN 处理,局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力
#LRN 对ReLU这种没有上限边界的激活函数比较试用，不适合于Sigmoid这种有固定边界并且能抑制过大值的激活函数。

# 卷积层2
weight2 = variable_with_weight_loss([5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, strides=[1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1], padding='SAME')
print(pool2)
# 全连接层3
# dim = 6*6*64  # 取每个样本的长度
shapeList = pool2.get_shape().as_list()
dim = shapeList[1] * shapeList[2] * shapeList[3]
reshape = tf.reshape(pool2, [-1, dim])  # 将每个样本reshape为一维向量
print(reshape)
weight3 = variable_with_weight_loss([dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 全连接层4
weight4 = variable_with_weight_loss([384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# 全连接层5
weight5 = variable_with_weight_loss([192, 10], stddev=1 / 192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.matmul(local4, weight5) + bias5


# 定义损失函数loss
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

# def train():
loss = loss(logits, label_holder)  # 定义loss
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)  # 定义优化器
top_k_op = tf.nn.in_top_k(logits, label_holder, 1, name='top_k_op')  # in_top_k？

# 定义会话并开始迭代训练
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
    # 启动图片数据增强的线程队列
tf.train.start_queue_runners()

saver = tf.train.Saver(tf.global_variables())

    # 迭代训练
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])  # 获取训练数据
    _, loss_value = sess.run([train_op, loss],
                                 feed_dict={image_holder: image_batch,
                                            label_holder: label_batch})
    duration = time.time() - start_time  # 计算每次迭代需要的时间
    if step % 10 == 0:
        examples_per_sec = batch_size / duration  # 每秒处理的样本数
        sec_per_batch = float(duration)  # 每批需要的时间
        format_str = ('step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

checkpoint_path = os.path.join('checkpoints/', 'model.ckpt')
if not os.path.exists('checkpoints/'):
    os.makedirs('checkpoints/')
saver.save(sess, checkpoint_path, global_step=step)
    # 在测试集上测评准确率


import math

def test():
    num_examples = 10000
    num_iter = int(math.ceil(num_examples / batch_size))
    true_count = 0
    total_sample_count = num_iter * batch_size
    step = 0
    while step < num_iter:
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op],
                               feed_dict={image_holder: image_batch,
                                          label_holder: label_batch})
        true_count += np.sum(predictions)
        step += 1

    precision = true_count / total_sample_count
    print('precision @ 1 =%.3f' % precision)


def predicate():

    session = tf.Session()

    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path='checkpoints/model.ckpt-299')  # 读取保存的模型
    print()
    pre_x , pre_y = load_predicate_data(sess=session)

    image_batch, label_batch = session.run([pre_x, pre_y])

    predictions = session.run([top_k_op], feed_dict={image_holder: np.reshape(image_batch,[1,24,24,3]), label_holder: label_batch})
    print(predictions)


def load_predicate_data(sess):

    images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                    data_dir='predicateImage/',
                                                    batch_size=batch_size)

    return images_test , labels_test

test()