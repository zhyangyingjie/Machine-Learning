import tensorflow as tf
import cifar10
import os
import argparse
import cifar10,cifar10_input
import numpy as np
import math

width = 24
height = 24

FLAGS = cifar10.FLAGS


def load_graph(frozen_graph_pb):
    with tf.gfile.GFile(frozen_graph_pb,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,name='prefix')
    return graph

def test_from_pbmodel():

    print("test from pb...")
    session = tf.Session()

    images_test = readFromLocal()
    images_test = tf.expand_dims(images_test, 0)
    images_test = session.run(images_test)
    print(images_test)

    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default='ckpt_pb_model.pb',
                        type=str, help='Frozen model file to import')
    args = parser.parse_args()
    graph = load_graph(args.frozen_model_filename)
    print(graph)

    for op in graph.get_operations():
        print(op.name, op.values())

    top_k_op = graph.get_tensor_by_name("prefix/top_k_op/top_k_op:0")
    session = tf.Session(graph=graph)
    x = graph.get_tensor_by_name("prefix/input_x:0")
    y = graph.get_tensor_by_name("prefix/input_y:0")

    # for i in range(0,10):
    #     out = session.run(top_k_op, feed_dict={x: images_test, y:[i]}) #(?,)
    #     print(out)
    #     true_count = np.sum(out)
    #     precision = true_count / 1
    #     print('precision @ 1 =%.3f' % precision)



    # return out

def readFromLocal():
    image_value = tf.read_file('predicateImage/icon.jpg')
    img = tf.image.decode_jpeg(image_value, channels=3)

    print(type(image_value))  # bytes
    print(type(img))  # Tensor
    return img


def test():
    print("test from pb...")
    data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'  # 数据所在路径
    batch_size = 128

    images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)


    print(images_test)

    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default='ckpt_pb_model.pb', type=str, help='Frozen model file to import')
    args = parser.parse_args()
    graph = load_graph(args.frozen_model_filename)
    print(graph)

    for op in graph.get_operations():
        print(op.name, op.values())

    top_k_op = graph.get_tensor_by_name("prefix/top_k_op/top_k_op:0")
    x = graph.get_tensor_by_name("prefix/input_x:0")
    y = graph.get_tensor_by_name("prefix/input_y:0")

    num_examples = 10000
    num_iter = int(math.ceil(num_examples / batch_size))
    true_count = 0
    total_sample_count = num_iter * batch_size
    step = 0

    session = tf.Session(graph=graph)
    image_batch = images_test.eval(session=session)
    label_batch = labels_test.eval(session=session)

    while step < num_iter:
        predictions = session.run([top_k_op], feed_dict={x: image_batch, y: label_batch})
        true_count += np.sum(predictions)
        step += 1

    precision = true_count / total_sample_count
    print('precision @ 1 =%.3f' % precision)


print(test())