import tensorflow as tf
import cifar10

width = 24
height = 24

FLAGS = cifar10.FLAGS

saver = tf.train.Saver(tf.global_variables())

# 1. GRAPH CREATION
input_img = tf.image.decode_jpeg(tf.read_file("/home/.../your_image.jpg"), channels=3)
reshaped_image = tf.image.resize_image_with_crop_or_pad(tf.cast(input_img, width, height), tf.float32)
float_image = tf.image.per_image_withening(reshaped_image)
images = tf.expand_dims(float_image, 0)  # create a fake batch of images (batch_size = 1)
logits = cifar10.inference(images)
_, top_k_pred = tf.nn.top_k(logits, k=5)

# 2. TENSORFLOW SESSION
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found')

    # sess.run(init_op)
    top_indices = sess.run([top_k_pred])
    print ("Predicted ", top_indices[0], " for your input image.")