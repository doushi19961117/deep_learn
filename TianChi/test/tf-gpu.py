import tensorflow as tf
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
tf.device('/gpu:2')
print(sess.run(hello))