# Introduction to Tensorflow as the Autodiff and low level Tensorflow API without Keras abstraction

import tensorflow as tf
a = tf.constant(3)
a

c = tf.Variable(0)
b = tf.constant(2)
c = a + b
c
A = tf.constant([[0, 1], [2, 3]], dtype = tf.float32)
A
A.numpy()
b = tf.Variable([1, 2], dtype=tf.float32)
b
tf.reshape(b, (-1, 1))
tf.matmul(A, tf.reshape(b, (-1, 1)))
