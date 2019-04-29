import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras

aa = tf.constant([[1.000,1.0,1]])
bb = tf.where(tf.equal(aa,1.001),tf.ones_like(aa),tf.zeros_like(aa))

sess= tf.Session()
print(sess.run(bb))


