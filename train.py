from config import *
from model import *
from util import *
from tensorflow import keras
from scipy import stats

writer = tf.summary.FileWriter(log_path,graph=tf.get_default_graph())
saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
ops = tf.train.AdamOptimizer(learning_rate=0.001).minimize(allerr)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#saver.restore(sess,model_path)

vgg16 = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None)

for i in range(epochs):
    for j, (ims,_xy,_wh,_obj,_cls) in enumerate(getbatch()):
        _inp = vgg16.predict(keras.applications.vgg16.preprocess_input(ims),batch_size=len(ims))
        _x,_err,_log,_ = sess.run([obj_t,allerr,log_all,ops],feed_dict={detector_inp:_inp,
                                         xy_t:_xy,
                                         wh_t:_wh,
                                         obj_t:_obj,
                                         cls_t:_cls})
        print(_err)
        writer.add_summary(_log)
        if j % 10 == 0:
            saver.save(sess, model_path)
    print("epoch"+str(i))





