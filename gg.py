import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import *
import re

a = ['conv2d/kernel:0', 'conv2d/bias:0', 'conv2d_1/kernel:0', 'conv2d_1/bias:0', 'conv2d_2/kernel:0', 'conv2d_2/bias:0']

sess = tf.Session()
saver = tf.train.import_meta_graph(model_path+'.meta')
saver.restore(sess,model_path)

b = [z for z in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)][:6]
print(b)

cc = tf.Graph().as_default()
saver1 = tf.train.Saver(var_list={_a:_b for _a,_b in zip(a,b)})

detector_inp = tf.placeholder(dtype=tf.float32,shape=[None,None,None,512],name='input')#[None,224,224,3]
xy_t = tf.placeholder(dtype=tf.float32,shape=[None,None,None,2])
wh_t = tf.placeholder(dtype=tf.float32,shape=[None,None,None,2])
obj_t = tf.placeholder(dtype=tf.float32,shape=[None,None,None,1])
cls_t = tf.placeholder(dtype=tf.float32,shape=[None,None,None,20])
tmp = tf.layers.conv2d(detector_inp,256,(3,3),activation=tf.nn.leaky_relu,padding='SAME')
tmp = tf.layers.conv2d(tmp,512,(3,3),activation=tf.nn.leaky_relu,padding='SAME')
boxnum = tf.reduce_sum(obj_t)

detector_out = tf.layers.conv2d(tmp,25,(1,1))#[None,7,7,25]
saver1.save(sess,os.path.join(basedir,'mdl2','mdl'))

#print({gg(z.name):z.name for z in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)})
#saver1 = tf.train.Saver(var_list={z.name:z for z in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)})
