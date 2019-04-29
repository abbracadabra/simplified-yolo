import tensorflow as tf
from config import *

detector_inp = tf.placeholder(dtype=tf.float32,shape=[None,None,None,512],name='input')#[None,224,224,3]
xy_t = tf.placeholder(dtype=tf.float32,shape=[None,None,None,2])
wh_t = tf.placeholder(dtype=tf.float32,shape=[None,None,None,2])
obj_t = tf.placeholder(dtype=tf.float32,shape=[None,None,None,1])
cls_t = tf.placeholder(dtype=tf.float32,shape=[None,None,None,20])
boxnum = tf.reduce_sum(obj_t)

detector_out = tf.layers.conv2d(detector_inp,25,(1,1))#[None,7,7,25]
xy = tf.nn.sigmoid(detector_out[...,0:2])
wh = tf.sqrt(tf.clip_by_value(tf.nn.sigmoid(detector_out[...,2:4]),1e-3,1.))
obj = tf.nn.sigmoid(detector_out[...,4:5])
cls = tf.nn.softmax(detector_out[...,5:])
xyerr = tf.reduce_sum((xy_t-xy)**2 * obj_t)/boxnum
wherr = tf.reduce_sum((wh_t-wh)**2 * obj_t)/boxnum
objerr = tf.reduce_mean((obj-obj_t)**2 * tf.where(tf.equal(obj_t,1.),tf.ones_like(obj_t)*3,tf.ones_like(obj_t)))
clserr = tf.reduce_sum((cls_t-cls)**2 * cls_t)/boxnum
allerr = xyerr+wherr+objerr+clserr

tf.summary.scalar('xyerr',xyerr)
tf.summary.scalar('wherr',wherr)
tf.summary.scalar('objerr',objerr)
tf.summary.scalar('clserr',clserr)
tf.summary.scalar('allerr',allerr)
log_all = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_path,graph=tf.get_default_graph())











