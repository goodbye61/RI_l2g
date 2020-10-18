import os
import sys
import numpy as np
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
#sys.path.append(os.path.join(BASE_DIR, '../nndistance'))
#import tf_nndistance
import pdb
from sampling.tf_sampling import gather_point 
from grouping.tf_grouping import group_point
from rinet_util import * 

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, num_pool, pool_k1, is_training, bn_decay=None, flag=0, flag2=0, flag3=0, gcn1=0, gcn2=0, gcn3=0, dilation=0):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    # stage 1 
    prev1, eig_vector, centering, out1, fps_points = encoder(point_cloud, num_point, gcn1, flag, batch_size, dilation, mlp=[64, 64, 128], is_training=is_training, bn_decay=bn_decay,
                                                sampling_points = num_pool, mlp_merge = [128, 256, 256], stage=1) 

    # stage 2 &  
    prev2, eig_vector, centering, out2, fps_points  = encoder(fps_points, num_pool, gcn2, flag2, batch_size, dilation,  mlp = [256, 256, 256], is_training=is_training, bn_decay=bn_decay,
                                                sampling_points = (num_pool//2), eig_vector=eig_vector, centering=centering, prev_feature = prev1, stage=2)

    _, _, _, out3, _  = encoder(fps_points, (num_pool//2), gcn3, flag3, batch_size, dilation,  mlp = [256, 512, 1024], is_training=is_training, bn_decay=bn_decay,
                                                sampling_points = (num_pool//4), eig_vector=eig_vector, centering=centering, prev_feature = prev2, stage=3)

    out1 = tf.reduce_max(out1, axis=1, keep_dims=True)  
    out2 = tf.reduce_max(out2, axis=1, keep_dims=True)  
    out3 = tf.reduce_max(out3, axis=1, keep_dims=True)  

    net = tf.concat([out1, out2, out3], axis=-1)         

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        eps = 1e-6
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.log(pt_1)) - \
                    tf.reduce_sum((1-alpha) * tf.pow(pt_0, gamma) * tf.log(1. - pt_0))
    return focal_loss_fixed


def get_loss(pred, label, end_points, alpha=30.0, beta=30.0, margin=0.2):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify_loss', classify_loss)

    vars = tf.trainable_variables()
    loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * 8e-5
    tf.summary.scalar('l2 regloss', loss_reg)

    return classify_loss + loss_reg


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
