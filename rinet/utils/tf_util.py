""" Wrapper functions for TensorFlow layers.

Author: Charles R. Qi
Date: November 2016
"""

import numpy as np
import tensorflow as tf
import pdb
import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../rinet/tf_ops'))
from sampling.tf_sampling import farthest_point_sample, gather_point
from grouping.tf_grouping import query_ball_point



def distance_to_centroid(point_cloud):
    
    centroid = tf.reduce_mean(point_cloud, axis=1, keepdims=True) # (B, 1, 3)
    centroid = tf.tile(centroid, [1, 1024, 1])
    
def fps_normal(point_cloud, sampling_points):
    idd = farthest_point_sample(sampling_points, point_cloud[:, :, :3])
    xyz = gather_point(point_cloud[:, :, :3],  idd)
    normal = gather_point(point_cloud[:, :, 3:], idd)

    return xyz, normal, idd 


def fps(point_cloud, sampling_points):
    idxes = farthest_point_sample(sampling_points, point_cloud)
    xyz = gather_point(point_cloud, idxes)
    return idxes, xyz

def fps_tsne(point_cloud, tsne, sampling_points):
    idxes = farthest_point_sample(sampling_points, tsne)
    xyz = gather_point(point_cloud, idxes)
    return idxes, xyz, new_tsne


def qbp(radius, nsample, centroid, point_cloud):
    '''
    Output:
        idx : (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt : (batch_size, npoint) int32 array, number of unique points in each local region 
    '''
    idx, pts_cnt = query_ball_point(radius, nsample, centroid, point_cloud)
    return idx, pts_cnt

def point_search(inner_points, point_cloud, tiler ):
    '''
    inner_points : divided point sets. 
    point_cloud  : input point cloud
    tiler        : the number of points farthest sampling.
    '''
    
    inner_cnt = {}
    data_number = len(inner_points)
    for d in range(data_number):
        data = inner_points[d] # data << inner points are put in sequentially. 
        pts = [] 
        for k in range(data.shape[0]):
            whole_point_cloud = tf.expand_dims(point_cloud[k, :, :], axis=0)
            whole_point_cloud = tf.tile(whole_point_cloud, [tiler, 1,1])
            unit_point_cloud  = data[k , : ,: ,:]
            _, pts_cnt = qbp(0.1, 16, whole_point_cloud, unit_point_cloud)
            pts.append(pts_cnt)
        inner_cnt[d] = tf.stack(pts)
    
    return inner_cnt


def graph_genertation(cnt_matrix):

    keys = list(cnt_matrix.keys())
    refer = cnt_matrix[0]
    dense_thresh = (tf.zeros((refer.shape[0], refer.shape[1], refer.shape[2]), tf.float32)) + 8
    masker_thresh = (tf.zeros((refer.shape[0], refer.shape[1], refer.shape[2]), tf.float32)) 
    
    for i in range(len(keys)):
        cnts = cnt_matrix[i]
        bool_idx = tf.greater_equal(tf.cast(cnts, tf.float32), dense_thresh)
        bin_idx = tf.cast(bool_idx, tf.float32)

    pass




def pinet_normalize(adj, p, q):
    I = tf.expand_dims(tf.eye(int(adj.shape[1])), axis=0)
    #I = tf.tile(I, [int(adj.shape[0]), 1, 1])
    rowsum = tf.reduce_sum(adj, axis=2, keep_dims=True)
    #D = tf.linalg.diag(tf.squeeze(rowsum), 2)
    D = tf.linalg.diag(tf.squeeze(rowsum, axis=2))
    
    A_hat_1 = tf.pow( (p*I + (1-p)*D), -0.5)
    A_hat_1 = tf.where(tf.is_inf(A_hat_1), tf.ones_like(A_hat_1) * 0, A_hat_1)
    A_hat_2 = (adj + q * I)
    
    A_hat = tf.matmul(tf.matmul(A_hat_1, A_hat_2), A_hat_1)
    
    return A_hat, p, q







def internal_points(point_cloud, m, n):
    '''
    Derive internal point which divide the point into m:n for pointwise. 
    point_cloud : (Batch, Points, 3) 
    '''


    num_points = (point_cloud.shape)[1] 
    # Point Cloning.
    pc1 = tf.expand_dims(point_cloud, axis=-1)
    pc1 = tf.tile(pc1, [1, 1, 1, num_points])
    pc1 = tf.transpose(pc1, (0, 1, 3, 2))

    pc2 = tf.expand_dims(point_cloud, axis=1)
    pc2 = tf.tile(pc2, [1, num_points, 1, 1])
    
    #assert m+n==5;
    inner_points = ((m * pc1 ) + (n * pc2)) / ( m + n )

    return inner_points
    

def _variable_on_cpu(name, shape, initializer, use_fp16=False):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True, board=False):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.contrib.layers.xavier_initializer()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('reg_loss', weight_decay)

  #if board:
  #  tf.summary.scalar('regularization loss', weight_decay)

  return var


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 1D convolution with non-linear operation.

  Args:
    inputs: 3-D tensor variable BxLxC
    num_output_channels: int
    kernel_size: int
    scope: string
    stride: int
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    num_in_channels = inputs.get_shape()[-1].value
    kernel_shape = [kernel_size,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    outputs = tf.nn.conv1d(inputs, kernel,
                           stride=stride,
                           padding=padding)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)

    if bn:
      outputs = batch_norm_for_conv1d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


  


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           activation_fn=tf.nn.relu,
           weight_decay=0.0,
           bn=False,
           bn_decay=None,
           is_training=None,
           bn_init_values=(0.0, 1.0),
           board=False,
           reuse=tf.AUTO_REUSE
           ):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope, reuse=reuse) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_in_channels, num_output_channels]
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay, 
                                           board=board)
      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(inputs, kernel,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn', init_values=bn_init_values, reuse=reuse)

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs


def conv2d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1],
                     padding='SAME',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=0.0,
                     activation_fn=tf.nn.relu,
                     bn=False,
                     bn_decay=None,
                     is_training=None,
                     bn_init_values=(0.0, 1.0)):
  """ 2D convolution transpose with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor

  Note: conv2d(conv2d_transpose(a, num_out, ksize, stride), a.shape[-1], ksize, stride) == a
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_output_channels, num_in_channels] # reversed to conv2d
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      
      # from slim.convolution2d_transpose
      def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
          dim_size *= stride_size

          if padding == 'VALID' and dim_size is not None:
            dim_size += max(kernel_size - stride_size, 0)
          return dim_size

      # caculate output shape
      batch_size = inputs.get_shape()[0].value
      height = inputs.get_shape()[1].value
      width = inputs.get_shape()[2].value
      out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
      out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
      output_shape = [batch_size, out_height, out_width, num_output_channels]

      outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn', init_values=bn_init_values)

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs

   

def conv3d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 3D convolution with non-linear operation.

  Args:
    inputs: 5-D tensor variable BxDxHxWxC
    num_output_channels: int
    kernel_size: a list of 3 ints
    scope: string
    stride: a list of 3 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    num_in_channels = inputs.get_shape()[-1].value
    kernel_shape = [kernel_d, kernel_h, kernel_w,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.conv3d(inputs, kernel,
                           [1, stride_d, stride_h, stride_w, 1],
                           padding=padding)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)
    
    if bn:
      outputs = batch_norm_for_conv3d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs

def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None,
                    reuse= False):
  """ Fully connected layer with non-linear operation.
  
  Args:
    inputs: 2-D tensor BxN
    num_outputs: int
  
  Returns:
    Variable tensor of size B x num_outputs.
  """
  with tf.variable_scope(scope, reuse=reuse) as sc:
    num_input_units = inputs.get_shape()[-1].value
    weights = _variable_with_weight_decay('weights',
                                          shape=[num_input_units, num_outputs],
                                          use_xavier=use_xavier,
                                          stddev=stddev,
                                          wd=weight_decay)
    outputs = tf.matmul(inputs, weights)
    biases = _variable_on_cpu('biases', [num_outputs],
                             tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)
     
    if bn:
      outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn', reuse=reuse)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D max pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs

def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D avg pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.avg_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs

def reduce_max(inputs,
            scope,
            axis=1,
            keep_dims=True):

  with tf.variable_scope(scope) as sc:
    outputs = tf.reduce_max(inputs, axis=axis, keep_dims=keep_dims, name=sc.name)

    return outputs

def max_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D max pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.max_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs

def avg_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D avg pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.avg_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs



def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay, init_values=(0.0,1.0), reuse=False):
  """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
  Return:
      normed:        batch-normalized maps
  """
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
    num_channels = inputs.get_shape()[-1].value
    beta = tf.Variable(tf.constant(init_values[0], shape=[num_channels]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(init_values[1], shape=[num_channels]),
                        name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    # Operator that maintains moving averages of variables.
    ema_apply_op = tf.cond(is_training,
                           lambda: ema.apply([batch_mean, batch_var]),
                           lambda: tf.no_op())
    
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
  return normed


def batch_norm_for_fc(inputs, is_training, bn_decay, scope, reuse=False):
  """ Batch normalization on FC data.
  
  Args:
      inputs:      Tensor, 2D BxC input
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,], bn_decay, reuse=reuse)


def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 1D convolutional maps.
  
  Args:
      inputs:      Tensor, 3D BLC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1], bn_decay)



  
def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope, init_values=(0.0,1.0), reuse=False):
  """ Batch normalization on 2D convolutional maps.
  
  Args:
      inputs:      Tensor, 4D BHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay, init_values, reuse=reuse)



def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 3D convolutional maps.
  
  Args:
      inputs:      Tensor, 5D BDHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1,2,3], bn_decay)


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
  """ Dropout layer.

  Args:
    inputs: tensor
    is_training: boolean tf.Variable
    scope: string
    keep_prob: float in [0,1]
    noise_shape: list of ints

  Returns:
    tensor variable
  """
  with tf.variable_scope(scope) as sc:
    outputs = tf.cond(is_training,
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs)
    return outputs

#def graph_lap(point_cloud, num_point):
    
def renormalize_adj_lamb(adj, batch_size, num_pool, lamb):
    I = tf.eye(num_pool)
    I = tf.tile(tf.expand_dims(I, axis=0), [batch_size, 1,1])
    adj = adj + (I*lamb) 
    rowsum = tf.reduce_sum(adj, axis=2, keep_dims=True)
    d = tf.pow(rowsum, -0.5)
    d = tf.where(tf.is_inf(d), tf.ones_like(d) * 0, d)

    #inf_mask = tf.is_inf(d)
    #inf_mask = tf.cast(inf_mask, tf.float32)
    #inf_mask = 1-inf_mask
    #d = d * inf_mask 
    d_d = tf.linalg.diag(tf.squeeze(d, axis=2))

    adj = tf.matmul(tf.matmul(d_d, adj), d_d)
    return adj 



def renormalize_adj(adj, batch_size, num_pool):
    I = tf.eye(num_pool)
    I = tf.tile(tf.expand_dims(I, axis=0), [batch_size, 1,1])
    adj = adj + I 
    rowsum = tf.reduce_sum(adj, axis=2, keep_dims=True)
    d = tf.pow(rowsum, -0.5)
    d = tf.where(tf.is_inf(d), tf.ones_like(d) * 0, d)

    #inf_mask = tf.is_inf(d)
    #inf_mask = tf.cast(inf_mask, tf.float32)
    #inf_mask = 1-inf_mask
    #d = d * inf_mask 
    d_d = tf.linalg.diag(tf.squeeze(d, axis=2))

    adj = tf.matmul(tf.matmul(d_d, adj), d_d)
    return adj 

    


    

def normalize_adj(adj, batch_size, num_pool):
    rowsum = tf.reduce_sum(adj, axis=2)
    d = tf.pow(rowsum, -0.5)
    d = tf.where(tf.is_inf(d), tf.ones_like(d) * 0, d)
    if batch_size==1:
        d_d = tf.linalg.diag(d)
    else:
        d_d = tf.linalg.diag(tf.squeeze(d))
        
    normed_adj = tf.matmul(tf.matmul(d_d, adj), d_d)
    return normed_adj


def scaled_laplacian(adj):
    adj_normalized = normalize_adj(adj, int(adj.shape[0]), int(adj.shape[1]))
    laplacian = tf.eye(int(adj.shape[1])) - adj_normalized
    eig = tf.linalg.eigvalsh(laplacian)
    max_eig = tf.reduce_max(eig, axis=1)
    #max_eig = 1.5
    fh = tf.reshape(( 2./max_eig), (-1, 1,1))
    I = tf.eye(int(adj.shape[1]))
    I = tf.tile(tf.expand_dims(I, axis=0), [int(adj.shape[0]), 1,1])
    scaled_laplacian = (fh * laplacian - I)

    return scaled_laplacian
    #return adj





def spherecal_transformation(point_cloud):
    
    r = tf.linalg.norm(point_cloud, axis=2, keep_dims=True)
    psi = tf.acos(tf.expand_dims(point_cloud[:,:,2], axis=-1)/ r)
    theta = tf.expand_dims(tf.atan(point_cloud[:, :, 1] / point_cloud[:, :, 0]), axis=-1)
    
    new_coords = tf.concat([r, psi, theta], axis=-1)
    return new_coords




def pairwise_angle(point_cloud):

    dots = tf.matmul(point_cloud, tf.transpose(point_cloud, (0,2,1)))
    point_norms = tf.linalg.norm(point_cloud, axis=2, keep_dims=True)
    point_norms = tf.matmul(point_norms, tf.transpose(point_norms, (0,2,1)))
    #cosine = -(dots / point_norms)
    
    #angle_matrix = tf.acos(tf.maximum(tf.minimum(cosine, 1.0), -1.0)) / np.pi
    #cosine = ( 1.0 / (1.0 + tf.exp(-cosine)))
    cosine = (dots / point_norms)
    return cosine


def pairwise_distance(point_cloud):
    """Compute pairwise distance of a point cloud.

    Args:
      point_cloud: tensor (batch_size, num_points, num_dims)

    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    #point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_transpose = tf.transpose(point_cloud, perm  =[0, 2, 1])
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
    point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose


def pairwise_distance_2(point_cloud1, point_cloud2):
    """Compute pairwise distance of a point cloud.

    Args:
      point_cloud1: tensor (batch_size, num_points1, num_dims)
      point_cloud2: tensor (batch_size, num_points2, num_dims)

    Returns:
      pairwise distance: (batch_size, num_points1, num_points2)
    """
    og_batch_size = point_cloud1.get_shape().as_list()[0]
    #point_cloud1 = tf.squeeze(point_cloud1)
    #point_cloud2 = tf.squeeze(point_cloud2)
    if og_batch_size == 1:
        point_cloud1 = tf.expand_dims(point_cloud1, 0)
        point_cloud2 = tf.expand_dims(point_cloud2, 0)

    point_cloud_inner = tf.matmul(point_cloud1, point_cloud2, transpose_b=True)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud1_square = tf.reduce_sum(tf.square(point_cloud1), axis=-1, keep_dims=True)
    point_cloud2_square = tf.reduce_sum(tf.square(point_cloud2), axis=-1, keep_dims=True)
    point_cloud2_square_tranpose = tf.transpose(point_cloud2_square, perm=[0, 2, 1])
    return point_cloud1_square + point_cloud_inner + point_cloud2_square_tranpose

def pairwise_distance_3(point_cloud1, point_cloud2):
    og_batch_size = point_cloud1.get_shape().as_list()[0]
    point_cloud1 = tf.squeeze(point_cloud1)
    point_cloud2 = tf.squeeze(point_cloud2)
    point_cloud_inner = tf.matmul(point_cloud1, tf.transpose(point_cloud2, (0,1,3,2)))
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud1_square = tf.linalg.norm(point_cloud1, axis=-1, keep_dims=True)
    point_cloud2_square = tf.linalg.norm(point_cloud2, axis=-1, keep_dims=True)
    point_cloud2_square_transpose = tf.transpose(point_cloud2_square, perm=[0,1,3,2])
    return point_cloud1_square + point_cloud_inner + point_cloud2_square_transpose

def knn(adj_matrix, k=20):
    """Get KNN based on the pairwise distance.
    Args:
      pairwise distance: (batch_size, num_points1, num_points2)
      k: int

    Returns:
      nearest neighbors: (batch_size, num_points1, k)
    """
    neg_adj = -adj_matrix
    _, nn_idx = tf.nn.top_k(neg_adj, k=k)
    return nn_idx


def gather_neighbors_4d(point_cloud, nn_idx, k=20):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k)
        k: int

    Returns:
        neighbors: (batch_size, num_points, k, num_dims)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud,axis=2)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)

    return point_cloud_neighbors




def gather_neighbors_stoc(point_cloud, nn_idx,checker,  k=20):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k)
        k: int

    Returns:
        neighbors: (batch_size, num_points, k, num_dims)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud,axis=2)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    #num_dims = point_cloud_shape[2].value
    num_dims = checker * 3 

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)

    return point_cloud_neighbors





def gather_neighbors(point_cloud, nn_idx, k=20):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k)
        k: int

    Returns:
        neighbors: (batch_size, num_points, k, num_dims)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud,axis=2)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)

    return point_cloud_neighbors

def get_neighbor_feature(point_cloud, nn_idx, k=20, no_central=False):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k)
        k: int

    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_central = point_cloud

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    return point_cloud_neighbors




def get_edge_feature(point_cloud, nn_idx, k=20, no_central=False):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k)
        k: int

    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_central = point_cloud

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])
    # point_cloud_central = tf.reduce_mean(point_cloud_neighbors, axis=2, keep_dims=True)

    if no_central:
        edge_feature = point_cloud_neighbors - point_cloud_central
    else:
        edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
    return edge_feature


def batch_mean_covar(inputs, is_training, scope, moments_dims, batch_decay):
    """ Batch ema of mean and covariance
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        moments_dims:  a list of ints, indicating dimensions for moments calculation
        bn_decay:      float or float tensor variable, controling moving average weight
    Return:
        normed:        batch-normalized maps
    """
    with tf.variable_scope(scope) as sc:
        batch_mean, _ = tf.nn.moments(inputs, moments_dims, name='moments')
        # compute covariance of batch-point data
        num_channels = inputs.get_shape()[-1].value
        flatten_input = tf.reshape(inputs, [-1, num_channels])
        num_data = flatten_input.get_shape()[0].value
        zero_mean_input = flatten_input - tf.reshape(batch_mean, [-1, num_channels])
        batch_covar = tf.matmul(tf.transpose(zero_mean_input, perm=(1, 0)), zero_mean_input)
        batch_covar = tf.divide(batch_covar, num_data-1)

        decay = batch_decay if batch_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_covar]),
                               lambda: tf.no_op())

        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_covar)

        # ema.average returns the Variable holding the average of var.
        mean, covar = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_covar)))
    return mean, covar




def distance_to_centroid(point_cloud):
    
    centroid = tf.reduce_mean(point_cloud, axis=1, keep_dims=True) # (B, 1, 3)
    centroid = tf.tile(centroid, [1, 1024, 1])

    return centroid


def stoc_K(att_coords, point_cloud, k1, d=0, pa = None, dilation=False, axis=False):

    k_d = k1 * d
    adj1 = pairwise_distance_2(att_coords, point_cloud)

    if dilation == True:
        neg_dists1, knn1 = tf.nn.top_k(-adj1, k=k1 * d)
        jumping_idx = tf.range(k1) * d
        knn1 = tf.transpose(knn1, (2,0,1))
        knn1 = tf.transpose(tf.gather(knn1, jumping_idx), (1,2,0))
    else:
        neg_dists, knn1 = tf.nn.top_k(-adj1, k=k1)
    
    neighbors1 = gather_neighbors(tf.expand_dims(point_cloud, axis=2), knn1, k=k1)
    central1 = tf.tile(tf.expand_dims(att_coords, axis=2), [1, 1, k1, 1])
    rel = neighbors1 - central1
    if axis==True:
       rel_mean = tf.reduce_mean(rel, axis=2, keep_dims=True)
       local_co = (tf.matmul(tf.transpose(rel-rel_mean, (0,1,3,2)), rel-rel_mean)) / tf.cast((k1-1), tf.float32)
       with tf.device('/cpu:0'):
            v, u = tf.linalg.eigh(local_co)
       u = tf.reverse(u, axis=[3])
       #encoding = tf.matmul(rel, u)
       #encoding = encoding - tf.reduce_mean(encoding, axis=2, keep_dims=True)

       return u

    else:
        encoding = tf.matmul(rel, pa)
        encoding = encoding - tf.reduce_mean(encoding, axis=2, keep_dims=True)

        return encoding 

def generate_knn(point_cloud,batch_size, num_pool, k=16):


    pairwise_distance = pairwise_distance_2(point_cloud, point_cloud)
    neg2, knn2 = tf.nn.top_k(-pairwise_distance, k=k)
    neg2_max = tf.reduce_min(neg2, axis=2, keep_dims=True)
    neg_max_tiled = tf.tile(neg2_max, [1, 1, num_pool])
    
    knn_graph = tf.greater_equal(pairwise_distance, -neg_max_tiled)
    knn_graph = 1.0 - tf.cast(knn_graph, tf.float32)
    knn_graph = renormalize_adj(knn_graph, batch_size, num_pool)

    return knn_graph


