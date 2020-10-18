import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../attention_pooling/tf_ops'))
sys.path.append(os.path.join(BASE_DIR, '../attention_pooling/models'))
import tf_util
import pdb
from sampling.tf_sampling import farthest_point_sample, gather_point
from grouping.tf_grouping import query_ball_point, group_point


def minimize_ambi_global(eigen, X):
    
    arr = [] 

    for i in range(3):
        
        eig =  tf.expand_dims(eigen[:, :, :, i], axis=-1) 
        eig_vs = eig * (-1.0)
        
        A = tf.cast(tf.greater_equal(tf.matmul(X, eig), 0), tf.float32) 
        B = tf.cast(tf.greater_equal(tf.matmul(X, eig_vs), 0), tf.float32) 
        
        A_sum = tf.reduce_sum(A, axis=1, keep_dims=True)
        B_sum = tf.reduce_sum(B, axis=1, keep_dims=True)
        cond = tf.tile((B_sum>=A_sum), [1, 1024, 3, 1])

        e = tf.where(cond, eig_vs, eig) 
        
        arr.append(e)


    
    arr = tf.squeeze(tf.stack(arr))
    arr = tf.transpose(arr, (1,2,3,0))
    

    return arr 


def minimize_ambi(eigen, X):
    
    arr = [] 

    for i in range(3):
        
        eig =  tf.expand_dims(eigen[:, :, :, i], axis=-1) 
        eig_vs = eig * (-1.0)
        
        A = tf.cast(tf.greater_equal(tf.matmul(X, eig), 0), tf.float32) 
        B = tf.cast(tf.greater_equal(tf.matmul(X, eig_vs), 0), tf.float32) 
        
        A_sum = tf.reduce_sum(A, axis=2, keep_dims=True)
        B_sum = tf.reduce_sum(B, axis=2, keep_dims=True)
        cond = tf.tile((B_sum>=A_sum), [1, 1, 3, 1])

        e = tf.where(cond, eig_vs, eig) # all-or-nothing
        
        arr.append(e)
    
    arr = tf.squeeze(tf.stack(arr))
    arr = tf.transpose(arr, (1,2,3,0))
    return arr 


def get_eigvec(eigvec, idxes):

    e1 = tf.expand_dims(tf_util.gather_point(eigvec[:, :, 0, :], idxes), axis=2)
    e2 = tf.expand_dims(tf_util.gather_point(eigvec[:, :, 1, :], idxes), axis=2)
    e3 = tf.expand_dims(tf_util.gather_point(eigvec[:, :, 2, :], idxes), axis=2)
    
    gathered_eigvec= tf.concat([e1, e2, e3], axis=2)

    return gathered_eigvec
    


def graph_generation(num_edges, num_points,  points, batch_size):

    '''
    Generate a graph which is based on sampled points. 
    
    args: 
        num_edges : the number of edges
        num_points : the number of points 
        points : the input points (B, N, 3) 

    return:
        A graph (B, N, N) 

    '''

    k1 = num_edges
    sampling_points = num_points

    pairwise_distance2 = tf_util.pairwise_distance_2(points, points)
    neg2, knn2 = tf.nn.top_k(-pairwise_distance2, k=k1)
    neg2_max = tf.reduce_min(neg2, axis=2, keep_dims=True)
    neg_max_tiled = tf.tile(neg2_max, [1, 1, sampling_points])
    knn_graph = tf.greater_equal(pairwise_distance2, -neg_max_tiled)
    knn_graph = 1.0 - tf.cast(knn_graph, tf.float32)
    _, variance = tf.nn.moments(pairwise_distance2, axes=[2], keep_dims=True)
    dist = tf.exp(-pairwise_distance2 / variance)
    knn_graph = (knn_graph * dist)
    knn_graph = tf.matrix_set_diag(knn_graph, tf.linalg.diag_part(knn_graph) * 0)
    mask = tf.cast((tf.transpose(knn_graph, (0,2,1)) > knn_graph), tf.float32)
    knn_graph = (knn_graph - knn_graph * mask + tf.transpose(knn_graph, (0,2,1)) * mask)
    knn_graph = tf_util.renormalize_adj(knn_graph, batch_size, sampling_points)


    return knn_graph




def RILD(num_search, prev, curr, prev_feature=None):

    adj1 = tf_util.pairwise_distance_2(curr, prev)
    neg_dists, knn1 = tf.nn.top_k(-adj1, k=num_search)
    neighbors1 = tf_util.gather_neighbors(tf.expand_dims(prev, axis=2), knn1, k=num_search)
    central1   = tf.tile(tf.expand_dims(curr, axis=2), [1, 1, num_search, 1])
    rel       = neighbors1 - central1

    if prev_feature is not None:
        neighbor_feature = tf_util.get_neighbor_feature(prev_feature, knn1, k=num_search)
        return rel, neighbor_feature

    return rel



def RILD_stoc(num_search, prev, curr, dilation):

    '''
    Generate stochastic local area with dilation rate for local descriptor

    args:
        num_search: the number of search
        prev : the point set of previous stage.
        curr : the point set of current stage.
        dilation : the dilation rate.
    
    return:
        local_region : (B, N, k, 3)

    '''

    k_stoc = num_search * dilation
    adj1 = tf_util.pairwise_distance_2(curr, prev)
    neg_dists, knn2 = tf.nn.top_k(-adj1, k=k_stoc)
    jumping_idx = tf.range(num_search) * dilation
    
    # 'knn_stoc' is for local area, 'knn2' is for extracting axes 
    knn_stoc = tf.transpose(knn2, (2,0,1))
    knn_stoc = tf.transpose(tf.gather(knn_stoc, jumping_idx), (1,2,0))

    neighbors_axis = tf_util.gather_neighbors(tf.expand_dims(prev, axis=2), knn_stoc, k=k_stoc)
    central_axis   = tf.tile(tf.expand_dims(curr, axis=2), [1, 1, num_search, 1])
    rel        = neighbors_axis - central_axis
    rel_mean   = tf.reduce_mean(rel, axis=2, keep_dims=True)
    local_co   = (tf.matmul(tf.transpose(rel - rel_mean, (0,1,3,2)), rel-rel_mean)) / tf.cast((num_search-1), tf.float32)
    with tf.device('/cpu:0'):
        v, eig_vector = tf.linalg.eigh(local_co)
    eig_vector = tf.reverse(eig_vector, axis=[3])
    eig_vector = minimize_ambi(eig_vector, rel) 
    # Extract stochastic local area (num_search)
    neighbors = tf_util.gather_neighbors(tf.expand_dims(prev, axis=2), knn_stoc, k=num_search)
    central   = tf.tile(tf.expand_dims(curr, axis=2), [1, 1, num_search, 1])
    rel = neighbors - central    
    rel_mean = tf.reduce_mean(rel, axis=2, keep_dims=True)

    return eig_vector, rel, rel_mean




def encoder(prev_set, num_points, num_edges, num_search, batch_size, dilation,
            mlp, is_training, bn_decay, sampling_points, mlp_merge=None, eig_vector=None, 
            centering=None, prev_feature=None, stage=None):

    idxer, fps_points = tf_util.fps(prev_set, sampling_points)
    graph = graph_generation(num_edges, sampling_points, fps_points, batch_size)

    if stage == 1:
        local_non_dil = RILD(num_search, prev_set, fps_points) 
        eig_vector, local_dil, centering = RILD_stoc(num_search, prev_set, fps_points, dilation) 
        
        local_non_dil = tf.matmul((local_non_dil-centering), eig_vector)
        local_dil = tf.matmul((local_dil-centering), eig_vector)

        for i in range(len(mlp)):
            local_non_dil = tf_util.conv2d(local_non_dil, mlp[i], [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope = 'en_non_dil_{}_{}'.format(stage, i), bn_decay=bn_decay)

            local_dil =  tf_util.conv2d(local_dil, mlp[i], [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope = 'en_dil_{}_{}'.format(stage, i), bn_decay=bn_decay)

        en_non_dil = tf.reduce_max(local_non_dil, axis=2, keep_dims=True)
        en_dil = tf.reduce_max(local_dil, axis=2, keep_dims=True)
        net = tf.concat([en_non_dil, en_dil], axis=-1)

        for i in range(len(mlp_merge)):
            net =   tf_util.conv2d(net, mlp_merge[i], [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope = 'net_{}_{}'.format(stage, i), bn_decay=bn_decay)

        gcn = tf.matmul(graph, tf.squeeze(net))
        gcn = tf.expand_dims(gcn, axis=2)
        gcn = tf_util.conv2d(gcn, 1024, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope = 'gcn_{}'.format(stage), bn_decay=bn_decay)

        return net, eig_vector, centering, gcn, fps_points


    else:
        centering = tf_util.gather_point(tf.squeeze(centering), idxer)
        local_non_dil, neighbor_feature = RILD(num_search, prev_set, fps_points, prev_feature)

        eig_vector = get_eigvec(eig_vector, idxer)
        local_non_dil = tf.matmul((local_non_dil - tf.expand_dims(centering, axis=2)), eig_vector)
        net = tf.concat([local_non_dil, neighbor_feature], axis=-1)

        for i in range(len(mlp)):
             net = tf_util.conv2d(net, mlp[i], [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope = 'net_{}_{}'.format(stage, i), bn_decay=bn_decay)


        net = tf.reduce_max(net, axis=2, keep_dims=True)
        gcn = tf.matmul(graph, tf.squeeze(net))
        gcn = tf.expand_dims(gcn, axis=2)
        gcn = tf_util.conv2d(gcn, 1024, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope = 'gcn_{}'.format(stage), bn_decay=bn_decay)

        return net, eig_vector, centering, gcn, fps_points


