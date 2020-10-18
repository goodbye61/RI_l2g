import os
import sys
import numpy as np
import h5py
import pdb
import tensorflow as tf
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:,idx,:]


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data




def rotate_point_cloud_with_normal(batch_xyz_normal):
    ''' Randomly rotate XYZ, normal point cloud.
        Input:
            batch_xyz_normal: B,N,6, first three channels are XYZ, last 3 all normal
        Output:
            B,N,6, rotated XYZ, normal point cloud
    '''
    for k in xrange(batch_xyz_normal.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_xyz_normal[k,:,0:3]
        shape_normal = batch_xyz_normal[k,:,3:6]
        batch_xyz_normal[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        batch_xyz_normal[k,:,3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return batch_xyz_normal


def rotate_perturbation_point_cloud_with_normal(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx6 array, original batch of point clouds and point normals
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in xrange(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k,:,0:3]
        shape_normal = batch_data[k,:,3:6]
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
        rotated_data[k,:,3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
    return rotated_data



def so3_rotate(batch_data):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    rotated_matrix = np.zeros(((batch_data.shape)[0], 3,3), dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        angles = np.random.uniform(-np.pi, np.pi, 3)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), np.transpose(R))
        rotated_matrix[k, ...] = R
    return rotated_data, rotated_matrix




def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18, seed = None):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #np.random.seed(seed)
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def rotate_unit_z(unit_z, angle_sigma=0.26, angle_clip=0.78):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                     [0,np.cos(angles[0]),-np.sin(angles[0])],
                     [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                      [0,1,0],
                     [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
               [np.sin(angles[2]),np.cos(angles[2]),0],
               [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    #shape_pc = batch_data[k, ...]
    #rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    R = np.expand_dims(np.asarray(R, np.float32), axis=0)
    rot_z_vec = tf.matmul(unit_z, np.transpose(R, (0,2,1))) 
    return rot_z_vec





def tf_rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18, seed = None):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    #rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    rotated_data = [] 
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        #rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
        #pdb.set_trace()
        R = np.asarray(R, np.float32)
        #rotated_data[k, ... ] = np.dot(tf.reshape(shape_pc, (-1,3)) ,R)
        rotated_data.append(tf.matmul(shape_pc, np.transpose(R)))
    
    rotated_data = tf.stack(rotated_data)
    return rotated_data

def random_rot(batch_data):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = []
    rotated_matrix = [] 
    for k in range(batch_data.shape[0]):
        #angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        angles = np.random.uniform(-np.pi, np.pi, 3)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        R = np.asarray(R, np.float32)
        rotated_data.append(tf.matmul(shape_pc, np.transpose(R)))
        #rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), np.transpose(R))
        #artated_matrix[k, ...] = R
        rotated_matrix.append(R)
    rotated_data = tf.stack(rotated_data)
    rotated_matrix = tf.stack(rotated_matrix)
    return rotated_data, rotated_matrix




def tf_random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """

    scaled_data = [] 

    #pdb.set_trace()
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    
    for batch_index in range(B):
        #batch_data[batch_index,:,:] *= scales[batch_index]
        scaled_data.append(batch_data[batch_index, :, :] * scales[batch_index])

    #return batch_data
    return tf.stack(scaled_data)



def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data


def tf_shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    shifted_data = [] 

    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        #batch_data[batch_index,:,:] += shifts[batch_index,:]
        shifted_data.append(batch_data[batch_index, :, :] + shifts[batch_index,:])
    #return batch_data
    return tf.stack(shifted_data)


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx6 array, original batch of point clouds with normal
          scalar, angle of rotation
        Return:
          BxNx6 array, rotated batch of point clouds iwth normal
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in xrange(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k,:,0:3]
        shape_normal = batch_data[k,:,3:6]
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k,:,3:6] = np.dot(shape_normal.reshape((-1,3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_axis_and_angle(batch_data, axis, rotation_angle):
    """Rotate the point cloud along given direction with certain angle
    Input:
        batch_data: (batch_size, ..., num_point, 3)
        axis: (ux, uy, uz) where ux^2 + uy^2 + uz^2 = 1
        rotation_angle: angle in randians
    Return: (batch_size, ..., num_point, 3) the same size as batch_data
    """
    # rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    ux, uy, uz = axis
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([
            [cosval+ux*ux*(1-cosval), ux*uy*(1-cosval)-uz*sinval, ux*uz*(1-cosval)+uy*sinval],
            [uy*ux*(1-cosval)+uz*sinval, cosval+uy*uy*(1-cosval), uy*uz*(1-cosval)-ux*sinval],
            [uz*ux*(1-cosval)-uy*sinval, uz*uy*(1-cosval)+ux*sinval, cosval+uz*uz*(1-cosval)]])
    rotated_data = np.matmul(batch_data, rotation_matrix)
    return rotated_data


def gaussian_rbf(x):
    epsilon = 2 
    return tf.exp(-tf.pow(epsilon*x, 2))




def eigen_rotation(batch_data, x_rad, y_rad, z_rad):
    '''
    batch_data : (batch_size, points, dim =3)
    
    '''
    #rotated_data = tf.zeros(batch_data.shape, dtype=np.float32)
    rotated_data = []
    # point_cloud : (batch, 2048, 3)
    # transformation (batch, 3 x 3 ) 
    
    for k in range(batch_data.shape[0]):
        Rx = tf.concat([[1.0,0.0,0.0],
                        [0.0, tf.cos(x_rad[k]), -tf.sin(x_rad[k])], 
                        [0.0, tf.sin(x_rad[k]), tf.cos(x_rad[k])]], axis=-1)
        
        Ry = tf.concat([[tf.cos(y_rad[k]), 0.0, tf.sin(y_rad[k])],
                        [0.0,1.0,0.0],
                        [-tf.sin(y_rad[k]), 0.0, tf.cos(y_rad[k])]],axis=-1)

        Rz = tf.concat([[tf.cos(z_rad[k]), -tf.sin(z_rad[k]), 0.0],
                        [tf.sin(z_rad[k]), tf.cos(z_rad[k]), 0.0], 
                        [0.0,0.0,1.0]],axis=-1)
        
        Rx = tf.reshape(Rx, [3,-1])
        Ry = tf.reshape(Ry, [3,-1])
        Rz = tf.reshape(Rz, [3,-1])


        #R = np.dot(Rz, np.dot(Ry, Rx))
        R = tf.matmul(Rz, tf.matmul(Ry, Rx))
        shape_pc = batch_data[k, ... ]
        #rotated_data[k, ...] = tf.matmul(shape_pc, R)
        rotated_data.append(tf.matmul(shape_pc, R))

    rotated_data = tf.stack(rotated_data)

    
    return rotated_data




def eigen_to_z_axis(batch_data, dots_z, cross_vec, num_point):

    # Should matching the eigen-vector to z-axis 
    rotated_data = [] 
    epsilon = tf.constant(1e-6, tf.float32)
    R_set = [] 
    eye = np.eye(3,dtype=np.float32) 
    for k in range(batch_data.shape[0]):
        vec = tf.squeeze(cross_vec[k])
        v_x = tf.concat([[0, -vec[2], vec[1]], 
                        [vec[2], 0, -vec[0]],
                        [-vec[1], vec[0], 0]], axis=-1)

    
        v_x = tf.reshape(v_x, [3, -1])
        #v_x_2 = tf.matmul(v_x, tf.transpose(v_x))
        v_x_x = tf.matmul(v_x, v_x)
        #v_x_x = np.dot(v_x, v_x)
        
        const = 1.0 / (1.0 + dots_z[k] + epsilon)
        #const = tf.clip_by_value(const, 0.0, 199.0)

        
        R = eye + v_x + v_x_x * (const)
        R = tf.clip_by_value(R, -1.0, 1.0)
        shape_pc = batch_data[k, ...]
        #shape_pc = tf.expand_dims(shape_pc, axis=-1)
        rot = tf.matmul(shape_pc, tf.transpose(R))
        #rot = tf.matmul(R, tf.transpose(shape_pc, (0,2,1)))
        rotated_data.append(tf.squeeze(rot))
        R_set.append(R)

    rotated_data = tf.stack(rotated_data)
    R_set = tf.stack(R_set)

    return rotated_data, R_set

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc


def octahedral_group_of_point_cloud(batch_data):
    """Rotate the point cloud according to octahedral equivalence
    Input:
        batch_data: (batch_size, num_point, 3)
    Return: (batch_size, 24, num_point, 3)
    """
    sqrt_2 = np.sqrt(2.0) / 2.0
    sqrt_3 = np.sqrt(3.0) / 3.0
    octa_group_configs = [((1.0, 0.0, 0.0), 0.0),
                          ((1.0, 0.0, 0.0), 0.5*np.pi), ((1.0, 0.0, 0.0), 1.0*np.pi), ((1.0, 0.0, 0.0), 1.5*np.pi),
                          ((0.0, 1.0, 0.0), 0.5*np.pi), ((0.0, 1.0, 0.0), 1.0*np.pi), ((0.0, 1.0, 0.0), 1.5*np.pi),
                          ((0.0, 0.0, 1.0), 0.5*np.pi), ((0.0, 0.0, 1.0), 1.0*np.pi), ((0.0, 0.0, 1.0), 1.5*np.pi),
                          ((0.0, sqrt_2, sqrt_2), np.pi), ((0.0, sqrt_2, -sqrt_2), np.pi),
                          ((sqrt_2, 0.0, sqrt_2), np.pi), ((-sqrt_2, 0.0, sqrt_2), np.pi),
                          ((sqrt_2, sqrt_2, 0.0), np.pi), ((sqrt_2, -sqrt_2, 0.0), np.pi),
                          ((sqrt_3, sqrt_3, sqrt_3), 2.0*np.pi/3.0), ((sqrt_3, sqrt_3, sqrt_3), 4.0*np.pi/3.0),
                          ((-sqrt_3, sqrt_3, sqrt_3), 2.0*np.pi/3.0), ((-sqrt_3, sqrt_3, sqrt_3), 4.0*np.pi/3.0),
                          ((sqrt_3, -sqrt_3, sqrt_3), 2.0*np.pi/3.0), ((sqrt_3, -sqrt_3, sqrt_3), 4.0*np.pi/3.0),
                          ((sqrt_3, sqrt_3, -sqrt_3), 2.0*np.pi/3.0), ((sqrt_3, sqrt_3, -sqrt_3), 4.0*np.pi/3.0),
                         ]
    batch_size, num_point, _ = batch_data.shape
    octa_batch_data = np.zeros((batch_size, 24, num_point, 3), dtype=np.float32)

    for i in range(24):
        u, angle = octa_group_configs[i]
        octa_batch_data[:, i, :, :] = rotate_point_cloud_by_axis_and_angle(batch_data, u, angle)
    return octa_batch_data

def octahedral_rotate_point_cloud(batch_data):
    """Rotate the point cloud according to octahedral equivalence
    Input:
        batch_data: (batch_size, num_point, 3)
    Return: (batch_size, num_point, 3)
    """
    sqrt_2 = np.sqrt(2.0) / 2.0
    sqrt_3 = np.sqrt(3.0) / 3.0
    octa_group_configs = [((1.0, 0.0, 0.0), 0.0),
                          ((1.0, 0.0, 0.0), 0.5*np.pi), ((1.0, 0.0, 0.0), 1.0*np.pi), ((1.0, 0.0, 0.0), 1.5*np.pi),
                          ((0.0, 1.0, 0.0), 0.5*np.pi), ((0.0, 1.0, 0.0), 1.0*np.pi), ((0.0, 1.0, 0.0), 1.5*np.pi),
                          ((0.0, 0.0, 1.0), 0.5*np.pi), ((0.0, 0.0, 1.0), 1.0*np.pi), ((0.0, 0.0, 1.0), 1.5*np.pi),
                          ((0.0, sqrt_2, sqrt_2), np.pi), ((0.0, sqrt_2, -sqrt_2), np.pi),
                          ((sqrt_2, 0.0, sqrt_2), np.pi), ((-sqrt_2, 0.0, sqrt_2), np.pi),
                          ((sqrt_2, sqrt_2, 0.0), np.pi), ((sqrt_2, -sqrt_2, 0.0), np.pi),
                          ((sqrt_3, sqrt_3, sqrt_3), 2.0*np.pi/3.0), ((sqrt_3, sqrt_3, sqrt_3), 4.0*np.pi/3.0),
                          ((-sqrt_3, sqrt_3, sqrt_3), 2.0*np.pi/3.0), ((-sqrt_3, sqrt_3, sqrt_3), 4.0*np.pi/3.0),
                          ((sqrt_3, -sqrt_3, sqrt_3), 2.0*np.pi/3.0), ((sqrt_3, -sqrt_3, sqrt_3), 4.0*np.pi/3.0),
                          ((sqrt_3, sqrt_3, -sqrt_3), 2.0*np.pi/3.0), ((sqrt_3, sqrt_3, -sqrt_3), 4.0*np.pi/3.0),
                         ]
    batch_size, num_point, _ = batch_data.shape
    rotated_data = np.zeros((batch_size, num_point, 3), dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_config = octa_group_configs[np.random.randint(len(octa_group_configs))]
        u, angle = rotation_config
        rotated_data[k, ...] = rotate_point_cloud_by_axis_and_angle(batch_data[k, ...], u, angle)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_normal(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    normal = f['normal'][:]
    data_normal = np.concatenate([data, normal], axis=2)
    return (data_normal, label)

def loadDataFile_with_normal(filename):
    return load_h5_data_label_normal(filename)

def load_h5_data_label_graph(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    graph = f['graph'][:]
    return (data, graph, label)

def loadDataFile_with_graph(filename):
    return load_h5_data_label_graph(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)

def load_h5_data_with_keys(h5_filename, keys):
    f = h5py.File(h5_filename)
    return [f[key][:] for key in keys]

def loadDataFile_with_keys(filename, keys):
    return load_h5_data_with_keys(filename, keys)


