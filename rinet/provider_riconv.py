import os
import sys
import numpy as np
import h5py
import pdb
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    point_num = data.shape[1]
    idx = np.arange(point_num)
    np.random.shuffle(idx)
    data = data[:,idx,:]

    idx = np.arange(len(labels))
    np.random.shuffle(idx)

    return data[idx, ...], labels[idx]


def so3_rotate(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle_A = np.random.uniform() * 2 * np.pi
        rotation_angle_B = np.random.uniform() * 2 * np.pi
        rotation_angle_C = np.random.uniform() * 2 * np.pi

        cosval_A = np.cos(rotation_angle_A)
        sinval_A = np.sin(rotation_angle_A)
        cosval_B = np.cos(rotation_angle_B)
        sinval_B = np.sin(rotation_angle_B)
        cosval_C = np.cos(rotation_angle_C)
        sinval_C = np.sin(rotation_angle_C)
        rotation_matrix = np.array([[cosval_B*cosval_C, -cosval_B*sinval_C, sinval_B],
                                    [sinval_A*sinval_B*cosval_C+cosval_A*sinval_C, -sinval_A*sinval_B*sinval_C+cosval_A*cosval_C, -sinval_A*cosval_B],
                                    [-cosval_A*sinval_B*cosval_C+sinval_A*sinval_C, cosval_A*sinval_B*sinval_C+sinval_A*cosval_C, cosval_A*cosval_B]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def so3_rotate_with_normal(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros((batch_data.shape[0], batch_data.shape[1], 3), dtype=np.float32)
    rotated_normal = np.zeros((batch_data.shape[0], batch_data.shape[1], 3), dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle_A = np.random.uniform() * 2 * np.pi
        rotation_angle_B = np.random.uniform() * 2 * np.pi
        rotation_angle_C = np.random.uniform() * 2 * np.pi

        cosval_A = np.cos(rotation_angle_A)
        sinval_A = np.sin(rotation_angle_A)
        cosval_B = np.cos(rotation_angle_B)
        sinval_B = np.sin(rotation_angle_B)
        cosval_C = np.cos(rotation_angle_C)
        sinval_C = np.sin(rotation_angle_C)
        rotation_matrix = np.array([[cosval_B*cosval_C, -cosval_B*sinval_C, sinval_B],
                                    [sinval_A*sinval_B*cosval_C+cosval_A*sinval_C, -sinval_A*sinval_B*sinval_C+cosval_A*cosval_C, -sinval_A*cosval_B],
                                    [-cosval_A*sinval_B*cosval_C+sinval_A*sinval_C, cosval_A*sinval_B*sinval_C+sinval_A*cosval_C, cosval_A*cosval_B]])

        shape_pc = batch_data[k, :, :3]
        shape_nm = batch_data[k, :, 3:]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_normal[k, ...] = np.dot(shape_nm.reshape((-1,3)), rotation_matrix)
    rotated_data = np.concatenate((rotated_data, rotated_normal), axis=-1)
    return rotated_data

def azi_rotate_with_normal(batch_data, rotation_axis="z"):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    '''
    if np.ndim(batch_data) != 6:
        raise ValueError("np.ndim(batch_data) != 3, must be (b, n, 3)")
    if batch_data.shape[2] != 6:
        raise ValueError("batch_data.shape[2] != 3, must be (x, y, z)")
    '''

    rotated_data = np.zeros((batch_data.shape[0], batch_data.shape[1], 3), dtype=np.float32)
    rotated_normal = np.zeros((batch_data.shape[0], batch_data.shape[1], 3), dtype=np.float32)

    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        if rotation_axis == "x":
            rotation_matrix = np.array(
                [[1, 0, 0], [0, cosval, sinval], [0, -sinval, cosval]]
            )
        elif rotation_axis == "y":
            rotation_matrix = np.array(
                [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
            )
        elif rotation_axis == "z":
            rotation_matrix = np.array(
                [[cosval, sinval, 0], [-sinval, cosval, 0], [0, 0, 1]]
            )
        else:
            raise ValueError("Wrong rotation axis")
        shape_pc = batch_data[k, :, :3]
        shape_nm = batch_data[k, :, 3:]
        

        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_normal[k, ...] = np.dot(shape_nm.reshape((-1,3)), rotation_matrix)
    
    rotated_data = np.concatenate((rotated_data, rotated_normal) , axis=-1)

    return rotated_data


def rot_rand():
    R = np.random.rand(3, 3)
    U, S, V = np.linalg.svd(R)
    det = np.linalg.det(U.dot(V))

    return U.dot(V).dot([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, det]])

def rot_rand_point_cloud(point_cloud):
    
    R = np.random.randn(3, 3)
    U, S, V = np.linalg.svd(R)
    det = np.linalg.det(U.dot(V))
    rotate = U.dot(V).dot([[ 1, 0, 0],
                           [ 0, 1, 0],
                           [0, 0, det]])

    num_batch = np.shape(point_cloud)[0]
    num_point = np.shape(point_cloud)[1]
    rotate = np.tile(np.expand_dims(rotate, axis=0), [num_batch, 1, 1])

    return np.matmul(point_cloud, rotate)




def azi_rotate(batch_data, rotation_axis="z"):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    if np.ndim(batch_data) != 3:
        raise ValueError("np.ndim(batch_data) != 3, must be (b, n, 3)")
    if batch_data.shape[2] != 3:
        raise ValueError("batch_data.shape[2] != 3, must be (x, y, z)")

    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        if rotation_axis == "x":
            rotation_matrix = np.array(
                [[1, 0, 0], [0, cosval, sinval], [0, -sinval, cosval]]
            )
        elif rotation_axis == "y":
            rotation_matrix = np.array(
                [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
            )
        elif rotation_axis == "z":
            rotation_matrix = np.array(
                [[cosval, sinval, 0], [-sinval, cosval, 0], [0, 0, 1]]
            )
        else:
            raise ValueError("Wrong rotation axis")
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_with_normal(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    pointcloud_data = batch_data[:, :, 0:3]
    normal_data = batch_data[:, :, 3:6]
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    rotated_data_pc = np.zeros(pointcloud_data.shape, dtype=np.float32)
    rotated_data_nor = np.zeros(normal_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = pointcloud_data[k, ...]
        shape_nor = normal_data[k, ...]
        rotated_data_pc[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data_nor[k, ...] = np.dot(shape_nor.reshape((-1, 3)), rotation_matrix)
    rotated_data = np.concatenate((rotated_data_pc, rotated_data_nor), 2)
    return rotated_data




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

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)


def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)


def loadDataFile_with_normal(filename):
    f = h5py.File(filename)
    data = f['data'][:]
    label = f['label'][:]
    normal = f['normal'][:]
    return (data, label, normal)
