import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
import pdb
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import provider
import provider_riconv
import tf_util
import pdb
import time
import scipy
import re
import pickle
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='rinet', help='Model name')
parser.add_argument('--load_dir', required=True, default='rinet')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=251, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--rotation', action='store_true', help='Whether to apply rotation during training [default: False]')
parser.add_argument('--finetune', action='store_true', help='Whether to finetune [default: False]')
parser.add_argument('--checkpoint', default='log/model.ckpt', help='Checkpoint directory to finetune [default: log/model.ckpt]')
parser.add_argument('--num_pool', type=int, default=256, help='Number of pooling [default: 64]')
parser.add_argument('--pool_knn1', type=int, default=64, help='Number of neighbors for lf [default: 128]')
parser.add_argument('--num_votes', type=int, default=12)
parser.add_argument('--so3', action='store_true', default=True, help='Whether training in SO3 setting')
parser.add_argument('--azi', action='store_true', help='Whether training in azimuthal rotation')

FLAGS = parser.parse_args()
LOAD_DIR = FLAGS.load_dir
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
FINETUNE = FLAGS.finetune
CHECKPOINT = FLAGS.checkpoint
NUM_VOTES = FLAGS.num_votes


sys.path.append(os.path.join(BASE_DIR, FLAGS.load_dir))
MODEL = importlib.import_module(FLAGS.model)
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')

print(MODEL_FILE)
LOG_FOUT = open(os.path.join(LOAD_DIR, 'log_test.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

BASE_REG_WEIGHT = 0.001
REG_WEIGHT_DECAY_RATE = 0.5
REG_WEIGHT_DECAY_STEP = float(DECAY_STEP)

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles(
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

# Load data beforehand
KEYS = ['data', 'label']
TRAIN_DATA, TRAIN_LABEL = \
    zip(*[provider.loadDataFile_with_keys(fn, KEYS) for fn in TRAIN_FILES])
TEST_DATA, TEST_LABEL = \
    zip(*[provider.loadDataFile_with_keys(fn, KEYS) for fn in TEST_FILES])

# concatenate batches
TRAIN_DATA = np.concatenate(TRAIN_DATA, axis=0)
TRAIN_LABEL = np.squeeze(np.concatenate(TRAIN_LABEL, axis=0))

TEST_DATA = np.concatenate(TEST_DATA, axis=0)
TEST_LABEL = np.squeeze(np.concatenate(TEST_LABEL, axis=0))


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def get_reg_weight(batch):
    reg_weight = tf.train.exponential_decay(
                      BASE_REG_WEIGHT,
                      batch * BATCH_SIZE,
                      REG_WEIGHT_DECAY_STEP,
                      REG_WEIGHT_DECAY_RATE,
                      staircase=False)
    reg_weight = tf.maximum(reg_weight, 0.00001)
    return reg_weight


def atoi(text):
    return int(text) if text.isdigit() else text 

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, 1024)
            input_graph = tf.placeholder(tf.float32, shape = (BATCH_SIZE, NUM_POINT, NUM_POINT))
            is_training_pl = tf.placeholder(tf.bool, shape=())
            flag_pl = tf.placeholder(tf.int32, shape=())
            flag1 =  tf.placeholder(tf.int32, shape=())
            flag2 =  tf.placeholder(tf.int32, shape=())
            flag3 =  tf.placeholder(tf.int32, shape=())
            dilation = tf.placeholder(tf.int32, shape=())

            gcn1 = tf.placeholder(tf.int32, shape=())
            gcn2 = tf.placeholder(tf.int32, shape=())
            gcn3 = tf.placeholder(tf.int32, shape=())


            print(is_training_pl)
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0, trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points, po, tr, po2, tr2 = MODEL.get_model(pointclouds_pl, FLAGS.num_pool, FLAGS.pool_knn1,
                                     is_training_pl, bn_decay=bn_decay, flag=flag_pl, flag2=flag2, flag3=flag3,gcn1=gcn1, gcn2=gcn2, gcn3=gcn3, dilation=dilation)
            reg_weight = get_reg_weight(batch)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            if FINETUNE:
                """THIS IS NOT WORKING CURRENTLY"""
                finetune_var_names = ['fc1', 'fc2', 'fc3']
                finetuning_vars = [v for v in tf.trainable_variables() if v.name.split('/')[0] in finetune_var_names]
                orig_vars = [v for v in tf.trainable_variables() if v.name.split('/')[0] not in finetune_var_names]
                gvs = optimizer.compute_gradients(loss, [orig_vars, finetuning_vars])
                scaled_gvs = [(grad * 0.1, var) for (grad, var) in gvs[:len(orig_vars)]] + gvs[len(orig_vars):]
                train_op = optimizer.apply_gradients(scaled_gvs, global_step=batch)
            else:
                gvs = optimizer.compute_gradients(loss)
                train_op = optimizer.apply_gradients(gvs, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

            # Load parameters before finetuning
            if FINETUNE:
                variables_to_restore = [v for v in tf.all_variables() if 'rel' not in v.name.split('/')[0]]
                variables_to_restore = [v for v in variables_to_restore if not v.name == 'batch']
                pre_saver = tf.train.Saver(variables_to_restore)
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        saver.restore(sess, LOAD_DIR + '/model.ckpt')

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter(os.path.join(LOAD_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'gvs': gvs,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'flag': flag_pl,
               'flag2': flag2,
               'flag3': flag3,
               'dilation' : dilation,
               'po' : po,
               'tr' : tr,
               'po2' : po2,
               'tr2' : tr2,
               'gcn1' : gcn1,
               'gcn2' : gcn2,
               'gcn3' : gcn3
               }


        acc, cls_avg = eval_one_epoch(sess, ops, test_writer, NUM_VOTES)
        print('Overall accuracy: ', acc)
 
        
def eval_one_epoch(sess, ops, test_writer, num_votes):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    current_data = TEST_DATA[:, 0:NUM_POINT, :]
    current_label = TEST_LABEL
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    pred_conf = np.zeros((0, 40))
 
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    shape_txt = open('data/modelnet40_ply_hdf5_2048/shape_names.txt', 'r')
    label_to_class = shape_txt.read().split('\n')

    flag1 = 64
    flag2 = 32
    flag3 = 16
    gcn1 = 16
    gcn2 = 8
    gcn3 = 4 

    log_string('----------------')
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1 ) * BATCH_SIZE
        batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES))
        for vote_idx in range(NUM_VOTES):
            shuffle = np.arange(NUM_POINT)
            np.random.shuffle(shuffle)
            rot_data = provider_riconv.so3_rotate(current_data[start_idx:end_idx, shuffle, :])
            feed_dict = {
                ops['pointclouds_pl']: rot_data,
                ops['labels_pl']: current_label[start_idx:end_idx],
                ops['is_training_pl']: is_training,
                ops['flag'] : flag1,
                ops['flag2'] : flag2,
                ops['flag3'] : flag3,
                ops['dilation'] : 3,
                ops['gcn1'] : gcn1,
                ops['gcn2'] : gcn2,
                ops['gcn3'] : gcn3
            }

            summary, step, loss_val, pred_val, point_cloud, att_coords1, att_coords2, att_coords3 = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred'], ops['po'], ops['tr'], ops['po2'], ops['tr2']], feed_dict=feed_dict)


            batch_pred_sum += pred_val 

        pred_conf = np.argmax(batch_pred_sum, 1)
        test_writer.add_summary(summary, step)
        correct = np.sum(pred_conf == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            l = current_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_conf[i-start_idx] == l)

    # Handle remaining
    if file_size - num_batches * BATCH_SIZE > 0:

        start_idx = num_batches * BATCH_SIZE
        end_idx = file_size

        input_data = np.zeros((BATCH_SIZE, 1024, 3))
        input_label = np.zeros(BATCH_SIZE)
        input_label[0:end_idx-start_idx] = current_label[start_idx:end_idx]
        batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES))
        for vote_idx in range(NUM_VOTES):
            shuffle = np.arange(NUM_POINT)
            np.random.shuffle(shuffle)
            input_data[0:end_idx - start_idx, ...] = provider_riconv.so3_rotate(current_data[start_idx:end_idx, 0:NUM_POINT, :])

            feed_dict = {
                ops['pointclouds_pl']: input_data,
                ops['labels_pl']: input_label,
                ops['is_training_pl']: is_training,
                ops['flag'] : flag1,
                ops['flag2'] : flag2,
                ops['flag3'] : flag3,
                ops['dilation'] : 3,
                ops['gcn1'] : gcn1,
                ops['gcn2'] : gcn2,
                ops['gcn3'] : gcn3

            }

            summary, step, loss_val, pred_val= sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']], feed_dict=feed_dict)

            batch_pred_sum += pred_val 

        pred_conf = np.argmax(batch_pred_sum, 1)


        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, axis=1)
        correct = np.sum(pred_conf[0:end_idx-start_idx] == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += end_idx - start_idx
        loss_sum += (loss_val * (end_idx - start_idx))
        for i in range(start_idx, end_idx):
            l = current_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_conf[i - start_idx] == l)

            
         
    return (total_correct / float(total_seen)), np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))


if __name__ == "__main__":
    start_time = time.time()
    train()
    end_time = time.time()


    LOG_FOUT.close()
