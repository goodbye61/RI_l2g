import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import provider_riconv
import provider
import tf_util
import pdb
import time
import scipy
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='rinet', help='Model name')
parser.add_argument('--log_dir', default='rinet',  help='Log dir [default: required]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=251, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--rotation', type=bool, default=True, help='Whether to apply rotation during training [default: False]')
parser.add_argument('--finetune', action='store_true', help='Whether to finetune [default: False]')
parser.add_argument('--checkpoint', default='log/model.ckpt', help='Checkpoint directory to finetune [default: log/model.ckpt]')
parser.add_argument('--num_pool', type=int, default=256, help='Number of pooling [default: 64]')
parser.add_argument('--pool_knn1', type=int, default=64, help='Number of neighbors for lf [default: 128]')
parser.add_argument('--so3', action='store_true', help='Whether training in SO3 setting')
parser.add_argument('--azi', action='store_true', default=True, help='Whether training in azimuthal rotation')


FLAGS = parser.parse_args()

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


print(FLAGS.model)
MODEL = importlib.import_module(FLAGS.model) # import network module

MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
print(MODEL_FILE)
LOG_DIR = FLAGS.log_dir
#if not os.path.exists(FLAGS.vis_dir): os.makedirs(FLAGS.vis_dir)
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_new.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
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


def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)


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


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            flag_pl = tf.placeholder(tf.int32, shape=())
            flag2   = tf.placeholder(tf.int32, shape=())
            flag3   = tf.placeholder(tf.int32, shape=())
            dilation = tf.placeholder(tf.int32, shape=())

            gcn1 = tf.placeholder(tf.int32, shape=())
            gcn2 = tf.placeholder(tf.int32, shape=())
            gcn3 = tf.placeholder(tf.int32, shape=())


            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0, trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, FLAGS.num_pool, FLAGS.pool_knn1,
                                     is_training_pl, bn_decay=bn_decay, flag=flag_pl, flag2=flag2, flag3=flag3, gcn1=gcn1, gcn2=gcn2, gcn3=gcn3, dilation=dilation)
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
                #added
                #gvs = [(tf.clip_by_value(grad, -0.5, 0.5),var) for grad, var in gvs]
                train_op = optimizer.apply_gradients(gvs, global_step=batch)
            # train_op = optimizer.minimize(loss, global_step=batch)
            
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

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})
        # Load paramters from checkpoint
        if FINETUNE:
            pre_saver.restore(sess, CHECKPOINT)

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
               'flag2':flag2,
               'flag3':flag3,
               'gcn1' : gcn1,
               'gcn2' : gcn2,
               'gcn3' : gcn3,
               'dilation' : dilation
               }

        best_acc = 0
        best_cls_avg = 0 
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer, epoch)
            acc, cls_avg_acc = eval_one_epoch(sess, ops, test_writer)
            if acc > best_acc:
                best_acc = acc
                best_cls_avg = cls_avg_acc
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)
 
            print('The best acc from now: ', best_acc)

        log_string("The best acc : %s" % best_acc)
        print('The corresponding cls avg acc : ', cls_avg_acc)
        print('The model is saved in ', FLAGS.log_dir)
        print(' corresponding model is : ', FLAGS.model)



def train_one_epoch(sess, ops, train_writer, epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    train_file_idxs = np.arange(0, len(TRAIN_DATA))
    current_data = TRAIN_DATA[:, 0:NUM_POINT, :]

    current_label = TRAIN_LABEL

    current_data, current_label, shuffle_idx = provider.shuffle_data(current_data, np.squeeze(current_label))
    current_label = np.squeeze(current_label)

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    shape_txt = open('data/modelnet40_ply_hdf5_2048/shape_names.txt', 'r')
    label_to_class = shape_txt.read().split('\n')

    for batch_idx in range(num_batches):
        if batch_idx % 10 == 0:
            log_string('Current batch / total_batch_num: {} / {}'.format(batch_idx, num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        if FLAGS.rotation:
            rotated_data = current_data[start_idx:end_idx, :, :]
            if FLAGS.so3:
                rotated_data = provider_riconv.so3_rotate(rotated_data)
            elif FLAGS.azi:
                rotated_data  = provider_riconv.azi_rotate(rotated_data)
        else:
            rotated_data = current_data[start_idx : end_idx, :, :]


        flag= int(np.random.randint(32, 97, 1))
        flag2 = int(np.random.randint(16, 49, 1)) 
        flag3 = int(np.random.randint(8, 25, 1))  
        gcn1 = int(np.random.randint(8, 25, 1))   
        gcn2 = int(np.random.randint(4, 13, 1))   
        gcn3 = int(np.random.randint(2, 9, 1))  
        dilation = int(np.random.randint(2, 5))
        
        idx = np.arange(1024)
        np.random.shuffle(idx)
        rotated_data = rotated_data[:, idx, :]


        feed_dict = {
            ops['pointclouds_pl']: rotated_data,
            ops['labels_pl']: current_label[start_idx:end_idx],
            ops['is_training_pl']: is_training,
            ops['flag'] : flag,
            ops['flag2'] : flag2,
            ops['flag3'] : flag3,
            ops['gcn1'] : gcn1,
            ops['gcn2'] : gcn2,
            ops['gcn3'] : gcn3,
            ops['dilation'] : dilation
        }

        summary, step, _, loss_val, pred_val =  sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'],
                                                         ops['pred']], feed_dict=feed_dict)

        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, axis=1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        if (loss_val >= 50):
            pdb.set_trace()
        if np.isnan(loss_val):
            pdb.set_trace()
            continue

        loss_sum += loss_val

    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    current_data = TEST_DATA[:, 0:NUM_POINT, :]
    current_label = TEST_LABEL
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    flag = 64
    flag2 = 32
    flag3 = 16
    gcn1 = 16
    gcn2 = 8
    gcn3 = 4
    dilation = 3 

    log_string('----------------')
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {
            ops['pointclouds_pl']: provider_riconv.so3_rotate(current_data[start_idx:end_idx, :, :]),
            ops['labels_pl']: current_label[start_idx:end_idx],
            ops['is_training_pl']: is_training,
            ops['flag'] : flag,
            ops['flag2'] : flag2,
            ops['flag3'] : flag3,
            ops['gcn1'] : gcn1,
            ops['gcn2'] : gcn2,
            ops['gcn3'] : gcn3,
            ops['dilation'] : dilation
        }

        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, axis=1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            l = current_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i-start_idx] == l)

    # Handle remaining
    if file_size - num_batches * BATCH_SIZE > 0:

        start_idx = num_batches * BATCH_SIZE
        end_idx = file_size

        input_data = np.random.normal(size=(BATCH_SIZE, NUM_POINT, 3))
        input_data[0:end_idx-start_idx, ...] = current_data[start_idx:end_idx, 0:NUM_POINT, :]

        input_label = np.zeros(BATCH_SIZE)
        input_label[0:end_idx-start_idx] = current_label[start_idx:end_idx]

        feed_dict = {
            ops['pointclouds_pl']: provider_riconv.so3_rotate(input_data),
            ops['labels_pl']: input_label,
            ops['is_training_pl']: is_training,
            ops['flag'] : flag,
            ops['flag2'] : 32,
            ops['flag3'] : 16,
            ops['gcn1'] : gcn1,
            ops['gcn2'] : gcn2,
            ops['gcn3'] : gcn3,
            ops['dilation']: dilation
            
        }

        summary, step, loss_val, pred_val= sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, axis=1)
        correct = np.sum(pred_val[0:end_idx-start_idx] == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += end_idx - start_idx
        loss_sum += (loss_val * (end_idx - start_idx))
        for i in range(start_idx, end_idx):
            l = current_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i - start_idx] == l)

            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
         
    return (total_correct / float(total_seen)),  (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)))


if __name__ == "__main__":
    start_time = time.time()
    train()
    end_time = time.time()

    print('elapsed time : ', end_time - start_time)

    LOG_FOUT.close()
