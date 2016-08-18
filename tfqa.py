from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import cPickle
import os
import sys
import time
from collections import defaultdict
import subprocess
from collections import defaultdict

import numpy as np
import tensorflow as tf


flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "mode", "TRAIN",
    "The data set used. Possible options are: TRAIN, TRAIN_ALL.")
flags.DEFINE_string("config", "default", "The config names")
FLAGS = flags.FLAGS

class DefaultConfig(object):
    """The hyper parameters."""
    n_outs = 2
    n_epochs = 25
    batch_size = 50
    learning_rate = 0.1
    max_norm = 0
    ndim = 50
    dropout_rate = 0.5
    nkernels = 100
    q_k_max = 1
    a_k_max = 1
    q_filter_widths = 5
    a_filter_widths = 5
    init_scale = 0.25

    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5


class DeepQAModel(object):
    def __init__(self, is_training, config):
        self.q_data = tf.placeholder(tf.int32, [None, 33])
        self.a_data = tf.placeholder(tf.int32, [None, 40])
        self.q_overlap = tf.placeholder(tf.int32, [None, 33])
        self.a_overlap = tf.placeholder(tf.int32, [None, 40])
    
        self.embedding = tf.placeholder(tf.float32, [None, 50])
        self.embedding_overlap = tf.placeholder(tf.float32, [None, 5])
        with tf.device("/cpu:0"):
            self.q_inputs = tf.nn.embedding_lookup(self.embedding, self.q_data)
            self.a_inputs = tf.nn.embedding_lookup(self.embedding, self.a_data)
            self.q_overlp = tf.nn.embedding_lookup(self.embedding_overlap, self.q_overlap)
            self.a_overlp = tf.nn.embedding_lookup(self.embedding_overlap, self.a_overlap)


        # expand dims to [batch, height, width, channel] = [#sentence, #word, dim, 1]
        self.q_tf_in = tf.expand_dims(self.q_inputs, 3)
        self.a_tf_in = tf.expand_dims(self.a_inputs, 3)
        
        # define the filter: [filter_height, filter_width, in_channels, out_channels]
        q_filter = tf.get_variable("q_filter", [config.q_filter_widths, 
                                                     config.ndim,
                                                     1,
                                                     config.nkernels])
        q_stride = [1, 1, config.ndim, 1]
        q_conv = tf.nn.conv2d(input=self.q_tf_in, filter=q_filter, strides=q_stride, padding="SAME")
        q_bias = tf.get_variable("q_bias", [config.nkernels], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
        q_act = tf.tanh(tf.nn.bias_add(q_conv, q_bias))
        q_pool = tf.nn.max_pool(value=q_act, ksize=[1, q_act.get_shape()[1], 1, 1], strides=[1, q_act.get_shape()[1], 1, 1], padding='SAME')
        self.q_feat = tf.squeeze(q_pool)
         

        # define the filter: [filter_height, filter_width, in_channels, out_channels]
        a_filter = tf.get_variable("a_filter", [config.a_filter_widths, 
                                                     config.ndim,
                                                     1,
                                                     config.nkernels])
        a_stride = [1, 1, config.ndim, 1]
        a_conv = tf.nn.conv2d(input=self.a_tf_in, filter=a_filter, strides=a_stride, padding="SAME")
        a_bias = tf.get_variable("a_bias", [config.nkernels], initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))
        a_act = tf.tanh(tf.nn.bias_add(a_conv, a_bias))
        a_pool = tf.nn.max_pool(value=a_act, ksize=[1, a_act.get_shape()[1], 1, 1], strides=[1, a_act.get_shape()[1], 1, 1], padding='SAME')
        self.a_feat = tf.squeeze(a_pool)

        sim_mat = tf.get_variable("sim_mat", [config.nkernels, config.nkernels])
        self.sim_value = tf.squeeze(tf.batch_matmul(tf.expand_dims(tf.matmul(self.q_feat, sim_mat), 1), tf.expand_dims(self.a_feat, 2)), [2])
        self.concat = tf.concat(1, [self.q_feat, self.sim_value, self.a_feat])
        
        ndim = config.nkernels * 2 + 1 
        W1 = tf.get_variable("W1", [ndim, ndim], initializer=tf.random_uniform_initializer(minval=-np.sqrt(6.0/ndim), maxval=np.sqrt(6.0/ndim), dtype=tf.float32))
        b1 = tf.get_variable("b1", [ndim], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
        l1_out = tf.tanh(tf.nn.bias_add(tf.matmul(self.concat, W1), b1))

        W2 = tf.get_variable("W2", [ndim, 2], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
        b2 = tf.get_variable("b2", [2], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

        logits = tf.nn.bias_add(tf.matmul(l1_out, W2), b2)
        self.labels = tf.placeholder(tf.int32, [None])
        onehot_labels = tf.one_hot(self.labels, 2)
        self.loss = tf.contrib.losses.softmax_cross_entropy(logits, onehot_labels)

        # apply L2 regularization
        self.reg_loss = self.loss + tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-2), [q_filter, q_bias, a_filter, a_bias, sim_mat, W1, b1, W2, b2])

        self.y_hat = tf.argmax(tf.nn.softmax(logits),1)
        self.y_score = tf.nn.softmax(logits)
        correct_prediction = tf.equal(tf.cast(self.y_hat, tf.int32), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if not is_training:
            return

        self.train_step = tf.train.AdadeltaOptimizer(learning_rate=config.learning_rate, rho=0.95, epsilon=1e-6).minimize(self.reg_loss)

                
def get_config(): 
    if FLAGS.config == "default":
        return DefaultConfig()
    else:
        raise ValueError("Invalid config name: %s", FLAGS.config)


def batch_iter(data, batch_size, shuffle=False):
    data_size = len(data[0])
    num_batches = int(data_size / batch_size)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = [data[i][shuffle_indices] for i in range(len(data))]
    else:
        shuffled_data = data
    
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield [shuffled_data[i][start_index:end_index] for i in range(len(shuffled_data))]

def map_score(qids, labels, preds):
    qid2cand = defaultdict(list)
    for qid, label, pred in zip(qids, labels, preds):
        qid2cand[qid].append((pred, label))

    average_precs = []
    for qid, candidates in qid2cand.iteritems():
        average_prec = 0
        running_correct_count = 0
        for i, (score, label) in enumerate(sorted(candidates, reverse=True), 1):
            if label > 0:
                running_correct_count += 1
                average_prec += float(running_correct_count) / i
        average_precs.append(average_prec / (running_correct_count + 1e-6))
    map_score = sum(average_precs) / len(average_precs)
    return map_score


def main(_):
    mode = FLAGS.mode
    if not mode in ['TRAIN', 'TRAIN-ALL']:
        print("ERROR! The two possible training settings are: ['TRAIN', 'TRAIN-ALL']")
        sys.exit(1)

    # load the training, dev and test data 
    data_dir = "data/" + mode
    if mode in ['TRAIN-ALL']:
        q_train = np.load(os.path.join(data_dir, 'train-all.questions.npy'))
        a_train = np.load(os.path.join(data_dir, 'train-all.answers.npy'))
        q_overlap_train = np.load(os.path.join(data_dir, 'train-all.q_overlap_indices.npy'))
        a_overlap_train = np.load(os.path.join(data_dir, 'train-all.a_overlap_indices.npy'))
        y_train = np.load(os.path.join(data_dir, 'train-all.labels.npy'))
    else:
        q_train = np.load(os.path.join(data_dir, 'train.questions.npy'))
        a_train = np.load(os.path.join(data_dir, 'train.answers.npy'))
        q_overlap_train = np.load(os.path.join(data_dir, 'train.q_overlap_indices.npy'))
        a_overlap_train = np.load(os.path.join(data_dir, 'train.a_overlap_indices.npy'))
        y_train = np.load(os.path.join(data_dir, 'train.labels.npy'))
    
    q_dev = np.load(os.path.join(data_dir, 'dev.questions.npy'))
    a_dev = np.load(os.path.join(data_dir, 'dev.answers.npy'))
    q_overlap_dev = np.load(os.path.join(data_dir, 'dev.q_overlap_indices.npy'))
    a_overlap_dev = np.load(os.path.join(data_dir, 'dev.a_overlap_indices.npy'))
    y_dev = np.load(os.path.join(data_dir, 'dev.labels.npy'))
    qids_dev = np.load(os.path.join(data_dir, 'dev.qids.npy'))

    q_test = np.load(os.path.join(data_dir, 'test.questions.npy'))
    a_test = np.load(os.path.join(data_dir, 'test.answers.npy'))
    q_overlap_test = np.load(os.path.join(data_dir, 'test.q_overlap_indices.npy'))
    a_overlap_test = np.load(os.path.join(data_dir, 'test.a_overlap_indices.npy'))
    y_test = np.load(os.path.join(data_dir, 'test.labels.npy'))
    qids_test = np.load(os.path.join(data_dir, 'test.qids.npy'))

    print("overlap:", q_overlap_train[-10:, -10:], a_overlap_train[-10:, -10:])    

    print('y_train:', np.unique(y_train, return_counts=True), y_train.shape, y_train[:10])
    print('y_dev:', np.unique(y_dev, return_counts=True))
    print('y_test:', np.unique(y_test, return_counts=True))

    #y_train2 = np.vstack([y_train, 1-y_train]).T
    print('q_train:', q_train.shape)
    print('q_dev', q_dev.shape)
    print('q_test', q_test.shape)
    print('qids', qids_dev.shape, qids_test.shape)

    print('a_train:', a_train.shape)
    print('a_dev:', a_dev.shape)
    print('a_test:', a_test.shape)

    ## Get the word embeddings
    q_max_sent_size = q_train.shape[1]
    a_max_sent_size = a_train.shape[1]
    numpy_rng = np.random.RandomState(123)

    ndim = 5
    print("Generating random vocabulary for word overlap indicator features with dim:", ndim)
    dummy_word_id = np.max(a_overlap_train)
    print("Gaussian")
    vocab_emb_overlap = numpy_rng.randn(dummy_word_id+1, ndim) * 0.25
    vocab_emb_overlap[-1] = 0

    # Load word2vec embeddings
    fname = os.path.join(data_dir, 'emb_aquaint+wiki.txt.gz.ndim=50.bin.npy')

    print("Loading word embeddings from", fname)
    vocab_emb = np.load(fname)
    ndim = vocab_emb.shape[1]
    dummpy_word_idx = np.max(a_train)
    print("Word embedding matrix size:", vocab_emb.shape)


    config = get_config()
    eval_config = get_config()

    #with tf.Graph().as_default, tf.Session() as session:
    with tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.device('/gpu:0'):
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = DeepQAModel(is_training=True, config=config)
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mv = DeepQAModel(is_training=False, config=config)

        tf.initialize_all_variables().run()

        i = 0
        for epoch in range(1000):
            train_data = [q_train, a_train, y_train]
            batches = enumerate(batch_iter(train_data, config.batch_size))
            for _, batch_data in batches:
                q_batch, a_batch, y_batch = batch_data
                step = session.run([m.train_step], 
                        {m.q_data : q_batch,
                         m.a_data : a_batch,
                         m.embedding : vocab_emb,
                         m.labels : y_batch})

                if i % 100 == 0:
                    loss, acc = session.run([mv.reg_loss, mv.accuracy], 
                                {mv.q_data : q_train,
                                 mv.a_data : a_train,
                                 mv.embedding : vocab_emb,
                                 mv.labels : y_train})
                    print("[TRAIN] Step %4d, acc=%5.5f, loss=%5.5f" %(i, acc, loss))
                    
                    y_score, acc = session.run([mv.y_score, mv.accuracy], 
                                {mv.q_data : q_dev,
                                 mv.a_data : a_dev,
                                 mv.embedding : vocab_emb,
                                 mv.labels : y_dev})
                    #print(qids_dev.shape, y_dev.shape, y_score.shape)
                    map_sc = map_score(qids_dev, y_dev, y_score[:,1])
                    print("[TRAIN] Step %4d, ============ MAP =======: %5.5f" %(i, map_sc))
                i += 1

        session.close()

    

if __name__ == "__main__":
    tf.app.run()
