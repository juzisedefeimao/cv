import tensorflow as tf
import numpy as np
from lib.networks.network import Network
from lib.networks.netconfig import cfg
cls_num = cfg.RELATION_NET.TRAIN.CLASSIFY_NUM

query_num = 1

class Relation_Network_test(Network) :
    def __init__(self):
        self.query_num = tf.placeholder(tf.int32, shape=[1], name='query_num')
        self.sample_data = tf.placeholder(tf.float32,
                                   shape=[cfg.RELATION_NET.TRAIN.CLASSIFY_SAMPLE * cls_num,
                                          cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE[0],
                                          cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE[1], 3], name='sample_data')
        self.query_data = tf.placeholder(tf.float32,
                                   shape=[None,
                                          cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE[0],
                                          cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE[1], 3], name='query_data')
        self.sample_label = tf.placeholder(tf.int32, shape=[cls_num],
                                           name='sample_label')
        self.query_label = tf.placeholder(tf.int32, shape=[None],
                                          name='query_label')

        self.test_sample_data = tf.placeholder(tf.float32,
                                        shape=[cfg.RELATION_NET.TRAIN.CLASSIFY_SAMPLE * cls_num,
                                               cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE[0],
                                               cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE[1], 3], name='test_sample_data')
        self.test_query_data = tf.placeholder(tf.float32,
                                               shape=[cfg.RELATION_NET.TRAIN.CLASSIFY_QUERY * cls_num,
                                                      cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE[0],
                                                      cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE[1], 3],
                                               name='test_query_data')
        self.test_sample_label = tf.placeholder(tf.int32, shape=[cls_num],
                                                name='test_sample_label')
        self.test_query_label = tf.placeholder(tf.int32, shape=[cfg.RELATION_NET.TRAIN.CLASSIFY_QUERY * cls_num],
                                               name='test_query_label')
        self.layers = {}
        self.inputs = []
        self.setup()

    def get_label(self, sample_label, query_label):
        query_label = tf.reshape(query_label, (-1,1))
        label_bool = tf.reshape(tf.equal(query_label, sample_label), (-1, 1))
        label = tf.cast(label_bool, dtype=tf.float32)

        return label

    def get_pre_label(self, scores):

        split_list = tf.split(scores, num_or_size_splits=self.test_query_label.get_shape()[0])
        pre_label = []
        for split in split_list:
            pre_label.append(tf.argmax(split))

        pre_label = tf.stack(pre_label, axis=0)

        return pre_label

    def get_test_label(self, sample_label, query_label):
        query_label = tf.reshape(query_label, (-1, 1))
        label_bool = tf.equal(query_label, sample_label)
        label = tf.argmax(tf.cast(label_bool, dtype=tf.float32), axis=1)
        label = tf.reshape(label, (-1,1))

        return label

    def test_net(self):
        test_sample_encoder = self.cnn_encoder(self.test_sample_data, reuse=True)
        test_query_encoder = self.cnn_encoder(self.test_query_data, reuse=True)
        (
            self.feed(test_sample_encoder, test_query_encoder)
                .relation_concat(C_WAY=cls_num, name='test_concattion')
        )
        scores = self.relation_net(self.get_output('test_concattion'), reuse=True)

        label = self.get_test_label(self.test_sample_label, self.test_query_label)
        pre_label = self.get_pre_label(scores)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(label, pre_label), dtype=tf.float32))

        return accuracy

    def demo_net(self):
        # split_list = tf.split(self.layers['scores'], num_or_size_splits=self.query_num[0])
        # print('spkl', split_list)
        # pre_label = []
        # for split in split_list:
        #     print(split)
        #     pre_label.append(tf.argmax(split))
        #
        # pre_label = tf.stack(pre_label, axis=0)
        return self.layers['scores']

    def setup(self):
        sample_encoder = self.cnn_encoder(self.sample_data, bn_trainable = False)
        query_encoder = self.cnn_encoder(self.query_data, reuse=True, bn_trainable = False)
        print(query_encoder)
        (
            self.feed(sample_encoder, query_encoder)
                .relation_concat(C_WAY=cls_num, query_num=self.query_num, name='concattion')
        )
        scores = self.relation_net(self.get_output('concattion'), bn_trainable = False)
        self.layers['scores'] = scores

    def cnn_encoder(self, data, reuse=False, bn_trainable = True):
        with tf.variable_scope('cnn_encoder') as scope:

            if reuse:
                scope.reuse_variables()
            (
                self.feed(data)
                    .conv(3, 3, 64, 1, 1, name='conv1', relu=False, padding='VALID', trainable=True)
                    .batch_normalization(name='bn1', relu=True, trainable=bn_trainable)
                    .max_pool(2, 2, 2, 2, name='pool1', padding='VALID')
                    .conv(3, 3, 64, 1, 1, name='conv2', relu=False, padding='VALID', trainable=True)
                    .batch_normalization(name='bn2', relu=True, trainable=bn_trainable)
                    .max_pool(2, 2, 2, 2, name='pool2', padding='VALID')
                    .conv(3, 3, 64, 1, 1, name='conv3', relu=False, padding='SAME', trainable=True)
                    .batch_normalization(name='bn3', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 64, 1, 1, name='conv4', relu=False, padding='SAME', trainable=True)
                    .batch_normalization(name='bn4', relu=True, trainable=bn_trainable)
            )

        return self.get_output('bn4')

    def relation_net(self, data, reuse=False, bn_trainable = True):
        with tf.variable_scope('relation_net') as scope:

            if reuse:
                scope.reuse_variables()
            (
                self.feed(data)
                    .conv(3, 3, 64, 1, 1, name='conv1', relu=False, padding='VALID', trainable=True)
                    .batch_normalization(name='bn1', relu=True, trainable=bn_trainable)
                    .max_pool(2, 2, 2, 2, name='pool1', padding='VALID')
                    .conv(3, 3, 64, 1, 1, name='conv2', relu=False, padding='VALID', trainable=True)
                    .batch_normalization(name='bn2', relu=True, trainable=bn_trainable)
                    .max_pool(2, 2, 2, 2, name='pool2', padding='VALID')
                    .fc(8, name='fc1', relu=True, trainable=True)
                    .fc(1, name='fc2', relu=False, trainable=True)
                    .sigmoid(name='score')
            )

        return self.get_output('score')