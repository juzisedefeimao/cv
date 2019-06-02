from lib.networks.network import Network
import numpy as np
from lib.networks.netconfig import cfg
import tensorflow as tf

cls_num = cfg.ZLRM.TRAIN.CLASSIFY_NUM

class Triple_classifier_train(Network):
    def __init__(self):
        self.inputs = []
        self.data = tf.placeholder(tf.float32,
                                   shape=[64,
                                          cfg.ZLRM.TRAIN.CLASSIFY_IMAGE_SIZE[0],
                                          cfg.ZLRM.TRAIN.CLASSIFY_IMAGE_SIZE[1], 3], name='data')
        self.label = tf.placeholder(tf.int32, shape=[64], name='label')
        self.layers = {'data': self.data}
        self.setup()

    def build_loss(self):
        # cls_score = tf.reshape(self.get_output('cls_score'), [-1, cls_num])
        cls_score = self.get_output('cls_score')
        label = tf.cast(tf.reshape(self.label, [-1,1]), dtype=tf.float32)
        cross_entropy_n = tf.nn.sigmoid_cross_entropy_with_logits(logits=cls_score, labels=label)
        cross_entropy = tf.reduce_mean(cross_entropy_n)
        if cfg.TRAIN.WEIGHT_DECAY > 0:
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            cross_entropy = tf.add_n(regularization_losses) + cross_entropy
        return cross_entropy

    def setup(self):
        bn_trainable = False
        init_GAN_weights = True
        spectral_norm = False
        reuse = False

        label = tf.one_hot(self.label, depth=cls_num)
        (
            self.feed(label)
                .reshape_layer([-1, 1, 1, cls_num], name='d_reshape_label')
        )
        (
            self.feed('data')
                .dropout(0.2, name='d_dropout')
        )
        (
            self.feed('d_dropout', 'd_reshape_label')
                .conv_concat(name='d_conv_concat_1')
                .conv(3, 3, 32, 1, 1, name='d_conv1', reuse=reuse, biased=False, relu=False,
                      init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm, trainable=True)
                .batch_normalization(name='d_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.2, name='d_lrelu1')
        )
        (
            self.feed('d_lrelu1', 'd_reshape_label')
                .conv_concat(name='d_conv_concat_2')
                .conv(3, 3, 32, 2, 2, name='d_conv2', reuse=reuse, biased=False, relu=False,
                      init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm, trainable=True)
                .batch_normalization(name='d_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.2, name='d_lrelu2')
                .dropout(0.2, name='d_dropout2')
        )
        (
            self.feed('d_dropout2', 'd_reshape_label')
                .conv_concat(name='d_conv_concat_3')
                .conv(3, 3, 64, 1, 1, name='d_conv3', reuse=reuse, biased=False, relu=False,
                      init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm, trainable=True)
                .batch_normalization(name='d_bn3', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.2, name='d_lrelu3')
        )
        (
            self.feed('d_lrelu3', 'd_reshape_label')
                .conv_concat(name='d_conv_concat_4')
                .conv(3, 3, 64, 2, 2, name='d_conv4', reuse=reuse, biased=False, relu=False,
                      init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm, trainable=True)
                .batch_normalization(name='d_bn4', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.2, name='d_lrelu4')
                .dropout(0.2, name='d_dropout4')
        )
        (
            self.feed('d_dropout4', 'd_reshape_label')
                .conv_concat(name='d_conv_concat_5')
                .conv(3, 3, 128, 1, 1, name='d_conv5', reuse=reuse, biased=False, relu=False,
                      init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm, trainable=True)
                .batch_normalization(name='d_bn5', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.2, name='d_lrelu5')
        )
        (
            self.feed('d_lrelu5', 'd_reshape_label')
                .conv_concat(name='d_conv_concat_6')
                .conv(3, 3, 128, 1, 1, name='d_conv6', reuse=reuse, biased=False, relu=False,
                      init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm, trainable=True)
                .batch_normalization(name='d_bn6', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.2, name='d_lrelu6')
        )
        (
            self.feed('d_lrelu6', 'd_reshape_label')
                .conv_concat(name='d_conv_concat_7')
                .global_avg_pool(name='d_global_ave_pool')
        )
        (
            self.feed('d_global_ave_pool', label)
                .concat(axis=1, name='d_concat')
                .fc(1, name='cls_score', relu=False)
                .sigmoid(name='cls_prob')
        )

    def setup_(self):
        bn_trainable = False
        (
            self.feed('data')
                .gaussian_noise_layer(std=0.15, name= 'gaussian_noise')
                .conv(3, 3, 128, 1, 1, name='c_conv1', biased=False, relu=False,
                      trainable=True)
                .batch_normalization(name='c_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.2, name='c_lrelu1')
                .conv(3, 3, 128, 1, 1, name='c_conv2', biased=False, relu=False,
                      trainable=True)
                .batch_normalization(name='c_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.2, name='c_lrelu2')
                .conv(3, 3, 128, 1, 1, name='c_conv3', biased=False, relu=False,
                      trainable=True)
                .batch_normalization(name='c_bn3', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.2, name='c_lrelu3')
                .max_pool(2, 2, 2, 2, name='c_maxpool1')
                .dropout(0.5, name='c_dropout1')
                .conv(3, 3, 256, 1, 1, name='c_conv4', biased=False, relu=False,
                      trainable=True)
                .batch_normalization(name='c_bn4', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.2, name='c_lrelu4')
                .conv(3, 3, 256, 1, 1, name='c_conv5', biased=False, relu=False,
                      trainable=True)
                .batch_normalization(name='c_bn5', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.2, name='c_lrelu5')
                .conv(3, 3, 256, 1, 1, name='c_conv6', biased=False, relu=False,
                      trainable=True)
                .batch_normalization(name='c_bn6', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.2, name='c_lrelu6')
                .max_pool(2, 2, 2, 2, name='c_maxpool2')
                .dropout(0.5, name='c_dropout2')
                .conv(3, 3, 512, 1, 1, name='c_conv7', biased=False, relu=False,
                      trainable=True)
                .batch_normalization(name='c_bn7', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.2, name='c_lrelu7')
                .conv(1, 1, 256, 1, 1, name='c_conv8', biased=False, relu=False,
                      trainable=True)
                .batch_normalization(name='c_bn8', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.2, name='c_lrelu8')
                .conv(1, 1, 128, 1, 1, name='c_conv9', biased=False, relu=False,
                      trainable=True)
                .batch_normalization(name='c_bn9', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.2, name='c_lrelu9')
                .max_pool(2, 2, 2, 2, name='c_maxpool1')
                .dropout(0.5, name='c_dropout1')
                .global_avg_pool(name='c_global_ave_pool')
                .fc(cls_num, name='cls_score', relu=False)
                .softmax(name='cls_prob')
        )

