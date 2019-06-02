from lib.networks.network import Network
import numpy as np
from lib.networks.netconfig import cfg
import tensorflow as tf

cls_num = cfg.ZLRM.TRAIN.CLASSIFY_NUM

class Triple_classifier_test(Network):
    def __init__(self):
        self.inputs = []
        self.data = tf.placeholder(tf.float32,
                                   shape=[None,
                                          cfg.ZLRM.TRAIN.CLASSIFY_IMAGE_SIZE[0],
                                          cfg.ZLRM.TRAIN.CLASSIFY_IMAGE_SIZE[1], 3], name='data')
        self.layers = {'data': self.data}
        self.setup()

    def setup(self):
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

