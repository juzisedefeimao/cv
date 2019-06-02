from lib.networks.network import Network
import numpy as np
from lib.networks.netconfig import cfg
import tensorflow as tf

cls_num = cfg.ZLRM.TRAIN.CLASSIFY_NUM

class Resnet18_fcn_classifier_test(Network):
    def __init__(self):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None ,None, 3], name='data')
        self.layers = {'data': self.data}
        self.setup()

    def setup(self):
        bn_trainable = False
        (
            self.feed('data')
                .conv(7, 7, 64, 2, 2, name='conv1', relu=False, trainable=True)
                .batch_normalization(name='bn1', relu=True, trainable=bn_trainable)
                .max_pool(3, 3, 2, 2, name='pool1', padding='VALID')
        )
        # ======================变换形状适应第一组模块=======================
        (
            self.feed('pool1')
                .conv(1, 1, 64, 1, 1, name='transform1_conv', relu=False, trainable=True)
                .batch_normalization(name='transform1_bn', relu=False, trainable=bn_trainable)
        )
        # ======================第一组模块===========================
        (
            self.feed('pool1')
                .conv(3, 3, 64, 1, 1, name='res1_1_conv1', relu=False, trainable=True)
                .batch_normalization(name='res1_1_bn1', relu=True, trainable=bn_trainable)
                .conv(3, 3, 64, 1, 1, name='res1_1_conv2', relu=False, trainable=True)
                .batch_normalization(name='res1_1_bn2', relu=False, trainable=bn_trainable)
        )
        (
            self.feed('transform1_bn', 'res1_1_bn2')
                .add(name='res1_1_add')
                .relu(name='res1_1_relu')
                .conv(3, 3, 64, 1, 1, name='res1_2_conv1', relu=False, trainable=True)
                .batch_normalization(name='res1_2_bn1', relu=True, trainable=bn_trainable)
                .conv(3, 3, 64, 1, 1, name='res1_2_conv2', relu=False, trainable=True)
                .batch_normalization(name='res1_2_bn2', relu=False, trainable=bn_trainable)
        )

        # ======================计算残差变换形状适应第二组模块=======================
        (
            self.feed('transform1_bn', 'res1_2_bn2')
                .add(name='res1_2_add')
                .relu(name='res1_2_relu')
                .conv(1, 1, 128, 2, 2, name='transform2_conv', relu=False, trainable=True)
                .batch_normalization(name='transform2_bn', relu=False, trainable=bn_trainable)
        )
        # ======================第二组模块===========================
        (
            self.feed('res1_2_relu')
                .conv(3, 3, 128, 2, 2, name='res2_1_conv1', relu=False, trainable=True)
                .batch_normalization(name='res2_1_bn1', relu=True, trainable=bn_trainable)
                .conv(3, 3, 128, 1, 1, name='res2_1_conv2', relu=False, trainable=True)
                .batch_normalization(name='res2_1_bn2', relu=False, trainable=bn_trainable)
        )
        (
            self.feed('transform2_bn', 'res2_1_bn2')
                .add(name='res2_1_add')
                .relu(name='res2_1_relu')
                .conv(3, 3, 128, 1, 1, name='res2_2_conv1', relu=False, trainable=True)
                .batch_normalization(name='res2_2_bn1', relu=True, trainable=bn_trainable)
                .conv(3, 3, 128, 1, 1, name='res2_2_conv2', relu=False, trainable=True)
                .batch_normalization(name='res2_2_bn2', relu=True, trainable=bn_trainable)
        )

        # ======================计算残差变换形状适应第三组模块=======================
        (
            self.feed('transform2_bn', 'res2_2_bn2')
                .add(name='res2_2_add')
                .relu(name='res2_2_relu')
                .conv(1, 1, 256, 2, 2, name='transform3_conv', relu=False, trainable=True)
                .batch_normalization(name='transform3_bn', relu=False, trainable=bn_trainable)
        )
        # ======================第三组模块===========================
        (
            self.feed('res2_2_relu')
                .conv(3, 3, 256, 2, 2, name='res3_1_conv1', relu=False, trainable=True)
                .batch_normalization(name='res3_1_bn1', relu=True, trainable=bn_trainable)
                .conv(3, 3, 256, 1, 1, name='res3_1_conv2', relu=False, trainable=True)
                .batch_normalization(name='res3_1_bn2', relu=True, trainable=bn_trainable)
        )
        (
            self.feed('transform3_bn', 'res3_1_bn2')
                .add(name='res3_1_add')
                .relu(name='res3_1_relu')
                .conv(3, 3, 256, 1, 1, name='res3_2_conv1', relu=False, trainable=True)
                .batch_normalization(name='res3_2_bn1', relu=True, trainable=bn_trainable)
                .conv(3, 3, 256, 1, 1, name='res3_2_conv2', relu=False, trainable=True)
                .batch_normalization(name='res3_2_bn2', relu=True, trainable=bn_trainable)
        )

        # ======================计算残差变换形状适应第四组模块=======================
        (
            self.feed('transform3_bn', 'res3_2_bn2')
                .add(name='res3_2_add')
                .relu(name='res3_2_relu')
                .conv(1, 1, 512, 2, 2, name='transform4_conv', relu=False, trainable=True)
                .batch_normalization(name='transform4_bn', relu=False, trainable=bn_trainable)
        )
        # ======================第四组模块===========================
        (
            self.feed('res3_2_relu')
                .conv(3, 3, 512, 2, 2, name='res4_1_conv1', relu=False, trainable=True)
                .batch_normalization(name='res4_1_bn1', relu=True, trainable=bn_trainable)
                .conv(3, 3, 512, 1, 1, name='res4_1_conv2', relu=False, trainable=True)
                .batch_normalization(name='res4_1_bn2', relu=True, trainable=bn_trainable)
        )
        (
            self.feed('transform4_bn', 'res4_1_bn2')
                .add(name='res4_1_add')
                .relu(name='res4_1_relu')
                .conv(3, 3, 512, 1, 1, name='res4_2_conv1', relu=False, trainable=True)
                .batch_normalization(name='res4_2_bn1', relu=True, trainable=bn_trainable)
                .conv(3, 3, 512, 1, 1, name='res4_2_conv2', relu=False, trainable=True)
                .batch_normalization(name='res4_2_bn2', relu=True, trainable=bn_trainable)
        )
        # ======================计算残差变换结束模块=======================
        (
            self.feed('transform4_bn', 'res4_2_bn2')
                .add(name='res4_2_add')
                .relu(name='res4_2_relu')
                .conv(1, 1, cls_num * cfg.ZLRM.PSROIPOOL * cfg.ZLRM.PSROIPOOL, 1, 1, name='fcn_cls', trainable=True)
                .ps_pool(output_dim=cls_num, group_size=cfg.ZLRM.PSROIPOOL, name='pspooled_cls_rois')
                .avg_pool(cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, name='cls_score')
                .softmax(name='cls_prob')
        )