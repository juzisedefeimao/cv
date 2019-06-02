from lib.networks.network import Network
import numpy as np
from lib.networks.netconfig import cfg
import tensorflow as tf
classes_num = cfg.SIAMSE.N_CLASSES
batch = cfg.SIAMSE.TEST.BATCH_SIZE * cfg.SIAMSE.IMAGE_TRANSFORM_NUM
class Siammask_test(Network):
    def __init__(self):
        self.inputs = []
        self.anchor_num = 9
        self.search_data = tf.placeholder(tf.float32, shape=[batch, 255, 255, 3], name='search_data')
        self.template_data = tf.placeholder(tf.float32, shape=[classes_num, 127, 127, 3], name='template_data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.layers = {'search_data': self.search_data, 'template_data': self.template_data,
                       'im_info': self.im_info}
        self.setup()

    def get_rois(self):
        rpn_rois, scores = self.roi_proposal(self.layers['feature_sum_cls'], self.layers['feature_sum_loc'])
        return rpn_rois, scores

    def setup(self):
        search_feature_2, search_feature_3, search_feature_4 = self.resnet50('search_data')
        template_feature_2, template_feature_3, template_feature_4 = self.resnet50('template_data', reuse=True)
        search_feature_resdown_2 = self.resdown(search_feature_2, name='2p', reuse=False)
        search_feature_resdown_3 = self.resdown(search_feature_3, name='3p', reuse=False)
        search_feature_resdown_4 = self.resdown(search_feature_4, name='4p', reuse=False)
        template_feature_resdown_2 = self.resdown(template_feature_2, name='2p', reuse=True)
        template_feature_resdown_3 = self.resdown(template_feature_3, name='3p', reuse=True)
        template_feature_resdown_4 = self.resdown(template_feature_4, name='4p', reuse=True)
        feature_2_cls, feature_2_loc = self.rpn(search_feature_resdown_2, template_feature_resdown_2,
                                                self.anchor_num, name='fearch_2')
        feature_3_cls, feature_3_loc = self.rpn(search_feature_resdown_3, template_feature_resdown_3,
                                                self.anchor_num, name='fearch_3', reuse=True)
        feature_4_cls, feature_4_loc = self.rpn(search_feature_resdown_4, template_feature_resdown_4,
                                                self.anchor_num, name='fearch_4', reuse=True)
        feature_sum_cls = self.feature_sum([feature_2_cls, feature_3_cls, feature_4_cls])
        feature_sum_loc = self.feature_sum([feature_2_loc, feature_3_loc, feature_4_loc])
        self.layers['feature_sum_cls'] = feature_sum_cls #[batch*classes, h, w, a*2]
        self.layers['feature_sum_loc'] = feature_sum_loc #[batch*classes, h, w, a*4]

    def resnet50(self, data, reuse=False, bn_trainable = False):
        with tf.variable_scope('restnet50') as scope:

            if reuse:
                scope.reuse_variables()
            (
             self.feed(data)
                 .conv(7, 7, 64, 2, 2,name='conv1', biased=False, relu=False, padding='VALID', trainable=True)
                 .batch_normalization(name='bn1', relu=True, trainable=bn_trainable)
                 .max_pool(3, 3, 2, 2, name='pool1', padding='SAME')
            )
            # ======================变换形状适应第一组模块=======================
            (
             self.feed('pool1')
                 .conv(1 ,1, 256, 1, 1, name='transform1_conv', biased=False, relu=False, trainable=True)
                 .batch_normalization(name='transform1_bn', relu=False, trainable=bn_trainable)
            )
            # ======================第一组模块===========================
            (
            self.feed('pool1')
                 .conv(1, 1, 64, 1, 1, name='res1_1_conv1', biased=False, relu=False, trainable=True)
                 .batch_normalization(name='res1_1_bn1', relu=True, trainable=bn_trainable)
                 .conv(3, 3, 64, 1, 1, name='res1_1_conv2', biased=False, relu=False, trainable=True)
                 .batch_normalization(name='res1_1_bn2', relu=True,trainable=bn_trainable)
                 .conv(1, 1, 256, 1, 1, name='res1_1_conv3', biased=False, relu=False, trainable=True)
                 .batch_normalization(name='res1_1_bn3', relu=False, trainable=bn_trainable)
            )
            (
            self.feed('transform1_bn', 'res1_1_bn3')
                .add(name='res1_1_add')
                .relu(name='res1_1_relu')
                .conv(1, 1, 64, 1, 1, name='res1_2_conv1', biased=False, relu=False, trainable=True)
                .batch_normalization(name='res1_2_bn1', relu=True, trainable=bn_trainable)
                .conv(3, 3, 64, 1, 1, name='res1_2_conv2', biased=False, relu=False, trainable=True)
                .batch_normalization(name='res1_2_bn2', relu=True, trainable=bn_trainable)
                .conv(1, 1, 256, 1, 1, name='res1_2_conv3', biased=False, relu=False, trainable=True)
                .batch_normalization(name='res1_2_bn3', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('transform1_bn', 'res1_2_bn3')
                    .add(name='res1_2_add')
                    .relu(name='res1_2_relu')
                    .conv(1, 1, 64, 1, 1, name='res1_3_conv1', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res1_3_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 64, 1, 1, name='res1_3_conv2', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res1_3_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 256, 1, 1, name='res1_3_conv3', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res1_3_bn3', relu=False, trainable=bn_trainable)
            )
            # ======================计算残差变换形状适应第二组模块=======================
            (
            self.feed('transform1_bn', 'res1_3_bn3')
                .add(name='res1_3_add')
                .relu(name='res1_3_relu')
                .conv(1, 1, 512, 2, 2, name='transform2_conv', biased=False, relu=False, trainable=True)
                .batch_normalization(name='transform2_bn', relu=False, trainable=bn_trainable)
            )
            # ======================第二组模块===========================
            (
            self.feed('res1_3_relu')
                .conv(1, 1, 128, 2, 2, name='res2_1_conv1', biased=False, relu=False, trainable=True)
                .batch_normalization(name='res2_1_bn1', relu=True, trainable=bn_trainable)
                .conv(3, 3, 128, 1, 1, name='res2_1_conv2', biased=False, relu=False, trainable=True)
                .batch_normalization(name='res2_1_bn2', relu=True, trainable=bn_trainable)
                .conv(1, 1, 512, 1, 1, name='res2_1_conv3', biased=False, relu=False, trainable=True)
                .batch_normalization(name='res2_1_bn3', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('transform2_bn', 'res2_1_bn3')
                    .add(name='res2_1_add')
                    .relu(name='res2_1_relu')
                    .conv(1, 1, 128, 1, 1, name='res2_2_conv1', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res2_2_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 128, 1, 1, name='res2_2_conv2', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res2_2_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 512, 1, 1, name='res2_2_conv3', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res2_2_bn3', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('transform2_bn', 'res2_2_bn3')
                    .add(name='res2_2_add')
                    .relu(name='res2_2_relu')
                    .conv(1, 1, 128, 1, 1, name='res2_3_conv1', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res2_3_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 128, 1, 1, name='res2_3_conv2', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res2_3_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 512, 1, 1, name='res2_3_conv3', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res2_3_bn3', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('transform2_bn', 'res2_3_bn3')
                    .add(name='res2_3_add')
                    .relu(name='res2_3_relu')
                    .conv(1, 1, 128, 1, 1, name='res2_4_conv1', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res2_4_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 128, 1, 1, name='res2_4_conv2', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res2_4_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 512, 1, 1, name='res2_4_conv3', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res2_4_bn3', relu=False, trainable=bn_trainable)
            )
            # ======================计算残差变换形状适应第三组模块=======================
            (
                self.feed('transform2_bn', 'res2_4_bn3')
                    .add(name='res2_4_add')
                    .relu(name='res2_4_relu')
                    .conv(1, 1, 1024, 1, 1, name='transform3_conv', biased=False, relu=False, dilation=[1,2,2,1], trainable=True)
                    .batch_normalization(name='transform3_bn', relu=False, trainable=bn_trainable)
            )
            # ======================第三组模块===========================
            (
                self.feed('res2_4_relu')
                    .conv(1, 1, 256, 1, 1, name='res3_1_conv1', biased=False, relu=False, dilation=[1,2,2,1], trainable=True)
                    .batch_normalization(name='res3_1_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 256, 1, 1, name='res3_1_conv2', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res3_1_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 1024, 1, 1, name='res3_1_conv3', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res3_1_bn3', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('transform3_bn', 'res3_1_bn3')
                    .add(name='res3_1_add')
                    .relu(name='res3_1_relu')
                    .conv(1, 1, 256, 1, 1, name='res3_2_conv1', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res3_2_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 256, 1, 1, name='res3_2_conv2', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res3_2_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 1024, 1, 1, name='res3_2_conv3', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res3_2_bn3', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('transform3_bn', 'res3_2_bn3')
                    .add(name='res3_2_add')
                    .relu(name='res3_2_relu')
                    .conv(1, 1, 256, 1, 1, name='res3_3_conv1', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res3_3_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 256, 1, 1, name='res3_3_conv2', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res3_3_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 1024, 1, 1, name='res3_3_conv3', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res3_3_bn3', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('transform3_bn', 'res3_3_bn3')
                    .add(name='res3_3_add')
                    .relu(name='res3_3_relu')
                    .conv(1, 1, 256, 1, 1, name='res3_4_conv1', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res3_4_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 256, 1, 1, name='res3_4_conv2', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res3_4_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 1024, 1, 1, name='res3_4_conv3', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res3_4_bn3', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('transform3_bn', 'res3_4_bn3')
                    .add(name='res3_4_add')
                    .relu(name='res3_4_relu')
                    .conv(1, 1, 256, 1, 1, name='res3_5_conv1', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res3_5_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 256, 1, 1, name='res3_5_conv2', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res3_5_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 1024, 1, 1, name='res3_5_conv3', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res3_5_bn3', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('transform3_bn', 'res3_5_bn3')
                    .add(name='res3_5_add')
                    .relu(name='res3_5_relu')
                    .conv(1, 1, 256, 1, 1, name='res3_6_conv1', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res3_6_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 256, 1, 1, name='res3_6_conv2', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res3_6_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 1024, 1, 1, name='res3_6_conv3', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res3_6_bn3', relu=False, trainable=bn_trainable)
            )
            # ======================计算残差变换形状适应第四组模块=======================
            (
                self.feed('transform3_bn', 'res3_6_bn3')
                    .add(name='res3_6_add')
                    .relu(name='res3_6_relu')
                    .conv(1, 1, 2048, 1, 1, name='transform4_conv', biased=False, relu=False, dilation=[1,4,4,1], trainable=True)
                    .batch_normalization(name='transform4_bn', relu=False, trainable=bn_trainable)
            )
            # ======================第四组模块===========================
            (
                self.feed('res3_6_relu')
                    .conv(1, 1, 512, 1, 1, name='res4_1_conv1', biased=False, relu=False, dilation=[1,4,4,1], trainable=True)
                    .batch_normalization(name='res4_1_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 512, 1, 1, name='res4_1_conv2', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res4_1_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 2048, 1, 1, name='res4_1_conv3', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res4_1_bn3', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('transform4_bn', 'res4_1_bn3')
                    .add(name='res4_1_add')
                    .relu(name='res4_1_relu')
                    .conv(1, 1, 512, 1, 1, name='res4_2_conv1', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res4_2_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 512, 1, 1, name='res4_2_conv2', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res4_2_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 2048, 1, 1, name='res4_2_conv3', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res4_2_bn3', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('transform4_bn', 'res4_2_bn3')
                    .add(name='res4_2_add')
                    .relu(name='res4_2_relu')
                    .conv(1, 1, 512, 1, 1, name='res4_3_conv1', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res4_3_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 512, 1, 1, name='res4_3_conv2', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res4_3_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 2048, 1, 1, name='res4_3_conv3', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='res4_3_bn3', relu=False, trainable=bn_trainable)
            )
            # ======================计算残差变换结束模块=======================
            (
                self.feed('transform4_bn', 'res4_3_bn3')
                    .add(name='res4_3_add')
                    .relu(name='res4_3_relu')
            )

        return self.get_output('res2_3_relu'), self.get_output('res3_6_relu'), self.get_output('res4_3_relu')


    # =========================RPN============================================
    def rpn(self, search, template, anchor_num, name, reuse=False, bn_trainable=False):
        with tf.variable_scope('rpn') as scope:

            if reuse:
                scope.reuse_variables()
            # ======================================cls===================================
            (
                self.feed(search)
                    .conv(3, 3, 256, 1, 1, name='search_adjust_cov_cls', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='search_adjust_bn_cls', relu=True, trainable=bn_trainable)
            )
            (
                self.feed(template)
                    .conv(3, 3, 256, 1, 1, name='template_adjust_cov_cls', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='template_adjust_bn_cls', relu=True, trainable=bn_trainable)
            )
            (
                self.feed('search_adjust_bn_cls', 'template_adjust_bn_cls')
                    .conv_dw_group(name='conv_dw_cls')
                    .conv(1, 1, 256, 1, 1, name='head_conv_cls', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='head_bn_cls', relu=True, trainable=bn_trainable)
                    .conv(1, 1, anchor_num*2, 1, 1, name='cls', relu=True, trainable=True)
            )
            # ==============================loc==================================================
            (
                self.feed(search)
                    .conv(3, 3, 256, 1, 1, name='search_adjust_cov_loc', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='search_adjust_bn_loc', relu=True, trainable=bn_trainable)
            )
            (
                self.feed(template)
                    .conv(3, 3, 256, 1, 1, name='template_adjust_cov_loc', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='template_adjust_bn_loc', relu=True, trainable=bn_trainable)
            )
            (
                self.feed('search_adjust_bn_loc', 'template_adjust_bn_loc')
                    .conv_dw_group(name='conv_dw_loc')
                    .conv(1, 1, 256, 1, 1, name='head_conv_loc', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='head_bn_loc', relu=True, trainable=bn_trainable)
                    .conv(1, 1, anchor_num*4, 1, 1, name='loc', relu=True, trainable=True)
            )
        return self.get_output('cls'), self.get_output('loc')

    def resdown(self, feature, name, reuse=False, bn_trainable=True):
        with tf.variable_scope(name + 'resdown') as scope:
            if reuse:
                scope.reuse_variables()
            (
                self.feed(feature)
                    .conv(1, 1, 256, 1, 1, name='resdown_conv', biased=False, relu=False, trainable=True)
                    .batch_normalization(name='resdown_bn', relu=False, trainable=bn_trainable)
            )

        return self.get_output('resdown_bn')

    def feature_sum(self, feature_list):
        feature_sum = tf.zeros_like(feature_list[0])
        for i in range(len(feature_list)):
            feature_sum = tf.add(feature_sum, feature_list[i])

        return feature_sum

    def roi_proposal(self, feature_sum_cls, feature_sum_loc):
        # ========= RoI Proposal ============
        with tf.variable_scope('roi_proposal') as scope:

            (
                self.feed(feature_sum_cls)
                    .spatial_reshape_layer(2, name='rpn_cls_score_reshape')
                    .spatial_softmax(name='rpn_cls_prob')
            )

            (
                self.feed('rpn_cls_prob')
                    .spatial_reshape_layer(self.anchor_num*2,
                                           name='rpn_cls_prob_reshape')
            )

            (
                self.feed('rpn_cls_prob_reshape', feature_sum_loc, 'im_info')
                    .siammase_proposal_layer(cfg_key=True, _feat_stride=cfg.SIAMSE.FEAT_STRIDE,
                                    anchor_scales=cfg.SIAMSE.ANCHOR_SCALE, name='rpn_rois')
            )


            return self.get_output('rpn_rois')