from lib.networks.network import Network
import numpy as np
from lib.networks.netconfig import cfg
import tensorflow as tf

class FPN_Resnet50_train_detect(Network):
    def __init__(self):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.rois_p2 = tf.placeholder(tf.float32, shape=[None, 5], name='roi-data/P2')
        self.rois_p3 = tf.placeholder(tf.float32, shape=[None, 5], name='roi-data/P3')
        self.rois_p4 = tf.placeholder(tf.float32, shape=[None, 5], name='roi-data/P4')
        self.rois_p5 = tf.placeholder(tf.float32, shape=[None, 5], name='roi-data/P5')
        self.rois_p6 = tf.placeholder(tf.float32, shape=[None, 5], name='roi-data/P6')
        self.labels = tf.placeholder(tf.int32, shape=[None, 1], name='labels')
        self.bbox_targets = tf.placeholder(tf.float32, shape=[None, (cfg.ZLRM.N_CLASSES + 1) * 4], name='bbox_targets')
        self.bbox_inside_weights = tf.placeholder(tf.float32, shape=[None, (cfg.ZLRM.N_CLASSES + 1) * 4],
                                                  name='bbox_inside_weights')
        self.bbox_outside_weights = tf.placeholder(tf.float32, shape=[None, (cfg.ZLRM.N_CLASSES + 1) * 4],
                                                   name='bbox_outside_weights')
        self.layers = {'data': self.data, 'im_info': self.im_info, 'roi-data/P2':self.rois_p2,
                       'roi-data/P3':self.rois_p3, 'roi-data/P4':self.rois_p4, 'roi-data/P5':self.rois_p5, 'roi-data/P6':self.rois_p6, }
        self.setup()

    def build_loss(self):

        ############# R-CNN
        # classification loss
        cls_score = tf.reshape(self.get_output('ave_cls_score_rois_reshape_concat'), [-1, cfg.ZLRM.N_CLASSES + 1]) # (R, C+1)
        label = tf.reshape(self.labels, [-1])  # (R)
        cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label)
        cross_entropy = tf.reduce_mean(cross_entropy_n)

        # bounding box regression L1 loss
        bbox_pred = tf.reshape(self.get_output('ave_bbox_pred_rois_reshape_concat'), [-1, (cfg.ZLRM.N_CLASSES + 1) * 4]) # (R, (C+1)x4)
        bbox_targets = self.bbox_targets  # (R, (C+1)x4)
        # each element is {0, 1}, represents background (0), objects (1)
        bbox_inside_weights = self.bbox_inside_weights  # (R, (C+1)x4)
        bbox_outside_weights = self.bbox_outside_weights  # (R, (C+1)x4)

        loss_box_n = tf.reduce_sum( \
            bbox_outside_weights * self.smooth_l1_dist(bbox_inside_weights * (bbox_pred - bbox_targets)), \
            axis=[1])

        loss_n = loss_box_n + cross_entropy_n
        loss_n = tf.reshape(loss_n, [-1])

        loss_box = tf.reduce_mean(loss_box_n)


        loss = cross_entropy + loss_box

        # add regularizer
        if cfg.TRAIN.WEIGHT_DECAY > 0:
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.add_n(regularization_losses) + loss
        # loss = rpn_cross_entropy + rpn_loss_box
        return loss, cross_entropy, loss_box

    def setup(self):
        (
         self.feed('data')
             .conv(7, 7, 64, 2, 2,name='conv1', relu=False, trainable=True)
             .batch_normalization(name='bn1', relu=True, trainable=False)
             .max_pool(3, 3, 2, 2, name='pool1', padding='VALID')
        )
        # ======================变换形状适应第一组模块=======================
        (
         self.feed('pool1')
             .conv(1 ,1, 256, 1, 1, name='transform1_conv', relu=False, trainable=True)
             .batch_normalization(name='transform1_bn', relu=False, trainable=False)
        )
        # ======================第一组模块===========================
        (
        self.feed('pool1')
             .conv(1, 1, 64, 1, 1, name='res1_1_conv1', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
             .batch_normalization(name='res1_1_bn1', relu=True, trainable=False)
             .conv(3, 3, 64, 1, 1, name='res1_1_conv2', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
             .batch_normalization(name='res1_1_bn2', relu=True,trainable=False)
             .conv(1, 1, 256, 1, 1, name='res1_1_conv3', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
             .batch_normalization(name='res1_1_bn3', relu=False, trainable=False)
        )
        (
        self.feed('transform1_bn', 'res1_1_bn3')
            .add(name='res1_1_add')
            .relu(name='res1_1_relu')
            .conv(1, 1, 64, 1, 1, name='res1_2_conv1', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
            .batch_normalization(name='res1_2_bn1', relu=True, trainable=False)
            .conv(3, 3, 64, 1, 1, name='res1_2_conv2', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
            .batch_normalization(name='res1_2_bn2', relu=True, trainable=False)
            .conv(1, 1, 256, 1, 1, name='res1_2_conv3', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
            .batch_normalization(name='res1_2_bn3', relu=False, trainable=False)
        )
        (
            self.feed('transform1_bn', 'res1_2_bn3')
                .add(name='res1_2_add')
                .relu(name='res1_2_relu')
                .conv(1, 1, 64, 1, 1, name='res1_3_conv1', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res1_3_bn1', relu=True, trainable=False)
                .conv(3, 3, 64, 1, 1, name='res1_3_conv2', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res1_3_bn2', relu=True, trainable=False)
                .conv(1, 1, 256, 1, 1, name='res1_3_conv3', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res1_3_bn3', relu=False, trainable=False)
        )
        # ======================计算残差变换形状适应第二组模块=======================
        (
        self.feed('transform1_bn', 'res1_3_bn3')
            .add(name='res1_3_add')
            .relu(name='res1_3_relu')
            .conv(1, 1, 512, 2, 2, name='transform2_conv', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
            .batch_normalization(name='transform2_bn', relu=False, trainable=False)
        )
        # ======================第二组模块===========================
        (
        self.feed('res1_3_relu')
            .conv(1, 1, 128, 2, 2, name='res2_1_conv1', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
            .batch_normalization(name='res2_1_bn1', relu=True, trainable=False)
            .conv(3, 3, 128, 1, 1, name='res2_1_conv2', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
            .batch_normalization(name='res2_1_bn2', relu=True, trainable=False)
            .conv(1, 1, 512, 1, 1, name='res2_1_conv3', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
            .batch_normalization(name='res2_1_bn3', relu=True, trainable=False)
        )
        (
            self.feed('transform2_bn', 'res2_1_bn3')
                .add(name='res2_1_add')
                .relu(name='res2_1_relu')
                .conv(1, 1, 128, 1, 1, name='res2_2_conv1', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res2_2_bn1', relu=True, trainable=False)
                .conv(3, 3, 128, 1, 1, name='res2_2_conv2', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res2_2_bn2', relu=True, trainable=False)
                .conv(1, 1, 512, 1, 1, name='res2_2_conv3', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res2_2_bn3', relu=False, trainable=False)
        )
        (
            self.feed('transform2_bn', 'res2_2_bn3')
                .add(name='res2_2_add')
                .relu(name='res2_2_relu')
                .conv(1, 1, 128, 1, 1, name='res2_3_conv1', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res2_3_bn1', relu=True, trainable=False)
                .conv(3, 3, 128, 1, 1, name='res2_3_conv2', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res2_3_bn2', relu=True, trainable=False)
                .conv(1, 1, 512, 1, 1, name='res2_3_conv3', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res2_3_bn3', relu=False, trainable=False)
        )
        (
            self.feed('transform2_bn', 'res2_3_bn3')
                .add(name='res2_3_add')
                .relu(name='res2_3_relu')
                .conv(1, 1, 128, 1, 1, name='res2_4_conv1', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res2_4_bn1', relu=True, trainable=False)
                .conv(3, 3, 128, 1, 1, name='res2_4_conv2', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res2_4_bn2', relu=True, trainable=False)
                .conv(1, 1, 512, 1, 1, name='res2_4_conv3', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res2_4_bn3', relu=False, trainable=False)
        )
        # ======================计算残差变换形状适应第三组模块=======================
        (
            self.feed('transform2_bn', 'res2_4_bn3')
                .add(name='res2_4_add')
                .relu(name='res2_4_relu')
                .conv(1, 1, 1024, 2, 2, name='transform3_conv', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='transform3_bn', relu=False, trainable=False)
        )
        # ======================第三组模块===========================
        (
            self.feed('res2_4_relu')
                .conv(1, 1, 256, 2, 2, name='res3_1_conv1', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res3_1_bn1', relu=True, trainable=False)
                .conv(3, 3, 256, 1, 1, name='res3_1_conv2', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res3_1_bn2', relu=True, trainable=False)
                .conv(1, 1, 1024, 1, 1, name='res3_1_conv3', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res3_1_bn3', relu=True, trainable=False)
        )
        (
            self.feed('transform3_bn', 'res3_1_bn3')
                .add(name='res3_1_add')
                .relu(name='res3_1_relu')
                .conv(1, 1, 256, 1, 1, name='res3_2_conv1', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res3_2_bn1', relu=True, trainable=False)
                .conv(3, 3, 256, 1, 1, name='res3_2_conv2', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res3_2_bn2', relu=True, trainable=False)
                .conv(1, 1, 1024, 1, 1, name='res3_2_conv3', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res3_2_bn3', relu=False, trainable=False)
        )
        (
            self.feed('transform3_bn', 'res3_2_bn3')
                .add(name='res3_2_add')
                .relu(name='res3_2_relu')
                .conv(1, 1, 256, 1, 1, name='res3_3_conv1', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res3_3_bn1', relu=True, trainable=False)
                .conv(3, 3, 256, 1, 1, name='res3_3_conv2', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res3_3_bn2', relu=True, trainable=False)
                .conv(1, 1, 1024, 1, 1, name='res3_3_conv3', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res3_3_bn3', relu=False, trainable=False)
        )
        (
            self.feed('transform3_bn', 'res3_3_bn3')
                .add(name='res3_3_add')
                .relu(name='res3_3_relu')
                .conv(1, 1, 256, 1, 1, name='res3_4_conv1', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res3_4_bn1', relu=True, trainable=False)
                .conv(3, 3, 256, 1, 1, name='res3_4_conv2', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res3_4_bn2', relu=True, trainable=False)
                .conv(1, 1, 1024, 1, 1, name='res3_4_conv3', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res3_4_bn3', relu=False, trainable=False)
        )
        (
            self.feed('transform3_bn', 'res3_4_bn3')
                .add(name='res3_4_add')
                .relu(name='res3_4_relu')
                .conv(1, 1, 256, 1, 1, name='res3_5_conv1', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res3_5_bn1', relu=True, trainable=False)
                .conv(3, 3, 256, 1, 1, name='res3_5_conv2', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res3_5_bn2', relu=True, trainable=False)
                .conv(1, 1, 1024, 1, 1, name='res3_5_conv3', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res3_5_bn3', relu=False, trainable=False)
        )
        (
            self.feed('transform3_bn', 'res3_5_bn3')
                .add(name='res3_5_add')
                .relu(name='res3_5_relu')
                .conv(1, 1, 256, 1, 1, name='res3_6_conv1', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res3_6_bn1', relu=True, trainable=False)
                .conv(3, 3, 256, 1, 1, name='res3_6_conv2', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res3_6_bn2', relu=True, trainable=False)
                .conv(1, 1, 1024, 1, 1, name='res3_6_conv3', relu=False, trainable=cfg.ZLRM.TRAIN.FEATURE_TRAIN)
                .batch_normalization(name='res3_6_bn3', relu=False, trainable=False)
        )
        # ======================计算残差变换形状适应第四组模块=======================
        (
            self.feed('transform3_bn', 'res3_6_bn3')
                .add(name='res3_6_add')
                .relu(name='res3_6_relu')
                .conv(1, 1, 2048, 2, 2, name='transform4_conv', relu=False, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
                .batch_normalization(name='transform4_bn', relu=False, trainable=False)
        )
        # ======================第四组模块===========================
        (
            self.feed('res3_6_relu')
                .conv(1, 1, 512, 2, 2, name='res4_1_conv1', relu=False, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
                .batch_normalization(name='res4_1_bn1', relu=True, trainable=False)
                .conv(3, 3, 512, 1, 1, name='res4_1_conv2', relu=False, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
                .batch_normalization(name='res4_1_bn2', relu=True, trainable=False)
                .conv(1, 1, 2048, 1, 1, name='res4_1_conv3', relu=False, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
                .batch_normalization(name='res4_1_bn3', relu=True, trainable=False)
        )
        (
            self.feed('transform4_bn', 'res4_1_bn3')
                .add(name='res4_1_add')
                .relu(name='res4_1_relu')
                .conv(1, 1, 512, 1, 1, name='res4_2_conv1', relu=False, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
                .batch_normalization(name='res4_2_bn1', relu=True, trainable=False)
                .conv(3, 3, 512, 1, 1, name='res4_2_conv2', relu=False, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
                .batch_normalization(name='res4_2_bn2', relu=True, trainable=False)
                .conv(1, 1, 2048, 1, 1, name='res4_2_conv3', relu=False, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
                .batch_normalization(name='res4_2_bn3', relu=False, trainable=False)
        )
        (
            self.feed('transform4_bn', 'res4_2_bn3')
                .add(name='res4_2_add')
                .relu(name='res4_2_relu')
                .conv(1, 1, 512, 1, 1, name='res4_3_conv1', relu=False, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
                .batch_normalization(name='res4_3_bn1', relu=True, trainable=False)
                .conv(3, 3, 512, 1, 1, name='res4_3_conv2', relu=False, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
                .batch_normalization(name='res4_3_bn2', relu=True, trainable=False)
                .conv(1, 1, 2048, 1, 1, name='res4_3_conv3', relu=False, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
                .batch_normalization(name='res4_3_bn3', relu=False, trainable=False)
        )
        # ======================计算残差变换结束模块=======================
        (
            self.feed('transform4_bn', 'res4_3_bn3')
                .add(name='res4_3_add')
                .relu(name='res4_3_relu')
        )
        # =========================FPN============================================
        with tf.variable_scope('Top-Down'):

            # Top-Down
            (
                self.feed('res4_3_relu') # C5
                    .conv(1, 1, 256, 1, 1, name='P5', biased=True, relu=False)
            )

            (
                self.feed('P5')
                    .max_pool(2, 2, 2, 2, name='P6', padding='VALID')
            )

            (
                self.feed('res3_6_relu') # C4
                    .conv(1, 1, 256, 1, 1, name='C4_lateral', biased=True, relu=False)
            )

            (
                self.feed('P5', 'C4_lateral')
                    .upbilinear(name='C5_topdown')
            )

            (
                self.feed('C5_topdown', 'C4_lateral')
                    .add(name='P4_pre')
                    .conv(3, 3, 256, 1, 1, name='P4', biased=True, relu=False)
            )

            (
                self.feed('res2_4_relu') #C3
                    .conv(1, 1, 256, 1, 1, name='C3_lateral', biased=True, relu=False)
            )

            (
                self.feed('P4', 'C3_lateral')
                    .upbilinear(name='C4_topdown')
            )

            (
                self.feed('C4_topdown', 'C3_lateral')
                    .add(name='P3_pre')
                    .conv(3, 3, 256, 1, 1, name='P3', biased=True, relu= False)
            )

            (
                self.feed('res1_3_relu') #C2
                    .conv(1, 1, 256, 1, 1, name='C2_lateral', biased=True, relu=False)
            )

            (
                self.feed('P3', 'C2_lateral')
                    .upbilinear(name='C3_topdown')
            )

            (
                self.feed('C3_topdown', 'C2_lateral')
                    .add(name='P2_pre')
                    .conv(3, 3, 256, 1, 1, name='P2', biased=True, relu= False)
            )

        # ===============================================newconv==============================================================
        with tf.variable_scope('newconv') as scope:

            reuse = True
            # p2
            (
                self.feed('P2')
                    .conv(1, 1, 1024, 1, 1, name='conv_new_1/P2', reuse=reuse, relu=True, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
                    .conv(1, 1, (cfg.ZLRM.N_CLASSES + 1) * cfg.ZLRM.PSROIPOOL * cfg.ZLRM.PSROIPOOL, 1, 1,
                          name='rfcn_cls/P2', reuse=reuse, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
            )
            (
                self.feed('conv_new_1/P2')
                    .conv(1, 1, 4 * (cfg.ZLRM.N_CLASSES + 1) * cfg.ZLRM.PSROIPOOL * cfg.ZLRM.PSROIPOOL, 1, 1,
                          name='rfcn_bbox/P2', reuse=reuse, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
            )
            scope.reuse_variables()
            # p3
            (
                self.feed('P3')
                    .conv(1, 1, 1024, 1, 1, name='conv_new_1/P3', reuse=reuse, relu=True, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
                    .conv(1, 1, (cfg.ZLRM.N_CLASSES + 1) * cfg.ZLRM.PSROIPOOL * cfg.ZLRM.PSROIPOOL, 1, 1,
                          name='rfcn_cls/P3', reuse=reuse,
                          trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
            )
            (
                self.feed('conv_new_1/P3')
                    .conv(1, 1, 4 * (cfg.ZLRM.N_CLASSES + 1) * cfg.ZLRM.PSROIPOOL * cfg.ZLRM.PSROIPOOL, 1, 1,
                          name='rfcn_bbox/P3', reuse=reuse, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
            )
            # p4
            (
                self.feed('P4')
                    .conv(1, 1, 1024, 1, 1, name='conv_new_1/P4', reuse=reuse, relu=True, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
                    .conv(1, 1, (cfg.ZLRM.N_CLASSES + 1) * cfg.ZLRM.PSROIPOOL * cfg.ZLRM.PSROIPOOL, 1, 1,
                          name='rfcn_cls/P4', reuse=reuse,
                          trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
            )
            (
                self.feed('conv_new_1/P4')
                    .conv(1, 1, 4 * (cfg.ZLRM.N_CLASSES + 1) * cfg.ZLRM.PSROIPOOL * cfg.ZLRM.PSROIPOOL, 1, 1,
                          name='rfcn_bbox/P4', reuse=reuse, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
            )
            # p5
            (
                self.feed('P5')
                    .conv(1, 1, 1024, 1, 1, name='conv_new_1/P5', reuse=reuse, relu=True, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
                    .conv(1, 1, (cfg.ZLRM.N_CLASSES + 1) * cfg.ZLRM.PSROIPOOL * cfg.ZLRM.PSROIPOOL, 1, 1,
                          name='rfcn_cls/P5', reuse=reuse,
                          trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
            )
            (
                self.feed('conv_new_1/P5')
                    .conv(1, 1, 4 * (cfg.ZLRM.N_CLASSES + 1) * cfg.ZLRM.PSROIPOOL * cfg.ZLRM.PSROIPOOL, 1, 1,
                          name='rfcn_bbox/P5', reuse=reuse, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
            )
            # p6
            (
                self.feed('P6')
                    .conv(1, 1, 1024, 1, 1, name='conv_new_1/P6', reuse=reuse, relu=True, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
                    .conv(1, 1, (cfg.ZLRM.N_CLASSES + 1) * cfg.ZLRM.PSROIPOOL * cfg.ZLRM.PSROIPOOL, 1, 1,
                          name='rfcn_cls/P6', reuse=reuse,
                          trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
            )
            (
                self.feed('conv_new_1/P6')
                    .conv(1, 1, 4 * (cfg.ZLRM.N_CLASSES + 1) * cfg.ZLRM.PSROIPOOL * cfg.ZLRM.PSROIPOOL, 1, 1,
                          name='rfcn_bbox/P6', reuse=reuse, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
            )
            # ========================position sensitive RoI pooling======================
            # p2
            (
                self.feed('rfcn_cls/P2', 'roi-data/P2')
                    .psroi_pool(output_dim=(cfg.ZLRM.N_CLASSES + 1), group_size=cfg.ZLRM.PSROIPOOL,
                                spatial_scale=0.25, name='psroipooled_cls_rois/P2')
                    .avg_pool(cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL,
                              name='ave_cls_score_rois/P2')
                    .softmax(name='cls_prob/P2')
            )
            (
                self.feed('rfcn_bbox/P2', 'roi-data/P2')
                    .psroi_pool(output_dim=(cfg.ZLRM.N_CLASSES + 1) * 4, group_size=cfg.ZLRM.PSROIPOOL,
                                spatial_scale=0.25, name='psroipooled_loc_rois/P2')
                    .avg_pool(cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL,
                              name='ave_bbox_pred_rois/P2')
            )
            # p3
            (
                self.feed('rfcn_cls/P3', 'roi-data/P3')
                    .psroi_pool(output_dim=(cfg.ZLRM.N_CLASSES + 1), group_size=cfg.ZLRM.PSROIPOOL,
                                spatial_scale=0.125,
                                name='psroipooled_cls_rois/P3')
                    .avg_pool(cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL,
                              name='ave_cls_score_rois/P3')
                    .softmax(name='cls_prob/P3')
            )
            (
                self.feed('rfcn_bbox/P3', 'roi-data/P3')
                    .psroi_pool(output_dim=(cfg.ZLRM.N_CLASSES + 1) * 4, group_size=cfg.ZLRM.PSROIPOOL,
                                spatial_scale=0.125, name='psroipooled_loc_rois/P3')
                    .avg_pool(cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL,
                              name='ave_bbox_pred_rois/P3')
            )
            # p4
            (
                self.feed('rfcn_cls/P4', 'roi-data/P4')
                    .psroi_pool(output_dim=(cfg.ZLRM.N_CLASSES + 1), group_size=cfg.ZLRM.PSROIPOOL,
                                spatial_scale=0.0625,
                                name='psroipooled_cls_rois/P4')
                    .avg_pool(cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL,
                              name='ave_cls_score_rois/P4')
                    .softmax(name='cls_prob/P4')
            )
            (
                self.feed('rfcn_bbox/P4', 'roi-data/P4')
                    .psroi_pool(output_dim=(cfg.ZLRM.N_CLASSES + 1) * 4, group_size=cfg.ZLRM.PSROIPOOL,
                                spatial_scale=0.0625, name='psroipooled_loc_rois/P4')
                    .avg_pool(cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL,
                              name='ave_bbox_pred_rois/P4')
            )
            # p5
            (
                self.feed('rfcn_cls/P5', 'roi-data/P5')
                    .psroi_pool(output_dim=(cfg.ZLRM.N_CLASSES + 1), group_size=cfg.ZLRM.PSROIPOOL,
                                spatial_scale=0.03125,
                                name='psroipooled_cls_rois/P5')
                    .avg_pool(cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL,
                              name='ave_cls_score_rois/P5')
                    .softmax(name='cls_prob/P5')
            )
            (
                self.feed('rfcn_bbox/P5', 'roi-data/P5')
                    .psroi_pool(output_dim=(cfg.ZLRM.N_CLASSES + 1) * 4, group_size=cfg.ZLRM.PSROIPOOL,
                                spatial_scale=0.03125, name='psroipooled_loc_rois/P5')
                    .avg_pool(cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL,
                              name='ave_bbox_pred_rois/P5')
            )
            # p6
            (
                self.feed('rfcn_cls/P6', 'roi-data/P6')
                    .psroi_pool(output_dim=(cfg.ZLRM.N_CLASSES + 1), group_size=cfg.ZLRM.PSROIPOOL,
                                spatial_scale=0.015625,
                                name='psroipooled_cls_rois/P6')
                    .avg_pool(cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL,
                              name='ave_cls_score_rois/P6')
                    .softmax(name='cls_prob/P6')
            )
            (
                self.feed('rfcn_bbox/P6', 'roi-data/P6')
                    .psroi_pool(output_dim=(cfg.ZLRM.N_CLASSES + 1) * 4, group_size=cfg.ZLRM.PSROIPOOL,
                                spatial_scale=0.015625, name='psroipooled_loc_rois/P6')
                    .avg_pool(cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL,
                              name='ave_bbox_pred_rois/P6')
            )

            (
                self.feed('ave_cls_score_rois/P2')
                    .reshape_layer([-1, cfg.ZLRM.N_CLASSES + 1], name='ave_cls_score_rois_reshape/P2')
            )
            (
                self.feed('ave_cls_score_rois/P3')
                    .reshape_layer([-1, cfg.ZLRM.N_CLASSES + 1], name='ave_cls_score_rois_reshape/P3')
            )
            (
                self.feed('ave_cls_score_rois/P4')
                    .reshape_layer([-1, cfg.ZLRM.N_CLASSES + 1], name='ave_cls_score_rois_reshape/P4')
            )
            (
                self.feed('ave_cls_score_rois/P5')
                    .reshape_layer([-1, cfg.ZLRM.N_CLASSES + 1], name='ave_cls_score_rois_reshape/P5')
            )
            (
                self.feed('ave_cls_score_rois/P6')
                    .reshape_layer([-1, cfg.ZLRM.N_CLASSES + 1], name='ave_cls_score_rois_reshape/P6')
            )
            (
                self.feed('ave_cls_score_rois_reshape/P2',
                          'ave_cls_score_rois_reshape/P3',
                          'ave_cls_score_rois_reshape/P4',
                          'ave_cls_score_rois_reshape/P5',
                          'ave_cls_score_rois_reshape/P6')
                    .concat(0, name='ave_cls_score_rois_reshape_concat')
            )
            (
                self.feed('ave_bbox_pred_rois/P2')
                    .reshape_layer([-1, (cfg.ZLRM.N_CLASSES + 1) * 4], name='ave_bbox_pred_rois_reshape/P2')
            )
            (
                self.feed('ave_bbox_pred_rois/P3')
                    .reshape_layer([-1, (cfg.ZLRM.N_CLASSES + 1) * 4], name='ave_bbox_pred_rois_reshape/P3')
            )
            (
                self.feed('ave_bbox_pred_rois/P4')
                    .reshape_layer([-1, (cfg.ZLRM.N_CLASSES + 1) * 4], name='ave_bbox_pred_rois_reshape/P4')
            )
            (
                self.feed('ave_bbox_pred_rois/P5')
                    .reshape_layer([-1, (cfg.ZLRM.N_CLASSES + 1) * 4], name='ave_bbox_pred_rois_reshape/P5')
            )
            (
                self.feed('ave_bbox_pred_rois/P6')
                    .reshape_layer([-1, (cfg.ZLRM.N_CLASSES + 1) * 4], name='ave_bbox_pred_rois_reshape/P6')
            )
            (
                self.feed('ave_bbox_pred_rois_reshape/P2',
                          'ave_bbox_pred_rois_reshape/P3',
                          'ave_bbox_pred_rois_reshape/P4',
                          'ave_bbox_pred_rois_reshape/P5',
                          'ave_bbox_pred_rois_reshape/P6')
                    .concat(0, name='ave_bbox_pred_rois_reshape_concat')
            )

