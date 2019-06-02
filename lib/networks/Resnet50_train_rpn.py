from lib.networks.network import Network
import numpy as np
from lib.networks.netconfig import cfg
import tensorflow as tf

class Resnet50_train_rpn(Network):
    def __init__(self):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')
        self.layers = {'data': self.data, 'im_info': self.im_info, 'gt_boxes': self.gt_boxes}
        self.setup()

    def build_loss(self):
        # ================================RPN======================================
        rpn_cls_score = tf.reshape(self.get_output('rpn_cls_score_reshape'), [-1, 2])  # shape (HxWxA, 2)
        rpn_label = tf.reshape(self.get_output('rpn-data')[0], [-1])  # shape (HxWxA)
        # ignore_label(-1)
        fg_keep = tf.equal(rpn_label, 1)
        rpn_keep = tf.where(tf.not_equal(rpn_label, -1))
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_keep), [-1, 2])  # shape (N, 2)
        rpn_label = tf.reshape(tf.gather(rpn_label, rpn_keep), [-1])
        rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label)
        rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_n)

        # box loss
        rpn_bbox_pred = self.get_output('rpn_bbox_pred')  # shape (1, H, W, Ax4)
        rpn_bbox_targets = self.get_output('rpn-data')[1]
        rpn_bbox_inside_weights = self.get_output('rpn-data')[2]
        rpn_bbox_outside_weights = self.get_output('rpn-data')[3]
        rpn_bbox_pred = tf.reshape(tf.gather(tf.reshape(rpn_bbox_pred, [-1, 4]), rpn_keep), [-1, 4])  # shape (N, 4)
        rpn_bbox_targets = tf.reshape(tf.gather(tf.reshape(rpn_bbox_targets, [-1, 4]), rpn_keep), [-1, 4])
        rpn_bbox_inside_weights = tf.reshape(tf.gather(tf.reshape(rpn_bbox_inside_weights, [-1, 4]), rpn_keep), [-1, 4])
        rpn_bbox_outside_weights = tf.reshape(tf.gather(tf.reshape(rpn_bbox_outside_weights, [-1, 4]), rpn_keep),
                                              [-1, 4])

        rpn_loss_box_n = tf.reduce_sum(self.smooth_l1_dist(
            rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)), axis=[1])
        rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1.0)

        loss = rpn_cross_entropy + rpn_loss_box
        # add regularizer
        if cfg.TRAIN.WEIGHT_DECAY > 0:
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.add_n(regularization_losses) + loss
        return loss, rpn_cross_entropy, rpn_loss_box

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
             .conv(1, 1, 64, 1, 1, name='res1_1_conv1', relu=False, trainable=True)
             .batch_normalization(name='res1_1_bn1', relu=True, trainable=False)
             .conv(3, 3, 64, 1, 1, name='res1_1_conv2', relu=False, trainable=True)
             .batch_normalization(name='res1_1_bn2', relu=True,trainable=False)
             .conv(1, 1, 256, 1, 1, name='res1_1_conv3', relu=False, trainable=True)
             .batch_normalization(name='res1_1_bn3', relu=False, trainable=False)
        )
        (
        self.feed('transform1_bn', 'res1_1_bn3')
            .add(name='res1_1_add')
            .relu(name='res1_1_relu')
            .conv(1, 1, 64, 1, 1, name='res1_2_conv1', relu=False, trainable=True)
            .batch_normalization(name='res1_2_bn1', relu=True, trainable=False)
            .conv(3, 3, 64, 1, 1, name='res1_2_conv2', relu=False, trainable=True)
            .batch_normalization(name='res1_2_bn2', relu=True, trainable=False)
            .conv(1, 1, 256, 1, 1, name='res1_2_conv3', relu=False, trainable=True)
            .batch_normalization(name='res1_2_bn3', relu=False, trainable=False)
        )
        (
            self.feed('transform1_bn', 'res1_2_bn3')
                .add(name='res1_2_add')
                .relu(name='res1_2_relu')
                .conv(1, 1, 64, 1, 1, name='res1_3_conv1', relu=False, trainable=True)
                .batch_normalization(name='res1_3_bn1', relu=True, trainable=False)
                .conv(3, 3, 64, 1, 1, name='res1_3_conv2', relu=False, trainable=True)
                .batch_normalization(name='res1_3_bn2', relu=True, trainable=False)
                .conv(1, 1, 256, 1, 1, name='res1_3_conv3', relu=False, trainable=True)
                .batch_normalization(name='res1_3_bn3', relu=False, trainable=False)
        )
        # ======================计算残差变换形状适应第二组模块=======================
        (
        self.feed('transform1_bn', 'res1_3_bn3')
            .add(name='res1_3_add')
            .relu(name='res1_3_relu')
            .conv(1, 1, 512, 2, 2, name='transform2_conv', relu=False, trainable=True)
            .batch_normalization(name='transform2_bn', relu=False, trainable=False)
        )
        # ======================第二组模块===========================
        (
        self.feed('res1_3_relu')
            .conv(1, 1, 128, 2, 2, name='res2_1_conv1', relu=False, trainable=True)
            .batch_normalization(name='res2_1_bn1', relu=True, trainable=False)
            .conv(3, 3, 128, 1, 1, name='res2_1_conv2', relu=False, trainable=True)
            .batch_normalization(name='res2_1_bn2', relu=True, trainable=False)
            .conv(1, 1, 512, 1, 1, name='res2_1_conv3', relu=False, trainable=True)
            .batch_normalization(name='res2_1_bn3', relu=True, trainable=False)
        )
        (
            self.feed('transform2_bn', 'res2_1_bn3')
                .add(name='res2_1_add')
                .relu(name='res2_1_relu')
                .conv(1, 1, 128, 1, 1, name='res2_2_conv1', relu=False, trainable=True)
                .batch_normalization(name='res2_2_bn1', relu=True, trainable=False)
                .conv(3, 3, 128, 1, 1, name='res2_2_conv2', relu=False, trainable=True)
                .batch_normalization(name='res2_2_bn2', relu=True, trainable=False)
                .conv(1, 1, 512, 1, 1, name='res2_2_conv3', relu=False, trainable=True)
                .batch_normalization(name='res2_2_bn3', relu=False, trainable=False)
        )
        (
            self.feed('transform2_bn', 'res2_2_bn3')
                .add(name='res2_2_add')
                .relu(name='res2_2_relu')
                .conv(1, 1, 128, 1, 1, name='res2_3_conv1', relu=False, trainable=True)
                .batch_normalization(name='res2_3_bn1', relu=True, trainable=False)
                .conv(3, 3, 128, 1, 1, name='res2_3_conv2', relu=False, trainable=True)
                .batch_normalization(name='res2_3_bn2', relu=True, trainable=False)
                .conv(1, 1, 512, 1, 1, name='res2_3_conv3', relu=False, trainable=True)
                .batch_normalization(name='res2_3_bn3', relu=False, trainable=False)
        )
        (
            self.feed('transform2_bn', 'res2_3_bn3')
                .add(name='res2_3_add')
                .relu(name='res2_3_relu')
                .conv(1, 1, 128, 1, 1, name='res2_4_conv1', relu=False, trainable=True)
                .batch_normalization(name='res2_4_bn1', relu=True, trainable=False)
                .conv(3, 3, 128, 1, 1, name='res2_4_conv2', relu=False, trainable=True)
                .batch_normalization(name='res2_4_bn2', relu=True, trainable=False)
                .conv(1, 1, 512, 1, 1, name='res2_4_conv3', relu=False, trainable=True)
                .batch_normalization(name='res2_4_bn3', relu=False, trainable=False)
        )
        # ======================计算残差变换形状适应第三组模块=======================
        (
            self.feed('transform2_bn', 'res2_4_bn3')
                .add(name='res2_4_add')
                .relu(name='res2_4_relu')
                .conv(1, 1, 1024, 2, 2, name='transform3_conv', relu=False, trainable=True)
                .batch_normalization(name='transform3_bn', relu=False, trainable=False)
        )
        # ======================第三组模块===========================
        (
            self.feed('res2_4_relu')
                .conv(1, 1, 256, 2, 2, name='res3_1_conv1', relu=False, trainable=True)
                .batch_normalization(name='res3_1_bn1', relu=True, trainable=False)
                .conv(3, 3, 256, 1, 1, name='res3_1_conv2', relu=False, trainable=True)
                .batch_normalization(name='res3_1_bn2', relu=True, trainable=False)
                .conv(1, 1, 1024, 1, 1, name='res3_1_conv3', relu=False, trainable=True)
                .batch_normalization(name='res3_1_bn3', relu=True, trainable=False)
        )
        (
            self.feed('transform3_bn', 'res3_1_bn3')
                .add(name='res3_1_add')
                .relu(name='res3_1_relu')
                .conv(1, 1, 256, 1, 1, name='res3_2_conv1', relu=False, trainable=True)
                .batch_normalization(name='res3_2_bn1', relu=True, trainable=False)
                .conv(3, 3, 256, 1, 1, name='res3_2_conv2', relu=False, trainable=True)
                .batch_normalization(name='res3_2_bn2', relu=True, trainable=False)
                .conv(1, 1, 1024, 1, 1, name='res3_2_conv3', relu=False, trainable=True)
                .batch_normalization(name='res3_2_bn3', relu=False, trainable=False)
        )
        (
            self.feed('transform3_bn', 'res3_2_bn3')
                .add(name='res3_2_add')
                .relu(name='res3_2_relu')
                .conv(1, 1, 256, 1, 1, name='res3_3_conv1', relu=False, trainable=True)
                .batch_normalization(name='res3_3_bn1', relu=True, trainable=False)
                .conv(3, 3, 256, 1, 1, name='res3_3_conv2', relu=False, trainable=True)
                .batch_normalization(name='res3_3_bn2', relu=True, trainable=False)
                .conv(1, 1, 1024, 1, 1, name='res3_3_conv3', relu=False, trainable=True)
                .batch_normalization(name='res3_3_bn3', relu=False, trainable=False)
        )
        (
            self.feed('transform3_bn', 'res3_3_bn3')
                .add(name='res3_3_add')
                .relu(name='res3_3_relu')
                .conv(1, 1, 256, 1, 1, name='res3_4_conv1', relu=False, trainable=True)
                .batch_normalization(name='res3_4_bn1', relu=True, trainable=False)
                .conv(3, 3, 256, 1, 1, name='res3_4_conv2', relu=False, trainable=True)
                .batch_normalization(name='res3_4_bn2', relu=True, trainable=False)
                .conv(1, 1, 1024, 1, 1, name='res3_4_conv3', relu=False, trainable=True)
                .batch_normalization(name='res3_4_bn3', relu=False, trainable=False)
        )
        (
            self.feed('transform3_bn', 'res3_4_bn3')
                .add(name='res3_4_add')
                .relu(name='res3_4_relu')
                .conv(1, 1, 256, 1, 1, name='res3_5_conv1', relu=False, trainable=True)
                .batch_normalization(name='res3_5_bn1', relu=True, trainable=False)
                .conv(3, 3, 256, 1, 1, name='res3_5_conv2', relu=False, trainable=True)
                .batch_normalization(name='res3_5_bn2', relu=True, trainable=False)
                .conv(1, 1, 1024, 1, 1, name='res3_5_conv3', relu=False, trainable=True)
                .batch_normalization(name='res3_5_bn3', relu=False, trainable=False)
        )
        (
            self.feed('transform3_bn', 'res3_5_bn3')
                .add(name='res3_5_add')
                .relu(name='res3_5_relu')
                .conv(1, 1, 256, 1, 1, name='res3_6_conv1', relu=False, trainable=True)
                .batch_normalization(name='res3_6_bn1', relu=True, trainable=False)
                .conv(3, 3, 256, 1, 1, name='res3_6_conv2', relu=False, trainable=True)
                .batch_normalization(name='res3_6_bn2', relu=True, trainable=False)
                .conv(1, 1, 1024, 1, 1, name='res3_6_conv3', relu=False, trainable=True)
                .batch_normalization(name='res3_6_bn3', relu=False, trainable=False)
        )
        # ======================计算残差变换形状适应第四组模块=======================
        (
            self.feed('transform3_bn', 'res3_6_bn3')
                .add(name='res3_6_add')
                .relu(name='res3_6_relu')
                # .conv(1, 1, 2048, 2, 2, name='transform4_conv', relu=False, trainable=True)
                # .batch_normalization(name='transform4_bn', relu=False, trainable=False)
        )
        # # ======================第四组模块===========================
        # (
        #     self.feed('res3_6_relu')
        #         .conv(1, 1, 512, 2, 2, name='res4_1_conv1', relu=False, trainable=True)
        #         .batch_normalization(name='res4_1_bn1', relu=True, trainable=False)
        #         .conv(3, 3, 512, 1, 1, name='res4_1_conv2', relu=False, trainable=True)
        #         .batch_normalization(name='res4_1_bn2', relu=True, trainable=False)
        #         .conv(1, 1, 2048, 1, 1, name='res4_1_conv3', relu=False, trainable=True)
        #         .batch_normalization(name='res4_1_bn3', relu=True, trainable=False)
        # )
        # (
        #     self.feed('transform4_bn', 'res4_1_bn3')
        #         .add(name='res4_1_add')
        #         .relu(name='res4_1_relu')
        #         .conv(1, 1, 512, 1, 1, name='res4_2_conv1', relu=False, trainable=True)
        #         .batch_normalization(name='res4_2_bn1', relu=True, trainable=False)
        #         .conv(3, 3, 512, 1, 1, name='res4_2_conv2', relu=False, trainable=True)
        #         .batch_normalization(name='res4_2_bn2', relu=True, trainable=False)
        #         .conv(1, 1, 2048, 1, 1, name='res4_2_conv3', relu=False, trainable=True)
        #         .batch_normalization(name='res4_2_bn3', relu=False, trainable=False)
        # )
        # (
        #     self.feed('transform4_bn', 'res4_2_bn3')
        #         .add(name='res4_2_add')
        #         .relu(name='res4_2_relu')
        #         .conv(1, 1, 512, 1, 1, name='res4_3_conv1', relu=False, trainable=True)
        #         .batch_normalization(name='res4_3_bn1', relu=True, trainable=False)
        #         .conv(3, 3, 512, 1, 1, name='res4_3_conv2', relu=False, trainable=True)
        #         .batch_normalization(name='res4_3_bn2', relu=True, trainable=False)
        #         .conv(1, 1, 2048, 1, 1, name='res4_3_conv3', relu=False, trainable=True)
        #         .batch_normalization(name='res4_3_bn3', relu=False, trainable=False)
        # )
        # # ======================计算残差变换结束模块=======================
        # (
        #     self.feed('transform4_bn', 'res4_3_bn3')
        #         .add(name='res4_3_add')
        #         .relu(name='res4_3_relu')
        # )

        # ======================缺陷检测RPN===================================

        (
            self.feed('res3_6_relu')
                .conv(3, 3, 512, 1, 1, name='rpn_conv', relu=True, trainable=True)
                .conv(1, 1, len(cfg.ZLRM.ANCHOR_SCALE)*len(cfg.ZLRM.ANCHOR_RATIO)*2, 1, 1, name='rpn_cls_score', relu=False, padding='VALID', trainable=True)
        )
        (
            self.feed('rpn_conv')
                .conv(1, 1, len(cfg.ZLRM.ANCHOR_SCALE)*len(cfg.ZLRM.ANCHOR_RATIO)*4, 1, 1, name='rpn_bbox_pred', relu=False, padding='VALID', trainable=True)
        )
        (
            self.feed('rpn_cls_score', 'gt_boxes', 'im_info')
                .anchor_target_layer(cfg.ZLRM.RESNET_50_FEAT_STRIDE, cfg.ZLRM.ANCHOR_SCALE, name='rpn-data')
         )
        # ========= RoI Proposal ============

        (
            self.feed('rpn_cls_score')
                .spatial_reshape_layer(2, name='rpn_cls_score_reshape')
                .spatial_softmax(name='rpn_cls_prob')
        )

        (
            self.feed('rpn_cls_prob')
                .spatial_reshape_layer(len(cfg.ZLRM.ANCHOR_SCALE)*len(cfg.ZLRM.ANCHOR_RATIO) * 2, name='rpn_cls_prob_reshape')
        )

        (
            self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
                .proposal_layer(cfg_key=True, _feat_stride=cfg.ZLRM.RESNET_50_FEAT_STRIDE, anchor_scales=cfg.ZLRM.ANCHOR_SCALE, name='rpn_rois')
        )
        (

            self.feed('rpn_rois', 'gt_boxes')
                .proposal_target_layer((cfg.ZLRM.N_CLASSES + 1), name='roi-data')

        )