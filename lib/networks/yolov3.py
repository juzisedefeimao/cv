from lib.networks.network import Network
import numpy as np
from lib.networks.netconfig import cfg
import tensorflow as tf

class Resnet50_train(Network):
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

        ############# R-CNN
        # classification loss
        cls_score = tf.reshape(self.get_output('ave_cls_score_rois'), [-1, cfg.ZLRM.N_CLASSES + 1]) # (R, C+1)
        label = tf.reshape(self.get_output('roi-data')[1], [-1])  # (R)
        cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label)
        cross_entropy = tf.reduce_mean(cross_entropy_n)

        # bounding box regression L1 loss
        bbox_pred = tf.reshape(self.get_output('ave_bbox_pred_rois'), [-1, (cfg.ZLRM.N_CLASSES + 1) * 4]) # (R, (C+1)x4)
        bbox_targets = self.get_output('roi-data')[2]  # (R, (C+1)x4)
        # each element is {0, 1}, represents background (0), objects (1)
        bbox_inside_weights = self.get_output('roi-data')[3]  # (R, (C+1)x4)
        bbox_outside_weights = self.get_output('roi-data')[4]  # (R, (C+1)x4)

        loss_box_n = tf.reduce_sum( \
            bbox_outside_weights * self.smooth_l1_dist(bbox_inside_weights * (bbox_pred - bbox_targets)), \
            axis=[1])

        loss_n = loss_box_n + cross_entropy_n
        loss_n = tf.reshape(loss_n, [-1])

        loss_box = tf.reduce_mean(loss_box_n)


        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

        # add regularizer
        if cfg.TRAIN.WEIGHT_DECAY > 0:
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.add_n(regularization_losses) + loss
        # loss = rpn_cross_entropy + rpn_loss_box
        return loss, cross_entropy, loss_box, rpn_cross_entropy, rpn_loss_box

    def setup(self):
        bn_trainable = True
        (
         self.feed('data')
             .conv(3, 3, 32, 1, 1,name='conv1', relu=False, trainable=True)
             .batch_normalization(name='bn1', relu=False, trainable=bn_trainable)
             .lrelu(alpha=0.01, name='leaky_relu1')
        )
        # ======================变换形状适应第一组模块=======================
        (
         self.feed('leaky_relu1')
             .conv(3, 3, 64, 2, 2, name='transform1_conv', relu=False, trainable=True)
             .batch_normalization(name='transform1_bn', relu=False, trainable=bn_trainable)
             .lrelu(alpha=0.01, name='transform1_lr')
        )
        # ======================第一组模块===========================
        (
         self.feed('transform1_lr')
             .conv(1, 1, 32, 1, 1, name='dark1_1_conv1', relu=False, trainable=True)
             .batch_normalization(name='dark1_1_bn1', relu=False, trainable=bn_trainable)
             .lrelu(alpha=0.01, name='dark1_1_lr1')
             .conv(3, 3, 64, 1, 1, name='dark1_1_conv2', relu=False, trainable=True)
             .batch_normalization(name='dark1_1_bn2', relu=False,trainable=bn_trainable)
             .lrelu(alpha=0.01, name='dark1_1_lr2')
        )
        # ======================计算残差变换形状适应第二组模块=======================
        (
        self.feed('transform1_lr', 'dark1_1_lr2')
            .add(name='dark1_1_add')
            .relu(name='dark1_1_relu')
            .conv(3, 3, 128, 2, 2, name='transform2_conv', relu=False, trainable=True)
            .batch_normalization(name='transform2_bn', relu=False, trainable=bn_trainable)
            .lrelu(alpha=0.01, name='transform2_lr')
        )
        # ======================第二组模块===========================
        (
        self.feed('transform2_lr')
            .conv(1, 1, 64, 1, 1, name='dark2_1_conv1', relu=False, trainable=True)
            .batch_normalization(name='dark2_1_bn1', relu=False, trainable=bn_trainable)
            .lrelu(alpha=0.01, name='dark2_1_lr1')
            .conv(3, 3, 128, 1, 1, name='dark2_1_conv2', relu=False, trainable=True)
            .batch_normalization(name='dark2_1_bn2', relu=False, trainable=bn_trainable)
            .lrelu(alpha=0.01, name='dark2_1_lr2')
        )
        (
            self.feed('transform2_lr', 'dark2_1_lr2')
                .add(name='dark2_1_add')
                .relu(name='dark2_1_relu')
                .conv(1, 1, 64, 1, 1, name='dark2_2_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark2_2_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark2_2_lr1')
                .conv(3, 3, 128, 1, 1, name='dark2_2_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark2_2_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark2_2_lr2')
        )
        # ======================计算残差变换形状适应第三组模块=======================
        (
            self.feed('transform2_lr', 'dark2_2_lr2')
                .add(name='dark2_2_add')
                .relu(name='dark2_2_relu')
                .conv(3, 3, 256, 2, 2, name='transform3_conv', relu=False, trainable=True)
                .batch_normalization(name='transform3_bn', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='transform3_lr')
        )
        # ======================第三组模块===========================
        (
            self.feed('transform3_lr')
                .conv(1, 1, 128, 1, 1, name='dark3_1_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark3_1_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark3_1_lr1')
                .conv(3, 3, 256, 1, 1, name='dark3_1_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark3_1_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark3_1_lr2')
        )
        (
            self.feed('transform3_lr', 'dark3_1_lr2')
                .add(name='dark3_1_add')
                .relu(name='dark3_1_relu')
                .conv(1, 1, 128, 1, 1, name='dark3_2_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark3_2_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark3_2_lr1')
                .conv(3, 3, 256, 1, 1, name='dark3_2_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark3_2_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark3_2_lr2')
        )
        (
            self.feed('transform3_lr', 'dark3_2_lr2')
                .add(name='dark3_2_add')
                .relu(name='dark3_2_relu')
                .conv(1, 1, 128, 1, 1, name='dark3_3_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark3_3_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark3_3_lr1')
                .conv(3, 3, 256, 1, 1, name='dark3_3_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark3_3_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark3_3_lr2')
        )
        (
            self.feed('transform3_lr', 'dark3_3_lr2')
                .add(name='dark3_3_add')
                .relu(name='dark3_3_relu')
                .conv(1, 1, 128, 1, 1, name='dark3_4_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark3_4_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark3_4_lr1')
                .conv(3, 3, 256, 1, 1, name='dark3_4_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark3_4_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark3_4_lr2')
        )
        (
            self.feed('transform3_lr', 'dark3_4_lr2')
                .add(name='dark3_4_add')
                .relu(name='dark3_4_relu')
                .conv(1, 1, 128, 1, 1, name='dark3_5_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark3_5_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark3_5_lr1')
                .conv(3, 3, 256, 1, 1, name='dark3_5_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark3_5_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark3_5_lr2')
        )
        (
            self.feed('transform3_lr', 'dark3_5_lr2')
                .add(name='dark3_5_add')
                .relu(name='dark3_5_relu')
                .conv(1, 1, 128, 1, 1, name='dark3_6_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark3_6_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark3_6_lr1')
                .conv(3, 3, 256, 1, 1, name='dark3_6_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark3_6_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark3_6_lr2')
        )
        (
            self.feed('transform3_lr', 'dark3_6_lr2')
                .add(name='dark3_6_add')
                .relu(name='dark3_6_relu')
                .conv(1, 1, 128, 1, 1, name='dark3_7_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark3_7_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark3_7_lr1')
                .conv(3, 3, 256, 1, 1, name='dark3_7_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark3_7_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark3_7_lr2')
        )
        (
            self.feed('transform3_lr', 'dark3_7_lr2')
                .add(name='dark3_7_add')
                .relu(name='dark3_7_relu')
                .conv(1, 1, 128, 1, 1, name='dark3_8_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark3_8_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark3_8_lr1')
                .conv(3, 3, 256, 1, 1, name='dark3_8_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark3_8_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark3_8_lr2')
        )
        # ======================计算残差变换形状适应第四组模块=======================
        (
            self.feed('transform3_lr', 'dark3_8_lr2')
                .add(name='dark3_8_add')
                .relu(name='dark3_8_relu')
                .conv(3, 3, 512, 2, 2, name='transform4_conv', relu=False, trainable=True)
                .batch_normalization(name='transform4_bn', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='transform4_lr')
        )
        # ======================第四组模块===========================
        (
            self.feed('transform4_lr')
                .conv(1, 1, 256, 1, 1, name='dark4_1_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark4_1_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark4_1_lr1')
                .conv(3, 3, 512, 1, 1, name='dark4_1_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark4_1_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark4_1_lr2')
        )
        (
            self.feed('transform4_lr', 'dark4_1_lr2')
                .add(name='dark4_1_add')
                .relu(name='dark4_1_relu')
                .conv(1, 1, 256, 1, 1, name='dark4_2_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark4_2_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark4_2_lr1')
                .conv(3, 3, 512, 1, 1, name='dark4_2_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark4_2_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark4_2_lr2')
        )
        (
            self.feed('transform4_lr', 'dark4_2_lr2')
                .add(name='dark4_2_add')
                .relu(name='dark4_2_relu')
                .conv(1, 1, 256, 1, 1, name='dark4_3_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark4_3_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark4_3_lr1')
                .conv(3, 3, 512, 1, 1, name='dark4_3_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark4_3_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark4_3_lr2')
        )
        (
            self.feed('transform4_lr', 'dark4_3_lr2')
                .add(name='dark4_3_add')
                .relu(name='dark4_3_relu')
                .conv(1, 1, 256, 1, 1, name='dark4_4_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark4_4_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark4_4_lr1')
                .conv(3, 3, 512, 1, 1, name='dark4_4_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark4_4_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark4_4_lr2')
        )
        (
            self.feed('transform4_lr', 'dark4_4_lr2')
                .add(name='dark4_4_add')
                .relu(name='dark4_4_relu')
                .conv(1, 1, 256, 1, 1, name='dark4_5_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark4_5_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark4_5_lr1')
                .conv(3, 3, 512, 1, 1, name='dark4_5_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark4_5_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark4_5_lr2')
        )
        (
            self.feed('transform4_lr', 'dark4_5_lr2')
                .add(name='dark4_5_add')
                .relu(name='dark4_5_relu')
                .conv(1, 1, 256, 1, 1, name='dark4_6_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark4_6_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark4_6_lr1')
                .conv(3, 3, 512, 1, 1, name='dark4_6_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark4_6_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark4_6_lr2')
        )
        (
            self.feed('transform4_lr', 'dark4_6_lr2')
                .add(name='dark4_6_add')
                .relu(name='dark4_6_relu')
                .conv(1, 1, 256, 1, 1, name='dark4_7_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark4_7_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark4_7_lr1')
                .conv(3, 3, 512, 1, 1, name='dark4_7_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark4_7_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark4_7_lr2')
        )
        (
            self.feed('transform4_lr', 'dark4_7_lr2')
                .add(name='dark4_7_add')
                .relu(name='dark4_7_relu')
                .conv(1, 1, 256, 1, 1, name='dark4_8_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark4_8_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark4_8_lr1')
                .conv(3, 3, 512, 1, 1, name='dark4_8_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark4_8_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark4_8_lr2')
        )

        # ======================计算残差变换第五模块=======================
        (
            self.feed('transform4_lr', 'res4_8_lr3')
                .add(name='res4_8_add')
                .relu(name='res4_8_relu')
                .conv(3, 3, 1024, 2, 2, name='transform5_conv', relu=False, trainable=True)
                .batch_normalization(name='transform5_bn', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='transform5_lr')
        )
        # ======================第五组模块===========================
        (
            self.feed('transform5_lr')
                .conv(1, 1, 512, 1, 1, name='dark5_1_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark5_1_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark5_1_lr1')
                .conv(3, 3, 1024, 1, 1, name='dark5_1_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark5_1_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark5_1_lr2')
        )
        (
            self.feed('transform5_lr', 'dark5_1_lr2')
                .add(name='dark5_1_add')
                .relu(name='dark5_1_relu')
                .conv(1, 1, 512, 1, 1, name='dark5_2_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark5_2_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark5_2_lr1')
                .conv(3, 3, 1024, 1, 1, name='dark5_2_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark5_2_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark5_2_lr2')
        )
        (
            self.feed('transform5_lr', 'dark5_2_lr2')
                .add(name='dark5_2_add')
                .relu(name='dark5_2_relu')
                .conv(1, 1, 512, 1, 1, name='dark5_3_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark5_3_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark5_3_lr1')
                .conv(3, 3, 1024, 1, 1, name='dark5_3_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark5_3_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark5_3_lr2')
        )
        (
            self.feed('transform5_lr', 'dark5_3_lr2')
                .add(name='dark5_3_add')
                .relu(name='dark5_3_relu')
                .conv(1, 1, 512, 1, 1, name='dark5_4_conv1', relu=False, trainable=True)
                .batch_normalization(name='dark5_4_bn1', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark5_4_lr1')
                .conv(3, 3, 1024, 1, 1, name='dark5_4_conv2', relu=False, trainable=True)
                .batch_normalization(name='dark5_4_bn2', relu=False, trainable=bn_trainable)
                .lrelu(alpha=0.01, name='dark5_4_lr2')
        )
        (
            self.feed('transform5_lr', 'dark5_4_lr2')
                .add(name='dark5_4_add')
                .relu(name='dark5_4_relu')
        )