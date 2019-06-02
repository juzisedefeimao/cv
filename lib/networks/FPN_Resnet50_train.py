from lib.networks.network import Network
import numpy as np
from lib.networks.netconfig import cfg
import tensorflow as tf

n_classes = cfg.ZLRM.N_CLASSES
class FPN_Resnet50_train(Network):
    def __init__(self):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')
        self.layers = {'data': self.data, 'im_info': self.im_info, 'gt_boxes': self.gt_boxes}
        self.setup()

    def build_loss(self):
        # ================================RPN======================================
        rpn_cls_score = tf.reshape(self.get_output('rpn_cls_score_reshape_reshape_concat'), [-1, 2])  # shape (HxWxA, 2)
        rpn_label = tf.reshape(self.get_output('rpn-data')[0], [-1])  # shape (HxWxA)
        # ignore_label(-1)
        fg_keep = tf.equal(rpn_label, 1)
        rpn_keep = tf.where(tf.not_equal(rpn_label, -1))
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_keep), [-1, 2])  # shape (N, 2)
        rpn_label = tf.reshape(tf.gather(rpn_label, rpn_keep), [-1])
        rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label)
        rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_n)

        # box loss
        rpn_bbox_pred = self.get_output('rpn_bbox_pred_reshape_concat')  # shape (1, H, W, Ax4)
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
        cls_score = tf.reshape(self.get_output('cls_score'), [-1, cfg.ZLRM.N_CLASSES + 1]) # (R, C+1)
        label = tf.reshape(self.get_output('roi-data')[5], [-1])  # (R)
        cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label)
        cross_entropy = tf.reduce_mean(cross_entropy_n)

        # bounding box regression L1 loss
        bbox_pred = tf.reshape(self.get_output('bbox_pred'), [-1, (cfg.ZLRM.N_CLASSES + 1) * 4]) # (R, (C+1)x4)
        bbox_targets = self.get_output('roi-data')[6]  # (R, (C+1)x4)
        # each element is {0, 1}, represents background (0), objects (1)
        bbox_inside_weights = self.get_output('roi-data')[7]  # (R, (C+1)x4)
        bbox_outside_weights = self.get_output('roi-data')[8]  # (R, (C+1)x4)

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
        return loss, cross_entropy, loss_box, rpn_cross_entropy, rpn_loss_box, label

    def setup(self):
        bn_trainable = False
        (
         self.feed('data')
             .conv(7, 7, 64, 2, 2,name='conv1', relu=False, trainable=True)
             .batch_normalization(name='bn1', relu=True, trainable=bn_trainable)
             .max_pool(3, 3, 2, 2, name='pool1', padding='VALID')
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
            .batch_normalization(name='res2_1_bn3', relu=True, trainable=bn_trainable)
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
                .conv(1, 1, 1024, 2, 2, name='transform3_conv', biased=False, relu=False, trainable=True)
                .batch_normalization(name='transform3_bn', relu=False, trainable=bn_trainable)
        )
        # ======================第三组模块===========================
        (
            self.feed('res2_4_relu')
                .conv(1, 1, 256, 2, 2, name='res3_1_conv1', biased=False, relu=False, trainable=True)
                .batch_normalization(name='res3_1_bn1', relu=True, trainable=bn_trainable)
                .conv(3, 3, 256, 1, 1, name='res3_1_conv2', biased=False, relu=False, trainable=True)
                .batch_normalization(name='res3_1_bn2', relu=True, trainable=bn_trainable)
                .conv(1, 1, 1024, 1, 1, name='res3_1_conv3', biased=False, relu=False, trainable=True)
                .batch_normalization(name='res3_1_bn3', relu=True, trainable=bn_trainable)
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
                .conv(1, 1, 2048, 2, 2, name='transform4_conv', biased=False, relu=False, trainable=True)
                .batch_normalization(name='transform4_bn', relu=False, trainable=bn_trainable)
        )
        # ======================第四组模块===========================
        (
            self.feed('res3_6_relu')
                .conv(1, 1, 512, 2, 2, name='res4_1_conv1', biased=False, relu=False, trainable=True)
                .batch_normalization(name='res4_1_bn1', relu=True, trainable=bn_trainable)
                .conv(3, 3, 512, 1, 1, name='res4_1_conv2', biased=False, relu=False, trainable=True)
                .batch_normalization(name='res4_1_bn2', relu=True, trainable=bn_trainable)
                .conv(1, 1, 2048, 1, 1, name='res4_1_conv3', biased=False, relu=False, trainable=True)
                .batch_normalization(name='res4_1_bn3', relu=True, trainable=bn_trainable)
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


        with tf.variable_scope('RPN') as scope:
            #========= RPN ============
            # P2
            (
                self.feed('P2')
                    .conv(3,3,512,1,1,name='rpn_conv/P2',reuse=True)
                    .conv(1,1,len(cfg.ZLRM.ANCHOR_RATIO)*2, 1, 1, padding='VALID', relu = False, name='rpn_cls_score/P2', reuse=True)
            )

            (
                self.feed('rpn_conv/P2')
                    .conv(1,1,len(cfg.ZLRM.ANCHOR_RATIO)*4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred/P2', reuse=True)
            )

            (
                self.feed('rpn_cls_score/P2')
                    .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape/P2')
                    .spatial_softmax(name='rpn_cls_prob/P2')
            )

            (
                self.feed('rpn_cls_prob/P2')
                    .spatial_reshape_layer(len(cfg.ZLRM.ANCHOR_RATIO)*2, name = 'rpn_cls_prob_reshape/P2')
            )

            scope.reuse_variables()

            # P3
            (
                self.feed('P3')
                    .conv(3,3,512,1,1,name='rpn_conv/P3', reuse=True)
                    .conv(1,1,len(cfg.ZLRM.ANCHOR_RATIO)*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score/P3', reuse=True)
            )

            (
                self.feed('rpn_conv/P3')
                    .conv(1,1,len(cfg.ZLRM.ANCHOR_RATIO)*4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred/P3', reuse=True)
            )

            (
                self.feed('rpn_cls_score/P3')
                    .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape/P3')
                    .spatial_softmax(name='rpn_cls_prob/P3')
            )

            (
                self.feed('rpn_cls_prob/P3')
                    .spatial_reshape_layer(len(cfg.ZLRM.ANCHOR_RATIO)*2, name = 'rpn_cls_prob_reshape/P3')
            )

            # P4
            (
                self.feed('P4')
                    .conv(3,3,512,1,1,name='rpn_conv/P4',reuse=True)
                    .conv(1,1,len(cfg.ZLRM.ANCHOR_RATIO)*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score/P4', reuse=True)
            )

            (
                self.feed('rpn_conv/P4')
                    .conv(1,1,len(cfg.ZLRM.ANCHOR_RATIO)*4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred/P4', reuse=True)
            )

            (
                self.feed('rpn_cls_score/P4')
                    .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape/P4')
                    .spatial_softmax(name='rpn_cls_prob/P4')
            )

            (
                self.feed('rpn_cls_prob/P4')
                    .spatial_reshape_layer(len(cfg.ZLRM.ANCHOR_RATIO)*2, name = 'rpn_cls_prob_reshape/P4')
            )

            # P5
            (
                self.feed('P5')
                    .conv(3,3,512,1,1,name='rpn_conv/P5', reuse=True)
                    .conv(1,1,len(cfg.ZLRM.ANCHOR_RATIO)*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score/P5', reuse=True)
            )

            (
                self.feed('rpn_conv/P5')
                    .conv(1,1,len(cfg.ZLRM.ANCHOR_RATIO)*4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred/P5', reuse=True)
            )

            (
                self.feed('rpn_cls_score/P5')
                    .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape/P5')
                    .spatial_softmax(name='rpn_cls_prob/P5')
            )

            (
                self.feed('rpn_cls_prob/P5')
                    .spatial_reshape_layer(len(cfg.ZLRM.ANCHOR_RATIO)*2, name = 'rpn_cls_prob_reshape/P5')
            )

            # P6
            (
                self.feed('P6')
                    .conv(3,3,512,1,1,name='rpn_conv/P6', reuse=True)
                    .conv(1,1,len(cfg.ZLRM.ANCHOR_RATIO)*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score/P6', reuse=True)
            )

            (
                self.feed('rpn_conv/P6')
                    .conv(1,1,len(cfg.ZLRM.ANCHOR_RATIO)*4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred/P6', reuse=True)
            )

            (
                self.feed('rpn_cls_score/P6')
                    .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape/P6')
                    .spatial_softmax(name='rpn_cls_prob/P6')
            )

            (
                self.feed('rpn_cls_prob/P6')
                    .spatial_reshape_layer(len(cfg.ZLRM.ANCHOR_RATIO)*2, name = 'rpn_cls_prob_reshape/P6')
            )



            (
                self.feed('rpn_cls_score_reshape/P2')
                    .reshape_layer([-1, 2], name = 'rpn_cls_score_reshape_reshape/P2')
            )

            (
                self.feed('rpn_cls_score_reshape/P3')
                    .reshape_layer([-1, 2], name = 'rpn_cls_score_reshape_reshape/P3')
            )

            (
                self.feed('rpn_cls_score_reshape/P4')
                    .reshape_layer([-1, 2], name = 'rpn_cls_score_reshape_reshape/P4')
            )

            (
                self.feed('rpn_cls_score_reshape/P5')
                    .reshape_layer([-1, 2], name = 'rpn_cls_score_reshape_reshape/P5')
            )

            (
                self.feed('rpn_cls_score_reshape/P6')
                    .reshape_layer([-1, 2], name = 'rpn_cls_score_reshape_reshape/P6')
            )

            (
                self.feed('rpn_cls_score_reshape_reshape/P2',
                          'rpn_cls_score_reshape_reshape/P3',
                          'rpn_cls_score_reshape_reshape/P4',
                          'rpn_cls_score_reshape_reshape/P5',
                          'rpn_cls_score_reshape_reshape/P6')
                    .concat(0, name = 'rpn_cls_score_reshape_reshape_concat')
            )


            (
                self.feed('rpn_bbox_pred/P2')
                    .reshape_layer([-1, 4], name = 'rpn_bbox_pred_reshape/P2')
            )

            (
                self.feed('rpn_bbox_pred/P3')
                    .reshape_layer([-1, 4], name = 'rpn_bbox_pred_reshape/P3')
            )

            (
                self.feed('rpn_bbox_pred/P4')
                    .reshape_layer([-1, 4], name = 'rpn_bbox_pred_reshape/P4')
            )

            (
                self.feed('rpn_bbox_pred/P5')
                    .reshape_layer([-1, 4], name = 'rpn_bbox_pred_reshape/P5')
            )

            (
                self.feed('rpn_bbox_pred/P6')
                    .reshape_layer([-1, 4], name = 'rpn_bbox_pred_reshape/P6')
            )

            (
                self.feed('rpn_bbox_pred_reshape/P2',
                          'rpn_bbox_pred_reshape/P3',
                          'rpn_bbox_pred_reshape/P4',
                          'rpn_bbox_pred_reshape/P5',
                          'rpn_bbox_pred_reshape/P6')
                    .concat(0, name = 'rpn_bbox_pred_reshape_concat')
            )


            (
                self.feed('rpn_cls_score/P2',
                          'rpn_cls_score/P3',
                          'rpn_cls_score/P4',
                          'rpn_cls_score/P5',
                          'rpn_cls_score/P6',
                          'gt_boxes', 'im_info')
                    .fpn_anchor_target_layer(cfg.ZLRM.FPN_FEAT_STRIDE[2:], cfg.ZLRM.FPN_ANCHOR_SIZE[2:], name = 'rpn-data')
            )

            #========= RoI Proposal ============
            (
                self.feed('rpn_cls_prob_reshape/P2', 'rpn_bbox_pred/P2',
                          'rpn_cls_prob_reshape/P3', 'rpn_bbox_pred/P3',
                          'rpn_cls_prob_reshape/P4', 'rpn_bbox_pred/P4',
                          'rpn_cls_prob_reshape/P5', 'rpn_bbox_pred/P5',
                          'rpn_cls_prob_reshape/P6', 'rpn_bbox_pred/P6',
                          'im_info')
                    .fpn_proposal_layer(cfg_train_key=True, _feat_strides=cfg.ZLRM.FPN_FEAT_STRIDE[2:],
                                        anchor_sizes=cfg.ZLRM.FPN_ANCHOR_SIZE[2:], name = 'rpn_rois')
            )

            (
                self.feed('rpn_rois','gt_boxes')
                    .fpn_proposal_target_layer((cfg.ZLRM.N_CLASSES + 1),name = 'roi-data')
            )

        # ===============================================newconv==============================================================
        with tf.variable_scope('Fast-RCNN'):
            # ========= RCNN ============
            (self.feed('P2', 'P3', 'P4', 'P5', 'P6', 'roi-data')
             .fpn_roi_pool(7, 7, name='fpn_roi_pooling')
             .fc(1024, name='fc6')
             .fc(1024, name='fc7')
             .fc(n_classes+1, relu=False, name='cls_score')
             .softmax(name='cls_prob'))

            (self.feed('fc7')
             .fc((n_classes+1) * 4, relu=False, name='bbox_pred'))