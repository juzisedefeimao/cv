from lib.networks.network import Network
import numpy as np
from lib.networks.netconfig import cfg
import tensorflow as tf

class Resnet50_test_detect(Network):
    def __init__(self):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.layers = {'data': self.data, 'im_info': self.im_info}
        self.setup()

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
                .conv(1, 1, 2048, 1, 1, name='transform4_conv', relu=False, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
                .batch_normalization(name='transform4_bn', relu=False, trainable=False)
        )
        # ======================第四组模块===========================
        (
            self.feed('res3_6_relu')
                .conv(1, 1, 512, 1, 1, name='res4_1_conv1', relu=False, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
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

        # ======================缺陷检测RPN===================================

        (
            self.feed('res3_6_relu')
                .conv(3, 3, 512, 1, 1, name='rpn_conv', relu=True, trainable=cfg.ZLRM.TRAIN.RPN_TRAIN)
                .conv(1, 1, len(cfg.ZLRM.ANCHOR_SCALE)*len(cfg.ZLRM.ANCHOR_RATIO)*2, 1, 1, name='rpn_cls_score', relu=False, padding='VALID', trainable=cfg.ZLRM.TRAIN.RPN_TRAIN)
        )
        (
            self.feed('rpn_conv')
                .conv(1, 1, len(cfg.ZLRM.ANCHOR_SCALE)*len(cfg.ZLRM.ANCHOR_RATIO)*4, 1, 1, name='rpn_bbox_pred', relu=False, padding='VALID', trainable=cfg.ZLRM.TRAIN.RPN_TRAIN)
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
                .proposal_layer(cfg_key=False, _feat_stride=cfg.ZLRM.RESNET_50_FEAT_STRIDE, anchor_scales=cfg.ZLRM.ANCHOR_SCALE, name='rois')
        )
        # ===================newconv==============
        (
            self.feed('res4_3_relu')
                .conv(1, 1, 1024, 1, 1, name='conv_new_1', relu=True, trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
                .conv(1, 1, (cfg.ZLRM.N_CLASSES + 1) * cfg.ZLRM.PSROIPOOL * cfg.ZLRM.PSROIPOOL, 1, 1, name='rfcn_cls', trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
        )
        (
            self.feed('conv_new_1')
                .conv(1, 1, 4 * (cfg.ZLRM.N_CLASSES + 1) * cfg.ZLRM.PSROIPOOL * cfg.ZLRM.PSROIPOOL, 1, 1, name='rfcn_bbox', trainable=cfg.ZLRM.TRAIN.ROI_TRAIN)
        )
        # ========================position sensitive RoI pooling======================
        (
            self.feed('rfcn_cls', 'rois')
                .psroi_pool(output_dim=(cfg.ZLRM.N_CLASSES + 1), group_size=cfg.ZLRM.PSROIPOOL, spatial_scale=0.0625, name='psroipooled_cls_rois')
                .avg_pool(cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, name='ave_cls_score_rois')
                .softmax(name='cls_prob')
        )
        (
            self.feed('rfcn_bbox', 'rois')
                .psroi_pool(output_dim=(cfg.ZLRM.N_CLASSES + 1) * 4, group_size=cfg.ZLRM.PSROIPOOL, spatial_scale=0.0625, name='psroipooled_loc_rois')
                .avg_pool(cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, cfg.ZLRM.PSROIPOOL, name='ave_bbox_pred_rois')
        )