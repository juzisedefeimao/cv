# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import numpy as np
import numpy.random as npr

from lib.Siammask_rpn_msr.generate_anchors import generate_anchors

from lib.rpn_msr.bbox import bbox_overlaps, bbox_intersections

from lib.networks.netconfig import cfg
from lib.rpn_msr.bbox_transform import bbox_transform
# <<<< obsolete

DEBUG = False

def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride = [8,]):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    Parameters
    ----------
    rpn_cls_score: (batch*clases, H, W, Ax2) bg/fg scores of previous conv layer
    gt_boxes: (batch*classes, G, 5) vstack of [x1, y1, x2, y2, class]
    im_info: a list of [image_height, image_width, scale_ratios]
    _feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    rpn_labels : (HxWxA, 1), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
    rpn_bbox_targets: (HxWxA, 4), distances of the anchors to the gt_boxes(may contains some transform)
                            that are the regression objectives
    rpn_bbox_inside_weights: (HxWxA, 4) weights of each boxes, mainly accepts hyper param in cfg
    rpn_bbox_outside_weights: (HxWxA, 4) used to balance the fg/bg,
                            beacuse the numbers of bgs and fgs mays significiantly different
    """
    _anchors = generate_anchors(scales=cfg.SIAMSE.ANCHOR_SCALE) #[classes_num, anchors_num]
    _num_anchors = [_anchors[i].shape[0] for i in range(len(_anchors))] #[classes_num]


    # allow boxes to sit over the edge by a small amount
    _allowed_border =  0
    # map of shape (..., H, W)
    #height, width = rpn_cls_score.shape[1:3]

    im_info = im_info[0]

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap



    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]
    batch_classes = rpn_cls_score.shape[0]

    # reshape gt_boxes
    gt_boxes_shape = gt_boxes.shape
    gt_boxes_reshape = np.reshape(gt_boxes,[-1, cfg.SIAMSE.N_CLASSES, gt_boxes_shape[1], gt_boxes_shape[2]])
    gt_boxes_reshape_transpose = np.transpose(gt_boxes_reshape, [1, 0, 2, 3])
    gt_boxes_reshape_transpose_shape = gt_boxes_reshape_transpose.shape


    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y) # in W H order
    # K is H x W
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # all_anchors = []
    # total_anchors = []
    # overlaps = []
    labels = []
    bbox_targets = []
    bbox_inside_weights = []
    bbox_outside_weights = []
    for i in range(len(_num_anchors)):
        A = _num_anchors[i]
        K = shifts.shape[0]
        all_anchors_i = (_anchors[i].reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors_i = all_anchors_i.reshape((K * A, 4))
        total_anchors_i = int(K * A)
        # all_anchors.append(all_anchors_i)
        # total_anchors.append(total_anchors_i)

        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors_i[:, 0] >= -_allowed_border) &
            (all_anchors_i[:, 1] >= -_allowed_border) &
            (all_anchors_i[:, 2] < im_info[1] + _allowed_border) &  # width
            (all_anchors_i[:, 3] < im_info[0] + _allowed_border)    # height
        )[0]

        # keep only inside anchors
        anchors_i = all_anchors_i[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        # (A)
        labels_i = np.empty((gt_boxes_reshape_transpose_shape[1], len(inds_inside)), dtype=np.float32)
        labels_i.fill(-1)
        bbox_targets_i = np.zeros((gt_boxes_reshape_transpose_shape[1], len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights_i = np.zeros((gt_boxes_reshape_transpose_shape[1], len(inds_inside), 4), dtype=np.float32)
        bbox_outside_weights_i = np.zeros((gt_boxes_reshape_transpose_shape[1], len(inds_inside), 4), dtype=np.float32)
        # 因为numpy矩阵赋值不能赋给形状不同的变量所以生成新形状的矩阵用来赋值
        labels_i_shape = np.empty((gt_boxes_reshape_transpose_shape[1], total_anchors_i), dtype=np.float32)
        labels_i_shape.fill(-1)
        bbox_targets_i_shape = np.zeros((gt_boxes_reshape_transpose_shape[1], total_anchors_i, 4), dtype=np.float32)
        bbox_inside_weights_i_shape = np.zeros((gt_boxes_reshape_transpose_shape[1], total_anchors_i, 4), dtype=np.float32)
        bbox_outside_weights_i_shape = np.zeros((gt_boxes_reshape_transpose_shape[1], total_anchors_i, 4), dtype=np.float32)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt), shape is A x G
        for j in range(len(gt_boxes_reshape_transpose[i])):
            overlaps = bbox_overlaps(
                np.ascontiguousarray(anchors_i, dtype=np.float),
                np.ascontiguousarray(gt_boxes_reshape_transpose[i][j], dtype=np.float))

            argmax_overlaps = overlaps.argmax(axis=1)  # (A)
            max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
            if (overlaps > 0).any():
                gt_argmax_overlaps = overlaps.argmax(axis=0) # G
                gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                           np.arange(overlaps.shape[1])]
                gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

                # fg label: for each gt, anchor with highest overlap
                labels_i[j][gt_argmax_overlaps] = 1
            # fg label: above threshold IOU
            labels_i[j][max_overlaps >= cfg.SIAMSE.TRAIN.RPN_POSITIVE_OVERLAP] = 1
            # assign bg labels first so that positive labels can clobber them
            labels_i[j][max_overlaps < cfg.SIAMSE.TRAIN.RPN_NEGATIVE_OVERLAP] = 0


            # subsample positive labels if we have too many
            num_fg = int(cfg.SIAMSE.TRAIN.RPN_FG_FRACTION * cfg.SIAMSE.TRAIN.RPN_BATCHSIZE)
            fg_inds = np.where(labels_i[j] == 1)[0]
            if len(fg_inds) > num_fg:
                disable_inds = npr.choice(
                    fg_inds, size=(len(fg_inds) - num_fg), replace=False)
                labels_i[j][disable_inds] = -1

            # subsample negative labels if we have too many
            num_bg = cfg.SIAMSE.TRAIN.RPN_BATCHSIZE - np.sum(labels_i[j] == 1)
            bg_inds = np.where(labels_i[j] == 0)[0]
            if len(bg_inds) > num_bg:
                disable_inds = npr.choice(
                    bg_inds, size=(len(bg_inds) - num_bg), replace=False)
                labels_i[j][disable_inds] = -1
                #print "was %s inds, disabling %s, now %s inds" % (
                    #len(bg_inds), len(disable_inds), np.sum(labels == 0))


            bbox_targets_i[j] = _compute_targets(anchors_i, gt_boxes_reshape_transpose[i][j][argmax_overlaps, :])


            bbox_inside_weights_i[j][labels_i[j] == 1, :] = np.array(cfg.SIAMSE.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

            positive_weights = np.ones((1, 4))
            negative_weights = np.zeros((1, 4))
            bbox_outside_weights_i[j][labels_i[j] == 1, :] = positive_weights
            bbox_outside_weights_i[j][labels_i[j] == 0, :] = negative_weights

            # map up to original set of anchors
            labels_i_shape[j] = _unmap(labels_i[j], total_anchors_i, inds_inside, fill=-1)
            bbox_targets_i_shape[j] = _unmap(bbox_targets_i[j], total_anchors_i, inds_inside, fill=0)
            bbox_inside_weights_i_shape[j] = _unmap(bbox_inside_weights_i[j], total_anchors_i, inds_inside, fill=0)
            bbox_outside_weights_i_shape[j] = _unmap(bbox_outside_weights_i[j], total_anchors_i, inds_inside, fill=0)

        labels.append(labels_i_shape)
        bbox_targets.append(bbox_targets_i_shape)
        bbox_inside_weights.append(bbox_inside_weights_i_shape)
        bbox_outside_weights.append(bbox_outside_weights_i_shape)

    # labels
    #pdb.set_trace()
    batch_merge = np.empty((0), dtype=np.float32)
    for i in range(gt_boxes_reshape_transpose_shape[1]):
        classes_merge = labels[0][i]
        for j in range(len(labels) - 1):
            classes_merge = np.hstack((classes_merge, labels[j+1][i]))
        batch_merge = np.hstack((batch_merge, classes_merge))

    labels = batch_merge.reshape((-1)) #[batch*classes*h*w]
    rpn_labels = labels

    # bbox_targets
    batch_merge = np.empty((0, 4), dtype=np.float32)
    for i in range(gt_boxes_reshape_transpose_shape[1]):
        classes_merge = bbox_targets[0][i]
        for j in range(len(bbox_targets) - 1):
            classes_merge = np.vstack((classes_merge, bbox_targets[j + 1][i]))
        batch_merge = np.vstack((batch_merge, classes_merge))
    bbox_targets = batch_merge.reshape((-1, 4))
    rpn_bbox_targets = bbox_targets

    # bbox_inside_weights
    batch_merge = np.empty((0, 4), dtype=np.float32)
    for i in range(gt_boxes_reshape_transpose_shape[1]):
        classes_merge = bbox_inside_weights[0][i]
        for j in range(len(bbox_inside_weights) - 1):
            classes_merge = np.vstack((classes_merge, bbox_inside_weights[j + 1][i]))
        batch_merge = np.vstack((batch_merge, classes_merge))
    bbox_inside_weights = batch_merge.reshape((-1, 4))
    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    batch_merge = np.empty((0, 4), dtype=np.float32)
    for i in range(gt_boxes_reshape_transpose_shape[1]):
        classes_merge = bbox_outside_weights[0][i]
        for j in range(len(bbox_outside_weights) - 1):
            classes_merge = np.vstack((classes_merge, bbox_outside_weights[j + 1][i]))
        batch_merge = np.vstack((batch_merge, classes_merge))
    bbox_outside_weights = batch_merge.reshape((-1, 4))

    rpn_bbox_outside_weights = bbox_outside_weights

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights



def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
