import numpy as np
import yaml

from .generate_anchors import generate_anchors

from lib.networks.netconfig import cfg
from lib.rpn_msr.bbox_transform import bbox_transform_inv, clip_boxes
from lib.nms.nms_wrapper import nms
# <<<< obsolete


DEBUG = False
"""
Outputs object detection proposals by applying estimated bounding-box
transformations to a set of regular boxes (called "anchors").
"""
def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key=True,
                   _feat_stride = cfg.SIAMSE.FEAT_STRIDE):
    """
    Parameters
    ----------
    rpn_cls_prob_reshape: (1 , H , W , Ax2) outputs of RPN, prob of bg or fg
                         NOTICE: the old version is ordered by (1, H, W, 2, A) !!!!
    rpn_bbox_pred: (1 , H , W , Ax4), rgs boxes output of RPN
    im_info: a list of [image_height, image_width, scale_ratios]
    cfg_key: 'TRAIN' or 'TEST'
    _feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    rpn_rois : (1 x H x W x A, 5) e.g. [0, x1, y1, x2, y2]

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)
    #layer_params = yaml.load(self.param_str_)

    """
    _anchors = generate_anchors(scales=cfg.SIAMSE.ANCHOR_SCALE)
    _num_anchors = [_anchors[i].shape[0] for i in range(len(_anchors))]  # [classes_num]

    im_info = im_info[0]
    class_num = len(cfg.SIAMSE.CLASSES)

    if cfg_key==True:
        pre_nms_topN = cfg.SIAMSE.TRAIN.RPN_PRE_NMS_TOP_N  # 64
        post_nms_topN = cfg.SIAMSE.TRAIN.RPN_POST_NMS_TOP_N  # 3
        nms_thresh = cfg.SIAMSE.TRAIN.RPN_NMS_THRESH  # 0.7
        min_size = cfg.SIAMSE.TRAIN.RPN_MIN_SIZE  # 16
    else:
        pre_nms_topN = cfg.SIAMSE.TEST.RPN_PRE_NMS_TOP_N  # 64
        post_nms_topN = cfg.SIAMSE.TEST.RPN_POST_NMS_TOP_N  # 3
        nms_thresh = cfg.SIAMSE.TEST.RPN_NMS_THRESH  # 0.7
        min_size = cfg.SIAMSE.TEST.RPN_MIN_SIZE  # 16



    batch, height, width = rpn_cls_prob_reshape.shape[0:3]

    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want
    # (1, H, W, A)
    scores = np.reshape(np.reshape(rpn_cls_prob_reshape, [batch, height, width, _num_anchors[0], 2])[:,:,:,:,1],
                        [batch, height, width, _num_anchors[0]])

    # TODO: NOTICE: the old version is ordered by (1, H, W, 2, A) !!!!
    # TODO: if you use the old trained model, VGGnet_fast_rcnn_iter_70000.ckpt, uncomment this line
    # scores = rpn_cls_prob_reshape[:,:,:,_num_anchors:]

    bbox_deltas = rpn_bbox_pred
    #im_info = bottom[2].data[0, :]

    if DEBUG:
        print ('im_size: ({}, {})'.format(im_info[0], im_info[1]))
        print ('scale: {}'.format(im_info[2]))

    # 1. Generate proposals from bbox deltas and shifted anchors
    if DEBUG:
        print ('score map size: {}'.format(scores.shape))

    # Enumerate all shifts
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    anchors = []
    for i in range(len(_num_anchors)):
        A = _num_anchors[i]
        K = shifts.shape[0]
        anchors_i = (_anchors[i].reshape((1, A, 4)) +
                         shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        anchors_i = anchors_i.reshape((K * A, 4))
        anchors.append(anchors_i)
        total_anchors_i = int(K * A)
    # A = _num_anchors
    # K = shifts.shape[0]
    # anchors = _anchors.reshape((1, A, 4)) + \
    #           shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    # anchors = anchors.reshape((K * A, 4))

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    bbox_deltas = bbox_deltas.reshape((batch, -1, 4)) #(batch, HxWxA, 4)

    # Same story for the scores:
    #
    # scores are (1, A, H, W) format
    # transpose to (1, H, W, A)
    # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
    scores = scores.reshape((batch, -1, 1))
    batch_scores = []
    batch_blob = []

    # Convert anchors into proposals via bbox transformations
    # 这里的batch是输入的图片数量x类别数量
    for i in range(batch):
        proposals_i = bbox_transform_inv(anchors[int(i%class_num)], bbox_deltas[i])

        # 2. clip predicted boxes to image
        proposals_i = clip_boxes(proposals_i, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals_i, min_size * im_info[2])
        proposals_i = proposals_i[keep, :]
        scores_i = scores[i][keep]

        # # remove irregular boxes, too fat too tall
        # keep = _filter_irregular_boxes(proposals)
        # proposals = proposals[keep, :]
        # scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores_i.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals_i = proposals_i[order, :]
        scores_i = scores_i[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((proposals_i, scores_i)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals_i = proposals_i[keep, :]
        scores_i = scores_i[keep]
        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.empty((proposals_i.shape[0], 1), dtype=np.float32)
        batch_inds.fill(i)
        blob_i = np.hstack((batch_inds, proposals_i.astype(np.float32, copy=False)))

        batch_scores.append(scores_i)
        batch_blob.append(blob_i)
    # dets = np.hstack((blob, scores)).astype(np.float32)
    # print(dets.shape)
    # print('jjjjj=============', dets[:, -1])
    return batch_blob, batch_scores
    #top[0].reshape(*(blob.shape))
    #top[0].data[...] = blob

    # [Optional] output scores blob
    #if len(top) > 1:
    #    top[1].reshape(*(scores.shape))
    #    top[1].data[...] = scores

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

def _filter_irregular_boxes(boxes, min_ratio = 0.2, max_ratio = 5):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    rs = ws / hs
    keep = np.where((rs <= max_ratio) & (rs >= min_ratio))[0]
    return keep
