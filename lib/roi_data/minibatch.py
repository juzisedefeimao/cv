# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN networks."""

import numpy as np
import numpy.random as npr
import cv2
import os
from PIL import Image, ImageDraw

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from lib.networks.netconfig import cfg
# <<<< obsolete
from lib.utils.blob import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.ZLRM.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.SIAMSE.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.SIAMSE.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.ZLRM.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.ZLRM.TRAIN.FG_FRACTION * rois_per_image)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    # 为样本图片增加边缘部分，像素设为样本像素的均值。以方便从下采样过的图片里切割用来训练的图片。缺陷样本通过均匀平移增强。
    crop_im_blob, gt_boxes = crop_image(im_blob, roidb, im_scales, num_images, num_classes)
    blobs = {'data': crop_im_blob}

    # gt boxes: (x1, y1, x2, y2, cls)
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
        [[255, 255, 1]],
        dtype=np.float32)
    blobs['im_name'] = os.path.basename(roidb[0]['image'])



    return blobs

def max_classes_num(roidb, num_images, num_classes):
    batch_classes_max_num = -1
    for i in range(num_images):
        classes_max_list = []
        for j in range(num_classes):
            max_num = len(np.where(roidb[i]['gt_classes'] == j)[0])
            classes_max_list.append(max_num)
        classes_max = max(classes_max_list)
        if batch_classes_max_num < classes_max:
            batch_classes_max_num = classes_max

    return int(batch_classes_max_num)

xm = 0

def crop_image_(im_blob, roidb, im_scale, num_images, num_classes):
    boundary = cfg.SIAMSE.SEARCH_IMAGE_SIZE[0]
    max_num = max_classes_num(roidb, num_images, num_classes)
    background = np.zeros((im_blob.shape[1] + 2 * boundary, im_blob.shape[2] + 2 * boundary, 3))
    crop_im_blob = []
    gt_boxes = []
    global xm
    for i in range(num_images):
        background[boundary:im_blob.shape[1] + boundary, boundary:im_blob.shape[2] + boundary] = im_blob[i]

        # print(roidb[i]['image'], len(roidb[i]['boxes']))
        for j in [np.random.randint(0, len(roidb[i]['boxes']) )]:
            # crop_im_blob_i = np.zeros((boundary, boundary, 3), dtype=np.float32)
            # crop_im_blob_bag_i = np.zeros((2*boundary, 2*boundary, 3), dtype=np.float32)

            gt_boxes_i = np.empty((num_classes-1, max_num, 5), dtype=np.int32)
            gt_boxes_i.fill(-1)
            box = roidb[i]['boxes'][j] * im_scale[i]
            cls = roidb[i]['gt_classes'][j]

            # b = im_blob[i] + cfg.SIAMSE.PIXEL_MEANS
            # b = Image.fromarray(b.astype(np.uint8))
            # vis(b, box, str(xm) + 'a.bmp')


            h = box[3] - box[1]
            w = box[2] - box[0]
            if h < 127 and w < 127:
                h_bias_max = int((boundary - h) / 2)
                w_bias_max = int((boundary - w) / 2)
                # print(h_bias_max)
                h_bias = np.random.randint(-h_bias_max, h_bias_max)
                w_bias = np.random.randint(-w_bias_max, w_bias_max)

                x1_crop = max(int((box[0] + box[2])/2) + w_bias + int(boundary/2), 0)
                x2_crop = x1_crop + boundary
                y1_crop = max(int((box[1] + box[3])/2) + h_bias + int(boundary/2), 0)
                y2_crop = y1_crop + boundary
                crop_im_blob_i = background[y1_crop:y2_crop, x1_crop:x2_crop]

                x1_box = box[0] - x1_crop + boundary
                x2_box = x1_box + w
                y1_box = box[1] - y1_crop + boundary
                y2_box = y1_box + h
                crop_box = [x1_box, y1_box, x2_box, y2_box]

                # b = crop_im_blob_i + cfg.SIAMSE.PIXEL_MEANS
                # b = Image.fromarray(b.astype(np.uint8))
                # vis(b, crop_box, str(xm) + 'b.bmp')

                crop_im_blob.append(crop_im_blob_i)
                gt_boxes_i[cls-1][0]=np.array(crop_box + [cls])
            else:
                h_bias_max = int((2*boundary - h) / 2)
                w_bias_max = int((2*boundary - w) / 2)
                h_bias = np.random.randint(-h_bias_max, h_bias_max)
                w_bias = np.random.randint(-w_bias_max, w_bias_max)
                x1_crop = max(int((box[0] + box[2]) / 2) + w_bias, 0)
                x2_crop = x1_crop + 2*boundary
                y1_crop = max(int((box[1] + box[3]) / 2) + h_bias, 0)
                y2_crop = y1_crop + 2*boundary

                # print('jjj',y1_crop, y2_crop, x1_crop, x2_crop)
                crop_im_blob_bag_i = background[y1_crop:y2_crop, x1_crop:x2_crop]
                crop_im_blob_i_one = (crop_im_blob_bag_i.transpose(2, 0, 1))[0]
                # print('kkk', crop_im_blob_i_one)
                crop_im_blob_image = Image.fromarray(crop_im_blob_i_one.astype(np.float32))
                crop_im_blob_i_resize = crop_im_blob_image.resize((boundary, boundary), Image.ANTIALIAS)
                x1_box = int((box[0] - x1_crop + boundary)/2)
                x2_box = int(x1_box + w/2) + 1
                y1_box = int((box[1] - y1_crop + boundary)/2)
                y2_box = int(y1_box + h/2) + 1
                crop_box = [x1_box, y1_box, x2_box, y2_box]

                # b = three(crop_im_blob_i_resize) + cfg.SIAMSE.PIXEL_MEANS
                # b = Image.fromarray(b.astype(np.uint8))
                # vis(b, crop_box, str(xm) + 'c.bmp')

                crop_im_blob.append(three(crop_im_blob_i_resize))
                gt_boxes_i[cls - 1][0] = np.array(crop_box + [cls])

            gt_boxes.append(gt_boxes_i)

            xm = xm +1

    crop_im_blob = np.stack(crop_im_blob, axis=0)
    gt_boxes = np.stack(gt_boxes, axis=0)
    gt_boxes = gt_boxes.reshape((-1, max_num, 5))

    return crop_im_blob, gt_boxes

def crop_image(im_blob, roidb, im_scale, num_images, num_classes):
    boundary = cfg.SIAMSE.SEARCH_IMAGE_SIZE[0]
    max_num = max_classes_num(roidb, num_images, num_classes)
    background = np.zeros((im_blob.shape[1] + 2 * boundary, im_blob.shape[2] + 2 * boundary, 3))
    crop_im_blob = []
    gt_boxes = []
    global xm
    for i in range(num_images):
        background[boundary:im_blob.shape[1] + boundary, boundary:im_blob.shape[2] + boundary] = im_blob[i]


        gt_boxes_i = np.empty((num_classes-1, max_num, 5), dtype=np.int32)
        gt_boxes_i.fill(-1)
        box_all = roidb[i]['boxes'] * im_scale[i]
        cls_all = roidb[i]['gt_classes']

        h_all = box_all[:, 3] - box_all[:, 1]
        w_all = box_all[:, 2] - box_all[:, 0]

        j = np.random.randint(0, len(roidb[i]['boxes']))


        h = h_all[j]
        w = w_all[j]
        box = box_all[j]

        # b = im_blob[i] + cfg.SIAMSE.PIXEL_MEANS
        # b = Image.fromarray(b.astype(np.uint8))
        # vis(b, np.reshape(box,(-1,4)), str(xm) + 'a.bmp')
        if h < 127 and w < 127:
            h_bias_max = int((boundary - h) / 2)
            w_bias_max = int((boundary - w) / 2)
            # print(h_bias_max)
            h_bias = np.random.randint(-h_bias_max, h_bias_max)
            w_bias = np.random.randint(-w_bias_max, w_bias_max)
            # h_bias = 0
            # w_bias = 0
            # print('h,w', h_bias, w_bias)
            x1_crop = max(int((box[0] + box[2])/2) + w_bias + int(boundary/2), 0)
            x2_crop = x1_crop + boundary
            y1_crop = max(int((box[1] + box[3])/2) + h_bias + int(boundary/2), 0)
            y2_crop = y1_crop + boundary
            crop_im_blob_i = background[y1_crop:y2_crop, x1_crop:x2_crop]

            # x1_box = box[0] - x1_crop + boundary
            # x2_box = x1_box + w
            # y1_box = box[1] - y1_crop + boundary
            # y2_box = y1_box + h
            # crop_box = [x1_box, y1_box, x2_box, y2_box]

            # 计算所有box转换后的坐标，判断是否有box框在新切割的图片里
            x1_box_all = box_all[:, 0] - x1_crop + boundary
            x2_box_all = x1_box_all + w_all
            y1_box_all = box_all[:, 1] - y1_crop + boundary
            y2_box_all = y1_box_all + h_all
            # print('1', x1_box_all, y1_box_all, x2_box_all, y2_box_all)
            # crop_box_all = np.hstack([x1_box_all, y1_box_all, x2_box_all, y2_box_all]).reshape(-1,4)
            # print('2', crop_box_all)
            x1_box_all_boundary = np.where(x1_box_all < 0, 0, x1_box_all)
            y1_box_all_boundary = np.where(y1_box_all < 0, 0, y1_box_all)
            x2_box_all_boundary = np.where(x2_box_all > boundary, boundary, x2_box_all)
            y2_box_all_boundary = np.where(y2_box_all > boundary, boundary, y2_box_all)
            # print('0', [x1_box_all_boundary, y1_box_all_boundary,
            #                           x2_box_all_boundary, y2_box_all_boundary])
            crop_box_all = np.stack([x1_box_all_boundary, y1_box_all_boundary,
                                      x2_box_all_boundary, y2_box_all_boundary], axis=1)
            # print('2', crop_box_all)
            overlap = ((x2_box_all_boundary - x1_box_all_boundary) * (y2_box_all_boundary - y1_box_all_boundary)) / \
                      ((x2_box_all - x1_box_all) * (y2_box_all - y1_box_all))
            inds = np.where((x2_box_all_boundary > x1_box_all_boundary) &
                            (y2_box_all_boundary > y1_box_all_boundary)& (overlap > 0.8))[0]
            # print('inds', inds)
            # print('1', crop_box_all)
            for k in inds:
                for l in range(max_num):
                    if gt_boxes_i[cls_all[k] - 1][l][0] == -1:
                        # print('4', np.array(crop_box_all[k] + [cls_all[k]]))
                        # print(cls_all[k].reshape((-1)))
                        # print('5', np.vstack([crop_box_all[k], cls_all[k].reshape((-1))]))
                        gt_boxes_i[cls_all[k] - 1][l][:4] = crop_box_all[k]
                        gt_boxes_i[cls_all[k] - 1][l][4] = cls_all[k]
                        # print('4',  '  ', str(xm), gt_boxes_i[cls_all[k] - 1][l])
                        break

            # b = crop_im_blob_i + cfg.SIAMSE.PIXEL_MEANS
            # b = Image.fromarray(b.astype(np.uint8))
            # vis(b, crop_box_all[inds], str(xm) + 'b.bmp')

            crop_im_blob.append(crop_im_blob_i)
        else:
            h_bias_max = int((2*boundary - h) / 2)
            w_bias_max = int((2*boundary - w) / 2)
            h_bias = np.random.randint(-h_bias_max, h_bias_max)
            w_bias = np.random.randint(-w_bias_max, w_bias_max)
            x1_crop = max(int((box[0] + box[2]) / 2) + w_bias, 0)
            x2_crop = x1_crop + 2*boundary
            y1_crop = max(int((box[1] + box[3]) / 2) + h_bias, 0)
            y2_crop = y1_crop + 2*boundary

            # print('jjj',y1_crop, y2_crop, x1_crop, x2_crop)
            crop_im_blob_bag_i = background[y1_crop:y2_crop, x1_crop:x2_crop]
            crop_im_blob_i_one = (crop_im_blob_bag_i.transpose(2, 0, 1))[0]
            # print('kkk', crop_im_blob_i_one)
            crop_im_blob_image = Image.fromarray(crop_im_blob_i_one.astype(np.float32))
            crop_im_blob_i_resize = crop_im_blob_image.resize((boundary, boundary), Image.ANTIALIAS)
            # x1_box = int((box[0] - x1_crop + boundary)/2)
            # x2_box = int(x1_box + w/2) + 1
            # y1_box = int((box[1] - y1_crop + boundary)/2)
            # y2_box = int(y1_box + h/2) + 1
            # crop_box = [x1_box, y1_box, x2_box, y2_box]

            # 计算所有box转换后的坐标，判断是否有box框在新切割的图片里
            x1_box_all = ((box_all[:, 0] - x1_crop + boundary)/2).astype(np.int32)
            x2_box_all = (x1_box_all + w_all/2 + 1).astype(np.int32)
            y1_box_all = ((box_all[:, 1] - y1_crop + boundary)/2).astype(np.int32)
            y2_box_all = (y1_box_all + h_all/2 + 1).astype(np.int32)

            x1_box_all_boundary = np.where(x1_box_all < 0, 0, x1_box_all)
            y1_box_all_boundary = np.where(y1_box_all < 0, 0, y1_box_all)
            x2_box_all_boundary = np.where(x2_box_all > boundary, boundary, x2_box_all)
            y2_box_all_boundary = np.where(y2_box_all > boundary, boundary, y2_box_all)
            crop_box_all = np.stack([x1_box_all_boundary, y1_box_all_boundary,
                                     x2_box_all_boundary, y2_box_all_boundary], axis=1)
            overlap = ((x2_box_all_boundary - x1_box_all_boundary)*(y2_box_all_boundary - y1_box_all_boundary))/\
                      ((x2_box_all - x1_box_all)*(y2_box_all - y1_box_all))
            inds = np.where((x2_box_all_boundary > x1_box_all_boundary) &
                            (y2_box_all_boundary > y1_box_all_boundary) & (overlap > 0.8))[0]
            # print('0',str(xm) ,  overlap[inds])
            # print('11', crop_box_all[inds])
            for k in inds:
                for l in range(max_num):
                    if gt_boxes_i[cls_all[k] - 1][l][0] == -1:
                        gt_boxes_i[cls_all[k] - 1][l][:4] = crop_box_all[k]
                        gt_boxes_i[cls_all[k] - 1][l][4] = cls_all[k]
                        # print('44',str(xm), gt_boxes_i[cls_all[k] - 1][l])
                        break
            #
            # b = three(crop_im_blob_i_resize) + cfg.SIAMSE.PIXEL_MEANS
            # b = Image.fromarray(b.astype(np.uint8))
            # vis(b, crop_box_all[inds], str(xm) + 'c.bmp')

            crop_im_blob.append(three(crop_im_blob_i_resize))
            # gt_boxes_i[cls - 1][0] = np.array(crop_box + [cls])

        gt_boxes.append(gt_boxes_i)

        xm = xm +1

    crop_im_blob = np.stack(crop_im_blob, axis=0)
    gt_boxes = np.stack(gt_boxes, axis=0)
    gt_boxes = gt_boxes.reshape((-1, max_num, 5))

    return crop_im_blob, gt_boxes

def vis(image,  bboxes, name):
    print('3', bboxes)
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        image = Image.fromarray(np.array(image))
        draw = ImageDraw.Draw(image)
        draw.rectangle((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), outline=(255, 0, 0))

    path = os.path.join('D:\data', name)
    image.save(path)


# 单通道图像转为3通道
def three(im):
    c = []
    for i in range(3):
        c.append(im)
    im = np.stack(c, axis=0)
    im = im.transpose([1, 2, 0])

    return im

def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.ZLRM.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.ZLRM.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.ZLRM.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
                bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            roidb['bbox_targets'][keep_inds, :], num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_inside_weights

# 可读取中文路径的图片
def imread(file_path):
    im = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),-1)
    if len(im.shape) == 2:
        c = []
        for i in range(3):
            c.append(im)
        im = np.asarray(c)
        im = im.transpose([1, 2, 0])
    return im

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in range(num_images):

        im = imread(roidb[i]['image'])

        while im is None:
            print('roidb', i, 'image', roidb[i]['image'], '为空')
            if not os.path.exists(roidb[i]['image']):
                print('路径不存在')
            im = imread(roidb[i]['image'])

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        # print('flipped', roidb[i]['flipped'])

        target_size = cfg.SIAMSE.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.SIAMSE.PIXEL_MEANS, target_size,
                                        cfg.SIAMSE.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the networks (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.ZLRM.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in range(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print ('class: ', cls, ' overlap: ', overlaps[i])
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()
