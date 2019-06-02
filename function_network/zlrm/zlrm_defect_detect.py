import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import os.path as osp
import glob
from PIL import Image, ImageDraw, ImageFont

this_dir = osp.dirname(__file__)
print(this_dir)

from lib.networks.factory import get_network
from lib.networks.netconfig import cfg
from lib.nms.nms_wrapper import nms
from lib.rpn_msr.bbox_transform import clip_boxes, bbox_transform_inv
from lib.utils.timer import Timer




def readimage(filename):
    image = np.array(Image.open(filename))
    if len(image.shape) == 2:
        c = []
        for i in range(3):
            c.append(image)
        image = np.asarray(c)
        image = image.transpose([1, 2, 0])
    return image


# 检出时的二分类
CLASSES_DEFECT = ('__background__', # always index 0
           'defect')

CLASSES_CLASSIFY = ('vein', 'slag_inclusion', 'aluminium_skimmings', 'crack',
            'edge_crack', 'paint_smear')  # 背景，纹理，夹渣，铝屑，裂纹，边裂，油污
classify_dic = dict(zip(range(len(CLASSES_CLASSIFY)), CLASSES_CLASSIFY))
 # {0:'vein', 1:'slag_inclusion', 2:'aluminium_skimmings', 3:'crack',
 #     4:'edge_crack', 5:'paint_smear'}
image_ext = '.bmp'
# =================================================检出===========================================================


def detect(sess, net, image):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # Detect all object classes and regress object bounds
    image = image_transform_1_3(image)
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, image)
    timer.toc()
    # print('rois--------------', scores)
    print ('Detection took {:.3f}s for '
           '{:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    CONF_THRESH = 0.7
    NMS_THRESH = 0.1
    for cls_ind, cls in enumerate(CLASSES_DEFECT[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        dets = dets[inds, :]
    return dets


# 单通道图像转为3通道图像
def image_transform_1_3(image):
    assert len(image.shape) != 2 or len(image.shape) != 3, print('图像既不是3通道,也不是单通道')
    if len(image.shape) == 2:
        c = []
        for i in range(3):
            c.append(image)
        image = np.asarray(c)
        image = image.transpose([1, 2, 0])

    return image




def im_detect(sess, net, im):

    blobs, im_scales = get_blobs(im)

    if cfg.ZLRM.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
    # forward pass
    if cfg.ZLRM.TEST.HAS_RPN:
        feed_dict = {net.data: blobs['data'], net.im_info: blobs['im_info']}

    cls_prob, bbox_pred, rois = \
        sess.run([net.get_output('cls_prob'), net.get_output('ave_bbox_pred_rois'),
                  net.get_output('rois')], \
                 feed_dict=feed_dict)

    if isinstance(rois, tuple):
        rois = rois[0]

    cls_prob = np.reshape(cls_prob, [-1, cfg.ZLRM.N_CLASSES + 1])  # (R, C+1)
    bbox_pred = np.reshape(bbox_pred, [-1, (cfg.ZLRM.N_CLASSES + 1) * 4])  # (R, (C+1)x4)
    rois = np.array(rois)
    if cfg.ZLRM.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        boxes = rois[:, 1:5] / im_scales[0]

    scores = cls_prob

    if cfg.ZLRM.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)

    return scores, pred_boxes

def get_blobs(im):
    """Convert an image and RoIs within that image into networks inputs."""
    if cfg.ZLRM.TEST.HAS_RPN:
        blobs = {'data': None, 'rois': None}
        blobs['data'], im_scale_factors = get_detect_image_blob(im)

    return blobs, im_scale_factors

def get_detect_image_blob(im):
    """Converts an image into a networks input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.ZLRM.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.ZLRM.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.ZLRM.TEST.MAX_SIZE:
            im_scale = float(cfg.ZLRM.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def im_list_to_blob(ims):
    """Convert a list of images into a networks input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob
# =========================================================识别===============================================

def im_classify(sess, net, image_batch):

    blobs = get_classify_blobs(image_batch)

    # forward pass
    feed_dict = {net.data: blobs['data']}

    cls_prob, = \
        sess.run([net.get_output('cls_prob')], \
                 feed_dict=feed_dict)

    if isinstance(cls_prob, tuple):
        cls_prob = cls_prob[0]

    cls_prob = np.reshape(cls_prob, [-1, cfg.ZLRM.TRAIN.CLASSIFY_NUM])  # (R, C+1)

    scores = cls_prob

    return scores

def get_classify_blobs(image_batch):
    blobs = {'data': []}
    for i in range(image_batch.shape[0]):
        image = image_batch[i]
        image = Image.fromarray(image.astype(np.uint8))
        image = image.resize(cfg.ZLRM.TRAIN.CLASSIFY_IMAGE_SIZE, Image.ANTIALIAS)
        image = np.array(image)
        image = image.astype(np.float32, copy=False)
        image -= cfg.ZLRM.PIXEL_MEANS
        blobs['data'].append(image)
    blobs['data'] = np.stack(blobs['data'], axis=0)

    return blobs

def classify(sess, net, image_batch):
    timer = Timer()
    timer.tic()
    scores = im_classify(sess, net, image_batch)
    timer.toc()
    print('Detection took {:.3f}s '.format(timer.total_time))
    scores = np.array(scores)
    # 取最大概率的类别为分类类别，并赋予类别号
    scores = scores.argmax(axis=1)
    return scores


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='zlrm demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--detect_net', dest='detect_net', help='Network to detect',
                        default='FPN_Resnet50_test')
    parser.add_argument('--classify_net', dest='classify_net', help='Network to classify',
                        default='Resnet18_classifier_test')
    parser.add_argument('--detect_model', dest='detect_model', help='Detect model path',
                        default='D:\\jjj\\zlrm\\output\\fpn')
    parser.add_argument('--classify_model', dest='classify_model', help='Classify model path',
                        default='D:\\jjj\\zlrm\\output\\Resnet18_classifier')
    parser.add_argument('--save_detect_image', dest='save_detect_image', help='save detect image root',
                        default='D:\\jjj\\zlrm\\data\\result\\detect_image')
    parser.add_argument('--save_detect_visual_image', dest='save_detect_visual_image', help='save detect visual image root',
                        default='D:\\jjj\\zlrm\\data\\result\\output_detect_result')
    parser.add_argument('--save_classify_image', dest='save_classify_image', help='save classify image',
                        default='D:\\jjj\\zlrm\data\\result\\classify_image')

    args = parser.parse_args()

    return args

def load_network():

    args = parse_args()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    # 构建图
    detect_graph = tf.Graph()
    classify_graph = tf.Graph()

    # 为构建的图分别建立会话
    detect_sess = tf.Session(graph=detect_graph)
    classify_sess = tf.Session(graph=classify_graph)

    with detect_graph.as_default():
        detect_network = get_network(args.detect_net)
        print('Loading networks {:s}... '.format(args.detect_net))
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(args.detect_model)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), )
        saver.restore(detect_sess, ckpt.model_checkpoint_path)
        print(' done.')
    with classify_graph.as_default():
        classify_network = get_network(args.classify_net)
        print('Loading networks {:s}... '.format(args.classify_net))
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(args.classify_model)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), )
        saver.restore(classify_sess, ckpt.model_checkpoint_path)
        print(' done.')

    return detect_sess, classify_sess, detect_network, classify_network

def close_sess(detect_sess, classify_sess):
    detect_sess.close()
    classify_sess.close()


def defect_detect(sess, net, image):
    dets = detect(sess, net, image)
    return dets

def fetch_defect_image(dets, image):
    pre_image_batch = []
    image_bath = []
    for i in range(len(dets)):
        image = Image.fromarray(np.array(image).astype(np.uint8))
        cut_image = image.crop(dets[i][:4])
        image_bath.append(cut_image)

        cut_image = cut_image.resize(cfg.ZLRM.TRAIN.CLASSIFY_IMAGE_SIZE, Image.ANTIALIAS)
        cut_image = np.array(cut_image)
        cut_image = cut_image.astype(np.float32, copy=False)
        cut_image -= cfg.ZLRM.PIXEL_MEANS
        pre_image_batch.append(cut_image)
    pre_image_batch = np.stack(pre_image_batch, axis=0)
    return pre_image_batch, image_bath

def defect_classify(sess, net, dets, image, image_name):
    args = parse_args()

    pre_defect_image_batch, defect_image_batch = fetch_defect_image(dets, image)
    scores = classify(sess, net, pre_defect_image_batch)

    # 保存分类的缺陷图片并将类别序号放到dets里
    for i in range(len(dets)):
        image = Image.fromarray(np.array(defect_image_batch[i]).astype(np.uint8))
        save_defect_image_root = os.path.join(args.save_detect_image, image_name + '_' + str(i) + '.bmp')
        image.save(save_defect_image_root)
        if not os.path.exists(args.save_classify_image) :
            os.makedirs(args.save_classify_image)
        save_classify_image_root = os.path.join(args.save_classify_image,
                                                image_name + '_' + str(i) + '_' + classify_dic[int(scores[i])] + '.bmp')
        image.save(save_classify_image_root)

        dets[i][4] = int(scores[i])
    return dets

def vis(image, image_name, dets):
    args = parse_args()
    if len(dets) == 0:
        return
    for i in range(len(dets)):
        bbox = dets[i, :4]
        score = dets[i, -1]
        image = Image.fromarray(np.array(image))
        draw = ImageDraw.Draw(image)
        draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]), outline=(255,0,0))
        font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")
        draw.text((bbox[0], bbox[1] - 40), classify_dic[int(score)], (0, 0, 255), font=font)

    # 保存图片
    path = os.path.join(args.save_detect_visual_image, image_name+'.bmp')
    print(path)
    image.save(path+'.bmp')

def image_defect_detect(detect_sess, classify_sess, detect_network, classify_network, image, image_name):
    dets = defect_detect(detect_sess, detect_network, image)
    if dets.size > 0:
        dets = defect_classify(classify_sess, classify_network, dets, image, image_name)

    dets[:, 0:2] = np.floor(dets[:, 0:2])
    dets[:, 2:] = np.ceil(dets[:, 2:])
    dets = dets.astype(np.int32)

    vis(image, image_name, dets)
    return dets

def ProcessImages(detect_sess, classify_sess, detect_network, classify_network, image_arr, image_name):

    dets = image_defect_detect(detect_sess, classify_sess, detect_network, classify_network, image_arr, image_name)

    ##**************************************************##
    return dets, image_arr

if __name__ == '__main__':
    detect_sess, classify_sess, detect_network, classify_network = load_network()
    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.bmp'))

    im_name_root = os.path.join(cfg.DATA_DIR, 'demo')
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for {:s}'.format(im_name))
        image = readimage(im_name)
        dets, image = ProcessImages(detect_sess, classify_sess, detect_network, classify_network,
                                    image, (im_name.strip(im_name_root)).split('.')[0])
        # vis(image, im_name, dets)
        print(dets)