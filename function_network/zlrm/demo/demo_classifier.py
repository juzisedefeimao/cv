import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import os.path as osp
import glob
import os
from PIL import Image, ImageDraw, ImageFont

this_dir = osp.dirname(__file__)
print(this_dir)

from lib.networks.factory import get_network
from lib.networks.netconfig import cfg
from lib.utils.timer import Timer

# CLASSES = ('vein', 'slag_inclusion', 'aluminium_skimmings', 'crack',
#             'edge_crack', 'paint_smear')  # 背景，纹理，夹渣，铝屑，裂纹，边裂，油污

CLASSES = ('defect_tree_0_0', 'defect_tree_0_1')

def imread(file_path):
    im = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),-1)
    if len(im.shape) == 2:
        c = []
        for i in range(3):
            c.append(im)
        im = np.asarray(c)
        im = im.transpose([1, 2, 0])
    return im

def readimage(filename):
    image = np.array(Image.open(filename))
    if len(image.shape) == 2:
        c = []
        for i in range(3):
            c.append(image)
        image = np.asarray(c)
        image = image.transpose([1, 2, 0])
    return image



def im_detect(sess, net, im):

    blobs = _get_blobs(im)

    # forward pass
    feed_dict = {net.data: blobs['data']}

    cls_prob, = \
        sess.run([net.get_output('cls_prob')], \
                 feed_dict=feed_dict)

    if isinstance(cls_prob, tuple):
        cls_prob = cls_prob[0]

    cls_prob = np.reshape(cls_prob, [-1, cfg.ZLRM.TRAIN.CLASSIFY_NUM])

    scores = cls_prob

    return scores

def _get_blobs(image):
    """Convert an image and RoIs within that image into networks inputs."""
    if cfg.ZLRM.TEST.HAS_RPN:
        blobs = {'data': None}
        image = Image.fromarray(image.astype(np.uint8))
        image = image.resize(cfg.ZLRM.TRAIN.CLASSIFY_IMAGE_SIZE, Image.ANTIALIAS)
        image = np.array(image)
        image = image.astype(np.float32, copy=False)
        image -= cfg.ZLRM.PIXEL_MEANS
        blobs['data'] = np.reshape(image, (-1, image.shape[0], image.shape[1], image.shape[2]))

    return blobs

def vis(im, im_name, scores):
    # im = Image.open(im_name)
    classify = {}
    print(scores)
    scores = scores[0]
    for cls_ind, cls in enumerate(CLASSES):
        cls_scores = scores[cls_ind]
        classify[cls] = cls_scores
    # classify_scores = sorted(classify.items(), key=lambda item: item[1], reverse=True)
    im = Image.fromarray(np.array(im))
    # im = im.resize((1024,1024), Image.ANTIALIAS)
    # draw = ImageDraw.Draw(im)
    # font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")
    # i=0
    # for class_name, class_score in classify_scores:
    #     i = i + 1
    #     draw.text((10, 30*i + 40), str(class_name), (0, 0, 255), font=font)
    #     draw.text((300, 30*i + 40), str(class_score), (0, 0, 255), font=font)

    print('jjj',classify)
    if classify['defect_tree_0_0'] > 0.6:
        root = os.path.join(cfg.DATA_DIR, 'result', 'classify', 'output_classify_zero')
        if not os.path.exists(root):
            os.makedirs(root)
        path = os.path.join(root, im_name)
        im.save(path)
    elif classify['defect_tree_0_1'] > 0.6:
        root = os.path.join(cfg.DATA_DIR, 'result', 'classify', 'output_classify_one')
        if not os.path.exists(root):
            os.makedirs(root)
        path = os.path.join(root, im_name)
        im.save(path)
    else:
        root = os.path.join(cfg.DATA_DIR, 'result', 'classify' 'output_classify_two')
        if not os.path.exists(root):
            os.makedirs(root)
        path = os.path.join(root, im_name)
        im.save(path)



def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = readimage(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s '.format(timer.total_time))
    return im, scores


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='zlrm demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='Triple_classifier_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default='D:\\jjj\\zlrm\\output\\Triple_classifier')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load networks
    net = get_network(args.demo_net)
    # load model
    print ('Loading networks {:s}... '.format(args.demo_net))
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.model)
    print('Restoring from {}...'.format(ckpt.model_checkpoint_path), )
    saver.restore(sess, ckpt.model_checkpoint_path)
    print (' done.')

    root = os.path.join(cfg.DATA_DIR, 'demo_classify')
    for im_name in os.listdir(root):
        print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print ('Demo for {:s}'.format(im_name))
        image_root = os.path.join(root, im_name)
        im, scores = demo(sess, net, image_root)
        vis(im, im_name, scores)