import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import os.path as osp
import glob

this_dir = osp.dirname(__file__)
print(this_dir)

from lib.networks.factory import get_network
from lib.networks.netconfig import cfg
from lib.nms.nms_wrapper import nms
from lib.utils.timer import Timer

CLASSES = ('__background__', # always index 0
           'defect')


# CLASSES = ('__background__','person','bike','motorbike','car','bus')

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    # print('rois--------------', scores)
    print ('Detection took {:.3f}s for '
           '{:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    CONF_THRESH = 0.9
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        # cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        # cls_scores = scores[:, cls_ind]
        # dets = np.hstack((cls_boxes,
        #                   cls_scores[:, np.newaxis])).astype(np.float32)
        # keep = nms(dets, NMS_THRESH)
        # dets = dets[keep, :]
        dets = np.hstack((boxes, scores)).astype(np.float32)
        vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)




def imread(file_path):
    im = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),-1)
    if len(im.shape) == 2:
        c = []
        for i in range(3):
            c.append(im)
        im = np.asarray(c)
        im = im.transpose([1, 2, 0])
    return im




def im_detect(sess, net, im):
    """Detect object classes in an image given object proposals.
    Arguments:
        net (caffe.Net): Fast R-CNN networks to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    blobs, im_scales = _get_blobs(im)

    if cfg.ZLRM.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
    # forward pass
    if cfg.ZLRM.TEST.HAS_RPN:
        feed_dict = {net.data: blobs['data'], net.im_info: blobs['im_info']}

    rois = sess.run(net.get_output('rois'), feed_dict=feed_dict)
    # print('cls_score,  bbox_pred, rois形状', cls_score.shape, bbox_pred.shape, rois.shape)
    # scores = np.reshape(np.reshape(rpn_cls_prob_reshape, [1, height, width, _num_anchors, 2])[:, :, :, :, 1],
    #                     [1, height, width, _num_anchors])

    # rois, scores = rois
    scores = rois[:, -1].reshape((-1, 1))
    # print(rois.shape)
    # print('kkkkk', scores)
    if cfg.ZLRM.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.ZLRM.TEST.BBOX_REG:
        pred_boxes = boxes
        # Apply bounding-box regression deltas
        # box_deltas = bbox_pred
        # pred_boxes = bbox_transform_inv(boxes, box_deltas)
        # pred_boxes = _clip_boxes(pred_boxes, im.shape)


    return scores, pred_boxes

def _get_blobs(im):
    """Convert an image and RoIs within that image into networks inputs."""
    if cfg.ZLRM.TEST.HAS_RPN:
        blobs = {'data': None, 'rois': None}
        blobs['data'], im_scale_factors = _get_image_blob(im)

    return blobs, im_scale_factors

def _get_image_blob(im):
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



def vis_detections(im, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        # print('box======', bbox)
        # bbox[0]=2128
        # bbox[1]=82
        # bbox[2]=2162
        # bbox[3]=198
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        # ax.text(bbox[0], bbox[1] - 2,
        #         '{:s} {:.3f}'.format(class_name, score),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')

    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh),
    #              fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()





def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='zlrm demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='Resnet50_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default='C:\\Users\\jjj\\Desktop\\jjj\\zlrm\\output\\default\\zlrm_data_train')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.ZLRM.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    # if args.model == ' ' or not os.path.exists(args.model):
    #     print ('current path is ' + os.path.abspath(__file__))
    #     raise IOError(('Error: Model not found.\n'))

    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load networks
    net = get_network(args.demo_net)
    # load model
    print ('Loading networks {:s}... '.format(args.demo_net))
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.model)
    print('Restoring from {}...'.format(ckpt.model_checkpoint_path), )
    saver.restore(sess, ckpt.model_checkpoint_path)
    # saver = tf.train.Saver()
    # saver.restore(sess, args.model)
    print (' done.')

    # Warmup on a dummy image
    # im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    # for i in range(2):
    #     _, _ = im_detect(sess, net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.bmp'))

    im_name_root = os.path.join(cfg.DATA_DIR, 'demo')
    root = os.path.join(cfg.DATA_DIR, 'output')
    for im_name in im_names:
        print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print ('Demo for {:s}'.format(im_name))
        demo(sess, net, im_name)
        name = im_name.strip(im_name_root)
        name = name.strip('.bmp') + '简单'
        path = os.path.join(root, name)
        plt.savefig(path)

    # plt.show()