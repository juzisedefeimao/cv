import tensorflow as tf
import matplotlib.pyplot as plt
from time import strftime
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

# 保存图片
def saveimage(image, saveimage_name=None, image_ext='bmp', saveimage_root=None):
    if len(image.shape)==2:
        image = image_transform_1_3(image)
    if saveimage_name is None:
        saveimage_name = 'image_{}'.format(strftime("%Y_%m_%d_%H_%M_%S")) + '.' + image_ext
    else:
        saveimage_name = saveimage_name + '.' + image_ext
    if saveimage_root is None:
        saveimage_root = 'D:\\jjj\\zlrm\\data\\result\\default_root'
        print('未设置保存图片的路径，默认保存到_{}'.format(saveimage_root))
    root = os.path.join(saveimage_root, str(saveimage_name))
    image = Image.fromarray(image)
    image.save(root)

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



image_ext = '.bmp'
# =================================================检出===========================================================

class detector():

    def __init__(self, sess, net):
        self.sess = sess
        self.network = net
        self.classes_detect = ('__background__',  # always index 0
                             'crack', 'dirty_big', 'slag_inclusion', 'dirty')
    def detect(self, image):
        """Detect object classes in an image using pre-computed object proposals."""

        # Load the demo image
        # Detect all object classes and regress object bounds
        image = image_transform_1_3(image)
        timer = Timer()
        timer.tic()
        scores, boxes = self.im_detect(image)
        timer.toc()
        print('kkk', np.argmax(scores, axis=1))
        print('lll', scores[np.argmax(scores, axis=1)==4, 4])
        print ('Detection took {:.3f}s for '
               '{:d} object proposals'.format(timer.total_time, boxes.shape[0]))

        CONF_THRESH = 0.3
        NMS_THRESH = 0.5
        dets_list = []
        for cls_ind, cls in enumerate(self.classes_detect[1:]):
            inds = np.where(scores[:, cls_ind] > CONF_THRESH)[0]
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets[inds, :], NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            cls_ind_list = np.empty((len(inds), 1), np.int32)
            cls_ind_list.fill(cls_ind)
            dets = np.hstack((dets[inds, :-1], cls_ind_list))
            dets_list.append(dets)
        dets = np.vstack(dets_list)
        print('jjj',dets)
        return dets


    def im_detect(self, im):

        blobs, im_scales = self.get_blobs(im)

        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
        # forward pass
        feed_dict = {self.network.data: blobs['data'], self.network.im_info: blobs['im_info']}

        cls_prob, bbox_pred, rois = \
            self.sess.run([self.network.get_output('cls_prob'), self.network.get_output('bbox_pred'),
                      self.network.get_output('rois')], \
                     feed_dict=feed_dict)

        if isinstance(rois, tuple):
            rois = rois[0]

        cls_prob = np.reshape(cls_prob, [-1, cfg.ZLRM.N_CLASSES + 1])  # (R, C+1)
        bbox_pred = np.reshape(bbox_pred, [-1, (cfg.ZLRM.N_CLASSES + 1) * 4])  # (R, (C+1)x4)
        rois = np.array(rois)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        boxes = rois[:, 1:5] / im_scales[0]

        scores = cls_prob

        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)

        return scores, pred_boxes

    def get_blobs(self, im):
        blobs = {'data': None, 'rois': None}
        blobs['data'], im_scale_factors = self.get_detect_image_blob(im)

        return blobs, im_scale_factors

    def get_detect_image_blob(self, im):
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
        blob = self.im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def im_list_to_blob(self, ims):
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

class zlrm_defect_detect():
    def __init__(self, detect_net, detect_model, output_dir):


        # self.classes_classify = ('vein', 'slag_inclusion', 'aluminium_skimmings', 'crack',
        #                     'black_ground', 'paint_smear')  # 背景，纹理，夹渣，铝屑，裂纹，边裂，油污
        self.classes_classify =  ('__background__',  # always index 0
                             'crack', 'dirty_big', 'slag_inclusion', 'dirty')
        self._ind_to_class = dict(zip(range(len(self.classes_classify)), self.classes_classify))

        self.detect_sess, self.detect_network,  = self.load_network(detect_net, detect_model)

        self.detector = detector(self.detect_sess, self.detect_network)
        self.output_dir = output_dir



    def load_network(self, detect_net, detect_model):

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        # 构建图
        detect_graph = tf.Graph()

        # 为构建的图分别建立会话
        detect_sess = tf.Session(graph=detect_graph)

        with detect_graph.as_default():
            detect_network = get_network(detect_net)
            print('Loading networks {:s}... '.format(detect_net))
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(detect_model)
            print('Restoring from {}...'.format(ckpt.model_checkpoint_path), )
            saver.restore(detect_sess, ckpt.model_checkpoint_path)
            print(' done.')

        return detect_sess, detect_network

    def close_sess(self):
        self.detect_sess.close()

    def defect_detect(self, image):
        dets = self.detector.detect(image)
        return dets



    def image_defect_detect(self, image, image_name):
        dets = self.defect_detect(image)

        dets[:, 0:2] = np.floor(dets[:, 0:2])
        dets[:, 2:] = np.ceil(dets[:, 2:])
        dets = dets.astype(np.int32)

        self.vis(image, image_name, dets)
        # if len(dets) > 0:
        #     zeros = np.zeros(len(dets), dtype=np.int32)
        #     extend_dets_0 = np.stack([dets[:, 0] - 20, zeros], axis=1)
        #     dets[:, 0] = np.max(extend_dets_0, axis=1)
        #     extend_dets_1 = np.stack([dets[:, 1] - 20, zeros], axis=1)
        #     dets[:, 1] = np.max(extend_dets_1, axis=1)
        #     extend_dets_2 = np.stack([dets[:, 2] + 20, zeros + 4095], axis=1)
        #     dets[:, 2] = np.min(extend_dets_2, axis=1)
        #     extend_dets_3 = np.stack([dets[:, 3] + 20, zeros + 1023], axis=1)
        #     dets[:, 3] = np.min(extend_dets_3, axis=1)
        return dets


    def vis(self, image, image_name, dets):
            if len(dets) == 0:
                return
            for i in range(len(dets)):
                bbox = dets[i, :4]
                score = dets[i, -1]
                image = Image.fromarray(np.array(image))
                draw = ImageDraw.Draw(image)
                draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]), outline=(255, 0, 0))
                font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")
                draw.text((bbox[0], bbox[1] - 40), self._ind_to_class[int(score)], (0, 0, 255), font=font)

            # 保存图片
            save_detect_dir = os.path.join(self.output_dir, 'detect')
            if not os.path.exists(save_detect_dir):
                os.makedirs(save_detect_dir)
            path = os.path.join(save_detect_dir, image_name + '.bmp')
            image.save(path + '.bmp')



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
    parser.add_argument('--detect_model', dest='detect_model', help='Detect model path',
                        default='D:\\jjj\\zlrm\\output\\fpn_new')
    parser.add_argument('--output_dir', dest='output_dir', help='output dir root',
                        default='D:\\jjj\\zlrm\\data\\result')
    args = parser.parse_args()

    return args


def defect_detect():
    args = parse_args()
    defect_detect = zlrm_defect_detect(args.detect_net, args.detect_model, args.output_dir)
    return defect_detect


def ProcessImages(defect_detect, image_arr, image_name):

    dets = defect_detect.image_defect_detect(image_arr, image_name)

    ##**************************************************##
    return dets, image_arr

if __name__ == '__main__':

    defect_detect_ = defect_detect()

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.bmp'))

    im_name_root = os.path.join(cfg.DATA_DIR, 'demo')
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for {:s}'.format(im_name))
        image = readimage(im_name)
        dets, image = ProcessImages(defect_detect_,
                                    image, (im_name.strip(im_name_root)).split('.')[0])
        # vis(image, im_name, dets)
        print('kkk',dets)