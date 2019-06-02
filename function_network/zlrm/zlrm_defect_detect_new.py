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


# 检出时的二分类
# CLASSES_DEFECT = ('__background__', # always index 0
#            'defect')
#
# CLASSES_CLASSIFY = ('vein', 'slag_inclusion', 'aluminium_skimmings', 'crack',
#             'edge_crack', 'paint_smear')  # 背景，纹理，夹渣，铝屑，裂纹，边裂，油污
# classify_dic = dict(zip(range(len(CLASSES_CLASSIFY)), CLASSES_CLASSIFY))
 # {0:'vein', 1:'slag_inclusion', 2:'aluminium_skimmings', 3:'crack',
 #     4:'edge_crack', 5:'paint_smear'}
image_ext = '.bmp'
# =================================================检出===========================================================

class detector():

    def __init__(self, sess, net):
        self.sess = sess
        self.network = net
        self.classes_detect = ('__background__',  # always index 0
                               'defect')

    def detect(self, image):
        """Detect object classes in an image using pre-computed object proposals."""

        # Load the demo image
        # Detect all object classes and regress object bounds
        image = image_transform_1_3(image)
        timer = Timer()
        timer.tic()
        scores, boxes = self.im_detect(image)
        timer.toc()
        # print('rois--------------', scores)
        print ('Detection took {:.3f}s for '
               '{:d} object proposals'.format(timer.total_time, boxes.shape[0]))

        CONF_THRESH = 0.7
        NMS_THRESH = 0.1
        for cls_ind, cls in enumerate(self.classes_detect[1:]):
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


    def im_detect(self, im):

        blobs, im_scales = self.get_blobs(im)

        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
        # forward pass
        feed_dict = {self.network.data: blobs['data'], self.network.im_info: blobs['im_info']}

        cls_prob, bbox_pred, rois = \
            self.sess.run([self.network.get_output('cls_prob'), self.network.get_output('ave_bbox_pred_rois'),
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
# =========================================================识别===============================================
class classifier():

    def __init__(self, sess, net):
        self.sess = sess
        self.network = net

    def im_classify(self, image_batch):

        blobs = self.get_classify_blobs(image_batch)

        # forward pass
        feed_dict = {self.network.data: blobs['data']}

        cls_prob, = \
            self.sess.run([self.network.get_output('cls_prob')], \
                     feed_dict=feed_dict)

        if isinstance(cls_prob, tuple):
            cls_prob = cls_prob[0]

        cls_prob = np.reshape(cls_prob, [-1, cfg.ZLRM.TRAIN.CLASSIFY_NUM])  # (R, C+1)

        scores = cls_prob

        return scores

    def get_classify_blobs(self, image_batch):
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

    def classify(self, image_batch):
        timer = Timer()
        timer.tic()
        scores = self.im_classify(image_batch)
        timer.toc()
        print('Detection took {:.3f}s '.format(timer.total_time))
        scores = np.array(scores)
        # 取最大概率的类别为分类类别，并赋予类别号
        scores = scores.argmax(axis=1)
        return scores

class relation_classifier():

    def __init__(self, sess, net):
        self.sess = sess
        self.classes = ('vein', 'black_ground', 'paint_smear', 'dirty_bar',
                        'aluminium_skimmings', 'slag_inclusion', 'crack', 'dirty')  # 背景，纹理，黑色背景，铝屑，裂纹，油污
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes())))

        self._ind_to_class = dict(zip(range(self.num_classes()), self.classes))
        self.net = net
        self.sample = self.get_sample('D:\\jjj\\zlrm\\data\\classifier_data\\datasets\\sample_data')


    def num_classes(self):
        return len(self.classes)


    def im_classify(self, image_batch):

        blobs = self.get_classify_blobs(image_batch)
        score = self.net.demo_net()
        query_num = np.array(blobs['data'].shape[0]).reshape([1])

        # forward pass
        feed_dict = {self.net.query_num:query_num,
                    self.net.sample_data:self.sample['sample_data'],
                     self.net.sample_label:self.sample['sample_label'],
                     self.net.query_data: blobs['data']}

        scores = self.sess.run(score, feed_dict=feed_dict)

        if isinstance(scores, tuple):
            scores = scores[0]

        scores = np.reshape(scores, (-1, 1))
        prob = np.argmax(np.split(scores, query_num[0]), axis=1)
        prob = self.sample['sample_label'][prob].reshape((-1,1))

        return prob

    def get_blobs(self, image):
        """Convert an image and RoIs within that image into networks inputs."""

        blobs = {'data': None}
        image = Image.fromarray(image.astype(np.uint8))
        image = image.resize(cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE, Image.ANTIALIAS)
        image = np.array(image)
        image = image.astype(np.float32, copy=False)
        image -= cfg.ZLRM.PIXEL_MEANS
        blobs['data'] = np.reshape(image, (-1, image.shape[0], image.shape[1], image.shape[2]))

        return blobs

    def get_classify_blobs(self, image_batch):
        blobs = {'data': []}
        for i in range(len(image_batch)):
            image = image_batch[i]
            # image = Image.fromarray(image.astype(np.uint8))
            image = image.resize(cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE, Image.ANTIALIAS)
            image = np.array(image)
            image = image.astype(np.float32, copy=False)
            image -= cfg.ZLRM.PIXEL_MEANS
            blobs['data'].append(image)
        blobs['data'] = np.stack(blobs['data'], axis=0)

        return blobs

    def get_sample(self, sample_dir):
        sample = {'sample_data':[], 'sample_label':[]}
        for classes_folder in os.listdir(sample_dir):
            classes_dir = os.path.join(sample_dir, classes_folder)
            sample['sample_label'].append(int(self._class_to_ind[classes_folder]))
            for classes_image_name in os.listdir(classes_dir):
                image_root = os.path.join(classes_dir, classes_image_name)
                image = readimage(image_root)
                image = Image.fromarray(image.astype(np.uint8))
                image = image.resize(cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE, Image.ANTIALIAS)
                image = np.array(image)
                image = image.astype(np.float32, copy=False)
                image -= cfg.ZLRM.PIXEL_MEANS
                sample['sample_data'].append(image)

        sample['sample_data'] = np.stack(sample['sample_data'], axis=0)
        sample['sample_label'] = np.stack(sample['sample_label'], axis=0).reshape(-1)

        return sample

    def classify(self, image_batch):

        timer = Timer()
        timer.tic()
        scores = self.im_classify(image_batch)
        timer.toc()
        print('Detection took {:.3f}s '.format(timer.total_time))

        return scores

class zlrm_defect_detect():
    def __init__(self, detect_net, detect_model, classify_net, classify_model,
                 output_dir):


        # self.classes_classify = ('vein', 'slag_inclusion', 'aluminium_skimmings', 'crack',
        #                     'black_ground', 'paint_smear')  # 背景，纹理，夹渣，铝屑，裂纹，边裂，油污
        self.classes_classify = ('vein', 'black_ground', 'paint_smear', 'dirty_bar',
                        'aluminium_skimmings', 'slag_inclusion', 'crack', 'dirty')  # 背景，纹理，黑色背景，铝屑，裂纹，油污
        self._ind_to_class = dict(zip(range(len(self.classes_classify)), self.classes_classify))

        self.detect_sess, self.classify_sess, \
        self.detect_network, self.classify_network = self.load_network(detect_net, detect_model,
                                                                       classify_net, classify_model)

        self.detector = detector(self.detect_sess, self.detect_network)
        self.classifier = relation_classifier(self.classify_sess, self.classify_network)
        self.output_dir = output_dir
        # 类别分界线，类别索引小于classify_boundary的不是缺陷，大于等于的为需要报出的缺陷
        self.classify_boundary = 3



    def load_network(self, detect_net, detect_model, classify_net, classify_model):

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
            detect_network = get_network(detect_net)
            print('Loading networks {:s}... '.format(detect_net))
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(detect_model)
            print('Restoring from {}...'.format(ckpt.model_checkpoint_path), )
            saver.restore(detect_sess, ckpt.model_checkpoint_path)
            print(' done.')
        with classify_graph.as_default():
            classify_network = get_network(classify_net)
            print('Loading networks {:s}... '.format(classify_net))
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(classify_model)
            print('Restoring from {}...'.format(ckpt.model_checkpoint_path), )
            saver.restore(classify_sess, ckpt.model_checkpoint_path)
            print(' done.')

        return detect_sess, classify_sess, detect_network, classify_network

    def close_sess(self):
        self.detect_sess.close()
        self.classify_sess.close()

    def defect_detect(self, image):
        dets = self.detector.detect(image)
        return dets

    def fetch_defect_image(self, dets, image):
        pre_image_batch = []
        image_bath = []
        for i in range(len(dets)):
            image = Image.fromarray(np.array(image).astype(np.uint8))
            cut_image = image.crop(dets[i][:4])
            image_bath.append(cut_image)

            # cut_image = cut_image.resize(cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE, Image.ANTIALIAS)
            # cut_image = np.array(cut_image)
            # cut_image = cut_image.astype(np.float32, copy=False)
            # cut_image -= cfg.ZLRM.PIXEL_MEANS
            # pre_image_batch.append(cut_image)
        # pre_image_batch = np.stack(pre_image_batch, axis=0)
        return image_bath

    def defect_classify(self, dets, image, image_name):

        defect_image_batch = self.fetch_defect_image(dets, image)
        scores = self.classifier.classify(defect_image_batch)

        # 保存分类的缺陷图片并将类别序号放到dets里
        for i in range(len(dets)):
            image = Image.fromarray(np.array(defect_image_batch[i]).astype(np.uint8))
            save_classify_dir = os.path.join(self.output_dir, 'classify', self._ind_to_class[int(scores[i])])
            if not os.path.exists(save_classify_dir):
                os.makedirs(save_classify_dir)
            save_classify_image_root = os.path.join(save_classify_dir,
                                                    image_name + '_' + str(i) + '.bmp')
            image.save(save_classify_image_root)

            dets[i][4] = int(scores[i])
        return dets

    def image_defect_detect(self, image, image_name):
        dets = self.defect_detect(image)
        if dets.size > 0:
            dets = self.defect_classify(dets, image, image_name)
            dets = dets[np.where(dets[:, 4] > self.classify_boundary)]

        dets[:, 0:2] = np.floor(dets[:, 0:2])
        dets[:, 2:] = np.ceil(dets[:, 2:])
        dets = dets.astype(np.int32)

        self.vis(image, image_name, dets)
        if len(dets) > 0:
            zeros = np.zeros(len(dets), dtype=np.int32)
            extend_dets_0 = np.stack([dets[:, 0] - 20, zeros],axis=1)
            dets[:, 0] = np.max(extend_dets_0, axis=1)
            extend_dets_1 = np.stack([dets[:, 1] - 20, zeros], axis=1)
            dets[:, 1] = np.max(extend_dets_1, axis=1)
            extend_dets_2 = np.stack([dets[:, 2] + 20, zeros + 4095], axis=1)
            dets[:, 2] = np.min(extend_dets_2, axis=1)
            extend_dets_3 = np.stack([dets[:, 3] + 20, zeros + 1023], axis=1)
            dets[:, 3] = np.min(extend_dets_3, axis=1)
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
    parser.add_argument('--classify_net', dest='classify_net', help='Network to classify',
                        default='Relation_net_test')
    parser.add_argument('--detect_model', dest='detect_model', help='Detect model path',
                        default='D:\\jjj\\zlrm\\output\\fpn')
    parser.add_argument('--classify_model', dest='classify_model', help='Classify model path',
                        default='D:\\jjj\\zlrm\\output\\zlrm_relation_net_classifier')
    parser.add_argument('--output_dir', dest='output_dir', help='output dir root',
                        default='D:\\jjj\\zlrm\\data\\result')
    args = parser.parse_args()

    return args


def defect_detect():
    args = parse_args()
    defect_detect = zlrm_defect_detect(args.detect_net, args.detect_model, args.classify_net, args.classify_model,
                 args.output_dir)
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