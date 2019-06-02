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
        self.classes_detect = ['__background__'] + cfg.SIAMSE.CLASSES
        self.classes_num = len(cfg.SIAMSE.CLASSES)
        self.template_dir = 'D:\\jjj\\zlrm\\data\\siammask_data\\datasets\\template'

    def detect(self, image):
        """Detect object classes in an image using pre-computed object proposals."""

        # Load the demo image
        # Detect all object classes and regress object bounds
        image = image_transform_1_3(image)
        timer = Timer()
        timer.tic()
        scores, boxes = self.im_detect(image)
        timer.toc()
        print('rois--------------', scores)
        print ('Detection took {:.3f}s for '
               '{:d} object proposals'.format(timer.total_time, len(boxes)))

        CONF_THRESH = 0.3
        # print(scores)
        NMS_THRESH = 0.5
        dets = []
        for i in range(len(boxes)):
            # print('lll')
            cls_boxes = boxes[i]
            cls_scores = scores[i]
            dets_i_ = np.hstack([cls_boxes[:, 0:4], cls_scores])
            keep = nms(dets_i_, NMS_THRESH)
            dets_i = np.hstack([cls_boxes, cls_scores])
            dets_i = dets_i[keep, :]
            inds = np.where(dets_i[:, -1] >= CONF_THRESH)[0]
            dets_i = dets_i[inds, :]
            dets_i = dets_i[:, 0:5]
            dets.append(dets_i)
        return dets


    def im_detect(self, im):

        blobs, im_scales = self.get_blobs(im)
        template = self.get_template(self.template_dir)

        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
        # forward pass
        rpn_rois, scores = self.network.get_rois()
        feed_dict = {self.network.search_data: blobs['data'], self.network.template_data:template,
                     self.network.im_info: blobs['im_info']}

        rpn_rois_val, scores_val = \
            self.sess.run([rpn_rois, scores], feed_dict=feed_dict)

        # print('jjj',rpn_rois_val)
        batch_scores = []
        batch_boxes = []
        batch = int(len(rpn_rois_val) / (self.classes_num*cfg.SIAMSE.IMAGE_TRANSFORM_NUM))
        # print('3,', len(rpn_rois_val), self.classes_num, cfg.SIAMSE.IMAGE_TRANSFORM_NUM)
        # print('2', batch)
        for i in range(batch):
            scores_i = []
            boxes_i = []
            for j in range(cfg.SIAMSE.IMAGE_TRANSFORM_NUM):
                scores_j = []
                boxes_j = []
                for k in range(self.classes_num):
                    index = i*(self.classes_num*cfg.SIAMSE.IMAGE_TRANSFORM_NUM) + j*self.classes_num + k
                    # print('5,', index, i, j, k)
                    rois_k = np.reshape(rpn_rois_val[index], [-1,5])
                    scores_k = np.reshape(scores_val[index], [-1, 1])
                    boxes_k = rois_k[:, 1:5] / im_scales[i*cfg.SIAMSE.IMAGE_TRANSFORM_NUM + j]
                    # if j < cfg.SIAMSE.IMAGE_TRANSFORM_NUM - 1:
                    #     boxes_k = boxes_k + np.array([j*255, 0, j*255, 0])
                    #     boxes_k = boxes_k.astype(np.float32)
                    # else:
                    #     boxes_k = boxes_k
                    cls_k = np.empty_like(scores_val[index], dtype=np.float32)
                    cls_k.fill(k + 1)
                    boxes_k = np.hstack([boxes_k, cls_k])
                    boxes_j.append(boxes_k)
                    scores_j.append(scores_k)
                scores_j = np.vstack(scores_j)
                boxes_j = np.vstack(boxes_j)
                scores_i.append(scores_j)
                boxes_i.append(boxes_j)
            scores_i = np.vstack(scores_i)
            boxes_i = np.vstack(boxes_i)
            batch_scores.append(scores_i)
            batch_boxes.append(boxes_i)

        # print('1', batch_boxes)
        return batch_scores, batch_boxes

    def get_template(self, template_dir):
        template = []
        for template_classes in self.classes_detect[1:self.classes_num + 1]:
            classes_dir = os.path.join(template_dir, template_classes)
            # sample['sample_label'].append(int(self._class_to_ind[classes_folder]))
            # for classes_image_name in os.listdir(classes_dir):
            classes_image_name = os.listdir(classes_dir)[0]
            image_root = os.path.join(classes_dir, classes_image_name)
            image = readimage(image_root)
            im =  image - cfg.SIAMSE.PIXEL_MEANS
            template.append(im)

        template = np.stack(template, axis=0)

        return template

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
        im_orig -= cfg.SIAMSE.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in cfg.SIAMSE.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.SIAMSE.TEST.MAX_SIZE:
                im_scale = float(cfg.SIAMSE.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            # 将255*1020的图片分割成四张
            im_list = self.crop_image(im, 4)
            processed_ims += im_list
            for i in range(len(im_list)):
                im_scale_factors.append(im_scale)
            #  将255*1020的图片分割为2张255*510然后拼接为510*510，然后resize为255*255
            im = self.crop_image(im, 2, joint=True)
            im = cv2.resize(im, None, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_LINEAR)
            # saveimage(im.astype(np.uint8),saveimage_name='1')

            processed_ims.append(im)
            im_scale_factors.append(im_scale*0.5)

        # Create a blob to hold the input images
        blob = self.im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def crop_image(self, im, num, axis=1, joint=False):
        im_list = []
        im_shape = im.shape
        per_im_size = int(im_shape[axis] / num)
        for i in range(num):
            im_list.append(im[:, i * per_im_size:(i + 1) * per_im_size])
        if joint:
            im_list = np.vstack(im_list)
        return im_list

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
        self.classes_classify =  ['__background__'] + cfg.SIAMSE.CLASSES
        self._ind_to_class = dict(zip(range(len(self.classes_classify)), self.classes_classify))

        self.detect_sess, self.detect_network,  = self.load_network(detect_net, detect_model)

        self.detector = detector(self.detect_sess, self.detect_network)
        self.output_dir = output_dir
        # 类别分界线，类别索引小于classify_boundary的不是缺陷，大于等于的为需要报出的缺陷
        self.classify_boundary = 3



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
        for i in range(len(dets)):

            dets[i][:, 0:2] = np.floor(dets[i][:, 0:2])
            dets[i][:, 2:] = np.ceil(dets[i][:, 2:])
            dets[i] = dets[i].astype(np.int32)

        self.vis(image, image_name, dets[0])
        # if len(dets) > 0:
        #     zeros = np.zeros(len(dets), dtype=np.int32)
        #     extend_dets_0 = np.stack([dets[:, 0] - 20, zeros],axis=1)
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
                        default='Siammask_test')
    parser.add_argument('--detect_model', dest='detect_model', help='Detect model path',
                        default='D:\\jjj\\zlrm\\output\\siammask')
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