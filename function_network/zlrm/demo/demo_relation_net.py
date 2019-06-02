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

CLASSES = ('vein', 'black_ground', 'slag_inclusion', 'aluminium_skimmings', 'crack', 'paint_smear')  # 背景，纹理，黑色背景，铝屑，裂纹，油污


def imread(file_path):
    im = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
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

class demo():

    def __init__(self, sess, network_name, model_root, sample_dir, output_dir):
        self.sess = sess
        self.output_dir = output_dir
        self.classes = ('vein', 'black_ground', 'slag_inclusion', 'aluminium_skimmings', 'crack', 'paint_smear')  # 背景，纹理，黑色背景，铝屑，裂纹，油污
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes())))

        self._ind_to_class = dict(zip(range(self.num_classes()), self.classes))
        self.net = self.load_network(network_name, model_root)
        self.sample = self.get_sample(sample_dir)


    def num_classes(self):
        return len(self.classes)

    def load_network(self, network_name, model_root):
        net = get_network(network_name)
        # load model
        print('Loading networks {:s}... '.format(network_name))
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_root)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), )
        saver.restore(self.sess, ckpt.model_checkpoint_path)
        print(' done.')
        return net

    def classify(self, im):

        blobs = self.get_blobs(im)
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

        # scores = np.reshape(scores, (-1))
        # prob = np.argmax(np.split(scores, query_num[0]))
        # prob = self.sample['sample_label'][prob]
        scores = np.reshape(scores, (-1, 1))
        prob = np.argmax(np.split(scores, query_num[0]), axis=1)
        prob = self.sample['sample_label'][prob].reshape((-1, 1))

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

    def demo(self, image_root, image_name):

        # Load the demo image
        image = readimage(image_root)

        timer = Timer()
        timer.tic()
        prob = self.classify(image)
        timer.toc()
        print('Detection took {:.3f}s '.format(timer.total_time))

        save_classify_image_dir = os.path.join(self.output_dir, self._ind_to_class[prob])
        if not os.path.exists(save_classify_image_dir):
            os.makedirs(save_classify_image_dir)

        image = Image.fromarray(np.array(image))
        save_classify_image_root = os.path.join(save_classify_image_dir, image_name)
        image.save(save_classify_image_root)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='zlrm demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use',
                        default='Relation_net_test')
    parser.add_argument('--model_root', dest='model_root', help='Model path',
                        default='D:\\jjj\\zlrm\\output\\zlrm_relation_net_classifier')
    parser.add_argument('--sample_dir', dest='sample_dir', help='sample_dir',
                        default='D:\\jjj\\zlrm\\data\\classifier_data\\datasets\\sample_data')
    parser.add_argument('--save_result_root', dest='save_result_root', help='save_result_root',
                        default='D:\\jjj\\zlrm\data\\result\\classify_image')

    args = parser.parse_args()

    return args


if __name__ == '__main__':



    args = parse_args()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    demo = demo(sess, args.demo_net, args.model_root, args.sample_dir, args.save_result_root)



    root = os.path.join(cfg.DATA_DIR, 'classify_data', 'classify_image19')
    for im_name in os.listdir(root):
        print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print ('Demo for {:s}'.format(im_name))
        image_root = os.path.join(root, im_name)
        demo.demo(image_root, im_name)