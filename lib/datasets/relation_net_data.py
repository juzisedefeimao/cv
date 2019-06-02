import xml.dom.minidom as minidom

import os
import os.path as osp
import cv2
from PIL import Image
import numpy as np
import pickle as cPickle
from lib.networks.netconfig import cfg
import scipy.misc


class Relation_net_classify( ):
    def __init__(self, image_set, devkit_path=None):
        self.name = 'mini_imagenet'

        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'mini_imagenet')
        self._classes = self.get_classes()
        self._class_to_ind = dict(zip(self._classes, range(self.num_classes())))
        self._image_ext = '.jpg'
        self._image_set = image_set
        self._image_index = self._load_image_set_index()
        self.imagedb = self.get_imagedb()


        # 初始化各个集合的索引
        self.shuffle_inds_init()


        assert os.path.exists(self._devkit_path), \
                'ZLRMdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)


    def num_classes(self):
        return len(self._classes)
    def classes(self):
        return self._classes
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.ZLRM.DATA_DIR, 'mini_imagenet', 'datasets', 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    def get_classes(self):
        classes = []
        filename_dir = os.path.join(self._data_path, 'datasets', 'main')
        for filename in os.listdir(filename_dir):
            if filename.split('.')[-1] == 'txt':
                classes.append(filename.split('.')[0].split('_')[0])

        classes = sorted(classes)
        return classes

    def get_set_classes(self, image_set):
        classes = []
        filename_dir = os.path.join(self._data_path, 'datasets', 'main')
        for filename in os.listdir(filename_dir):
            if filename.split('.')[-1] == 'txt' and filename.split('.')[0].split('_')[-1] == image_set:
                classes.append(filename.split('.')[0].split('_')[0])

        classes = sorted(classes)
        return classes

    def shuffle_inds_init(self):
        self._classes_perm = {}
        self._classes_cur = {}
        for name in ['train', 'val', 'test']:
            self._shuffle_classes_inds(name)

        self._perm = {}
        self._cur = {}

    # ========================取下一批数据===========================================
    def _shuffle_inds(self, image_set, name):

        self._perm[name] = np.random.permutation(np.arange(len(self.imagedb[image_set][name])))
        self._cur[name] = 0

    def _shuffle_classes_inds(self, image_set):
        self._classes_perm[image_set] = np.random.permutation(np.arange(len(self.imagedb[image_set])))
        self._classes_cur[image_set] = 0

    def _get_next_minibatch_inds(self):
        db_inds = {}
        if self._image_set == 'train' or self._image_set == 'val':
            db_inds['train'] = {}
            # 取训练的类别
            classes_train_set = self.get_set_classes('train')
            classes_set = []
            if self._classes_cur['train'] + cfg.RELATION_NET.TRAIN.CLASSIFY_NUM >= len(self.imagedb['train']):
                self._shuffle_classes_inds('train')

            classes_indx_set = self._classes_perm['train'][self._classes_cur['train']:
                                                           self._classes_cur['train'] + cfg.RELATION_NET.TRAIN.CLASSIFY_NUM]
            self._classes_cur['train'] += cfg.RELATION_NET.TRAIN.CLASSIFY_NUM
            for classes_index in classes_indx_set:
                classes_set.append(classes_train_set[classes_index])

            # 取训练的相应类别的数据
            for name in classes_set:
                self._shuffle_inds('train', name)

                db_inds['train'][name] = self._perm[name][self._cur[name]:
                                                                   self._cur[name] + cfg.RELATION_NET.TRAIN.CLASSIFY_BATCH]

            if self._image_set == 'val':
                db_inds['val'] = {}

                # 取验证的类别
                classes_val_set = self.get_set_classes('val')
                classes_set = []
                if self._classes_cur['val'] + cfg.RELATION_NET.TRAIN.CLASSIFY_NUM >= len(self.imagedb['val']):
                    self._shuffle_classes_inds('val')

                classes_indx_set = self._classes_perm['val'][self._classes_cur['val']:
                                                               self._classes_cur[
                                                                   'val'] + cfg.RELATION_NET.TRAIN.CLASSIFY_NUM]
                self._classes_cur['val'] += cfg.RELATION_NET.TRAIN.CLASSIFY_NUM
                for classes_index in classes_indx_set:
                    classes_set.append(classes_val_set[classes_index])

                # 取验证的相应类别的数据
                for name in classes_set:
                    self._shuffle_inds('val', name)

                    db_inds['val'][name] = self._perm[name][self._cur[name]:
                                                              self._cur[name] + cfg.RELATION_NET.TRAIN.CLASSIFY_BATCH]

        elif self._image_set == 'test':
            db_inds['test'] = {}
            # 取测试的类别
            classes_test_set = self.get_set_classes('test')
            classes_set = []
            if self._classes_cur['test'] + cfg.RELATION_NET.TRAIN.CLASSIFY_NUM >= len(self.imagedb['test']):
                self._shuffle_classes_inds('test')

            classes_indx_set = self._classes_perm['test'][self._classes_cur['test']:
                                                           self._classes_cur[
                                                               'test'] + cfg.RELATION_NET.TRAIN.CLASSIFY_NUM]
            self._classes_cur['test'] += cfg.RELATION_NET.TRAIN.CLASSIFY_NUM
            for classes_index in classes_indx_set:
                classes_set.append(classes_test_set[classes_index])

            # 取测试的相应类别的数据
            for name in classes_set:
                self._shuffle_inds('test', name)

                db_inds['test'][name] = self._perm[name][self._cur[name]:
                                                          self._cur[name] + cfg.RELATION_NET.TRAIN.CLASSIFY_BATCH]


        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = {}
        for image_set in db_inds:
            minibatch_db[image_set] = {'sample': [], 'query': []}
            for classname in db_inds[image_set]:
                for i in db_inds[image_set][classname][:cfg.RELATION_NET.TRAIN.CLASSIFY_SAMPLE]:
                    minibatch_db[image_set]['sample'].append(self.imagedb[image_set][classname][i])
                for i in db_inds[image_set][classname][cfg.RELATION_NET.TRAIN.CLASSIFY_SAMPLE:]:
                    minibatch_db[image_set]['query'].append(self.imagedb[image_set][classname][i])

        return get_minibatch(minibatch_db)

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        # blob字典，有3个元素，im_name、data、label
        return blobs

    # =====================================取图片=============================================

    def image_path_at(self, image_set, classify, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[image_set][classify][i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'datasets', 'ImageSets',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_index_dic = {'train':{}, 'val':{}, 'test':{}}
        fail_file_root = []
        filename_dir = os.path.join(self._data_path, 'datasets', 'main')
        for filename in os.listdir(filename_dir):
            if filename.split('.')[-1] == 'txt':
                image_set_file = os.path.join(filename_dir, filename)
                if os.path.exists(image_set_file):
                    with open(image_set_file) as f:
                        image_index = [x.strip() for x in f.readlines()]

                    class_set = filename.split('.')[0].split('_')
                    image_index_dic[class_set[1]][class_set[0]] = image_index
                else:
                    fail_file_root.append(image_set_file)
        assert len(image_index_dic) > 0, 'Path does not exist: {}'.format(fail_file_root)
        return image_index_dic


    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.ZLRM.DATA_DIR)

    # =================================取roi==========================================

    def get_imagedb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        imagedb = {}
        for image_set in self._image_index:
            cache_file = os.path.join(self.cache_path(), self.name + image_set + '_gt_imagedb.pkl')
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as fid:
                    imagedb[image_set] = cPickle.load(fid)
                print('{} gt imagedb loaded from {}'.format(self.name, cache_file))
                continue
            imagedb[image_set] = {}
            for name in self._image_index[image_set]:
                imagedb[image_set][name] = []
                for i in range(len(self._image_index[image_set][name])):
                    image_path = self.image_path_at(image_set, name, i)

                    imagedb[image_set][name].append({'im_name':self._image_index[image_set][name][i],
                                                 'im_path':image_path,
                                                 'label':int(self._class_to_ind[name])})
            with open(cache_file, 'wb') as fid:
                cPickle.dump(imagedb[image_set], fid, cPickle.HIGHEST_PROTOCOL)
            print('wrote gt roidb to {}'.format(cache_file))

        return imagedb

    def one_hot(self, label):
        one_hot = np.zeros(self.num_classes(), dtype=np.float32)
        one_hot[label] = 1.0
        return one_hot


# 可读取中文路径的图片
def imread(file_path):
    im = np.array(Image.open(file_path))
    if len(im.shape) == 2:
        c = []
        for i in range(3):
            c.append(im)
        im = np.asarray(c)
        im = im.transpose([1, 2, 0])
    return im

def get_minibatch(minibatchdb):
    blob = {}
    for image_set in minibatchdb:
        blob[image_set] = {'sample':{}, 'query':{}}
        for sample_query in minibatchdb[image_set]:
            blob[image_set][sample_query]['im_name'] = []
            blob[image_set][sample_query]['data'] = []
            blob[image_set][sample_query]['label'] = []
            for i in range(len(minibatchdb[image_set][sample_query])):
                minibatch = minibatchdb[image_set][sample_query]
                blob[image_set][sample_query]['im_name'].append(minibatch[i]['im_name'])
                image = imread(minibatch[i]['im_path'])
                image = Image.fromarray(image.astype(np.uint8))
                image = image.resize(cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE, Image.ANTIALIAS)
                image = np.array(image)
                image = image.astype(np.float32, copy=False)
                image -= cfg.PIXEL_MEANS
                blob[image_set][sample_query]['data'].append(image)
                blob[image_set][sample_query]['label'].append(minibatch[i]['label'])
            blob[image_set][sample_query]['data'] = np.stack(blob[image_set][sample_query]['data'], axis=0)
            blob[image_set][sample_query]['label'] = np.stack(blob[image_set][sample_query]['label'], axis=0).reshape(-1)
        blob[image_set]['sample']['label'] = blob[image_set]['sample']['label'][0::cfg.RELATION_NET.TRAIN.CLASSIFY_SAMPLE]

    return blob



def image_transform_1_3(image):
    if len(image.shape) == 2:
        c = []
        for i in range(3):
            c.append(image)
        image = np.asarray(c)
        image = image.transpose([1, 2, 0])

    return image