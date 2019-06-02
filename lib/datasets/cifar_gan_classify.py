import xml.dom.minidom as minidom

import os
import os.path as osp
import cv2
from PIL import Image
import numpy as np
import pickle as cPickle
from lib.networks.netconfig import cfg
import scipy.misc


class cifar_gan_classify( ):
    def __init__(self, image_set, devkit_path=None):
        self.name = 'cifar'

        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'cifar')
        self._classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # 背景，纹理，夹渣，铝屑，裂纹，边裂，油污
        self._class_to_ind = dict(zip(self._classes, range(self.num_classes())))
        self._image_ext = '.bmp'
        # self._image_index = self._load_image_set_index()
        self.train_imagedb = self.get_train_imagedb()

        self.test_imagedb = self.get_test_imagedb()
        self.epoch_imagedb = self.get_epoch_imagedb()
        self.epoch_iter_num = self.get_epoch_iter_num()
        self._perm = {}
        self._cur = {}
        self._epoch_perm = {}
        self._epoch_cur = {}
        for name in self.train_imagedb:
            self._shuffle_inds(name)

        for name in ['label', 'unlabel', 'test']:
            self._shuffle_epoch_inds(name)

        assert os.path.exists(self._devkit_path), \
                'ZLRMdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)


    def num_classes(self):
        return len(self._classes)
    def classes(self):
        return self._classes
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.ZLRM.DATA_DIR, 'cifar', 'datasets', 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path
    # ==============================取训练数据================================================

    def get_train_imagedb(self):
        self._image_set = 'train'
        self._image_index = self._load_image_set_index()
        train_imagedb = self.gt_imagedb()
        return train_imagedb

    # ========================取下一批数据===========================================
    def _shuffle_inds(self, name):
        """Randomly permute the training roidb."""
        self._perm[name] = np.random.permutation(np.arange(len(self.train_imagedb[name])))
        self._cur[name] = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        db_inds = {}
        for name in self.train_imagedb:
            if self._cur[name] + cfg.GAN.TRAIN.CLASSIFY_BATCH[name] >= len(self.train_imagedb[name]):
                self._shuffle_inds(name)

            db_inds[name] = self._perm[name][self._cur[name]:self._cur[name] + cfg.GAN.TRAIN.CLASSIFY_BATCH[name]]
            self._cur[name] += cfg.GAN.TRAIN.CLASSIFY_BATCH[name]

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = {'label':[], 'unlabel':[]}
        for name in db_inds:
            if name == 'unlabel':
                for i in db_inds[name]:
                    minibatch_db['unlabel'].append(self.train_imagedb[name][i])
            else:
                for i in db_inds[name]:
                    minibatch_db['label'].append(self.train_imagedb[name][i])

        return get_minibatch(minibatch_db)

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        # blob字典，有3个元素，im_name、data、label
        return blobs

    # ==================================取下一个test数据==========================

    def get_test_imagedb(self):
        self._image_set = 'test'
        self._image_index = self._load_image_set_index()
        test_classify_imagedb = self.gt_imagedb()
        test_imagedb = []
        for name in test_classify_imagedb:
            test_imagedb = test_imagedb + test_classify_imagedb[name]

        return test_imagedb

    # ==================================取下一个epoch数据==========================

    def get_epoch_imagedb(self):
        epoch_imagedb = {}
        epoch_imagedb['label'] = []
        epoch_imagedb['unlabel'] = []
        epoch_imagedb['test'] = self.test_imagedb
        for name in self.train_imagedb:
            if name == 'unlabel':
                epoch_imagedb['unlabel'] = epoch_imagedb['unlabel'] + self.train_imagedb[name]
            else:
                for i in range(len(self.train_imagedb[name])):
                    if i < (cfg.GAN.TRAIN.LABEL_DATA_NUM // self.num_classes()):
                        epoch_imagedb['label'].append(self.train_imagedb[name][i])
                        epoch_imagedb['unlabel'].append(self.train_imagedb[name][i])
                    epoch_imagedb['unlabel'].append(self.train_imagedb[name][i])
        return epoch_imagedb

    def get_epoch_iter_num(self):
        iter_num = len(self.epoch_imagedb['label']) // cfg.GAN.TRAIN.CLASSIFY_BATCH['label']
        return int(iter_num)

    def _shuffle_epoch_inds(self, name):
        """Randomly permute the training roidb."""
        self._epoch_perm[name] = np.random.permutation(np.arange(len(self.epoch_imagedb[name])))
        self._epoch_cur[name] = 0

    def _get_next_epoch_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        db_inds = {}
        for name in ['label', 'unlabel', 'test']:
            if self._epoch_cur[name] + cfg.GAN.TRAIN.CLASSIFY_BATCH[name] >= len(self.epoch_imagedb[name]):
                self._shuffle_epoch_inds(name)

            db_inds[name] = self._epoch_perm[name][self._epoch_cur[name]:self._epoch_cur[name] +
                                                                         cfg.GAN.TRAIN.CLASSIFY_BATCH[name]]
            self._epoch_cur[name] += cfg.GAN.TRAIN.CLASSIFY_BATCH[name]

        return db_inds

    def _get_next_eopch_minibatch(self):
        db_inds = self._get_next_epoch_minibatch_inds()
        minibatch_db = {'label': [], 'unlabel': [], 'test':[]}
        for name in db_inds:
            for i in db_inds[name]:
                minibatch_db[name].append(self.epoch_imagedb[name][i])
        return get_minibatch(minibatch_db)

    def epoch_forward(self):
        blobs = self._get_next_eopch_minibatch()
        return  blobs

    # =====================================取图片=============================================

    def image_path_at(self, classify, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[classify][i])

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
        image_index_dic = {}
        fail_file_root = []
        filename_list = list(self._classes) + ['unlabel']
        for filename in filename_list:
            image_set_file = os.path.join(self._data_path, 'datasets', 'main',
                                          filename + '_' + self._image_set + '.txt')
            if os.path.exists(image_set_file):
                with open(image_set_file) as f:
                    image_index = [x.strip() for x in f.readlines()]
                image_index_dic[filename] = image_index
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

    def gt_imagedb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path(), self.name + self._image_set + '_gt_imagedb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                imagedb = cPickle.load(fid)
            print('{} gt imagedb loaded from {}'.format(self.name, cache_file))
            return imagedb
        gt_imagedb = {}
        for name in self._image_index:
            gt_imagedb[name] = []
            if name == 'unlabel':
                for i in range(len(self._image_index[name])):
                    image_path = self.image_path_at(name, i)

                    label = np.random.randint(0, self.num_classes())
                    gt_imagedb[name].append({'im_name': self._image_index[name][i],
                                             'im_path': image_path, 'label': label})
            else:
                for i in range(len(self._image_index[name])):
                    image_path = self.image_path_at(name, i)

                    gt_imagedb[name].append({'im_name':self._image_index[name][i],
                                             'im_path':image_path,
                                             'label':int(self._class_to_ind[name])})
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_imagedb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_imagedb

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
    blob['im_name'] = []
    blob['data'] = []
    blob['label'] = []
    blob['unlabel_data'] = []
    blob['unlabel'] = []
    blob['test_data'] = []
    blob['test_label'] = []
    for i in range(len(minibatchdb['label'])):
        minibatch = minibatchdb['label']
        blob['im_name'].append(minibatch[i]['im_name'])
        image = scipy.misc.imread(minibatch[i]['im_path']).astype(np.float)
        image = scipy.misc.imresize(image, [cfg.GAN.IMAGE_SIZE, cfg.GAN.IMAGE_SIZE])
        image = np.array(image) / 127.5 - 1.
        image = image_transform_1_3(image)
        blob['data'].append(image)
        blob['label'].append(minibatch[i]['label'])
    for i in range(len(minibatchdb['unlabel'])):
        minibatch = minibatchdb['unlabel']
        blob['im_name'].append(minibatch[i]['im_name'])
        image = scipy.misc.imread(minibatch[i]['im_path']).astype(np.float)
        image = scipy.misc.imresize(image, [cfg.GAN.IMAGE_SIZE, cfg.GAN.IMAGE_SIZE])
        image = np.array(image) / 127.5 - 1.
        image = image_transform_1_3(image)
        blob['unlabel_data'].append(image)
        blob['unlabel'].append(minibatch[i]['label'])
    for i in range(len(minibatchdb['test'])):
        minibatch = minibatchdb['test']
        blob['im_name'].append(minibatch[i]['im_name'])
        image = scipy.misc.imread(minibatch[i]['im_path']).astype(np.float)
        image = scipy.misc.imresize(image, [cfg.GAN.IMAGE_SIZE, cfg.GAN.IMAGE_SIZE])
        image = np.array(image) / 127.5 - 1.
        image = image_transform_1_3(image)
        blob['test_data'].append(image)
        blob['test_label'].append(minibatch[i]['label'])
    blob['data'] = np.stack(blob['data'], axis=0)
    blob['unlabel_data'] = np.stack(blob['unlabel_data'], axis=0)
    blob['test_data'] = np.stack(blob['test_data'], axis=0)
    blob['label'] = np.stack(blob['label'], axis=0).reshape(-1)
    blob['unlabel'] = np.stack(blob['unlabel'], axis=0).reshape(-1)
    blob['test_label'] = np.stack(blob['test_label'], axis=0).reshape(-1)
    blob['noise'] = np.random.uniform(-1, 1, [cfg.GAN.TRAIN.BATCH_SIZE, cfg.GAN.NOISE_DIM]).astype(np.float32)
    return blob



def image_transform_1_3(image):
    if len(image.shape) == 2:
        c = []
        for i in range(3):
            c.append(image)
        image = np.asarray(c)
        image = image.transpose([1, 2, 0])

    return image