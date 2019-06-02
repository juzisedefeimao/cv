import xml.dom.minidom as minidom

import os
import os.path as osp
import cv2
from PIL import Image
import numpy as np
import pickle as cPickle
from lib.networks.netconfig import cfg


class zlrmdata_classify( ):
    def __init__(self, image_set, devkit_path=None):
        self.name = 'zlrmdata_classify'
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'zlrmdata_classify')
        self._classes = ('vein', 'slag_inclusion', 'aluminium_skimmings', 'crack',
                         'edge_crack', 'paint_smear')  # 背景，纹理，夹渣，铝屑，裂纹，边裂，油污
        self._class_to_ind = dict(zip(self._classes, range(self.num_classes())))
        self._image_ext = '.bmp'
        self._image_index = self._load_image_set_index()
        self.imagedb = self.gt_imagedb()
        self._perm = {}
        self._cur = {}
        for name in self.imagedb:
            self._shuffle_inds(name)

        assert os.path.exists(self._devkit_path), \
                'ZLRMdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)


    def num_classes(self):
        return len(self._classes)
    def classes(self):
        return self._classes
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.ZLRM.DATA_DIR, 'zlrmdata_classify', 'datasets', 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path
    # ==============================接口================================================

    def _shuffle_inds(self, name):
        """Randomly permute the training roidb."""
        self._perm[name] = np.random.permutation(np.arange(len(self.imagedb[name])))
        self._cur[name] = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        db_inds = {}
        for name in self.imagedb:
            if self._cur[name] + cfg.ZLRM.TRAIN.CLASSIFY_BATCH[name] >= len(self.imagedb[name]):
                self._shuffle_inds(name)

            db_inds[name] = self._perm[name][self._cur[name]:self._cur[name] + cfg.ZLRM.TRAIN.CLASSIFY_BATCH[name]]
            self._cur[name] += cfg.ZLRM.TRAIN.CLASSIFY_BATCH[name]

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = []
        for name in db_inds:
            for i in db_inds[name]:
                minibatch_db.append(self.imagedb[name][i])

        return get_minibatch(minibatch_db)

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        # blob字典，有3个元素，im_name、data、label
        return blobs
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
        for filename in self._classes:
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
        cache_file = os.path.join(self.cache_path(), self.name + self._image_set + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                imagedb = cPickle.load(fid)
            print(imagedb)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return imagedb
        gt_imagedb = {}
        for name in self._image_index:
            gt_imagedb[name] = []
            for i in range(len(self._image_index[name])):
                image_path = self.image_path_at(name, i)

                gt_imagedb[name].append({'im_name':self._image_index[name][i],
                                         'im_path':image_path, 'label':int(self._class_to_ind[name])})
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_imagedb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_imagedb


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
    for i in range(len(minibatchdb)):
        blob['im_name'].append(minibatchdb[i]['im_name'])
        image = imread(minibatchdb[i]['im_path'])
        image = Image.fromarray(image.astype(np.uint8))
        image = image.resize(cfg.ZLRM.TRAIN.CLASSIFY_IMAGE_SIZE, Image.ANTIALIAS)
        image = np.array(image)
        image = image.astype(np.float32, copy=False)
        image -= cfg.ZLRM.PIXEL_MEANS
        blob['data'].append(image)
        blob['label'].append(minibatchdb[i]['label'])
    blob['data'] = np.stack(blob['data'], axis=0)
    blob['label'] = np.array(blob['label']).reshape(-1)
    return blob