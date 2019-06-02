import xml.dom.minidom as minidom

import os
import os.path as osp
from PIL import Image
import numpy as np
import scipy.sparse
import subprocess
import pickle as cPickle
import uuid
import xml.etree.ElementTree as ET

from lib.datasets.imdb import imdb
from lib.networks.netconfig import cfg


class siammask_data(imdb):
    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, 'siammask_data_' + image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'siammask_data')

        self._classes = ['__background__'] + cfg.SIAMSE.CLASSES
        self.defect = cfg.SIAMSE.CLASSES
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = '.bmp'
        self._image_index = self._load_image_set_index()

        self._roidb_handler = self.gt_roidb
        self.templatedb = self.get_templatedb()

        self._shuffle_inds()
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._devkit_path), \
                'ZLRMdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.SIAMSE.DATA_DIR, 'siammask_data', 'datasets', 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    # =====================================取图片=============================================
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

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
        image_set_file = os.path.join(self._data_path, 'datasets', 'main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.SIAMSE.DATA_DIR)




    # =================================取roi==========================================
    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path(), self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print ('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb
        gt_roidb = [self._load_zlrm_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print ('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb


    def _load_zlrm_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        # boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        # gt_classes = np.zeros((num_objs), dtype=np.int32)
        # overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # # "Seg" area for pascal is just the box area
        # seg_areas = np.zeros((num_objs), dtype=np.float32)
        # ishards = np.zeros((num_objs), dtype=np.int32)
        boxes = []
        gt_classes = []
        overlaps = []
        seg_areas = []
        ishards = []

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            # ishards[ix] = difficult

            if (obj.find('name').text in cfg.SIAMSE.CLASSES) and difficult==0:
                bbox = obj.find('bndbox')
                # Make pixel indexes 0-based
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1


                if cfg.SIAMSE.N_CLASSES == 1:
                    cls = self._class_to_ind['defect'.lower().strip()]
                else:
                    cls = self._class_to_ind[obj.find('name').text.lower().strip()]
                boxes.append([x1, y1, x2, y2])
                gt_classes.append(cls)
                overlap = np.zeros((self.num_classes), dtype=np.float32)
                overlap[cls] = 1.0
                overlaps.append(overlap)
                seg_areas.append((x2 - x1 + 1) * (y2 - y1 + 1))
                ishards.append(difficult)

        boxes = np.array(boxes, dtype=np.uint16).reshape((-1, 4))
        gt_classes =np.array(gt_classes, dtype=np.int32).reshape(-1)
        overlaps = np.array(overlaps, dtype=np.float32).reshape((-1, self.num_classes))
        seg_areas = np.array(seg_areas, dtype=np.float32).reshape(-1)
        ishards = np.array(ishards, dtype=np.int32).reshape(-1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    # =================================取template data==========================================
    def _shuffle_inds(self):
        self._perm = {}
        self._cur = {}
        for classes in self.defect:
            self._shuffle_inds_classes(classes)

    def _shuffle_inds_classes(self, classes):

        self._perm[classes] = np.random.permutation(np.arange(len(self.templatedb[classes])))
        self._cur[classes] = 0

    def _get_next_template_inds(self):
        db_inds = {}
        for classes in self.defect:
            if self._cur[classes] + 1 >= len(self.templatedb[classes]):
                self._shuffle_inds_classes(classes)

            db_inds[classes] = self._perm[classes][self._cur[classes]:self._cur[classes] + 1]
        return db_inds

    def _get_next_template_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """

        db_inds = self._get_next_template_inds()
        template = []
        for classes in self.defect:
            template_image = crop_template(self.templatedb[classes][db_inds[classes][0]]['image_path'],
                                           self.templatedb[classes][db_inds[classes][0]]['boxes'][0:4])
            template.append(template_image)

        template = np.stack(template, axis=0)

        return template

    def template(self):
        """Get blobs and copy them into this layer's top blob vector."""
        template = self._get_next_template_minibatch()
        # blob字典，有3个元素，im_name、data、label
        return template

    # 输出所有的模板
    def template_all(self):
        path = 'D:\\jjj\\zlrm\\data\\siammask_data\\datasets\\template'
        for classes in self.defect:
            print(classes, '开始')
            for i in range(len(self.templatedb[classes])):
                crop_template_(self.templatedb[classes][i]['image_path'],
                                               self.templatedb[classes][i]['boxes'][0:4], str(i), os.path.join(path, classes))


    def get_templatedb(self):
        templatedb = {}
        cache_file = os.path.join(self.cache_path(), self.name + '_template_imagedb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                templatedb = cPickle.load(fid)
            print('{} gt templatedb loaded from {}'.format(self.name, cache_file))
            return templatedb

        for classes in self._classes:
            templatedb[classes] = []
            for index in self.image_index:
                templatedb[classes] = templatedb[classes] + self._load_template(index, classes)

        with open(cache_file, 'wb') as fid:
            cPickle.dump(templatedb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return templatedb

    def _load_template(self, index, classes):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        template_classes = []
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            # ishards[ix] = difficult

            if difficult==0 and obj.find('name').text == classes:
                template_classes_ = {}
                bbox = obj.find('bndbox')
                # Make pixel indexes 0-based
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1

                cls = self._class_to_ind[obj.find('name').text.lower().strip()]
                boxes = [x1, y1, x2, y2, cls]
                template_classes_['boxes'] = np.array(boxes, dtype=np.uint16)
                template_classes_['image_path'] = self.image_path_from_index(index)
                template_classes.append(template_classes_)

        return template_classes

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

def crop_template(image_path, boxes):
    im = Image.open(image_path)
    template_crop = im.crop(boxes)
    w = int(boxes[2] - boxes[0])
    h = int(boxes[3] - boxes[1])
    if h > 120:
        w = int(w / (h / 120))
        h = 120
    if w > 120:
        h = int(h / (w / 120))
        w = 120
    template_resize = template_crop.resize((w,h), Image.ANTIALIAS)
    background = Image.new('L', cfg.SIAMSE.TEMPLATE_IMAGE_SIZE, (138))
    template_resize_arr = np.array(template_resize)
    background_arr = np.array(background)


    y1 = int(background_arr.shape[0] / 2 - template_resize_arr.shape[0] / 2)
    x1 = int(background_arr.shape[1] / 2 - template_resize_arr.shape[1] / 2)
    y2 = int(y1 + template_resize_arr.shape[0])
    x2 = int(x1 + template_resize_arr.shape[1])
    background_arr[y1:y2, x1:x2] = template_resize_arr

    b = Image.fromarray(background_arr.astype(np.uint8))
    outpath = os.path.join('D:\data', 'test.bmp')
    b.save(outpath)

    c = []
    for i in range(3):
        c.append(background_arr)
    background_arr = np.asarray(c)
    background_arr = background_arr.transpose([1, 2, 0])
    background_arr = background_arr.astype(np.float32, copy=False)
    background_arr -= cfg.SIAMSE.PIXEL_MEANS
    return background_arr
def crop_template_(image_path, boxes, save_name, path):
    im = Image.open(image_path)
    template_crop = im.crop(boxes)
    w = int(boxes[2] - boxes[0])
    h = int(boxes[3] - boxes[1])

    if h >120:
        w = int(w/(h/120))
        h = 120
    if w > 120:
        h = int(h/(w/120))
        w=120
    template_resize = template_crop.resize((w,h), Image.ANTIALIAS)
    background = Image.new('L', cfg.SIAMSE.TEMPLATE_IMAGE_SIZE, (138))
    template_resize_arr = np.array(template_resize)
    background_arr = np.array(background)


    y1 = int(background_arr.shape[0] / 2 - template_resize_arr.shape[0] / 2)
    x1 = int(background_arr.shape[1] / 2 - template_resize_arr.shape[1] / 2)
    y2 = int(y1 + template_resize_arr.shape[0])
    x2 = int(x1 + template_resize_arr.shape[1])
    background_arr[y1:y2, x1:x2] = template_resize_arr

    b = Image.fromarray(background_arr.astype(np.uint8))
    if not os.path.exists(path):
        os.mkdir(path)
    outpath = os.path.join(path, save_name + '.bmp')
    b.save(outpath)

if __name__ == '__main__':

    siammask = siammask_data('train')
    siammask.template_all()