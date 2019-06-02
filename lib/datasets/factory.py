# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from lib.datasets.siammask import siammask_data
from lib.datasets.zlrmdata import zlrm_data
from lib.datasets.zlrmdata_classify import zlrmdata_classify
from lib.datasets.zlrmdata_fcn_classify import zlrmdata_fcn_classify
from lib.datasets.zlrmdata_gan_classify import zlrmdata_gan_classify
from lib.datasets.cifar_gan_classify import cifar_gan_classify
from lib.datasets.zlrmdata_triple_classifiy import zlrmdata_triple_classify
from lib.datasets.relation_net_data import Relation_net_classify
from lib.datasets.zlrm_relation_net_data import zlrm_Relation_net_classify
from lib.datasets.testdata import test_data
from lib.datasets.voc import VOC2007


for split in ['train', 'val', 'trainval', 'test']:
    name = 'siammask_data_{}'.format(split)
    __sets[name] = (lambda split=split:
            siammask_data(split))


for split in ['train', 'val', 'trainval', 'test']:
    name = 'detector_data_{}'.format(split)
    __sets[name] = (lambda split=split:
            zlrm_data(split))

for split in ['train', 'val', 'trainval', 'test']:
    name = 'zlrmdata_classify_{}'.format(split)
    __sets[name] = (lambda split=split:
            zlrmdata_classify(split))

for split in ['train', 'val', 'trainval', 'test']:
    name = 'zlrmdata_gan_classify_{}'.format(split)
    __sets[name] = (lambda split=split:
            zlrmdata_gan_classify(split))

for split in ['train', 'val', 'trainval', 'test']:
    name = 'zlrmdata_fcn_classify_{}'.format(split)
    __sets[name] = (lambda split=split:
            zlrmdata_fcn_classify(split))

for split in ['train', 'val', 'trainval', 'test']:
    name = 'cifar_gan_classify_{}'.format(split)
    __sets[name] = (lambda split=split:
            cifar_gan_classify(split))

for split in ['train', 'val', 'trainval', 'test']:
    name = 'classifier_data_{}'.format(split)
    __sets[name] = (lambda split=split:
            zlrm_Relation_net_classify(split))

for split in ['train', 'val', 'trainval', 'test']:
    name = 'mini_imagenet_{}'.format(split)
    __sets[name] = (lambda split=split:
            Relation_net_classify(split))

for split in ['train', 'val', 'trainval', 'test']:
    name = 'zlrmdata_triple_classify_{}'.format(split)
    __sets[name] = (lambda split=split:
            zlrmdata_triple_classify(split))

for split in ['train', 'val', 'trainval', 'test']:
    name = 'VOC2007_{}'.format(split)
    __sets[name] = (lambda split=split:
            VOC2007(split))



for split in ['train', 'val']:
    name = 'test_data_{}'.format(split)
    __sets[name] = (lambda split=split: test_data(split))

# Set up voc_<year>_<split> using selective search "fast" mode
# for year in ['2007', '2012', '0712']:
#     for split in ['train', 'val', 'trainval', 'test']:
#         name = 'voc_{}_{}'.format(year, split)
#         __sets[name] = (lambda split=split, year=year:
#                 pascal_voc(split, year))
#
#
# # Set up kittivoc
#     for split in ['train', 'val', 'trainval', 'test']:
#         name = 'kittivoc_{}'.format(split)
#         print (name)
#         __sets[name] = (lambda split=split: kittivoc(split))
#
# # # KITTI dataset
# # for split in ['train', 'val', 'trainval', 'test']:
# #     name = 'kitti_{}'.format(split)
# #     print name
# #     __sets[name] = (lambda split=split: kitti(split))
#
# # Set up coco_2014_<split>
# for year in ['2014']:
#     for split in ['train', 'val', 'minival', 'valminusminival']:
#         name = 'coco_{}_{}'.format(year, split)
#         __sets[name] = (lambda split=split, year=year: coco(split, year))
#
# # Set up coco_2015_<split>
# for year in ['2015']:
#     for split in ['test', 'test-dev']:
#         name = 'coco_{}_{}'.format(year, split)
#         __sets[name] = (lambda split=split, year=year: coco(split, year))
#
# # NTHU dataset
# for split in ['71', '370']:
#     name = 'nthu_{}'.format(split)
#     print (name)
#     __sets[name] = (lambda split=split: nthu(split))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    # if not __sets.has_key(name):
    if name not in __sets:
        print (list_imdbs())
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
