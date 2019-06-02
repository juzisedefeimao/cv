from easydict import EasyDict as edict
from time import strftime, localtime
import os
import os.path as osp
import numpy as np

__C = edict()
cfg = __C


__C.LRELU_DECAY = 0.2
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

__C.TRAIN = edict()
__C.TRAIN.WEIGHT_DECAY = 0.000
__C.TRAIN.BATCH_SIZE = 4

#===================GAN classifier训练参数=====================
__C.GAN = edict()
__C.GAN.GAN_TYPE = 'gan'  # gan 、 lsgan 、 wgan-gp 、 wgan-lp 、 dragan 、 hinge
__C.GAN.LD = 10  # 'The gradient penalty lambda'
__C.GAN.NOISE_DIM = 100  # 'Dimension of noise vector'
__C.GAN.IMAGE_SIZE = 32

#generator生成图片的通道数，默认为3通道（rgb）
__C.GAN.IMAGE_DIM = 3

__C.GAN.TRAIN = edict()
__C.GAN.TRAIN.SOLVER = 'Adam'
__C.GAN.TRAIN.ALPHA = 0.5
__C.GAN.TRAIN.ALPHA_CLA_ADV = 0.01
__C.GAN.TRAIN.GENERATOR_LEARNING_RATE = 0.0002 #'learning rate for generator'
__C.GAN.TRAIN.DISCRIMINATOR_LEARNING_RATE = 0.0002 # 'learning rate for discriminator'
__C.GAN.TRAIN.CLASSIFIER_LEARNING_RATE = 0.0002 # 'learning rate for classifier'
__C.GAN.TRAIN.GENERATOR_MOMENTUM = 0.9
__C.GAN.TRAIN.DISCRIMINATOR_MOMENTUM = 0.9
__C.GAN.TRAIN.CLASSIFIER_MOMENTUM = 0.9
__C.GAN.TRAIN.GENERATOR_GAMMA = 0.1
__C.GAN.TRAIN.DISCRIMINATOR_GAMMA = 0.1
__C.GAN.TRAIN.CLASSIFIER_GAMMA = 0.1
# 半监督分类用来衰减学习率
__C.GAN.TRAIN.GENERATOR_DECAY = 0.995
__C.GAN.TRAIN.DISCRIMINATOR_DECAY = 0.995
__C.GAN.TRAIN.CLASSIFIER_DECAY = 0.99

__C.GAN.TRAIN.CLASSIFY_NUM = 10
__C.GAN.TRAIN.LABEL_DATA_NUM = 4000 #有标签的数据数量，用于半监督时标签样本的量
__C.GAN.TRAIN.BATCH_SIZE = 20  # 'The size of have label batch per gpu'
__C.GAN.TRAIN.UNLABEL_BATCH_SIZE = 250  # 'The size of unlabel batch per gpu'
__C.GAN.TRAIN.TEST_BATCH_SIZE = 1000
# __C.GAN.TRAIN.CLASSIFY_BATCH = {'defect_zero':20, 'paint_smear':0, 'aluminium_skimmings':0,
#                                 'slag_inclusion':0, 'crack':0, 'label':20, 'unlabel':250}
__C.GAN.TRAIN.CLASSIFY_BATCH = {'airplane':2, 'automobile':2, 'bird':2, 'cat':2, 'deer':2,
                                'dog':2, 'frog':2, 'horse':2, 'ship':2, 'truck':2, 'label':20, 'unlabel':250, 'test':1000}
# __C.GAN.TRAIN.CLASSIFY_BATCH = {'defect_tree_0_0':32, 'defect_tree_0_1':32, 'label':64, 'unlabel':64}

__C.GAN.TRAIN.SNAPSHOT_ITERS = 1000
__C.GAN.TRAIN.SAVE_IMAGE_ITERS = 1000
__C.GAN.TRAIN.SNAPSHOT_EPOCH = 1
__C.GAN.TRAIN.SAVE_IMAGE_NUM = 100
__C.GAN.TRAIN.SAVE_IMAGE_EPOCH = 1

#====================关系网络参数====================
__C.RELATION_NET = edict()
__C.RELATION_NET.CLASSIFY_IMAGE_SIZE = (84, 84)

__C.RELATION_NET.TRAIN = edict()
__C.RELATION_NET.TRAIN.CLASSIFY_NUM = 8
__C.RELATION_NET.TRAIN.CLASSIFY_BATCH = 15
__C.RELATION_NET.TRAIN.CLASSIFY_SAMPLE = 5
__C.RELATION_NET.TRAIN.CLASSIFY_QUERY = 10 # sample+query=batch;batch*num=每一iter使用的图片数量

__C.RELATION_NET.TEST = edict()

#====================元对抗生产网络参数====================
__C.METAGAN = edict()
__C.METAGAN.CLASSIFY_IMAGE_SIZE = (84, 84)
__C.METAGAN.NOISE_DIM = 100  # 'Dimension of noise vector'

__C.METAGAN.TRAIN = edict()
__C.METAGAN.TRAIN.CLASSIFY_NUM = 5
__C.METAGAN.TRAIN.CLASSIFY_BATCH = 15
__C.METAGAN.TRAIN.CLASSIFY_SAMPLE = 5
__C.METAGAN.TRAIN.CLASSIFY_QUERY = 10 # sample+query=batch;batch*num=每一iter使用的图片数量

__C.METAGAN.TRAIN.BATCH_SIZE = 10  # 元对抗生产每一类生成的图片数量

__C.METAGAN.TRAIN.GENERATOR_LEARNING_RATE = 0.0002 #'learning rate for generator'
__C.METAGAN.TRAIN.DISCRIMINATOR_LEARNING_RATE = 0.0002 # 'learning rate for discriminator'
__C.METAGAN.TRAIN.GENERATOR_MOMENTUM = 0.9
__C.METAGAN.TRAIN.DISCRIMINATOR_MOMENTUM = 0.9
__C.METAGAN.TRAIN.GENERATOR_GAMMA = 0.1
__C.METAGAN.TRAIN.DISCRIMINATOR_GAMMA = 0.1
# 半监督分类用来衰减学习率
__C.METAGAN.TRAIN.GENERATOR_DECAY = 0.995
__C.METAGAN.TRAIN.DISCRIMINATOR_DECAY = 0.995



#====================孪生检测网络参数====================
__C.SIAMSE = edict()
__C.SIAMSE.DATA_DIR = 'D:\\jjj\\zlrm\\data'
__C.SIAMSE.PIXEL_MEANS = np.array([[[138.5479869970236, 138.5479869970236, 138.5479869970236]]])
__C.SIAMSE.ANCHOR_SCALE = {'crack':[4, 8, 15], 'dirty_big':[4, 6, 8], 'slag_inclusion':[1, 2, 4], 'dirty':[2, 4, 6]}
__C.SIAMSE.ANCHOR_RATIO = {'crack':[0.5, 1, 5], 'dirty_big':[2, 4, 8], 'slag_inclusion':[1, 2, 4], 'dirty':[0.5, 1, 2]}# crack第三个数字5代表crack的高度，前两个代表宽度。其他缺陷数值代表h/w
__C.SIAMSE.FEAT_STRIDE = [8,]
__C.SIAMSE.CLASSES = ['crack', 'dirty_big', 'slag_inclusion', 'dirty']
__C.SIAMSE.N_CLASSES = 4
__C.SIAMSE.SEARCH_IMAGE_SIZE = [255, 255]
__C.SIAMSE.TEMPLATE_IMAGE_SIZE = [127, 127]
# 缺陷类别里是否包含纹理
__C.SIAMSE.VEIN = False
__C.SIAMSE.BIG_DEFECT_NUM = 2 #缺陷大小大于100的缺陷数量
__C.SIAMSE.IMAGE_TRANSFORM_NUM = 5  #一张大图转为多张小图的数量。其中将大图1024*4096->255*(4*255)->将图片从左至右依次拆为4个255*255.然后将大图1024*2096->127*（4*127）->将图左右等大小拆开上下拼接254*254。因此总共5张


__C.SIAMSE.TRAIN = edict()
__C.SIAMSE.TRAIN.BATCH_SIZE = 10
__C.SIAMSE.TRAIN.SNAPSHOT_PREFIX = 'Siammask_Resnet50'
__C.SIAMSE.TRAIN.SNAPSHOT_INFIX = ''
__C.SIAMSE.TRAIN.SOLVER = 'Momentum'
__C.SIAMSE.TRAIN.LEARNING_RATE = 0.001
__C.SIAMSE.TRAIN.MOMENTUM = 0.9
__C.SIAMSE.TRAIN.GAMMA = 0.1
__C.SIAMSE.TRAIN.STEPSIZE = 20000
# IOU >= thresh: positive example
__C.SIAMSE.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
__C.SIAMSE.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# Max number of foreground examples
__C.SIAMSE.TRAIN.RPN_FG_FRACTION = 0.25
# Total number of examples
__C.SIAMSE.TRAIN.RPN_BATCHSIZE = 64
# NMS threshold used on RPN proposals
__C.SIAMSE.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.SIAMSE.TRAIN.RPN_PRE_NMS_TOP_N = 64
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.SIAMSE.TRAIN.RPN_POST_NMS_TOP_N = 3
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.SIAMSE.TRAIN.RPN_MIN_SIZE = 16
# Deprecated (outside weights)
__C.SIAMSE.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Max pixel size of the longest side of a scaled input image
__C.SIAMSE.TRAIN.MAX_SIZE = 1020
# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.SIAMSE.TRAIN.SCALES = (255,)


__C.SIAMSE.TEST = edict()
__C.SIAMSE.TEST.BATCH_SIZE = 1
# NMS threshold used on RPN proposals
__C.SIAMSE.TEST.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.SIAMSE.TEST.RPN_PRE_NMS_TOP_N = 64
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.SIAMSE.TEST.RPN_POST_NMS_TOP_N = 3
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.SIAMSE.TEST.RPN_MIN_SIZE = 16
# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.SIAMSE.TEST.SCALES = (255,)
# Max pixel size of the longest side of a scaled input image
__C.SIAMSE.TEST.MAX_SIZE = 1020


#====================缺陷检测训练参数====================
__C.ZLRM = edict()
__C.ZLRM.ANCHOR_SCALE = [1, 2, 4]
__C.ZLRM.FPN_ANCHOR_SIZE = [None, None, 8, 16, 32, 64, 128]
__C.ZLRM.ANCHOR_RATIO = [1, 3, 20, 30]
__C.ZLRM.RESNET_50_FEAT_STRIDE = [16,]
__C.ZLRM.FPN_FEAT_STRIDE = [None, 2, 4, 8, 16, 32, 64]
__C.ZLRM.PSROIPOOL = 7
__C.ZLRM.N_CLASSES = 4
__C.ZLRM.DATA_DIR = 'D:\\jjj\\zlrm\\data'
# multiscale training and testing
__C.ZLRM.IS_MULTISCALE = False
__C.ZLRM.IS_EXTRAPOLATING = True
__C.ZLRM.USE_GPU_NMS = False
__C.ZLRM.RNG_SEED = 3
__C.ZLRM.ANNOTATION_EXT = 'xml'
# 缺陷类别里是否包含纹理
__C.ZLRM.VEIN = False
# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.ZLRM.PIXEL_MEANS = np.array([[[138.5479869970236, 138.5479869970236, 138.5479869970236]]])

# A small number that's used many times
__C.EPS = 1e-148


__C.ZLRM.TRAIN = edict()
# 是否训练基础网络
__C.ZLRM.TRAIN.FEATURE_TRAIN = True
# 是否训练RPN
__C.ZLRM.TRAIN.RPN_TRAIN = True
# 是否训练ROI网络
__C.ZLRM.TRAIN.ROI_TRAIN = True
__C.ZLRM.TRAIN.LEARNING_RATE = 0.001
__C.ZLRM.TRAIN.WEIGHT_DECAY = 0.0005
__C.ZLRM.TRAIN.MOMENTUM = 0.9
__C.ZLRM.TRAIN.STEPSIZE = 20000
# 每间隔DISPLAY次显示图片名字
__C.ZLRM.TRAIN.DISPLAY = 1
__C.ZLRM.TRAIN.SOLVER = 'Momentum'
__C.ZLRM.TRAIN.GAMMA = 0.1
__C.ZLRM.TRAIN.SNAPSHOT_ITERS = 1000
__C.ZLRM.TRAIN.SNAPSHOT_EPOCH = 10
__C.ZLRM.TRAIN.USE_FLIPPED = True
__C.ZLRM.TRAIN.RANDOM_DOWNSAMPLE = False
# Use RPN to detect objects
__C.ZLRM.TRAIN.HAS_RPN = True
# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.ZLRM.TRAIN.BBOX_THRESH = 0.5
# Train bounding-box regressors
__C.ZLRM.TRAIN.BBOX_REG = True
# Deprecated (inside weights)
# used for assigning weights for each coords (x1, y1, w, h)
__C.ZLRM.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.ZLRM.TRAIN.BBOX_NORMALIZE_TARGETS = True
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.ZLRM.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
__C.ZLRM.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.ZLRM.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.ZLRM.TRAIN.SNAPSHOT_PREFIX = 'Resnet50_ZLRM'
__C.ZLRM.TRAIN.SNAPSHOT_INFIX = ''
# Minibatch size (number of regions of interest [ROIs])
__C.ZLRM.TRAIN.BATCH_SIZE = 128
# Max pixel size of the longest side of a scaled input image
__C.ZLRM.TRAIN.MAX_SIZE = 1000
# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.ZLRM.TRAIN.SCALES = (600,)
# Scales to compute real features
__C.ZLRM.TRAIN.SCALES_BASE = (0.25, 0.5, 1.0, 2.0, 3.0)
# Aspect ratio to use during training
# __C.TRAIN.ASPECTS = (1, 0.75, 0.5, 0.25)
__C.ZLRM.TRAIN.ASPECTS= (1,)
# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.ZLRM.TRAIN.FG_THRESH = 0.5
# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.ZLRM.TRAIN.FG_FRACTION = 0.25
# Images to use per minibatch
__C.ZLRM.TRAIN.IMS_PER_BATCH = 1
# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.ZLRM.TRAIN.BG_THRESH_HI = 0.5
__C.ZLRM.TRAIN.BG_THRESH_LO = 0.0
# IOU >= thresh: positive example
__C.ZLRM.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
__C.ZLRM.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# If an anchor statisfied by positive and negative conditions set to negative
__C.ZLRM.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
__C.ZLRM.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples
__C.ZLRM.TRAIN.RPN_BATCHSIZE = 256
# NMS threshold used on RPN proposals
__C.ZLRM.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.ZLRM.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.ZLRM.TRAIN.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.ZLRM.TRAIN.RPN_MIN_SIZE = 8
# Deprecated (outside weights)
__C.ZLRM.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
__C.ZLRM.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
# __C.TRAIN.RPN_POSITIVE_WEIGHT = 0.5





__C.ZLRM.TEST = edict()

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.ZLRM.TEST.NMS = 0.1
# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.ZLRM.TEST.SCALES = (600,)
# Max pixel size of the longest side of a scaled input image
__C.ZLRM.TEST.MAX_SIZE = 1024
# Propose boxes
__C.ZLRM.TEST.HAS_RPN = True
# Test using bounding-box regressors
__C.ZLRM.TEST.BBOX_REG = True
## NMS threshold used on RPN proposals
__C.ZLRM.TEST.RPN_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.ZLRM.TEST.RPN_PRE_NMS_TOP_N = 2000
#__C.TEST.RPN_PRE_NMS_TOP_N = 12000
## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.ZLRM.TEST.RPN_POST_NMS_TOP_N = 20
#__C.TEST.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.ZLRM.TEST.RPN_MIN_SIZE = 16

# ================================分类器的参数==============================================
__C.ZLRM.TRAIN.CLASSIFY_NUM = 6
__C.ZLRM.TRAIN.CLASSIFY_IMAGE_SIZE = (224, 224)
__C.ZLRM.TRAIN.CLASSIFY_BATCH = {'vein':8, 'slag_inclusion':8, 'aluminium_skimmings':16, 'crack':8,
                         'edge_crack':8, 'paint_smear':16}
# __C.ZLRM.TRAIN.CLASSIFY_BATCH = {'defect_tree_0_0':32, 'defect_tree_0_1':32, 'label':64}
__C.ZLRM.TRAIN.CLASSIFY_BATCH_SUM = 64



#
# MISC
#

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.ZLRM.DEDUP_BOXES = 1./16.
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Model directory
__C.MODELS_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'models', 'pascal_voc'))


# Place outputs under an experiments directory
__C.EXP_DIR = 'default'
__C.LOG_DIR = 'default'

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True

# Default GPU device id
__C.GPU_ID = 0

def get_log_dir(imdb):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a networks
    (if not None).
    """
    log_dir = osp.abspath(\
        osp.join(__C.ROOT_DIR, 'logs', __C.LOG_DIR, imdb.name, strftime("%Y-%m-%d-%H-%M-%S", localtime())))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def get_output_dir(imdb, weights_filename):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a networks
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if weights_filename is not None:
        outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value

