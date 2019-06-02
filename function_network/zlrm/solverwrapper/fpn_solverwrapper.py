import numpy as np
import os
import tensorflow as tf
import cv2
from PIL import Image, ImageDraw, ImageFont
from time import strftime
from lib.roi_data_layer.layer import RoIDataLayer
from lib.utils.timer import Timer
from lib.gt_data_layer import roidb as gdl_roidb
from lib.roi_data_layer import roidb as rdl_roidb
from lib.nms.nms_wrapper import nms

# >>>> obsolete, because it depends on sth outside of this project
from lib.networks.netconfig import cfg
from lib.rpn_msr.bbox_transform import clip_boxes, bbox_transform_inv
# <<<< obsolete

# _DEBUG = False

class SolverWrapper(object):

    def __init__(self, sess, network, imdb, roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print ('Computing bounding-box regression targets...')
        if cfg.ZLRM.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print ('done')

        self.saver = tf.train.Saver(max_to_keep=100)
        self.rpn_restor_saver = tf.train.Saver()

    def snapshot(self, sess, iter):
        """Take a snapshot of the networks after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if cfg.ZLRM.TRAIN.BBOX_REG and ('bbox_pred' in net.layers) and cfg.ZLRM.TRAIN.BBOX_NORMALIZE_TARGETS:
            # save original values
            with tf.variable_scope('Fast-RCNN', reuse=True):
                with tf.variable_scope('bbox_pred'):
                    weights = tf.get_variable("weights")
                    biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(weights.assign(orig_0 * np.tile(self.bbox_stds, (weights_shape[0],1))))
            sess.run(biases.assign(orig_1 * self.bbox_stds + self.bbox_means))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.ZLRM.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.ZLRM.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        # self.saver = tf.train.Saver(max_to_keep=100)
        self.saver.save(sess, filename)
        # tf.train.Saver.save(sess=sess, save_path=filename)
        print ('Wrote snapshot to: {:s}'.format(filename))

        if cfg.ZLRM.TRAIN.BBOX_REG and ('bbox_pred' in net.layers):
            # restore net to original state
            sess.run(weights.assign(orig_0))
            sess.run(biases.assign(orig_1))


    def train_model(self, sess, max_iters, restore=False):
        """Network training loop."""

        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)

        loss, cross_entropy, loss_box, rpn_cross_entropy, rpn_loss_box, label = self.net.build_loss()

        # optimizer
        if cfg.ZLRM.TRAIN.SOLVER == 'Adam':
            opt = tf.train.AdamOptimizer(cfg.ZLRM.TRAIN.LEARNING_RATE)
        elif cfg.ZLRM.TRAIN.SOLVER == 'RMS':
            opt = tf.train.RMSPropOptimizer(cfg.ZLRM.TRAIN.LEARNING_RATE)
        else:
            lr = tf.Variable(cfg.ZLRM.TRAIN.LEARNING_RATE, trainable=False)
            # lr = tf.Variable(0.0, trainable=False)
            momentum = cfg.ZLRM.TRAIN.MOMENTUM
            opt = tf.train.MomentumOptimizer(lr, momentum)

        global_step = tf.Variable(0, trainable=False)
        with_clip = True
        if with_clip:
            tvars = tf.trainable_variables()
            grads, norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), 10.0)
            train_op = opt.apply_gradients(zip(grads, tvars), global_step=global_step)
        else:
            train_op = opt.minimize(loss, global_step=global_step)

        # intialize variables
        sess.run(tf.global_variables_initializer())
        restore_iter = 0

        # load Resnet50
        if self.pretrained_model is not None and not restore:
            try:
                print ('Loading pretrained model '
                   'weights from {:s}'.format(self.pretrained_model))
                self.net.load(self.pretrained_model, sess, True)
            except:
                raise BaseException('Check your pretrained model {:s}'.format(self.pretrained_model))

        # resuming a trainer
        if restore:
            try:
                print(self.output_dir)
                ckpt = tf.train.get_checkpoint_state(self.output_dir)
                print ('Restoring from {}...'.format(ckpt.model_checkpoint_path),)
                tvars = tf.trainable_variables()
                # tvars = [v for v in tvars if (v.name.split('-')[0] == 'rpn')]
                print(tvars)
                # rpn_restor_saver = tf.train.Saver()
                self.rpn_restor_saver.restore(sess, ckpt.model_checkpoint_path)
                stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
                restore_iter = int(stem.split('_')[-1])
                sess.run(global_step.assign(restore_iter))
                print ('done')
            except:
                raise BaseException('Check your pretrained {:s}'.format(ckpt.model_checkpoint_path))

        last_snapshot_iter = -1
        timer = Timer()

        for iter in range(restore_iter, max_iters):
            timer.tic()

            # learning rate
            if iter != 0 and iter % cfg.ZLRM.TRAIN.STEPSIZE == 0:
                sess.run(tf.assign(lr, lr.eval() * cfg.ZLRM.TRAIN.GAMMA))

            # get one batch
            blobs = data_layer.forward()

            if (iter + 1) % (cfg.ZLRM.TRAIN.DISPLAY) == 0:
                print ('image: %s' %(blobs['im_name']),)

            feed_dict={
                self.net.data: blobs['data'],
                self.net.im_info: blobs['im_info'],
                self.net.gt_boxes: blobs['gt_boxes']
            }

            res_fetches = []  # RPN rgs output

            fetch_list = [self.net.get_output('cls_prob'), self.net.get_output('bbox_pred'),
                      self.net.get_output('rois'),
                          rpn_cross_entropy,
                          rpn_loss_box,
                          cross_entropy,
                          label,
                          loss_box,
                          train_op] + res_fetches

            fetch_list += []
            cls_prob, bbox_pred, rois, rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, label_val, loss_box_value, \
            _,  = sess.run(fetches=fetch_list, feed_dict=feed_dict)


            _diff_time = timer.toc(average=False)

            # print('jjj', label_val)
            # print('kkk',np.argmax(cls_prob, axis=1))
            # # print('lll', cls_prob[:, np.argmax(cls_prob, axis=1)])
            # print(len(label_val[label_val==0]))
            # vi = vis(cls_prob, bbox_pred, rois, blobs['im_info'][0][2])
            # vi.detect(os.path.join('D:\\jjj\\zlrm\\data\\detector_data\\datasets\\ImageSets', blobs['im_name']))

            if (iter) % (cfg.ZLRM.TRAIN.DISPLAY) == 0:
                print(
                    'iter: %d / %d, total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, lr: %f' % \
                    (iter, max_iters, rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value, \
                     rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, lr.eval()))
                print ('speed: {:.3f}s / iter'.format(_diff_time))

            if (iter+1) % cfg.ZLRM.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

class vis():
    def __init__(self, cls_prob, bbox_pred, rois, im_scale):
        self.classes_detect = ('__background__',  # always index 0
                               'crack', 'dirty_big', 'slag_inclusion', 'dirty')
        self._ind_to_class = dict(zip(range(len(self.classes_detect)), self.classes_detect))
        self.output_dir = 'D:\\jjj\\zlrm\\data\\result'

        self.cls_prob = cls_prob
        self.bbox_pred = bbox_pred
        self.rois = rois
        self.im_scale = im_scale

    def detect(self, image_name):
        """Detect object classes in an image using pre-computed object proposals."""

        # Load the demo image
        # Detect all object classes and regress object bounds
        image = readimage(image_name)
        image = image_transform_1_3(image)
        timer = Timer()
        timer.tic()
        scores, boxes = self.im_detect(image)
        timer.toc()
        # print('kkk', np.argmax(scores, axis=1))
        # print('rois--------------', scores)
        print ('Detection took {:.3f}s for '
               '{:d} object proposals'.format(timer.total_time, boxes.shape[0]))

        CONF_THRESH = 0.7
        NMS_THRESH = 0.1
        dets_list=[]
        for cls_ind, cls in enumerate(self.classes_detect[1:]):
            inds = np.where(scores[:, cls_ind] > CONF_THRESH)[0]
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets[inds,:], NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            cls_ind_list = np.empty((len(inds), 1), np.int32)
            cls_ind_list.fill(cls_ind)
            dets = np.hstack((dets[inds, :-1], cls_ind_list))
            dets_list.append(dets)
        dets = np.vstack(dets_list)
        dets[:, 0:2] = np.floor(dets[:, 0:2])
        dets[:, 2:] = np.ceil(dets[:, 2:])
        dets = dets.astype(np.int32)
        print('jjj',dets)
        self.vis(image, image_name, dets)
        return dets


    def im_detect(self, im):

        cls_prob = self.cls_prob
        bbox_pred = self.bbox_pred
        rois = self.rois

        if isinstance(rois, tuple):
            rois = rois[0]

        cls_prob = np.reshape(cls_prob, [-1, cfg.ZLRM.N_CLASSES + 1])  # (R, C+1)
        bbox_pred = np.reshape(bbox_pred, [-1, (cfg.ZLRM.N_CLASSES + 1) * 4])  # (R, (C+1)x4)
        rois = np.array(rois)
        boxes = rois[:, 1:5] / self.im_scale

        scores = cls_prob

        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)

        return scores, pred_boxes


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

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.ZLRM.TRAIN.USE_FLIPPED:
        print ('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print ('done')

    print ('Preparing training data...')
    if cfg.ZLRM.TRAIN.HAS_RPN:
        if cfg.ZLRM.IS_MULTISCALE:
            # TODO: fix multiscale training (single scale is already a good trade-off)
            print ('#### warning: multi-scale has not been tested.')
            print ('#### warning: using single scale by setting IS_MULTISCALE: False.')
            gdl_roidb.prepare_roidb(imdb)
        else:
            rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print ('done')

    return imdb.roidb


def get_data_layer(roidb, num_classes):
    """return a data layer."""
    if cfg.ZLRM.TRAIN.HAS_RPN:
        if cfg.ZLRM.IS_MULTISCALE:
            # obsolete
            # layer = GtDataLayer(roidb)
            raise BaseException("Calling caffe modules...")
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer

def _process_boxes_scores(cls_prob, bbox_pred, rois, im_scale, im_shape):
    """
    process the output tensors, to get the boxes and scores
    """
    assert rois.shape[0] == bbox_pred.shape[0],\
        'rois and bbox_pred must have the same shape'
    boxes = rois[:, 1:5]
    scores = cls_prob
    if cfg.ZLRM.TEST.BBOX_REG:
        pred_boxes = bbox_transform_inv(boxes, deltas=bbox_pred)
        pred_boxes = clip_boxes(pred_boxes, im_shape)
    else:
        # Simply repeat the boxes, once for each class
        # boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes = clip_boxes(boxes, im_shape)
    return pred_boxes, scores

def _draw_boxes_to_image(im, res):
    colors = [(86, 0, 240), (173, 225, 61), (54, 137, 255),\
              (151, 0, 255), (243, 223, 48), (0, 117, 255),\
              (58, 184, 14), (86, 67, 140), (121, 82, 6),\
              (174, 29, 128), (115, 154, 81), (86, 255, 234)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = np.copy(im)
    cnt = 0
    for ind, r in enumerate(res):
        if r['dets'] is None: continue
        dets = r['dets']
        for i in range(0, dets.shape[0]):
            (x1, y1, x2, y2, score) = dets[i, :]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colors[ind % len(colors)], 2)
            text = '{:s} {:.2f}'.format(r['class'], score)
            cv2.putText(image, text, (x1, y1), font, 0.6, colors[ind % len(colors)], 1)
            cnt = (cnt + 1)
    return image

def _draw_gt_to_image(im, gt_boxes, gt_ishard):
    image = np.copy(im)

    for i in range(0, gt_boxes.shape[0]):
        (x1, y1, x2, y2, score) = gt_boxes[i, :]
        if gt_ishard[i] == 0:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
        else:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
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
    elif len(image.shape)==3:
        print('图像为3通道图像,不需要转换')

    return image

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
        saveimage_root = 'D:\\jjj\\zlrm\\data\\default_root'
        print('未设置保存图片的路径，默认保存到{}'.format(saveimage_root))
    root = os.path.join(saveimage_root, str(saveimage_name))
    image = Image.fromarray(image)
    image.save(root)

# 保存特征图
def savefeature(feature, roi):
    feature = np.array(feature)
    print('featureshape==', feature.shape)
    feature = feature[0]
    feature = feature.transpose(2,0,1)
    print('jjjshape', feature.shape)

    for a in range(roi.shape[0]):
        c = []
        for i in range(feature.shape[0]):
            n = int(i / 2)
            roi_i = int(roi[a][0])
            roi_j = int(roi[a][1])
            c.append(feature[i][roi_i][roi_j])

        print(c)


    for i in range(feature.shape[0]):
        for j in range(feature.shape[1]):
            for k in range(feature.shape[2]):
                if feature[i][j][k] > 0:
                    feature[i][j][k] = 255
                    # feature[i][j][k] = feature[i][j][k] * 100000

    feature = np.array(feature, dtype=np.uint8)

    for l in range(feature.shape[0]):

        saveimage(feature[l], saveimage_name='k'+str(l))

def train_net(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000, restore=False):
    """Train a Fast R-CNN networks."""

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)
        print ('Solving...')
        sw.train_model(sess, max_iters, restore=restore)
        print ('done solving')
