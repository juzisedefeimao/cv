import numpy as np
import os
import tensorflow as tf
import cv2
from PIL import Image
from time import strftime
from lib.roi_data_layer.layer import RoIDataLayer
from lib.utils.timer import Timer

# >>>> obsolete, because it depends on sth outside of this project
from lib.networks.netconfig import cfg
from lib.rpn_msr.bbox_transform import clip_boxes, bbox_transform_inv
# <<<< obsolete

# _DEBUG = False

class SolverWrapper(object):

    def __init__(self, sess, network, imdb_train, imdb_test, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb_train = imdb_train
        self.imdb_test = imdb_test
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        self.saver = tf.train.Saver(max_to_keep=100)
        self.restor_saver = tf.train.Saver()

    def snapshot(self, sess, iter):
        """Take a snapshot of the networks after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.ZLRM.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.ZLRM.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print ('Wrote snapshot to: {:s}'.format(filename))


    def train_model(self, sess, max_iters, restore=False):
        """Network training loop."""

        train_data_layer = self.imdb_train
        test_data_layer = self.imdb_test

        loss = self.net.build_loss()

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
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            tvars = tf.trainable_variables()
            grads, norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), 10.0)
            # with tf.control_dependencies(update_ops):
            train_op = opt.apply_gradients(zip(grads, tvars), global_step=global_step)
        else:
            train_op = opt.minimize(loss, global_step=global_step)

        # intialize variables
        sess.run(tf.global_variables_initializer())
        restore_iter = 0

        # load Resnet18
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
                print(tvars)
                self.restor_saver.restore(sess, ckpt.model_checkpoint_path)
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
            blobs_train = train_data_layer.forward()
            blobs_test = test_data_layer.forward()

            # if (iter + 1) % (cfg.ZLRM.TRAIN.DISPLAY) == 0:
            #     print ('image: %s' %(blobs['im_name']),)

            feed_train_dict={
                self.net.data: blobs_train['data'],
                self.net.label: blobs_train['label']
            }

            feed_test_dict = {
                self.net.data: blobs_test['data'],
                self.net.label: blobs_test['label']
            }

            fetch_train_list = [loss,
                          train_op]

            fetch_test_list = [loss]

            fetch_train_list += []

            fetch_test_list += []

            loss_train_val, _ = sess.run(fetches=fetch_train_list, feed_dict=feed_train_dict)
            loss_test_val, = sess.run(fetches=fetch_test_list, feed_dict=feed_test_dict)


            _diff_time = timer.toc(average=False)


            if (iter) % (cfg.ZLRM.TRAIN.DISPLAY) == 0:
                print(
                    'iter: %d / %d, loss: %.4f, test_loss: %.4f, lr: %f' % \
                    (iter, max_iters, loss_train_val, loss_test_val, lr.eval()))
                print ('speed: {:.3f}s / iter'.format(_diff_time))

            if (iter+1) % cfg.ZLRM.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

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

def train_net(network, imdb_train, imdb_test, output_dir, pretrained_model=None, max_iters=40000, restore=False):
    """Train a Fast R-CNN networks."""

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, network, imdb_train, imdb_test, output_dir, pretrained_model=pretrained_model)
        print ('Solving...')
        sw.train_model(sess, max_iters, restore=restore)
        print ('done solving')
