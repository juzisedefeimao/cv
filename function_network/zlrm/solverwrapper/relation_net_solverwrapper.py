import numpy as np
import os
import tensorflow as tf
import cv2
from PIL import Image
from time import strftime
from lib.utils.timer import Timer

# >>>> obsolete, because it depends on sth outside of this project
from lib.networks.netconfig import cfg
# <<<< obsolete

# _DEBUG = False

class SolverWrapper(object):

    def __init__(self, sess, network, imdb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
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

        data_layer = self.imdb

        loss = self.net.build_loss()
        accuracy = self.net.test_net()

        # optimizer
        global_step = tf.Variable(0, trainable=False)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = tf.train.AdamOptimizer(cfg.ZLRM.TRAIN.LEARNING_RATE).minimize(loss, global_step=global_step)

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

            # get one batch
            blobs_train = data_layer.forward()

            # if (iter + 1) % (cfg.ZLRM.TRAIN.DISPLAY) == 0:
            #     print ('image: %s' %(blobs['im_name']),)

            feed_train_dict={
                self.net.sample_data: blobs_train['train']['sample']['data'],
                self.net.query_data: blobs_train['train']['query']['data'],
                self.net.sample_label: blobs_train['train']['sample']['label'],
                self.net.query_label: blobs_train['train']['query']['label'],
                self.net.test_sample_data: blobs_train['val']['sample']['data'],
                self.net.test_query_data: blobs_train['val']['query']['data'],
                self.net.test_sample_label: blobs_train['val']['sample']['label'],
                self.net.test_query_label: blobs_train['val']['query']['label']
            }

            feed_val_dict = {
                self.net.test_sample_data: blobs_train['val']['sample']['data'],
                self.net.test_query_data: blobs_train['val']['query']['data'],
                self.net.test_sample_label: blobs_train['val']['sample']['label'],
                self.net.test_query_label: blobs_train['val']['query']['label']
            }

            fetch_train_list = [loss,
                          train_op]
            fetch_train_list += []


            loss_train_val, _ = sess.run(fetches=fetch_train_list, feed_dict=feed_train_dict)
            accuracy_val = sess.run(fetches=accuracy, feed_dict=feed_val_dict)

            _diff_time = timer.toc(average=False)

            #将iter， 损失和验证的精确度写到accuracy文本文件里
            save_data_txt = '{} {} {}'.format(iter, loss_train_val, accuracy_val)
            with open('D:\\jjj\\zlrm\\logs\\zlrm_relation_net_accuracy.txt', 'a') as f:
                f.write(save_data_txt + '\n')

            if (iter) % (cfg.ZLRM.TRAIN.DISPLAY) == 0:
                print(
                    'iter: %d / %d, loss: %.4f, test_loss: %.4f, lr: %f' % \
                    (iter, max_iters, loss_train_val, accuracy_val, cfg.ZLRM.TRAIN.LEARNING_RATE))
                print ('speed: {:.3f}s / iter'.format(_diff_time))

            if (iter+1) % cfg.ZLRM.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)


def train_net(network, imdb, output_dir, pretrained_model=None, max_iters=40000, restore=False):
    """Train a Fast R-CNN networks."""

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, network, imdb, output_dir, pretrained_model=pretrained_model)
        print ('Solving...')
        sw.train_model(sess, max_iters, restore=restore)
        print ('done solving')
