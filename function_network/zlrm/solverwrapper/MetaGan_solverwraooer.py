import numpy as np
import os
import tensorflow as tf
import cv2
import math
import scipy
from PIL import Image
from time import strftime
from lib.utils.timer import Timer

# >>>> obsolete, because it depends on sth outside of this project
from lib.networks.netconfig import cfg
# <<<< obsolete

# _DEBUG = False

class SolverWrapper(object):

    def __init__(self, sess, network, imdb, model_output_dir, generate_image_output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.model_output_dir = model_output_dir
        self.generate_image_output_dir = generate_image_output_dir
        self.pretrained_model = pretrained_model

        self.saver = tf.train.Saver(max_to_keep=100)
        self.restor_saver = tf.train.Saver()

    def snapshot(self, sess, iter):
        """Take a snapshot of the networks after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """

        if not os.path.exists(self.model_output_dir):
            os.makedirs(self.model_output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.ZLRM.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.ZLRM.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.model_output_dir, filename)

        self.saver.save(sess, filename)
        print ('Wrote snapshot to: {:s}'.format(filename))

    def visualize_results(self, sess, fake_images, epoch):
        # tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(cfg.GAN.TRAIN.SAVE_IMAGE_NUM)))
        noise = np.random.uniform(-1, 1, size=(cfg.GAN.TRAIN.SAVE_IMAGE_NUM, cfg.GAN.NOISE_DIM))

        """ random noise, random discrete code, fixed continuous code """
        label = np.random.choice(cfg.GAN.TRAIN.CLASSIFY_NUM, cfg.GAN.TRAIN.SAVE_IMAGE_NUM)


        feed_dict = {
            self.net.visual_noise: noise,
            self.net.visual_label: label
        }

        samples = sess.run(fake_images, feed_dict=feed_dict)

        save_images_root_dir =  os.path.join(self.generate_image_output_dir, 'all_classes')
        if not os.path.exists(save_images_root_dir):
            os.makedirs(save_images_root_dir)
        image_name = '_epoch%03d' % epoch + '_test_all_classes.png'
        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    os.path.join(save_images_root_dir, image_name))
        """ specified condition, random noise """

        np.random.seed()

        for l in range(cfg.GAN.TRAIN.CLASSIFY_NUM):
            label = np.zeros(cfg.GAN.TRAIN.SAVE_IMAGE_NUM, dtype=np.int64) + l

            feed_dict = {
                self.net.visual_noise: noise,
                self.net.visual_label: label
            }

            samples = sess.run(fake_images, feed_dict=feed_dict)

            save_images_root_dir = os.path.join(self.generate_image_output_dir, 'class_%d' % l)
            if not os.path.exists(save_images_root_dir):
                os.makedirs(save_images_root_dir)
            image_name = '_epoch%03d' % epoch + '_test_class_%d.png' % l

            save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        os.path.join(save_images_root_dir, image_name))

    def train_model(self, sess, max_iters, restore=False):
        """Network training loop."""

        data_layer = self.imdb

        discriminator_loss, generator_loss = self.net.build_loss()
        fake_images = self.net.test_net()

        # optimizer
        discriminator_tvars = tf.trainable_variables(scope='discriminator')
        generator_tvars = tf.trainable_variables(scope='generator')
        global_step = tf.Variable(0, trainable=False)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # train_op = tf.train.AdamOptimizer(cfg.METAGAN.TRAIN.LEARNING_RATE).minimize(d_loss, global_step=global_step)
            discriminator_opt = tf.train.AdamOptimizer(cfg.METAGAN.TRAIN.DISCRIMINATOR_LEARNING_RATE, beta1=0.5).minimize(discriminator_loss,
                                                                                                   global_step=global_step,
                                                                                                   var_list=discriminator_tvars)
            generator_opt = tf.train.AdamOptimizer(cfg.METAGAN.TRAIN.GENERATOR_LEARNING_RATE, beta1=0.5).minimize(generator_loss,
                                                                                               global_step=global_step,
                                                                                               var_list=generator_tvars)

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
                print(self.model_output_dir)
                ckpt = tf.train.get_checkpoint_state(self.model_output_dir)
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
        last_save_generate_image_iter = -1
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
                self.net.test_sample_data: blobs_train['val']['sample']['data']
            }

            feed_val_dict = {
                self.net.test_sample_data: blobs_train['val']['sample']['data']
            }

            fetch_discriminator_train_list = [discriminator_loss,
                          discriminator_opt]
            fetch_discriminator_train_list += []
            fetch_generator_train_list = [generator_loss,
                                              generator_opt]
            fetch_generator_train_list += []

            discriminator_loss_train_val, _ = sess.run(fetches=fetch_discriminator_train_list,
                                                       feed_dict=feed_train_dict)
            generator_loss_train_val, _ = sess.run(fetches=fetch_generator_train_list,
                                                       feed_dict=feed_train_dict)

            _diff_time = timer.toc(average=False)

            # #将iter， 损失和验证的精确度写到accuracy文本文件里
            # save_data_txt = '{} {} {}'.format(iter, loss_train_val, accuracy_val)
            # with open('D:\\jjj\\zlrm\\logs\\zlrm_relation_net_accuracy.txt', 'a') as f:
            #     f.write(save_data_txt + '\n')

            if (iter) % (cfg.ZLRM.TRAIN.DISPLAY) == 0:
                print(
                    'iter: %d / %d, discriminator_loss: %.4f, generator_loss: %.4f, discriminator_lr: %4f, generator_lr: %f' % \
                    (iter, max_iters, discriminator_loss_train_val, generator_loss_train_val,
                     cfg.METAGAN.TRAIN.DISCRIMINATOR_LEARNING_RATE, cfg.METAGAN.TRAIN.GENERATOR_LEARNING_RATE))
                print ('speed: {:.3f}s / iter'.format(_diff_time))

            if (iter+1) % cfg.ZLRM.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

            # show temporal results
            if (iter + 1) % cfg.GAN.TRAIN.SAVE_IMAGE_EPOCH == 0:
                last_save_generate_image_iter = iter
                self.visualize_results(sess, fake_images, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)
        if last_save_generate_image_iter != iter:
            self.visualize_results(sess, fake_images, iter)


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def inverse_transform(images):
    return (images+1.)/2.
    # return ((images + 1.) * 127.5).astype('uint8')

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def train_net(network, imdb, model_output_dir, generate_image_output_dir, pretrained_model=None, max_iters=40000, restore=False):
    """Train a Fast R-CNN networks."""

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, network, imdb, model_output_dir, generate_image_output_dir, pretrained_model=pretrained_model)
        print ('Solving...')
        sw.train_model(sess, max_iters, restore=restore)
        print ('done solving')
