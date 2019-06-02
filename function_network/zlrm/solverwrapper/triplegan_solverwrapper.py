import numpy as np
import os
import tensorflow as tf
import cv2
from PIL import Image
from time import strftime
from lib.utils.timer import Timer
import scipy.misc

# >>>> obsolete, because it depends on sth outside of this project
from lib.networks.netconfig import cfg
# <<<< obsolete

# _DEBUG = False

class SolverWrapper(object):

    def __init__(self, sess, network, imdb_train, output_ckpt_dir, output_generate_image_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb_train = imdb_train
        self.output_ckpt_dir = output_ckpt_dir
        self.output_generate_image_dir = output_generate_image_dir
        self.pretrained_model = pretrained_model

        self.saver = tf.train.Saver(max_to_keep=100)
        self.restor_saver = tf.train.Saver()

    def snapshot(self, sess, iter):
        """Take a snapshot of the networks after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """

        if not os.path.exists(self.output_ckpt_dir):
            os.makedirs(self.output_ckpt_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.ZLRM.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.ZLRM.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_ckpt_dir, filename)

        self.saver.save(sess, filename)
        print ('Wrote snapshot to: {:s}'.format(filename))

    def save_generate_image(self, batch_generate_image, iter):

        for i in range(len(batch_generate_image)):
            image = (batch_generate_image[i] + 1.) / 2.
            if not os.path.exists(self.output_generate_image_dir):
                os.makedirs(self.output_generate_image_dir)
            image_name = ('generate_image_iter_{:d}_id_{:d}'.format(iter + 1, i) + '.bmp')
            image_root = os.path.join(self.output_generate_image_dir, image_name)
            scipy.misc.imsave(image_root, image)

        print('Wrote generate image to: {:s}'.format(self.output_generate_image_dir + str(iter)))

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

        save_images_root_dir =  os.path.join(self.output_generate_image_dir, 'all_classes')
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

            save_images_root_dir = os.path.join(self.output_generate_image_dir, 'class_%d' % l)
            if not os.path.exists(save_images_root_dir):
                os.makedirs(save_images_root_dir)
            image_name = '_epoch%03d' % epoch + '_test_class_%d.png' % l

            save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        os.path.join(save_images_root_dir, image_name))

    def train_model(self, sess, max_epoch, restore=False):
        """Network training loop."""

        data_layer = self.imdb_train

        discriminator_loss, generator_loss, classifier_loss = self.net.build_loss()
        fake_images = self.net.generate_image()

        # optimizer
        if cfg.GAN.TRAIN.SOLVER == 'Adam':
            generator_lr = tf.Variable(cfg.GAN.TRAIN.GENERATOR_LEARNING_RATE,  trainable=False)
            discriminator_lr = tf.Variable(cfg.GAN.TRAIN.DISCRIMINATOR_LEARNING_RATE, trainable=False)
            classifier_lr = tf.Variable(cfg.GAN.TRAIN.CLASSIFIER_LEARNING_RATE, trainable=False)

            generator_opt = tf.train.AdamOptimizer(generator_lr, beta1=0.5)
            discriminator_opt = tf.train.AdamOptimizer(discriminator_lr, beta1=0.5)
            classifier_opt = tf.train.AdamOptimizer(classifier_lr)
        elif cfg.GAN.TRAIN.SOLVER == 'RMS':
            generator_lr = tf.Variable(cfg.GAN.TRAIN.GENERATOR_LEARNING_RATE, trainable=False)
            generator_opt = tf.train.RMSPropOptimizer(generator_lr)
            discriminator_lr = tf.Variable(cfg.GAN.TRAIN.DISCRIMINATOR_LEARNING_RATE, trainable=False)
            discriminator_opt = tf.train.RMSPropOptimizer(discriminator_lr)
            classifier_lr = tf.Variable(cfg.GAN.TRAIN.CLASSIFIER_LEARNING_RATE, trainable=False)
            classifier_opt = tf.train.RMSPropOptimizer(classifier_lr)
        elif cfg.GAN.TRAIN.SOLVER == 'Momentum':
            generator_lr = tf.Variable(cfg.GAN.TRAIN.GENERATOR_LEARNING_RATE, trainable=False)
            generator_momentum = cfg.GAN.TRAIN.GENERATOR_MOMENTUM
            generator_opt = tf.train.MomentumOptimizer(generator_lr, generator_momentum)
            discriminator_lr = tf.Variable(cfg.GAN.TRAIN.DISCRIMINATOR_LEARNING_RATE, trainable=False)
            discriminator_momentum = cfg.GAN.TRAIN.DISCRIMINATOR_MOMENTUM
            discriminator_opt = tf.train.MomentumOptimizer(discriminator_lr, discriminator_momentum)
            classifier_lr = tf.Variable(cfg.GAN.TRAIN.CLASSIFIER_LEARNING_RATE, trainable=False)
            classifier_momentum = cfg.GAN.TRAIN.CLASSIFIER_MOMENTUM
            classifier_opt = tf.train.MomentumOptimizer(classifier_lr, classifier_momentum)
        else:
            raise ModuleNotFoundError('不存在的优化器，可使用的优化器为Adam、RMS、Momentum')

        global_step = tf.Variable(0, trainable=False)
        with_clip = False
        if with_clip:
            discriminator_tvars = tf.trainable_variables(scope='discriminator')
            generator_tvars = tf.trainable_variables(scope='generator')
            classifier_tvars = tf.trainable_variables(scope='classifier')

            discriminator_grads, discriminator_norm = tf.clip_by_global_norm(
                tf.gradients(discriminator_loss, discriminator_tvars), 10.0)
            discriminator_train_op = discriminator_opt.apply_gradients(zip(discriminator_grads, discriminator_tvars),
                                                         global_step=global_step)
            generator_grads, generator_norm = tf.clip_by_global_norm(
                tf.gradients(generator_loss, generator_tvars), 10.0)
            generator_train_op = generator_opt.apply_gradients(zip(generator_grads, generator_tvars),
                                                         global_step=global_step)
            classifier_grads, classifier_norm = tf.clip_by_global_norm(
                tf.gradients(classifier_loss, classifier_tvars), 10.0)
            classifier_train_op = classifier_opt.apply_gradients(zip(classifier_grads, classifier_tvars),
                                                               global_step=global_step)
        else:
            discriminator_tvars = tf.trainable_variables(scope='discriminator')
            generator_tvars = tf.trainable_variables(scope='generator')
            classifier_tvars = tf.trainable_variables(scope='classifier')
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                discriminator_train_op = discriminator_opt.minimize(discriminator_loss, global_step=global_step,
                                                                    var_list=discriminator_tvars)
                generator_train_op = generator_opt.minimize(generator_loss, global_step=global_step,
                                                            var_list=generator_tvars)
                classifier_train_op = classifier_opt.minimize(classifier_loss, global_step=global_step,
                                                            var_list=classifier_tvars)

        # intialize variables
        sess.run(tf.global_variables_initializer())
        restore_epoch = 0

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
                print(self.output_ckpt_dir)
                ckpt = tf.train.get_checkpoint_state(self.output_ckpt_dir)
                print ('Restoring from {}...'.format(ckpt.model_checkpoint_path),)
                tvars = tf.trainable_variables()
                print(tvars)
                self.restor_saver.restore(sess, ckpt.model_checkpoint_path)
                stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
                restore_epoch = int(stem.split('_')[-1])
                sess.run(global_step.assign(restore_epoch))
                print ('done')
            except:
                raise BaseException('Check your pretrained {:s}'.format(ckpt.model_checkpoint_path))

        last_snapshot_epoch = -1
        last_save_generate_image_epoch = -1
        timer = Timer()
        for epoch in range(restore_epoch, max_epoch):

            if epoch >= 50 :
                sess.run(tf.assign(generator_lr, generator_lr.eval() * cfg.GAN.TRAIN.GENERATOR_DECAY))
                sess.run(tf.assign(discriminator_lr, discriminator_lr.eval() * cfg.GAN.TRAIN.DISCRIMINATOR_DECAY))
                sess.run(tf.assign(classifier_lr, classifier_lr.eval() * cfg.GAN.TRAIN.CLASSIFIER_DECAY))

            if epoch >= 200 :
                alpha_p = 0.1
            else :
                alpha_p = 0.0

            # for iter in range(2):
            for iter in range(data_layer.epoch_iter_num):
                timer.tic()
                # learning rate
                if cfg.GAN.TRAIN.SOLVER == 'Momentum':
                    if iter != 0 and iter % cfg.ZLRM.TRAIN.STEPSIZE == 0:
                        sess.run(tf.assign(generator_lr, generator_lr.eval() * cfg.GAN.TRAIN.GENERATOR_GAMMA))
                        sess.run(tf.assign(discriminator_lr, discriminator_lr.eval() * cfg.GAN.TRAIN.DISCRIMINATOR_GAMMA))
                        sess.run(tf.assign(classifier_lr, classifier_lr.eval() * cfg.GAN.TRAIN.CLASSIFIER_GAMMA))
                # get one batch
                blobs = data_layer.epoch_forward()

                feed_dict={
                    self.net.real_data: blobs['data'],
                    self.net.unlabel_data: blobs['unlabel_data'],
                    self.net.noise_data: blobs['noise'],
                    self.net.label: blobs['label'],
                    self.net.unlabel: blobs['unlabel'],
                    self.net.alpha_p: alpha_p
                }


                fetch_distriminator_list = [discriminator_loss,
                                            discriminator_train_op]
                fetch_generator_list = [generator_loss,
                                        generator_train_op]
                fetch_classifier_list = [classifier_loss,
                                            classifier_train_op]

                fetch_distriminator_list += []
                fetch_generator_list += []
                fetch_classifier_list += []

                discriminator_loss_val, _ = sess.run(fetches=fetch_distriminator_list, feed_dict=feed_dict)
                generator_loss_val, _ = sess.run(fetches=fetch_generator_list, feed_dict=feed_dict)
                classifier_loss_val, _ = sess.run(fetches=fetch_classifier_list, feed_dict=feed_dict)


                _diff_time = timer.toc(average=False)


                if (iter) % (cfg.ZLRM.TRAIN.DISPLAY) == 0:
                    if cfg.GAN.TRAIN.SOLVER == 'Momentum':
                        print(
                            'epoch: %d / %d, iter: %d / %d, '
                            'discriminator loss: %.4f, generator loss: %.4f, classifier loss: %.4f,'
                            ' generator lr: %f, discriminator lr: %f, classifier lr: %f' % \
                            (epoch, max_epoch, iter, data_layer.epoch_iter_num,
                             discriminator_loss_val, generator_loss_val, classifier_loss_val,
                             generator_lr.eval(), discriminator_lr.eval(), classifier_lr.eval()))
                    else:
                        print(
                            'epoch: %d / %d,iter: %d / %d, '
                            'discriminator loss: %.4f, generator loss: %.4f, classifier loss: %.4f'
                            ' generator lr: %f, discriminator lr: %f, classifier lr: %f'% \
                            (epoch, max_epoch, iter, data_layer.epoch_iter_num,
                             discriminator_loss_val, generator_loss_val, classifier_loss_val,
                             generator_lr.eval(), discriminator_lr.eval(), classifier_lr.eval()))
                    print ('speed: {:.3f}s / iter'.format(_diff_time))

            if (epoch+1) % cfg.GAN.TRAIN.SNAPSHOT_EPOCH == 0:
                last_snapshot_epoch = epoch
                self.snapshot(sess, epoch)

            if (epoch + 1) % cfg.GAN.TRAIN.SAVE_IMAGE_EPOCH == 0:
                last_save_generate_image_epoch = epoch
                self.visualize_results(sess, fake_images, epoch)
        if last_snapshot_epoch != epoch:
            self.snapshot(sess, epoch)
        if last_save_generate_image_epoch != epoch:
            self.visualize_results(sess, fake_images, epoch)


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

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def train_net(network, imdb_train, output_ckpt_dir, output_generate_image_dir,
              pretrained_model=None, max_epoch=1000, restore=False):
    """Train a Fast R-CNN networks."""

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, network, imdb_train, output_ckpt_dir, output_generate_image_dir,
                           pretrained_model=pretrained_model)
        print ('Solving...')
        sw.train_model(sess, max_epoch, restore=restore)
        print ('done solving')
