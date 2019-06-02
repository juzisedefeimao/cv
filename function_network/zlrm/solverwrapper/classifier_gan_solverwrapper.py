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

    def train_model(self, sess, max_iters, restore=False):
        """Network training loop."""

        data_layer = self.imdb_train

        discriminator_loss, generator_loss = self.net.build_loss()

        # optimizer
        if cfg.GAN.TRAIN.SOLVER == 'Adam':
            generator_opt = tf.train.AdamOptimizer(cfg.GAN.TRAIN.GENERATOR_LEARNING_RATE)
            discriminator_opt = tf.train.AdamOptimizer(cfg.GAN.TRAIN.DISCRIMINATOR_LEARNING_RATE)
        elif cfg.GAN.TRAIN.SOLVER == 'RMS':
            generator_opt = tf.train.RMSPropOptimizer(cfg.GAN.TRAIN.GENERATOR_LEARNING_RATE)
            discriminator_opt = tf.train.RMSPropOptimizer(cfg.GAN.TRAIN.DISCRIMINATOR_LEARNING_RATE)
        elif cfg.GAN.TRAIN.SOLVER == 'Momentum':
            generator_lr = tf.Variable(cfg.GAN.TRAIN.GENERATOR_LEARNING_RATE, trainable=False)
            generator_momentum = cfg.GAN.TRAIN.GENERATOR_MOMENTUM
            generator_opt = tf.train.MomentumOptimizer(generator_lr, generator_momentum)
            discriminator_lr = tf.Variable(cfg.GAN.TRAIN.DISCRIMINATOR_LEARNING_RATE, trainable=False)
            discriminator_momentum = cfg.GAN.TRAIN.DISCRIMINATOR_MOMENTUM
            discriminator_opt = tf.train.MomentumOptimizer(discriminator_lr, discriminator_momentum)
        else:
            raise ModuleNotFoundError('不存在的优化器，可使用的优化器为Adam、RMS、Momentum')

        global_step = tf.Variable(0, trainable=False)
        with_clip = False
        if with_clip:
            discriminator_tvars = tf.trainable_variables(scope='discriminator')
            generator_tvars = tf.trainable_variables(scope='generator')

            discriminator_grads, discriminator_norm = tf.clip_by_global_norm(
                tf.gradients(discriminator_loss, discriminator_tvars), 10.0)
            discriminator_train_op = discriminator_opt.apply_gradients(zip(discriminator_grads, discriminator_tvars),
                                                         global_step=global_step)
            generator_grads, generator_norm = tf.clip_by_global_norm(
                tf.gradients(generator_loss, generator_tvars), 10.0)
            generator_train_op = generator_opt.apply_gradients(zip(generator_grads, generator_tvars),
                                                         global_step=global_step)
        else:
            discriminator_tvars = tf.trainable_variables(scope='discriminator')
            generator_tvars = tf.trainable_variables(scope='generator')
            discriminator_train_op = discriminator_opt.minimize(discriminator_loss, global_step=global_step, var_list=discriminator_tvars)
            generator_train_op = generator_opt.minimize(generator_loss, global_step=global_step, var_list=generator_tvars)

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
                print(self.output_ckpt_dir)
                ckpt = tf.train.get_checkpoint_state(self.output_ckpt_dir)
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

            # learning rate
            if cfg.GAN.TRAIN.SOLVER == 'Momentum':
                if iter != 0 and iter % cfg.ZLRM.TRAIN.STEPSIZE == 0:
                    sess.run(tf.assign(generator_lr, generator_lr.eval() * cfg.GAN.TRAIN.GENERATOR_GAMMA))
                    sess.run(tf.assign(discriminator_lr, discriminator_lr.eval() * cfg.GAN.TRAIN.DISCRIMINATOR_GAMMA))
            # get one batch
            blobs = data_layer.forward()
            batch_noise = np.random.uniform(-1, 1, [cfg.GAN.TRAIN.BATCH_SIZE, 1, 1, cfg.GAN.NOISE_DIM]) \
                .astype(np.float32)

            feed_dict={
                self.net.real_data: blobs['data'],
                self.net.noise_data: batch_noise,
                self.net.label: blobs['label']
            }


            fetch_distriminator_list = [discriminator_loss,
                                        discriminator_train_op]
            fetch_generator_list = [generator_loss,
                                    self.net.get_output('fake_images'),
                                    generator_train_op]

            fetch_distriminator_list += []
            fetch_generator_list += []

            discriminator_loss_val, _ = sess.run(fetches=fetch_distriminator_list, feed_dict=feed_dict)
            generator_loss_val, generate_image_batch, _ = sess.run(fetches=fetch_generator_list, feed_dict=feed_dict)


            if (iter) % (cfg.ZLRM.TRAIN.DISPLAY) == 0:
                if cfg.GAN.TRAIN.SOLVER == 'Momentum':
                    print(
                        'one_iter: %d / %d, discriminator loss: %.4f, generator loss: %.4f, generator lr: %f, discriminator lr: %f' % \
                        (iter, max_iters, discriminator_loss_val, generator_loss_val, generator_lr.eval(), discriminator_lr.eval()))
                else:
                    print(
                        'one_iter: %d / %d, discriminator loss: %.4f, generator loss: %.4f' % \
                        (iter, max_iters, discriminator_loss_val, generator_loss_val))

            discriminator_iter = 0
            while discriminator_loss_val>0.6:
                # # get one batch
                # blobs = data_layer.forward()
                # batch_noise = np.random.uniform(-1, 1, [cfg.GAN.TRAIN.BATCH_SIZE, 1, 1, cfg.GAN.NOISE_DIM]) \
                #     .astype(np.float32)
                #
                # feed_dict = {
                #     self.net.real_data: blobs['data'],
                #     self.net.noise_data: batch_noise,
                #     self.net.label: blobs['label']
                # }
                #
                # fetch_distriminator_list = [discriminator_loss,
                #                             discriminator_train_op]
                #
                # fetch_distriminator_list += []
                discriminator_loss_val, _ = sess.run(fetches=fetch_distriminator_list, feed_dict=feed_dict)
                if cfg.GAN.TRAIN.SOLVER == 'Momentum':
                    print(
                        '    discriminator_iter: %d / %d, discriminator loss: %.4f, discriminator lr: %f' % \
                        (discriminator_iter, iter, discriminator_loss_val, discriminator_lr.eval()))
                else:
                    print(
                        '    discriminator_iter: %d / %d, discriminator loss: %.4f' % \
                        (discriminator_iter, iter, discriminator_loss_val))
                discriminator_iter = discriminator_iter + 1

            generator_iter = 0
            while generator_loss_val>1.45:
                # get one batch
                # blobs = data_layer.forward()
                # batch_noise = np.random.uniform(-1, 1, [cfg.GAN.TRAIN.BATCH_SIZE, 1, 1, cfg.GAN.NOISE_DIM]) \
                #     .astype(np.float32)
                #
                # feed_dict = {
                #     self.net.real_data: blobs['data'],
                #     self.net.noise_data: batch_noise,
                #     self.net.label: blobs['label']
                # }
                #
                # fetch_generator_list = [generator_loss,
                #                         self.net.get_output('fake_images'),
                #                         generator_train_op]
                #
                # fetch_generator_list += []
                generator_loss_val, generate_image_batch, _ = sess.run(fetches=fetch_generator_list,
                                                                           feed_dict=feed_dict)
                if cfg.GAN.TRAIN.SOLVER == 'Momentum':
                    print(
                        '    generator_iter: %d / %d, generator loss: %.4f, generator lr: %f' % \
                        (generator_iter, iter, generator_loss_val, generator_lr.eval()))
                else:
                    print(
                        '    generator_iter: %d / %d, generator loss: %.4f' % \
                        (generator_iter, iter, generator_loss_val))

                generator_iter = generator_iter + 1


            _diff_time = timer.toc(average=False)


            if (iter) % (cfg.ZLRM.TRAIN.DISPLAY) == 0:
                if cfg.GAN.TRAIN.SOLVER == 'Momentum':
                    print(
                        'two_iter: %d / %d, discriminator loss: %.4f, generator loss: %.4f, generator lr: %f, discriminator lr: %f' % \
                        (iter, max_iters, discriminator_loss_val, generator_loss_val, generator_lr.eval(), discriminator_lr.eval()))
                else:
                    print(
                        'two_iter: %d / %d, discriminator loss: %.4f, generator loss: %.4f' % \
                        (iter, max_iters, discriminator_loss_val, generator_loss_val))
                print ('speed: {:.3f}s / iter'.format(_diff_time))

            if (iter+1) % cfg.GAN.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

            if (iter + 1) % cfg.GAN.TRAIN.SAVE_IMAGE_ITERS == 0:
                last_save_generate_image_iter = iter
                self.save_generate_image(generate_image_batch, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)
        if last_save_generate_image_iter != iter:
            self.save_generate_image(generate_image_batch, iter)

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

def train_net(network, imdb_train, output_ckpt_dir, output_generate_image_dir,
              pretrained_model=None, max_iters=40000, restore=False):
    """Train a Fast R-CNN networks."""

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, network, imdb_train, output_ckpt_dir, output_generate_image_dir,
                           pretrained_model=pretrained_model)
        print ('Solving...')
        sw.train_model(sess, max_iters, restore=restore)
        print ('done solving')
