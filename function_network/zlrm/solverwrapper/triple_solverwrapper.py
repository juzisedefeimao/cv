from lib.networks.netconfig import cfg
from lib.datasets.factory import get_imdb
import numpy as np
import tensorflow as tf
import time
import math
import os
import scipy

class SolverWrapper(object):

    def __init__(self, network, imdb_train, output_ckpt_dir, output_generate_image_dir, output_log, pretrained_model=None):
        self.net = network
        self.imdb_train = imdb_train
        self.output_ckpt_dir = output_ckpt_dir
        self.output_generate_image_dir = output_generate_image_dir
        self.output_log = output_log
        self.pretrained_model = pretrained_model


        self.learning_rate = cfg.GAN.TRAIN.DISCRIMINATOR_LEARNING_RATE # 3e-4, 1e-3
        self.cla_learning_rate = cfg.GAN.TRAIN.CLASSIFIER_LEARNING_RATE # 3e-3, 1e-2 ?
        self.GAN_beta1 = 0.5
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.init_alpha_p = 0.0 # 0.1, 0.03
        self.apply_alpha_p = 0.1
        self.apply_epoch = 200 # 200, 300
        self.decay_epoch = 50

        self.sample_num = 64

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
        data_layer = self.imdb_train

        self.gan_lr = tf.placeholder(tf.float32, name='gan_lr')
        self.cla_lr = tf.placeholder(tf.float32, name='cla_lr')
        self.unsup_weight = tf.placeholder(tf.float32, name='unsup_weight')
        self.c_beta1 = tf.placeholder(tf.float32, name='c_beta1')

        # for test
        fake_images = self.net.generate_image()
        accuracy = self.net.test_classifier()

        """ Loss Function """
        discriminator_loss, generator_loss, classifier_loss = self.net.build_loss()



        """ Training """


        discriminator_tvars = tf.trainable_variables(scope='discriminator')
        generator_tvars = tf.trainable_variables(scope='generator')
        classifier_tvars = tf.trainable_variables(scope='classifier')

        global_step = tf.Variable(0, trainable=False)
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            discriminator_opt = tf.train.AdamOptimizer(self.gan_lr, beta1=self.GAN_beta1).minimize(discriminator_loss, global_step=global_step,
                                                                                              var_list=discriminator_tvars)
            generator_opt = tf.train.AdamOptimizer(self.gan_lr, beta1=self.GAN_beta1).minimize(generator_loss, global_step=global_step,
                                                                                              var_list=generator_tvars)
            classifier_opt = tf.train.AdamOptimizer(self.cla_lr, beta1=self.beta1, beta2=self.beta2,
                                                  epsilon=self.epsilon).minimize(classifier_loss, global_step=global_step,
                                                                                 var_list=classifier_tvars)

        # initialize all variables
        tf.global_variables_initializer().run()
        gan_lr = self.learning_rate
        cla_lr = self.cla_learning_rate


        # resuming a trainer
        restore_epoch = 0
        if restore:
            try:
                print(self.output_ckpt_dir)
                ckpt = tf.train.get_checkpoint_state(self.output_ckpt_dir)
                print('Restoring from {}...'.format(ckpt.model_checkpoint_path), )
                tvars = tf.trainable_variables()
                print(tvars)
                self.restor_saver.restore(sess, ckpt.model_checkpoint_path)
                stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
                restore_epoch = int(stem.split('_')[-1])
                sess.run(global_step.assign(restore_epoch))

                learning_rate_log = os.path.join(self.output_log, 'learning_rate.txt')
                with open(learning_rate_log, 'r') as f:
                    learning_list = f.readlines()
                for i in range(len(learning_list)):
                    if learning_list[i].split(' ')[0] == restore_epoch:
                        gan_lr = learning_list[i].split(' ')[1]
                        cla_lr = learning_list[i].split(' ')[2]
                print('done')
            except:
                raise BaseException('Check your pretrained {:s}'.format(ckpt.model_checkpoint_path))

        # loop for epoch
        last_snapshot_epoch = -1
        last_save_generate_image_epoch = -1
        start_time = time.time()
        for epoch in range(restore_epoch, max_epoch):

            if epoch >= self.decay_epoch :
                gan_lr *= 0.995
                cla_lr *= 0.99
                print("**** learning rate DECAY ****")
                print(gan_lr)
                print(cla_lr)

            if epoch >= self.apply_epoch :
                alpha_p = self.apply_alpha_p
            else :
                alpha_p = self.init_alpha_p

            rampup_value = rampup(epoch - 1)
            unsup_weight = rampup_value * 100.0 if epoch > 1 else 0

            # get batch data
            for idx in range(data_layer.epoch_iter_num):
                blobs = data_layer.epoch_forward()

                feed_dict = {
                    self.net.real_data: blobs['data'], self.net.label: blobs['label'],
                    self.net.unlabel_data: blobs['unlabel_data'],
                    self.net.unlabel: blobs['unlabel'],
                    self.net.noise_data: blobs['noise'], self.net.alpha_p: alpha_p,
                    self.gan_lr: gan_lr, self.cla_lr: cla_lr,
                    self.unsup_weight : unsup_weight
                }
                # update D network
                _, discriminator_loss_value = sess.run([discriminator_opt, discriminator_loss], feed_dict=feed_dict)

                # update G network
                _, generator_loss_value = sess.run([generator_opt, generator_loss], feed_dict=feed_dict)

                # update C network
                _, classifier_loss_value = sess.run([classifier_opt, classifier_loss], feed_dict=feed_dict)

                test_feed_dict = {
                    self.net.test_data: blobs['test_data'],
                    self.net.test_label: blobs['test_label']
                }

                accuracy_value = sess.run(accuracy, feed_dict=test_feed_dict)

                # display training status
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, c_loss: %.8f, test_accuracy: %.8f" \
                      % (epoch, idx, data_layer.epoch_iter_num, time.time() - start_time,
                         discriminator_loss_value, generator_loss_value, classifier_loss_value, accuracy_value))

                # save training results for every 100 steps

            lr = "{} {} {}".format(epoch, gan_lr, cla_lr)
            accuracy_ = "epoch{} accuracy{} gan_lr{} cla_lr{}".format(epoch, accuracy_value, gan_lr, cla_lr)
            if not os.path.exists(self.output_log):
                os.makedirs(self.output_log)

            accuracy_log = os.path.join(self.output_log, 'accuracy.txt')
            learning_rate_log = os.path.join(self.output_log, 'learning_rate.txt')

            if restore or epoch > 0:
                with open(accuracy_log, 'a') as f:
                    f.write(accuracy_ + '\n')
                with open(learning_rate_log, 'a') as f :
                    f.write(lr+'\n')
            else:
                with open(accuracy_log, 'a') as f:
                    f.write(time.strftime("%Y_%m_%d_%H_%M_%S") + '\n')
                    f.write(accuracy_ + '\n')
                with open(learning_rate_log, 'w') as f :
                    f.write(lr+'\n')

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model

            # save model
            if (epoch+1) % cfg.GAN.TRAIN.SNAPSHOT_EPOCH == 0:
                last_snapshot_epoch = epoch
                self.snapshot(sess, epoch)


            # show temporal results
            if (epoch + 1) % cfg.GAN.TRAIN.SAVE_IMAGE_EPOCH == 0:
                last_save_generate_image_epoch = epoch
                self.visualize_results(sess, fake_images, epoch)

            # save model for final step
        if last_snapshot_epoch != epoch:
            self.snapshot(sess, epoch)
        if last_save_generate_image_epoch != epoch:
            self.visualize_results(sess, fake_images, epoch)


def rampup(epoch):
    if epoch < 80:
        p = max(0.0, float(epoch)) / float(80)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0

def rampdown(epoch):
    if epoch >= (300 - 50):
        ep = (epoch - (300 - 50)) * 0.5
        return math.exp(-(ep * ep) / 50)
    else:
        return 1.0


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

def train_net(network, imdb_train, output_ckpt_dir, output_generate_image_dir, output_log,
              pretrained_model=None, max_epoch=1000, restore=False):
    """Train a Fast R-CNN networks."""

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(network, imdb_train, output_ckpt_dir, output_generate_image_dir, output_log,
                           pretrained_model=pretrained_model)
        print ('Solving...')
        sw.train_model(sess, max_epoch, restore=restore)
        print ('done solving')