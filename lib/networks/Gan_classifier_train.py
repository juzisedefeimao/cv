from lib.networks.network import Network
import numpy as np
from lib.networks.netconfig import cfg
import tensorflow as tf


class Gan_classifier_train(Network):
    def __init__(self):
        self.inputs = []
        self.noise_data = tf.placeholder(tf.float32, [cfg.GAN.TRAIN.BATCH_SIZE, 1, 1, cfg.GAN.NOISE_DIM], name='noise_data')
        self.real_data = tf.placeholder(tf.float32,
                                   shape=[cfg.GAN.TRAIN.BATCH_SIZE, cfg.GAN.IMAGE_SIZE,
                                          cfg.GAN.IMAGE_SIZE, cfg.GAN.IMAGE_DIM], name='real_data')
        self.label = tf.placeholder(tf.int32, shape=[cfg.GAN.TRAIN.BATCH_SIZE], name='label')
        self.layers = {'random_data':self.noise_data, 'real_data': self.real_data, 'label':self.label}
        self.setup()

    def build_loss(self):
        """ Loss Function """
        if cfg.GAN.GAN_TYPE.__contains__('wgan') or cfg.GAN.GAN_TYPE == 'dragan':
            GP = self.gradient_penalty(real=self.real_data, fake=self.layers['fake_images'])
        else:
            GP = 0

        # get loss for discriminator
        d_loss = self.discriminator_loss(cfg.GAN.GAN_TYPE, real=self.layers['real_logits'], fake=self.layers['fake_logits']) + GP

        # get loss for generator
        g_loss = self.generator_loss(cfg.GAN.GAN_TYPE, fake=self.layers['fake_logits'])
        return d_loss, g_loss

    def setup(self):
        # output of D for real images
        real_logits = self.discriminator(self.real_data)

        # output of D for fake images
        fake_images = self.generator(self.noise_data)
        fake_logits = self.discriminator(fake_images, reuse=True)

        self.layers['real_logits'] = real_logits
        self.layers['fake_images'] = fake_images
        self.layers['fake_logits'] = fake_logits

    def discriminator(self, data, reuse = False):
        with tf.variable_scope('discriminator') as scope:
            bn_trainable = False
            if reuse:
                scope.reuse_variables()
            (
                self.feed(data)
                    .conv(4, 4, 64, 2, 2, name='conv1', reuse=reuse, biased=False, relu=False, spectral_norm=True, trainable=True)
                    .batch_normalization(name='bn1', relu=False, trainable=bn_trainable)
                    .lrelu(alpha = 0.2,name='lrelu1')
                    .conv(4, 4, 128, 2, 2, name='conv2', reuse=reuse, biased=False, relu=False, spectral_norm=True, trainable=True)
                    .batch_normalization(name='bn2', relu=False, trainable=bn_trainable)
                    .lrelu(alpha=0.2, name='lrelu2')
                    .attention(128, name='attention')
                    .conv(4, 4, 256, 2, 2, name='conv3', reuse=reuse, biased=False, relu=False, spectral_norm=True, trainable=True)
                    .batch_normalization(name='bn3', relu=False, trainable=bn_trainable)
                    .lrelu(alpha=0.2, name='lrelu3')
                    .conv(4, 4, 4, 1, 1, name='discriminator_data', reuse=reuse, biased=False, relu=False, spectral_norm=True, trainable=True)
            )

        return self.get_output('discriminator_data')


    def generator(self, data):
        with tf.variable_scope('generator'):
            bn_trainable = False
            (
                self.feed(data)
                    .upconv(None, 1024, ksize=4, stride=4, name='upconv1', biased=False, relu=False,
                            spectral_norm=True, trainable=True)
                    .batch_normalization(name='bn1', relu=True, trainable=bn_trainable)
                    .upconv(None, 512, ksize=4, stride=2, name='upconv2', biased=False, relu=False,
                            spectral_norm=True, trainable=True)
                    .batch_normalization(name='bn2', relu=True, trainable=bn_trainable)
                    .attention(512, name='attention')
                    .upconv(None, 256, ksize=4, stride=2, name='upconv3', biased=False, relu=False,
                            spectral_norm=True, trainable=True)
                    .batch_normalization(name='bn3', relu=True, trainable=bn_trainable)
                    .upconv(None, cfg.GAN.IMAGE_DIM, ksize=4, stride=2, name='upconv4', biased=False, relu=False,
                            spectral_norm=True, trainable=True)
                    .batch_normalization(name='bn4', relu=True, trainable=bn_trainable)
                    .tanh(name='generate_data')
            )
        return self.get_output('generate_data')

    def discriminator_loss(self, loss_func, real, fake):
        real_loss = 0
        fake_loss = 0

        if loss_func.__contains__('wgan'):
            real_loss = -tf.reduce_mean(real)
            fake_loss = tf.reduce_mean(fake)

        if loss_func == 'lsgan':
            real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
            fake_loss = tf.reduce_mean(tf.square(fake))

        if loss_func == 'gan' or loss_func == 'dragan':
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

        if loss_func == 'hinge':
            real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real))
            fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake))

        loss = real_loss + fake_loss

        return loss

    def generator_loss(sellf, loss_func, fake):
        fake_loss = 0

        if loss_func.__contains__('wgan'):
            fake_loss = -tf.reduce_mean(fake)

        if loss_func == 'lsgan':
            fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))

        if loss_func == 'gan' or loss_func == 'dragan':
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

        if loss_func == 'hinge':
            fake_loss = -tf.reduce_mean(fake)

        loss = fake_loss

        return loss

    def gradient_penalty(self, real, fake):
        if cfg.GAN.GAN_TYPE == 'dragan' :
            shape = tf.shape(real)
            eps = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            x_mean, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
            noise = 0.5 * x_std * eps  # delta in paper

            # Author suggested U[0,1] in original paper, but he admitted it is bug in github
            # (https://github.com/kodalinaveen3/DRAGAN). It should be two-sided.

            alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
            interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X

        else :
            alpha = tf.random_uniform(shape=[cfg.GAN.TRAIN.BATCH_SIZE, 1, 1, 1], minval=0., maxval=1.)
            interpolated = alpha*real + (1. - alpha)*fake

        logit = self.discriminator(interpolated, reuse=True)

        grad = tf.gradients(logit, interpolated)[0]  # gradient of D(interpolated)
        grad_norm = tf.norm(self.flatten(grad), axis=1)  # l2 norm

        GP = 0

        # WGAN - LP
        if cfg.GAN.GAN_TYPE == 'wgan-lp':
            GP = cfg.GAN.LD * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

        elif cfg.GAN.GAN_TYPE == 'wgan-gp' or cfg.GAN.GAN_TYPE == 'dragan':
            GP = cfg.GAN.LD * tf.reduce_mean(tf.square(grad_norm - 1.))

        return GP