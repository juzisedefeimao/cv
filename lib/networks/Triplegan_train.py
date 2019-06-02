from lib.networks.network import Network
import numpy as np
from lib.networks.netconfig import cfg
import tensorflow as tf

cls_num = cfg.GAN.TRAIN.CLASSIFY_NUM

class Triplegan_train(Network):
    def __init__(self):
        self.inputs = []
        self.noise_data = tf.placeholder(tf.float32, [cfg.GAN.TRAIN.BATCH_SIZE, cfg.GAN.NOISE_DIM], name='noise_data')

        self.real_data = tf.placeholder(tf.float32,
                                   shape=[cfg.GAN.TRAIN.BATCH_SIZE, cfg.GAN.IMAGE_SIZE,
                                          cfg.GAN.IMAGE_SIZE, cfg.GAN.IMAGE_DIM], name='real_data')
        self.unlabel_data = tf.placeholder(tf.float32,
                                        shape=[cfg.GAN.TRAIN.UNLABEL_BATCH_SIZE, cfg.GAN.IMAGE_SIZE,
                                               cfg.GAN.IMAGE_SIZE, cfg.GAN.IMAGE_DIM], name='unlabel_data')
        self.label = tf.placeholder(tf.int32, shape=[cfg.GAN.TRAIN.BATCH_SIZE], name='label')
        self.unlabel = tf.placeholder(tf.int32, shape=[cfg.GAN.TRAIN.UNLABEL_BATCH_SIZE], name='unlabel')
        self.visual_label = tf.placeholder(tf.int32, [cfg.GAN.TRAIN.SAVE_IMAGE_NUM], name='visual_label')
        self.visual_noise = tf.placeholder(tf.float32, [cfg.GAN.TRAIN.SAVE_IMAGE_NUM, cfg.GAN.NOISE_DIM],
                                           name='visual_noise')

        self.alpha_p = tf.placeholder(tf.float32, name='alpha_p')
        self.layers = {}
        self.setup()

    def build_loss(self):
        """ Loss Function """
        # get loss for discriminator
        d_loss = self.discriminator_loss(self.layers['D_real_logits'], self.layers['D_fake_logits'],
                                         self.layers['D_cla_logits'])

        # get loss for generator
        g_loss = self.generator_loss(self.layers['D_fake_logits'])

        # get loss for classifier
        c_loss = self.classifier_loss(self.layers['C_label_logist'], self.layers['D_cla_logits'],
                                      self.layers['C_real_logits'], self.layers['C_fake_logits'],
                                      self.alpha_p)
        return d_loss, g_loss, c_loss

    def generate_image(self):
        G = self.generator(self.visual_noise, self.visual_label, reuse=True, bn_trainable=False)
        return G

    def setup(self):

        # output of D for real images
        D_real, D_real_logits = self.discriminator(self.real_data, self.label, reuse=False)

        # output of D for fake images
        G = self.generator(self.noise_data, self.label, reuse=False)
        D_fake, D_fake_logits = self.discriminator(G, self.label, reuse=True)

        # output of C for real images
        C_real_logits = self.classifier(self.real_data, reuse=False)

        # output of D for unlabelled images
        C_label_logist = self.classifier(self.unlabel_data, reuse=True)
        C_label = tf.reshape(tf.cast(tf.argmax(C_label_logist, axis=1), tf.int32), [-1])
        D_cla, D_cla_logits = self.discriminator(self.unlabel_data, C_label, reuse=True)

        # output of C for fake images
        C_fake_logits = self.classifier(G, reuse=True)

        self.layers['C_real_logits'] = C_real_logits
        self.layers['C_fake_logits'] = C_fake_logits
        self.layers['C_label_logist'] = C_label_logist
        self.layers['C_label'] = C_label
        self.layers['D_real'] = D_real
        self.layers['D_real_logits'] = D_real_logits
        self.layers['D_fake'] = D_fake
        self.layers['D_fake_logits'] = D_fake_logits
        self.layers['D_cla'] = D_cla
        self.layers['D_cla_logits'] = D_cla_logits

    def discriminator(self, data, label, reuse = False, bn_trainable = True):
        with tf.variable_scope('discriminator') as scope:

            init_GAN_weights = True
            spectral_norm = False
            if reuse:
                scope.reuse_variables()

            label = tf.one_hot(label, depth=cls_num)
            (
                self.feed(label)
                    .reshape_layer([-1, 1, 1, cls_num], name='d_reshape_label')
            )
            (
                self.feed(data)
                    .dropout(0.2, name='d_dropout')
            )
            (
                self.feed('d_dropout', 'd_reshape_label')
                    .conv_concat(name='d_conv_concat_1')
                    .conv(3, 3, 32, 1, 1, name='d_conv1', reuse=reuse, biased=False, relu=False,
                          init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm, trainable=True)
                    .batch_normalization(name='d_bn1', relu=False, trainable=bn_trainable)
                    .lrelu(alpha = 0.2,name='d_lrelu1')
            )
            (
                self.feed('d_lrelu1', 'd_reshape_label')
                    .conv_concat(name='d_conv_concat_2')
                    .conv(3, 3, 32, 2, 2, name='d_conv2', reuse=reuse, biased=False, relu=False,
                          init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm, trainable=True)
                    .batch_normalization(name='d_bn2', relu=False, trainable=bn_trainable)
                    .lrelu(alpha=0.2, name='d_lrelu2')
                    .dropout(0.2, name='d_dropout2')
            )
            (
                self.feed('d_dropout2', 'd_reshape_label')
                    .conv_concat(name='d_conv_concat_3')
                    .conv(3, 3, 64, 1, 1, name='d_conv3', reuse=reuse, biased=False, relu=False,
                          init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm, trainable=True)
                    .batch_normalization(name='d_bn3', relu=False, trainable=bn_trainable)
                    .lrelu(alpha=0.2, name='d_lrelu3')
            )
            (
                self.feed('d_lrelu3', 'd_reshape_label')
                    .conv_concat(name='d_conv_concat_4')
                    .conv(3, 3, 64, 2, 2, name='d_conv4', reuse=reuse, biased=False, relu=False,
                          init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm, trainable=True)
                    .batch_normalization(name='d_bn4', relu=False, trainable=bn_trainable)
                    .lrelu(alpha=0.2, name='d_lrelu4')
                    .dropout(0.2, name='d_dropout4')
            )
            (
                self.feed('d_dropout4', 'd_reshape_label')
                    .conv_concat(name='d_conv_concat_5')
                    .conv(3, 3, 128, 1, 1, name='d_conv5', reuse=reuse, biased=False, relu=False,
                          init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm, trainable=True)
                    .batch_normalization(name='d_bn5', relu=False, trainable=bn_trainable)
                    .lrelu(alpha=0.2, name='d_lrelu5')
            )
            (
                self.feed('d_lrelu5', 'd_reshape_label')
                    .conv_concat(name='d_conv_concat_6')
                    .conv(3, 3, 128, 2, 2, name='d_conv6', reuse=reuse, biased=False, relu=False,
                          init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm, trainable=True)
                    .batch_normalization(name='d_bn6', relu=False, trainable=bn_trainable)
                    .lrelu(alpha=0.2, name='d_lrelu6')
            )
            (
                self.feed('d_lrelu6', 'd_reshape_label')
                    .conv_concat(name='d_conv_concat_7')
                    .global_avg_pool(name='d_global_ave_pool')
            )
            (
                self.feed('d_global_ave_pool', label)
                    .concat(axis=1, name='d_concat')
                    .fc(1, name='discriminator_logit', relu=False)
                    .sigmoid(name='discriminator_data')
            )

        return self.get_output('discriminator_data'), self.get_output('discriminator_logit')


    def generator(self, data, label, reuse = False, bn_trainable=True):
        with tf.variable_scope('generator') as scope:
            spectral_norm = False
            if reuse:
                scope.reuse_variables()

            label = tf.one_hot(label, depth=cls_num)
            (
                self.feed(data, label)
                    .concat(axis=1, name='g_concat')
                    .fc(512*4*4, name='g_fc', relu=True)
                    .batch_normalization(name='bn', relu=False, trainable=bn_trainable)
                    .reshape_layer([-1, 4, 4, 512], name='g_reshape_data')
            )
            (
                self.feed(label)
                    .reshape_layer([-1, 1, 1, cls_num], name='g_reshape_label')
            )
            (
                self.feed('g_reshape_data', 'g_reshape_label')
                    .conv_concat(name='g_conv_concat_1')
                    .upconv(None, 256, ksize=5, stride=2, name='g_upconv1', biased=True, relu=True,
                            spectral_norm=spectral_norm, trainable=True)
                    .batch_normalization(name='g_bn1', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('g_bn1', 'g_reshape_label')
                    .conv_concat(name='g_conv_concat_2')
                    .upconv(None, 128, ksize=5, stride=2, name='g_upconv2', biased=True, relu=True,
                            spectral_norm=spectral_norm, trainable=True)
                    .batch_normalization(name='g_bn2', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('g_bn2', 'g_reshape_label')
                    .conv_concat(name='g_conv_concat_3')
                    .upconv(None, cfg.GAN.IMAGE_DIM, ksize=5, stride=2, name='g_upconv3', biased=True, relu=True,
                            spectral_norm=spectral_norm, trainable=True)
                    .batch_normalization(name='g_bn3', relu=False, trainable=bn_trainable)
                    .tanh(name='generate_data')
            )
        return self.get_output('generate_data')

    def classifier(self, data, reuse=False, bn_trainable = True):
        with tf.variable_scope('classifier') as scope:

            init_GAN_weights = True
            spectral_norm = False
            if reuse:
                scope.reuse_variables()
            (
                self.feed(data)
                    .gaussian_noise_layer(std=0.15, name= 'gaussian_noise')
                    .conv(3, 3, 128, 1, 1, name='c_conv1', reuse=reuse, biased=False, relu=False,
                          init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm,
                          trainable=True)
                    .batch_normalization(name='c_bn1', relu=False, trainable=bn_trainable)
                    .lrelu(alpha=0.2, name='c_lrelu1')
                    .conv(3, 3, 128, 1, 1, name='c_conv2', reuse=reuse, biased=False, relu=False,
                          init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm,
                          trainable=True)
                    .batch_normalization(name='c_bn2', relu=False, trainable=bn_trainable)
                    .lrelu(alpha=0.2, name='c_lrelu2')
                    .conv(3, 3, 128, 1, 1, name='c_conv3', reuse=reuse, biased=False, relu=False,
                          init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm,
                          trainable=True)
                    .batch_normalization(name='c_bn3', relu=False, trainable=bn_trainable)
                    .lrelu(alpha=0.2, name='c_lrelu3')
                    .max_pool(2, 2, 2, 2, name='c_maxpool1')
                    .dropout(0.5, name='c_dropout1')
                    .conv(3, 3, 256, 1, 1, name='c_conv4', reuse=reuse, biased=False, relu=False,
                          init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm,
                          trainable=True)
                    .batch_normalization(name='c_bn4', relu=False, trainable=bn_trainable)
                    .lrelu(alpha=0.2, name='c_lrelu4')
                    .conv(3, 3, 256, 1, 1, name='c_conv5', reuse=reuse, biased=False, relu=False,
                          init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm,
                          trainable=True)
                    .batch_normalization(name='c_bn5', relu=False, trainable=bn_trainable)
                    .lrelu(alpha=0.2, name='c_lrelu5')
                    .conv(3, 3, 256, 1, 1, name='c_conv6', reuse=reuse, biased=False, relu=False,
                          init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm,
                          trainable=True)
                    .batch_normalization(name='c_bn6', relu=False, trainable=bn_trainable)
                    .lrelu(alpha=0.2, name='c_lrelu6')
                    .max_pool(2, 2, 2, 2, name='c_maxpool2')
                    .dropout(0.5, name='c_dropout2')
                    .conv(3, 3, 512, 1, 1, name='c_conv7', reuse=reuse, biased=False, relu=False,
                          init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm,
                          trainable=True)
                    .batch_normalization(name='c_bn7', relu=False, trainable=bn_trainable)
                    .lrelu(alpha=0.2, name='c_lrelu7')
                    .conv(1, 1, 256, 1, 1, name='c_conv8', reuse=reuse, biased=False, relu=False,
                          init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm,
                          trainable=True)
                    .batch_normalization(name='c_bn8', relu=False, trainable=bn_trainable)
                    .lrelu(alpha=0.2, name='c_lrelu8')
                    .conv(1, 1, 128, 1, 1, name='c_conv9', reuse=reuse, biased=False, relu=False,
                          init_GAN_weights=init_GAN_weights, spectral_norm=spectral_norm,
                          trainable=True)
                    .batch_normalization(name='c_bn9', relu=False, trainable=bn_trainable)
                    .lrelu(alpha=0.2, name='c_lrelu9')
                    .max_pool(2, 2, 2, 2, name='c_maxpool1')
                    .dropout(0.5, name='c_dropout1')
                    .global_avg_pool(name='c_global_ave_pool')
                    .fc(cls_num, name='classifier_logit', relu=False)
                    .softmax(name='classifier_data')

            )
        return self.get_output('classifier_logit')

    def discriminator_loss(self, D_real_logits, D_fake_logits, D_cla_logits):
        # get loss for discriminator
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
        d_loss_fake = (1 - cfg.GAN.TRAIN.ALPHA) * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
        d_loss_cla = cfg.GAN.TRAIN.ALPHA * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_cla_logits, labels=tf.zeros_like(D_cla_logits)))
        d_loss = d_loss_real + d_loss_fake + d_loss_cla
        return d_loss

    def generator_loss(sellf, D_fake_logits):
        g_loss = (1 - cfg.GAN.TRAIN.ALPHA) * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))

        return g_loss



    def classifier_loss(self, C_label_logist, D_cla_logits, C_real_logits, C_fake_logits, alpha_p):
        # get loss for classify
        max_c = tf.cast(tf.reduce_max(C_label_logist, axis=1), tf.float32)
        c_loss_dis = tf.reduce_mean(
            max_c * tf.nn.sigmoid_cross_entropy_with_logits(logits=D_cla_logits, labels=tf.ones_like(D_cla_logits)))
        R_L = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=C_real_logits))
        R_P = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=C_fake_logits))
        c_loss = cfg.GAN.TRAIN.ALPHA_CLA_ADV * cfg.GAN.TRAIN.ALPHA * c_loss_dis + R_L + alpha_p * R_P

        return c_loss