import tensorflow as tf
from lib.networks.network import Network
from lib.networks.netconfig import cfg
cls_num = cfg.METAGAN.TRAIN.CLASSIFY_NUM

class MeatGan_Network(Network) :
    def __init__(self):
        self.sample_data = tf.placeholder(tf.float32,
                                   shape=[cfg.METAGAN.TRAIN.CLASSIFY_SAMPLE * cls_num,
                                          cfg.METAGAN.CLASSIFY_IMAGE_SIZE[0],
                                          cfg.METAGAN.CLASSIFY_IMAGE_SIZE[1], 3], name='sample_data')
        self.query_data = tf.placeholder(tf.float32,
                                   shape=[cfg.METAGAN.TRAIN.CLASSIFY_QUERY * cls_num,
                                          cfg.METAGAN.CLASSIFY_IMAGE_SIZE[0],
                                          cfg.METAGAN.CLASSIFY_IMAGE_SIZE[1], 3], name='query_data')
        self.sample_label = tf.placeholder(tf.int32, shape=[cls_num],
                                           name='sample_label')
        self.query_label = tf.placeholder(tf.int32, shape=[cfg.METAGAN.TRAIN.CLASSIFY_QUERY * cls_num],
                                          name='query_label')

        self.test_sample_data = tf.placeholder(tf.float32,
                                        shape=[cfg.METAGAN.TRAIN.CLASSIFY_SAMPLE * cls_num,
                                               cfg.METAGAN.CLASSIFY_IMAGE_SIZE[0],
                                               cfg.METAGAN.CLASSIFY_IMAGE_SIZE[1], 3], name='test_sample_data')

        # noises
        self.noise_data = tf.placeholder(tf.float32, shape=[cfg.METAGAN.TRAIN.BATCH_SIZE, cfg.METAGAN.NOISE_DIM],
                                         name='noise_data')

        self.layers = {}
        self.inputs = []
        self.setup()



    def build_loss(self):
        # get loss for discriminator
        d_loss = self.discriminator_loss(self.layers['D_real_logits'], self.layers['D_fake_logits'])

        # get loss for generator
        g_loss = self.generator_loss(self.layers['D_fake_logits'])
        return d_loss, g_loss


    def setup(self):
        label_batch = []

        sample_generator_encoder = self.generator_cnn_encoder(self.sample_data)
        (
            self.feed(sample_generator_encoder, self.noise_data)
                .concat(axis=1, name='sample_generate_data')
        )
        G = self.generator(self.get_output('sample_generate_data'), label_batch)

        # output of D for real images
        D_real, D_real_logits = self.discriminator(self.query_data, self.query_label, reuse=False)

        D_fake, D_fake_logits = self.discriminator(G, label_batch, reuse=True)

        self.layers['D_real'] = D_real
        self.layers['D_real_logits'] = D_real_logits
        self.layers['D_fake'] = D_fake
        self.layers['D_fake_logits'] = D_fake_logits

    def test_net(self):
        label_batch = []

        test_sample_generator_encoder = self.generator_cnn_encoder(self.test_sample_data)
        (
            self.feed(test_sample_generator_encoder, self.noise_data)
                .concat(axis=1, name='test_sample_generate_data')
        )
        G = self.generator(self.get_output('test_sample_generate_data'), label_batch)
        return G

    def generator_cnn_encoder(self, data, reuse=False, bn_trainable = True):
        with tf.variable_scope('generator_cnn_encoder') as scope:

            if reuse:
                scope.reuse_variables()
            (
                self.feed(data)
                    .conv(3, 3, 64, 1, 1, name='conv1', relu=False, padding='VALID', trainable=True)
                    .batch_normalization(name='bn1', relu=True, trainable=bn_trainable)
                    .max_pool(2, 2, 2, 2, name='pool1', padding='VALID')
                    .conv(3, 3, 64, 1, 1, name='conv2', relu=False, padding='VALID', trainable=True)
                    .batch_normalization(name='bn2', relu=True, trainable=bn_trainable)
                    .max_pool(2, 2, 2, 2, name='pool2', padding='VALID')
                    .conv(3, 3, 64, 1, 1, name='conv3', relu=False, padding='SAME', trainable=True)
                    .batch_normalization(name='bn3', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 64, 1, 1, name='conv4', relu=False, padding='SAME', trainable=True)
                    .batch_normalization(name='bn4', relu=True, trainable=bn_trainable)
            )

        return self.get_output('bn4')



    def discriminator(self, data, label, reuse = False, bn_trainable = True):
        with tf.variable_scope('discriminator') as scope:

            init_GAN_weights = True
            spectral_norm = False
            if reuse:
                scope.reuse_variables()

            # label = tf.one_hot(label, depth=cls_num)
            (
                self.feed(label)
                    .reshape_layer([-1, 1, 1, 10], name='d_reshape_label')
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
            if reuse:
                scope.reuse_variables()

            # label = tf.one_hot(label, depth=cls_num)
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
                    .deconv(filter_size=256, kernel=[5, 5], stride=2, name='g_upconv1', biased=True, relu=True,
                            trainable=True)
                    .batch_normalization(name='g_bn1', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('g_bn1', 'g_reshape_label')
                    .conv_concat(name='g_conv_concat_2')
                    .deconv(filter_size=128, kernel=[5, 5], stride=2, name='g_upconv2', biased=True, relu=True,
                            trainable=True)
                    .batch_normalization(name='g_bn2', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('g_bn2', 'g_reshape_label')
                    .conv_concat(name='g_conv_concat_3')
                    .deconv(filter_size=cfg.GAN.IMAGE_DIM, kernel=[5, 5], stride=2, name='g_upconv3', biased=True,
                            relu=True, trainable=True)
                    .batch_normalization(name='g_bn3', relu=False, trainable=bn_trainable)
                    .tanh(name='generate_data')
            )
        return self.get_output('generate_data')

    def discriminator_loss(self, D_real_logits, D_fake_logits):
        # get loss for discriminator
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
        d_loss = d_loss_real + d_loss_fake
        return d_loss

    def generator_loss(self, D_fake_logits):
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))

        return g_loss