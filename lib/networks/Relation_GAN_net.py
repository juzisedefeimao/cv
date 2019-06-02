import tensorflow as tf
from lib.networks.network import Network
from lib.networks.netconfig import cfg
cls_num = cfg.RELATION_NET.TRAIN.CLASSIFY_NUM

class Relation_Network(Network) :
    def __init__(self):
        self.sample_data = tf.placeholder(tf.float32,
                                   shape=[cfg.RELATION_NET.TRAIN.CLASSIFY_SAMPLE * cls_num,
                                          cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE[0],
                                          cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE[1], 3], name='sample_data')
        self.query_data = tf.placeholder(tf.float32,
                                   shape=[cfg.RELATION_NET.TRAIN.CLASSIFY_QUERY * cls_num,
                                          cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE[0],
                                          cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE[1], 3], name='query_data')
        self.sample_label = tf.placeholder(tf.int32, shape=[cls_num],
                                           name='sample_label')
        self.query_label = tf.placeholder(tf.int32, shape=[cfg.RELATION_NET.TRAIN.CLASSIFY_QUERY * cls_num],
                                          name='query_label')

        self.test_sample_data = tf.placeholder(tf.float32,
                                        shape=[cfg.RELATION_NET.TRAIN.CLASSIFY_SAMPLE * cls_num,
                                               cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE[0],
                                               cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE[1], 3], name='test_sample_data')
        self.test_query_data = tf.placeholder(tf.float32,
                                               shape=[cfg.RELATION_NET.TRAIN.CLASSIFY_QUERY * cls_num,
                                                      cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE[0],
                                                      cfg.RELATION_NET.CLASSIFY_IMAGE_SIZE[1], 3],
                                               name='test_query_data')
        self.test_sample_label = tf.placeholder(tf.int32, shape=[cls_num],
                                                name='test_sample_label')
        self.test_query_label = tf.placeholder(tf.int32, shape=[cfg.RELATION_NET.TRAIN.CLASSIFY_QUERY * cls_num],
                                               name='test_query_label')
        # noises
        self.noise_data = tf.placeholder(tf.float32, shape=[cfg.GAN.TRAIN.BATCH_SIZE, cfg.GAN.NOISE_DIM],
                                         name='noise_data')

        self.layers = {}
        self.inputs = []
        self.setup()

    def get_label(self, sample_label, query_label):
        query_label = tf.reshape(query_label, (-1,1))
        label_bool = tf.reshape(tf.equal(query_label, sample_label), (-1, 1))
        label = tf.cast(label_bool, dtype=tf.float32)

        return label

    def get_pre_label(self, scores):
        split_list = tf.split(scores, num_or_size_splits=self.test_query_label.get_shape()[0])
        pre_label = []
        for split in split_list:
            pre_label.append(tf.argmax(split))

        pre_label = tf.stack(pre_label, axis=0)

        return pre_label

    def get_test_label(self, sample_label, query_label):
        query_label = tf.reshape(query_label, (-1, 1))
        label_bool = tf.equal(query_label, sample_label)
        label = tf.argmax(tf.cast(label_bool, dtype=tf.float32), axis=1)
        label = tf.reshape(label, (-1,1))

        return label

    def build_loss(self):
        label = self.get_label(self.sample_label, self.query_label)
        loss = tf.reduce_mean(tf.squared_difference(self.layers['scores'], label))
        return loss

    def test_net(self):
        test_sample_encoder = self.classifier_cnn_encoder(self.test_sample_data, reuse=True)
        test_query_encoder = self.classifier_cnn_encoder(self.test_query_data, reuse=True)
        (
            self.feed(test_sample_encoder, test_query_encoder)
                .relation_concat(C_WAY=cls_num, name='test_concattion')
        )
        scores = self.relation_net(self.get_output('test_concattion'), reuse=True)

        label = self.get_test_label(self.test_sample_label, self.test_query_label)
        pre_label = self.get_pre_label(scores)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(label, pre_label), dtype=tf.float32))

        return accuracy

    def setup(self):
        label_batch = []

        sample_classifier_encoder = self.classifier_cnn_encoder(self.sample_data)
        query_encoder = self.classifier_cnn_encoder(self.query_data, reuse=True)
        (
            self.feed(sample_classifier_encoder, query_encoder)
                .relation_concat(C_WAY=cls_num, name='concattion')
        )
        C_real_logits = self.relation_net(self.get_output('concattion'))
        self.layers['C_real_logits'] = C_real_logits

        sample_generator_encoder = self.generator_cnn_encoder(self.sample_data)
        (
            self.feed(sample_generator_encoder, self.noise_data)
                .concat(axis=1, name='sample_generate_data')
        )
        G = self.generator(self.get_output('sample_generate_data'), label_batch)


        # output of D for real images
        D_real, D_real_logits = self.discriminator(self.query_data, self.label_one_hot, reuse=False)

        # output of D for fake images
        G = self.generator(self.noise_data, self.label_one_hot, reuse=False)
        D_fake, D_fake_logits = self.discriminator(G, self.label_one_hot, reuse=True)

        # output of C for real images
        C_real_logits = self.classifier(self.real_data, reuse=False)

        # output of D for unlabelled images
        C_label_logist = self.classifier(self.unlabel_data, reuse=True)
        # C_label = tf.reshape(tf.cast(tf.argmax(C_label_logist, axis=1), tf.int32), [-1])
        C_label_one_hot = C_label_logist
        D_cla, D_cla_logits = self.discriminator(self.unlabel_data, C_label_one_hot, reuse=True)

        # output of C for fake images
        C_fake_logits = self.classifier(G, reuse=True)


    def classifier_cnn_encoder(self, data, reuse=False, bn_trainable = True):
        with tf.variable_scope('cnn_encoder') as scope:

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

    def generator_cnn_encoder(self, data, reuse=False, bn_trainable = True):
        with tf.variable_scope('cnn_encoder') as scope:

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

    def relation_net(self, data, reuse=False, bn_trainable = True):
        with tf.variable_scope('relation_net') as scope:

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
                    .fc(8, name='fc1', relu=True, trainable=True)
                    .fc(1, name='fc2', relu=False, trainable=True)
                    .sigmoid(name='score')
            )

        return self.get_output('score')

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

class TripleGAN(Network) :
    def __init__(self):

        self.alpha_p = tf.placeholder(tf.float32, name='alpha_p')
        """ Graph Input """
        # images
        self.real_data = tf.placeholder(tf.float32, shape=[cfg.GAN.TRAIN.BATCH_SIZE, cfg.GAN.IMAGE_SIZE,
                                                           cfg.GAN.IMAGE_SIZE, cfg.GAN.IMAGE_DIM], name='real_images')
        self.unlabel_data = tf.placeholder(tf.float32, shape=[cfg.GAN.TRAIN.UNLABEL_BATCH_SIZE, cfg.GAN.IMAGE_SIZE,
                                                              cfg.GAN.IMAGE_SIZE, cfg.GAN.IMAGE_DIM],
                                           name='unlabel_data')
        self.test_data = tf.placeholder(tf.float32, shape=[cfg.GAN.TRAIN.TEST_BATCH_SIZE, cfg.GAN.IMAGE_SIZE,
                                                           cfg.GAN.IMAGE_SIZE, cfg.GAN.IMAGE_DIM], name='test_images')

        # labels
        self.label = tf.placeholder(tf.int32, shape=[cfg.GAN.TRAIN.BATCH_SIZE],
                                    name='label')
        self.unlabel = tf.placeholder(tf.int32, shape=[cfg.GAN.TRAIN.UNLABEL_BATCH_SIZE],
                                      name='unlabel')
        self.test_label = tf.placeholder(tf.int32, shape=[cfg.GAN.TRAIN.TEST_BATCH_SIZE],
                                           name='test_label')
        self.visual_label = tf.placeholder(tf.int32, [cfg.GAN.TRAIN.SAVE_IMAGE_NUM],
                                           name='visual_label')
        self.label_one_hot = tf.one_hot(self.label, depth=cls_num)
        self.unlabel_one_hot = tf.one_hot(self.unlabel, depth=cls_num)
        self.test_label_one_hot = tf.one_hot(self.test_label, depth=cls_num)
        self.visual_label_one_hot = tf.one_hot(self.visual_label, depth=cls_num)

        # noises
        self.noise_data = tf.placeholder(tf.float32, shape=[cfg.GAN.TRAIN.BATCH_SIZE, cfg.GAN.NOISE_DIM],
                                         name='noise_data')
        self.visual_noise = tf.placeholder(tf.float32, shape=[cfg.GAN.TRAIN.SAVE_IMAGE_NUM, cfg.GAN.NOISE_DIM],
                                           name='visual_noise')
        self.layers = {}
        self.inputs = []
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

    def setup(self):

        # output of D for real images
        D_real, D_real_logits = self.discriminator(self.real_data, self.label_one_hot, reuse=False)

        # output of D for fake images
        G = self.generator(self.noise_data, self.label_one_hot, reuse=False)
        D_fake, D_fake_logits = self.discriminator(G, self.label_one_hot, reuse=True)

        # output of C for real images
        C_real_logits = self.classifier(self.real_data, reuse=False)

        # output of D for unlabelled images
        C_label_logist = self.classifier(self.unlabel_data, reuse=True)
        # C_label = tf.reshape(tf.cast(tf.argmax(C_label_logist, axis=1), tf.int32), [-1])
        C_label_one_hot = C_label_logist
        D_cla, D_cla_logits = self.discriminator(self.unlabel_data, C_label_one_hot, reuse=True)

        # output of C for fake images
        C_fake_logits = self.classifier(G, reuse=True)

        self.layers['C_real_logits'] = C_real_logits
        self.layers['C_fake_logits'] = C_fake_logits
        self.layers['C_label_logist'] = C_label_logist
        self.layers['C_label'] = C_label_one_hot
        self.layers['D_real'] = D_real
        self.layers['D_real_logits'] = D_real_logits
        self.layers['D_fake'] = D_fake
        self.layers['D_fake_logits'] = D_fake_logits
        self.layers['D_cla'] = D_cla
        self.layers['D_cla_logits'] = D_cla_logits

    def test_classifier(self):
        # test loss for classify
        test_pred_label = self.classifier(self.test_data, bn_trainable=False, reuse=True)
        correct_prediction = tf.equal(tf.argmax(test_pred_label, axis=1), tf.argmax(self.test_label_one_hot, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy

    def generate_image(self):
        fake_images = self.generator(self.visual_noise, self.visual_label_one_hot, bn_trainable=False, reuse=True)

        return fake_images

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

    def generator_(self, data, label, reuse = False, bn_trainable=True):
        with tf.variable_scope('generator') as scope:
            spectral_norm = False
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
        d_loss_fake = (1 - 0.5) * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
        d_loss_cla = 0.5 * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_cla_logits, labels=tf.zeros_like(D_cla_logits)))
        d_loss = d_loss_real + d_loss_fake + d_loss_cla
        return d_loss

    def generator_loss(self, D_fake_logits):
        g_loss = (1 - 0.5) * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))

        return g_loss



    def classifier_loss(self, C_label_logist, D_cla_logits, C_real_logits, C_fake_logits, alpha_p):
        # get loss for classify
        max_c = tf.cast(tf.reduce_max(C_label_logist, axis=1), tf.float32)
        c_loss_dis = tf.reduce_mean(
            max_c * tf.nn.sigmoid_cross_entropy_with_logits(logits=D_cla_logits, labels=tf.ones_like(D_cla_logits)))
        R_L = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label_one_hot, logits=C_real_logits))
        R_P = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label_one_hot, logits=C_fake_logits))
        c_loss = cfg.GAN.TRAIN.ALPHA_CLA_ADV * cfg.GAN.TRAIN.ALPHA * c_loss_dis + R_L + alpha_p * R_P

        return c_loss