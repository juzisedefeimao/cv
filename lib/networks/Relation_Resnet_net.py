import tensorflow as tf
from lib.networks.network import Network
from lib.networks.netconfig import cfg
cls_num = cfg.RELATION_NET.TRAIN.CLASSIFY_NUM

class Relation_Resnet_Network(Network) :
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
        test_sample_encoder = self.cnn_encoder(self.test_sample_data, reuse=True)
        test_query_encoder = self.cnn_encoder(self.test_query_data, reuse=True)
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
        sample_encoder = self.cnn_encoder(self.sample_data)
        query_encoder = self.cnn_encoder(self.query_data, reuse=True)
        print(query_encoder)
        (
            self.feed(sample_encoder, query_encoder)
                .relation_concat(C_WAY=cls_num, name='concattion')
        )
        scores = self.relation_net(self.get_output('concattion'))
        self.layers['scores'] = scores



    def relation_net_(self, data, reuse=False, bn_trainable = True):
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

    def cnn_encoder_(self, data, reuse=False, bn_trainable = True):
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
            # ======================计算残差变换形状适应第四组模块=======================
            # (
            #     self.feed(data)
            #         .conv(1, 1, 2048, 2, 2, name='transform4_conv', relu=False, trainable=True)
            #         .batch_normalization(name='transform4_bn', relu=False, trainable=bn_trainable)
            # )
            # # ======================第四组模块===========================
            # (
            #     self.feed(data)
            #         .conv(1, 1, 512, 2, 2, name='res4_1_conv1', relu=False, trainable=True)
            #         .batch_normalization(name='res4_1_bn1', relu=True, trainable=bn_trainable)
            #         .conv(3, 3, 512, 1, 1, name='res4_1_conv2', relu=False, trainable=True)
            #         .batch_normalization(name='res4_1_bn2', relu=True, trainable=bn_trainable)
            #         .conv(1, 1, 2048, 1, 1, name='res4_1_conv3', relu=False, trainable=True)
            #         .batch_normalization(name='res4_1_bn3', relu=True, trainable=bn_trainable)
            # )
            # (
            #     self.feed('transform4_bn', 'res4_1_bn3')
            #         .add(name='res4_1_add')
            #         .relu(name='res4_1_relu')
            #         .conv(1, 1, 512, 1, 1, name='res4_2_conv1', relu=False, trainable=True)
            #         .batch_normalization(name='res4_2_bn1', relu=True, trainable=bn_trainable)
            #         .conv(3, 3, 512, 1, 1, name='res4_2_conv2', relu=False, trainable=True)
            #         .batch_normalization(name='res4_2_bn2', relu=True, trainable=bn_trainable)
            #         .conv(1, 1, 2048, 1, 1, name='res4_2_conv3', relu=False, trainable=True)
            #         .batch_normalization(name='res4_2_bn3', relu=False, trainable=bn_trainable)
            # )
            # (
            #     self.feed('transform4_bn', 'res4_2_bn3')
            #         .add(name='res4_2_add')
            #         .relu(name='res4_2_relu')
            #         .conv(1, 1, 512, 1, 1, name='res4_3_conv1', relu=False, trainable=True)
            #         .batch_normalization(name='res4_3_bn1', relu=True, trainable=bn_trainable)
            #         .conv(3, 3, 512, 1, 1, name='res4_3_conv2', relu=False, trainable=True)
            #         .batch_normalization(name='res4_3_bn2', relu=True, trainable=bn_trainable)
            #         .conv(1, 1, 2048, 1, 1, name='res4_3_conv3', relu=False, trainable=True)
            #         .batch_normalization(name='res4_3_bn3', relu=False, trainable=bn_trainable)
            # )
            # # ======================计算残差变换结束模块=======================
            # (
            #     self.feed('transform4_bn', 'res4_3_bn3')
            #         .add(name='res4_3_add')
            #         .relu(name='res4_3_relu')
            #         .fc(8, name='fc1', relu=True, trainable=True)
            #         .fc(1, name='fc2', relu=False, trainable=True)
            #         .sigmoid(name='score')
            # )
            (
                self.feed(data)
                    .conv(3, 3, 64, 1, 1, name='conv1', relu=False, trainable=True)
                    .batch_normalization(name='bn1', relu=True, trainable=bn_trainable)
                    .max_pool(2, 2, 2, 2, name='pool1', padding='SAME')
                    .conv(3, 3, 64, 1, 1, name='conv2', relu=False, trainable=True)
                    .batch_normalization(name='bn2', relu=True, trainable=bn_trainable)
                    .max_pool(2, 2, 2, 2, name='pool2', padding='SAME')
                    .fc(8, name='fc1', relu=True, trainable=True)
                    .fc(1, name='fc2', relu=False, trainable=True)
                    .sigmoid(name='score')
            )

        return self.get_output('score')

    def cnn_encoder(self, data, reuse=False, bn_trainable = True):
        with tf.variable_scope('cnn_encoder') as scope:

            if reuse:
                scope.reuse_variables()
            (
                self.feed(data)
                    .conv(7, 7, 64, 2, 2, name='conv1', relu=False, trainable=True)
                    .batch_normalization(name='bn1', relu=True, trainable=bn_trainable)
                    .max_pool(3, 3, 2, 2, name='pool1', padding='VALID')
            )
            # ======================变换形状适应第一组模块=======================
            (
                self.feed('pool1')
                    .conv(1, 1, 256, 1, 1, name='transform1_conv', relu=False, trainable=True)
                    .batch_normalization(name='transform1_bn', relu=False, trainable=bn_trainable)
            )
            # ======================第一组模块===========================
            (
                self.feed('pool1')
                    .conv(1, 1, 64, 1, 1, name='res1_1_conv1', relu=False, trainable=True)
                    .batch_normalization(name='res1_1_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 64, 1, 1, name='res1_1_conv2', relu=False, trainable=True)
                    .batch_normalization(name='res1_1_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 256, 1, 1, name='res1_1_conv3', relu=False, trainable=True)
                    .batch_normalization(name='res1_1_bn3', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('transform1_bn', 'res1_1_bn3')
                    .add(name='res1_1_add')
                    .relu(name='res1_1_relu')
                    .conv(1, 1, 64, 1, 1, name='res1_2_conv1', relu=False, trainable=True)
                    .batch_normalization(name='res1_2_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 64, 1, 1, name='res1_2_conv2', relu=False, trainable=True)
                    .batch_normalization(name='res1_2_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 256, 1, 1, name='res1_2_conv3', relu=False, trainable=True)
                    .batch_normalization(name='res1_2_bn3', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('transform1_bn', 'res1_2_bn3')
                    .add(name='res1_2_add')
                    .relu(name='res1_2_relu')
                    .conv(1, 1, 64, 1, 1, name='res1_3_conv1', relu=False, trainable=True)
                    .batch_normalization(name='res1_3_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 64, 1, 1, name='res1_3_conv2', relu=False, trainable=True)
                    .batch_normalization(name='res1_3_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 256, 1, 1, name='res1_3_conv3', relu=False, trainable=True)
                    .batch_normalization(name='res1_3_bn3', relu=False, trainable=bn_trainable)
            )
            # ======================计算残差变换形状适应第二组模块=======================
            (
                self.feed('transform1_bn', 'res1_3_bn3')
                    .add(name='res1_3_add')
                    .relu(name='res1_3_relu')
                    .conv(1, 1, 512, 2, 2, name='transform2_conv', relu=False, trainable=True)
                    .batch_normalization(name='transform2_bn', relu=False, trainable=bn_trainable)
            )
            # ======================第二组模块===========================
            (
                self.feed('res1_3_relu')
                    .conv(1, 1, 128, 2, 2, name='res2_1_conv1', relu=False, trainable=True)
                    .batch_normalization(name='res2_1_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 128, 1, 1, name='res2_1_conv2', relu=False, trainable=True)
                    .batch_normalization(name='res2_1_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 512, 1, 1, name='res2_1_conv3', relu=False, trainable=True)
                    .batch_normalization(name='res2_1_bn3', relu=True, trainable=bn_trainable)
            )
            (
                self.feed('transform2_bn', 'res2_1_bn3')
                    .add(name='res2_1_add')
                    .relu(name='res2_1_relu')
                    .conv(1, 1, 128, 1, 1, name='res2_2_conv1', relu=False, trainable=True)
                    .batch_normalization(name='res2_2_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 128, 1, 1, name='res2_2_conv2', relu=False, trainable=True)
                    .batch_normalization(name='res2_2_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 512, 1, 1, name='res2_2_conv3', relu=False, trainable=True)
                    .batch_normalization(name='res2_2_bn3', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('transform2_bn', 'res2_2_bn3')
                    .add(name='res2_2_add')
                    .relu(name='res2_2_relu')
                    .conv(1, 1, 128, 1, 1, name='res2_3_conv1', relu=False, trainable=True)
                    .batch_normalization(name='res2_3_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 128, 1, 1, name='res2_3_conv2', relu=False, trainable=True)
                    .batch_normalization(name='res2_3_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 512, 1, 1, name='res2_3_conv3', relu=False, trainable=True)
                    .batch_normalization(name='res2_3_bn3', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('transform2_bn', 'res2_3_bn3')
                    .add(name='res2_3_add')
                    .relu(name='res2_3_relu')
                    .conv(1, 1, 128, 1, 1, name='res2_4_conv1', relu=False, trainable=True)
                    .batch_normalization(name='res2_4_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 128, 1, 1, name='res2_4_conv2', relu=False, trainable=True)
                    .batch_normalization(name='res2_4_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 512, 1, 1, name='res2_4_conv3', relu=False, trainable=True)
                    .batch_normalization(name='res2_4_bn3', relu=False, trainable=bn_trainable)
            )
            # ======================计算残差变换形状适应第三组模块=======================
            (
                self.feed('transform2_bn', 'res2_4_bn3')
                    .add(name='res2_4_add')
                    .relu(name='res2_4_relu')
                    .conv(1, 1, 1024, 2, 2, name='transform3_conv', relu=False, trainable=True)
                    .batch_normalization(name='transform3_bn', relu=False, trainable=bn_trainable)
            )
            # ======================第三组模块===========================
            (
                self.feed('res2_4_relu')
                    .conv(1, 1, 256, 2, 2, name='res3_1_conv1', relu=False, trainable=True)
                    .batch_normalization(name='res3_1_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 256, 1, 1, name='res3_1_conv2', relu=False, trainable=True)
                    .batch_normalization(name='res3_1_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 1024, 1, 1, name='res3_1_conv3', relu=False, trainable=True)
                    .batch_normalization(name='res3_1_bn3', relu=True, trainable=bn_trainable)
            )
            (
                self.feed('transform3_bn', 'res3_1_bn3')
                    .add(name='res3_1_add')
                    .relu(name='res3_1_relu')
                    .conv(1, 1, 256, 1, 1, name='res3_2_conv1', relu=False, trainable=True)
                    .batch_normalization(name='res3_2_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 256, 1, 1, name='res3_2_conv2', relu=False, trainable=True)
                    .batch_normalization(name='res3_2_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 1024, 1, 1, name='res3_2_conv3', relu=False, trainable=True)
                    .batch_normalization(name='res3_2_bn3', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('transform3_bn', 'res3_2_bn3')
                    .add(name='res3_2_add')
                    .relu(name='res3_2_relu')
                    .conv(1, 1, 256, 1, 1, name='res3_3_conv1', relu=False, trainable=True)
                    .batch_normalization(name='res3_3_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 256, 1, 1, name='res3_3_conv2', relu=False, trainable=True)
                    .batch_normalization(name='res3_3_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 1024, 1, 1, name='res3_3_conv3', relu=False, trainable=True)
                    .batch_normalization(name='res3_3_bn3', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('transform3_bn', 'res3_3_bn3')
                    .add(name='res3_3_add')
                    .relu(name='res3_3_relu')
                    .conv(1, 1, 256, 1, 1, name='res3_4_conv1', relu=False, trainable=True)
                    .batch_normalization(name='res3_4_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 256, 1, 1, name='res3_4_conv2', relu=False, trainable=True)
                    .batch_normalization(name='res3_4_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 1024, 1, 1, name='res3_4_conv3', relu=False, trainable=True)
                    .batch_normalization(name='res3_4_bn3', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('transform3_bn', 'res3_4_bn3')
                    .add(name='res3_4_add')
                    .relu(name='res3_4_relu')
                    .conv(1, 1, 256, 1, 1, name='res3_5_conv1', relu=False, trainable=True)
                    .batch_normalization(name='res3_5_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 256, 1, 1, name='res3_5_conv2', relu=False, trainable=True)
                    .batch_normalization(name='res3_5_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 1024, 1, 1, name='res3_5_conv3', relu=False, trainable=True)
                    .batch_normalization(name='res3_5_bn3', relu=False, trainable=bn_trainable)
            )
            (
                self.feed('transform3_bn', 'res3_5_bn3')
                    .add(name='res3_5_add')
                    .relu(name='res3_5_relu')
                    .conv(1, 1, 256, 1, 1, name='res3_6_conv1', relu=False, trainable=True)
                    .batch_normalization(name='res3_6_bn1', relu=True, trainable=bn_trainable)
                    .conv(3, 3, 256, 1, 1, name='res3_6_conv2', relu=False, trainable=True)
                    .batch_normalization(name='res3_6_bn2', relu=True, trainable=bn_trainable)
                    .conv(1, 1, 1024, 1, 1, name='res3_6_conv3', relu=False, trainable=True)
                    .batch_normalization(name='res3_6_bn3', relu=False, trainable=bn_trainable)
            )
            # # ======================计算残差变换形状适应第四组模块=======================
            # (
            #     self.feed('transform3_bn', 'res3_6_bn3')
            #         .add(name='res3_6_add')
            #         .relu(name='res3_6_relu')
            #         .conv(1, 1, 2048, 2, 2, name='transform4_conv', relu=False, trainable=True)
            #         .batch_normalization(name='transform4_bn', relu=False, trainable=bn_trainable)
            # )
            # # ======================第四组模块===========================
            # (
            #     self.feed('res3_6_relu')
            #         .conv(1, 1, 512, 2, 2, name='res4_1_conv1', relu=False, trainable=True)
            #         .batch_normalization(name='res4_1_bn1', relu=True, trainable=bn_trainable)
            #         .conv(3, 3, 512, 1, 1, name='res4_1_conv2', relu=False, trainable=True)
            #         .batch_normalization(name='res4_1_bn2', relu=True, trainable=bn_trainable)
            #         .conv(1, 1, 2048, 1, 1, name='res4_1_conv3', relu=False, trainable=True)
            #         .batch_normalization(name='res4_1_bn3', relu=True, trainable=bn_trainable)
            # )
            # (
            #     self.feed('transform4_bn', 'res4_1_bn3')
            #         .add(name='res4_1_add')
            #         .relu(name='res4_1_relu')
            #         .conv(1, 1, 512, 1, 1, name='res4_2_conv1', relu=False, trainable=True)
            #         .batch_normalization(name='res4_2_bn1', relu=True, trainable=bn_trainable)
            #         .conv(3, 3, 512, 1, 1, name='res4_2_conv2', relu=False, trainable=True)
            #         .batch_normalization(name='res4_2_bn2', relu=True, trainable=bn_trainable)
            #         .conv(1, 1, 2048, 1, 1, name='res4_2_conv3', relu=False, trainable=True)
            #         .batch_normalization(name='res4_2_bn3', relu=False, trainable=bn_trainable)
            # )
            # (
            #     self.feed('transform4_bn', 'res4_2_bn3')
            #         .add(name='res4_2_add')
            #         .relu(name='res4_2_relu')
            #         .conv(1, 1, 512, 1, 1, name='res4_3_conv1', relu=False, trainable=True)
            #         .batch_normalization(name='res4_3_bn1', relu=True, trainable=bn_trainable)
            #         .conv(3, 3, 512, 1, 1, name='res4_3_conv2', relu=False, trainable=True)
            #         .batch_normalization(name='res4_3_bn2', relu=True, trainable=bn_trainable)
            #         .conv(1, 1, 2048, 1, 1, name='res4_3_conv3', relu=False, trainable=True)
            #         .batch_normalization(name='res4_3_bn3', relu=False, trainable=bn_trainable)
            # )
            # ======================计算残差变换结束模块=======================
            (
                self.feed('transform3_bn', 'res3_6_bn3')
                    .add(name='res3_6_add')
                    .relu(name='res3_6_relu')
            )

        return self.get_output('res3_6_relu')