import numpy as np
import os
import tensorflow as tf
import cv2
from PIL import Image
from time import strftime
from lib.roi_data_layer.layer import RoIDataLayer
from lib.utils.timer import Timer
from lib.gt_data_layer import roidb as gdl_roidb
from lib.roi_data_layer import roidb as rdl_roidb
from lib.networks.factory import get_network

# >>>> obsolete, because it depends on sth outside of this project
from lib.networks.netconfig import cfg
from lib.rpn_msr.bbox_transform import clip_boxes, bbox_transform_inv
# <<<< obsolete

# _DEBUG = False

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, imdb, roidb, rpn_graph, detect_graph, shared_conv_graph,
                 rpn_output_dir, detect_output_dir, shared_conv_rpn_output_dir, shared_conv_detect_output_dir,
                 train_step='train_step_rpn', restore=False, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.imdb = imdb
        self.roidb = roidb
        self.rpn_graph = rpn_graph
        self.detect_graph = detect_graph
        self.shared_conv_graph = shared_conv_graph
        self.rpn_output_dir = rpn_output_dir
        self.detect_output_dir = detect_output_dir
        self.shared_conv_rpn_output_dir = shared_conv_rpn_output_dir
        self.shared_conv_detect_output_dir = shared_conv_detect_output_dir
        self.train_step = train_step
        self.restore = restore
        self.pretrained_model = pretrained_model
        self.train_op = {} # 训练的操作节点，在会话中使用，比如 rpn_train_op rpn_init

        print ('Computing bounding-box regression targets...')
        if cfg.ZLRM.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print ('done')

        with self.rpn_graph.as_default():
            self.rpn_restore_saver = tf.train.Saver()
        with self.detect_graph.as_default():
            self.detect_restore_saver = tf.train.Saver()
        with self.shared_conv_graph.as_default():
            if self.restore:
                self.shared_conv_rpn_restore_saver = tf.train.Saver()
                self.shared_conv_detect_restore_saver = tf.train.Saver()
            elif self.train_step == 'train_step_shared_conv_rpn' or self.train_step == 'train_step_shared_conv_detect':
                tvars = tf.all_variables()
                # 从
                detect_ckpt = tf.train.get_checkpoint_state(self.detect_output_dir)
                reader = tf.train.NewCheckpointReader(detect_ckpt.model_checkpoint_path)
                detect_variables = reader.get_variable_to_shape_map()
                tvars_detect = [v for v in tvars if (v.name.split(':')[0] in detect_variables)]
                print('tvars_detect', tvars_detect)
                self.shared_conv_rpn_restore_detect_fraction_saver = tf.train.Saver(var_list=tvars_detect)

                tvars_rpn = [v for v in tvars if (v.name.split('/')[0] == 'rpn_conv' or
                                              v.name.split('/')[0] == 'rpn_cls_score' or
                                              v.name.split('/')[0] == 'rpn_bbox_pred'
                                              )]
                print('tvars_rpn', tvars_rpn)
                self.shared_conv_rpn_restore_rpn_fraction_saver = tf.train.Saver(var_list=tvars_rpn)

                self.shared_conv_rpn_restore_saver = tf.train.Saver()

                self.shared_conv_detect_restore_saver = tf.train.Saver()


    def snapshot(self, sess, network, restore_saver, iter, output_dir):
        """Take a snapshot of the networks after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = network

        if cfg.ZLRM.TRAIN.BBOX_REG and ('bbox_pred' in net.layers) and cfg.ZLRM.TRAIN.BBOX_NORMALIZE_TARGETS:
            # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(weights.assign(orig_0 * np.tile(self.bbox_stds, (weights_shape[0],1))))
            sess.run(biases.assign(orig_1 * self.bbox_stds + self.bbox_means))

        # 保存rpn网络参数
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.ZLRM.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.ZLRM.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(output_dir, filename)
        restore_saver.save(sess, filename)
        print ('Wrote Resnet50_train_rpn snapshot to: {:s}'.format(filename))

        if cfg.ZLRM.TRAIN.BBOX_REG and ('bbox_pred' in net.layers):
            # restore net to original state
            sess.run(weights.assign(orig_0))
            sess.run(biases.assign(orig_1))

    def rpn_train_model(self, rpn_network, restore=False):

        rpn_loss, rpn_cross_entropy, rpn_loss_box = rpn_network.build_loss()
        self.train_op['rpn_loss'] = rpn_loss
        self.train_op['rpn_cross_entropy'] = rpn_cross_entropy
        self.train_op['rpn_loss_box'] = rpn_loss_box
        # optimizer
        if cfg.ZLRM.TRAIN.SOLVER == 'Adam':
            opt = tf.train.AdamOptimizer(cfg.ZLRM.TRAIN.LEARNING_RATE)
        elif cfg.ZLRM.TRAIN.SOLVER == 'RMS':
            opt = tf.train.RMSPropOptimizer(cfg.ZLRM.TRAIN.LEARNING_RATE)
        else:
            rpn_lr = tf.Variable(cfg.ZLRM.TRAIN.LEARNING_RATE, trainable=False)
            # lr = tf.Variable(0.0, trainable=False)
            momentum = cfg.ZLRM.TRAIN.MOMENTUM
            opt = tf.train.MomentumOptimizer(rpn_lr, momentum)
            self.train_op['rpn_lr'] = rpn_lr

        rpn_global_step = tf.Variable(0, trainable=False)
        self.train_op['rpn_global_step'] = rpn_global_step
        with_clip = True
        if with_clip:
            tvars = tf.trainable_variables()
            print('所有的训练参数', tvars)
            grads, norm = tf.clip_by_global_norm(tf.gradients(rpn_loss, tvars), 10.0)
            rpn_train_op = opt.apply_gradients(zip(grads, tvars), global_step=rpn_global_step)
        else:
            tvars = tf.trainable_variables()
            print('所有的训练参数', tvars)
            rpn_train_op = opt.minimize(rpn_loss, global_step=rpn_global_step, var_list=tvars)
        self.train_op['rpn_train_op'] = rpn_train_op

        # intialize variables
        rpn_init = tf.global_variables_initializer()
        self.train_op['rpn_init'] = rpn_init
        rpn_restore_iter = 0
        self.train_op['rpn_restore_iter'] = rpn_restore_iter
        self.train_op['rpn_restore_saver'] = self.rpn_restore_saver
        if restore:
            try:
                # rpn网络参数恢复
                print('从 ', self.rpn_output_dir, '恢复rpn网络')
                rpn_ckpt = tf.train.get_checkpoint_state(self.rpn_output_dir)
                self.train_op['rpn_ckpt'] = rpn_ckpt
                print ('Restoring from {}...'.format(rpn_ckpt.model_checkpoint_path),)
                stem = os.path.splitext(os.path.basename(rpn_ckpt.model_checkpoint_path))[0]
                rpn_restore_iter = int(stem.split('_')[-1])
                self.train_op['rpn_restore_iter'] = rpn_restore_iter
                print ('done')
            except:
                raise BaseException('Check your pretrained {:s}'.format(rpn_ckpt.model_checkpoint_path))

    def detect_train_model(self, detect_network, restore=False):

        detect_loss, detect_cross_entropy, detect_loss_box = detect_network.build_loss()
        self.train_op['detect_loss'] = detect_loss
        self.train_op['detect_cross_entropy'] = detect_cross_entropy
        self.train_op['detect_loss_box'] = detect_loss_box
        # optimizer
        if cfg.ZLRM.TRAIN.SOLVER == 'Adam':
            opt = tf.train.AdamOptimizer(cfg.ZLRM.TRAIN.LEARNING_RATE)
        elif cfg.ZLRM.TRAIN.SOLVER == 'RMS':
            opt = tf.train.RMSPropOptimizer(cfg.ZLRM.TRAIN.LEARNING_RATE)
        else:
            detect_lr = tf.Variable(cfg.ZLRM.TRAIN.LEARNING_RATE, trainable=False)
            # lr = tf.Variable(0.0, trainable=False)
            momentum = cfg.ZLRM.TRAIN.MOMENTUM
            opt = tf.train.MomentumOptimizer(detect_lr, momentum)
            self.train_op['detect_lr'] = detect_lr

        detect_global_step = tf.Variable(0, trainable=False)
        self.train_op['detect_global_step'] = detect_global_step
        with_clip = True
        if with_clip:
            tvars = tf.trainable_variables()
            print('所有的训练参数', tvars)
            grads, norm = tf.clip_by_global_norm(tf.gradients(detect_loss, tvars), 10.0)
            detect_train_op = opt.apply_gradients(zip(grads, tvars), global_step=detect_global_step)
        else:
            tvars = tf.trainable_variables()
            print('所有的训练参数', tvars)
            detect_train_op = opt.minimize(detect_loss, global_step=detect_global_step, var_list=tvars)
        self.train_op['detect_train_op'] = detect_train_op

        # intialize variables
        detect_init = tf.global_variables_initializer()
        self.train_op['detect_init'] = detect_init
        detect_restore_iter = 0
        self.train_op['detect_restore_iter'] = detect_restore_iter
        self.train_op['detect_restore_saver'] = self.detect_restore_saver
        if restore:
            try:
                # detect网络参数恢复
                print('从 ', self.detect_output_dir, '恢复detectn网络')
                detect_ckpt = tf.train.get_checkpoint_state(self.detect_output_dir)
                self.train_op['detect_ckpt'] = detect_ckpt
                print ('Restoring from {}...'.format(detect_ckpt.model_checkpoint_path),)
                stem = os.path.splitext(os.path.basename(detect_ckpt.model_checkpoint_path))[0]
                detect_restore_iter = int(stem.split('_')[-1])
                self.train_op['detect_restore_iter'] = detect_restore_iter
                print ('done')
            except:
                raise BaseException('Check your pretrained {:s}'.format(detect_ckpt.model_checkpoint_path))

    def shared_conv_rpn_train_model(self, shared_conv_network, restore=False):

        shared_conv_total_loss, shared_conv_detect_cross_entropy, shared_conv_detect_loss_box,\
        shared_conv_rpn_cross_entropy, shared_conv_rpn_loss_box = shared_conv_network.build_loss()
        self.train_op['shared_conv_total_loss'] = shared_conv_total_loss
        shared_conv_rpn_loss = shared_conv_rpn_cross_entropy + shared_conv_rpn_loss_box
        self.train_op['shared_conv_rpn_loss'] = shared_conv_rpn_loss
        self.train_op['shared_conv_rpn_cross_entropy'] = shared_conv_rpn_cross_entropy
        self.train_op['shared_conv_rpn_loss_box'] = shared_conv_rpn_loss_box
        shared_conv_detect_loss = shared_conv_detect_cross_entropy + shared_conv_detect_loss_box
        self.train_op['shared_conv_detect_loss'] = shared_conv_detect_loss
        self.train_op['shared_conv_detect_cross_entropy'] = shared_conv_detect_cross_entropy
        self.train_op['shared_conv_detect_loss_box'] = shared_conv_detect_loss_box

        # optimizer
        if cfg.ZLRM.TRAIN.SOLVER == 'Adam':
            opt = tf.train.AdamOptimizer(cfg.ZLRM.TRAIN.LEARNING_RATE)
        elif cfg.ZLRM.TRAIN.SOLVER == 'RMS':
            opt = tf.train.RMSPropOptimizer(cfg.ZLRM.TRAIN.LEARNING_RATE)
        else:
            shared_conv_rpn_lr = tf.Variable(cfg.ZLRM.TRAIN.LEARNING_RATE, trainable=False)
            # lr = tf.Variable(0.0, trainable=False)
            momentum = cfg.ZLRM.TRAIN.MOMENTUM
            opt = tf.train.MomentumOptimizer(shared_conv_rpn_lr, momentum)
            self.train_op['shared_conv_rpn_lr'] = shared_conv_rpn_lr

        shared_conv_rpn_global_step = tf.Variable(0, trainable=False)
        self.train_op['shared_conv_rpn_global_step'] = shared_conv_rpn_global_step
        with_clip = True
        change_conv = True #是否通过调整共享特征提取层来学习rpn提取，若否，则只通过调整rpn层来学习rpn提取
        if with_clip:
            tvars = tf.trainable_variables()
            print('shared conv rpn network 所有的可训练参数', tvars)
            if change_conv:
                tvars = [v for v in tvars if (v.name.split('_')[0] != 'transform4' and
                                              v.name.split('_')[0] != 'res4' and
                                              v.name.split('/')[0] != 'conv_new_1' and
                                              v.name.split('/')[0] != 'rfcn_cls' and
                                              v.name.split('/')[0] != 'rfcn_bbox'
                                              )]
            else:
                tvars = [v for v in tvars if (v.name.split('/')[0] == 'rpn_conv' or
                                              v.name.split('/')[0] == 'rpn_cls_score' or
                                              v.name.split('/')[0] == 'rpn_bbox_pred'
                                              )]
            print('shared conv rpn network 训练的参数', tvars)
            grads, norm = tf.clip_by_global_norm(tf.gradients(shared_conv_rpn_loss, tvars), 10.0)
            shared_conv_rpn_train_op = opt.apply_gradients(zip(grads, tvars), global_step=shared_conv_rpn_global_step)
        else:
            tvars = tf.trainable_variables()
            print('shared conv rpn network 所有的可训练参数', tvars)
            if change_conv:
                tvars = [v for v in tvars if (v.name.split('_')[0] != 'transform4' and
                                              v.name.split('_')[0] != 'res4' and
                                              v.name.split('/')[0] != 'conv_new_1' and
                                              v.name.split('/')[0] != 'rfcn_cls' and
                                              v.name.split('/')[0] != 'rfcn_bbox'
                                              )]
            else:
                tvars = [v for v in tvars if (v.name.split('/')[0] == 'rpn_conv' and
                                              v.name.split('/')[0] == 'rpn_cls_score' and
                                              v.name.split('/')[0] == 'rpn_bbox_pred'
                                              )]
            print('shared conv rpn network 训练的参数', tvars)
            shared_conv_rpn_train_op = opt.minimize(shared_conv_rpn_loss, global_step=shared_conv_rpn_global_step, var_list=tvars)
        self.train_op['shared_conv_rpn_train_op'] = shared_conv_rpn_train_op

        # intialize variables
        shared_conv_rpn_init = tf.global_variables_initializer()
        self.train_op['shared_conv_rpn_init'] = shared_conv_rpn_init
        shared_conv_rpn_restore_iter = 0
        self.train_op['shared_conv_rpn_restore_iter'] = shared_conv_rpn_restore_iter
        self.train_op['shared_conv_rpn_restore_saver'] = self.shared_conv_rpn_restore_saver

        if restore:
            try:
                # shared_conv网络参数恢复
                print('从 ', self.shared_conv_rpn_output_dir, '恢复rpn网络')
                shared_conv_rpn_ckpt = tf.train.get_checkpoint_state(self.shared_conv_rpn_output_dir)
                self.train_op['shared_conv_rpn_ckpt'] = shared_conv_rpn_ckpt
                print ('Restoring from {}...'.format(shared_conv_rpn_ckpt.model_checkpoint_path),)
                stem = os.path.splitext(os.path.basename(shared_conv_rpn_ckpt.model_checkpoint_path))[0]
                rpn_restore_iter = int(stem.split('_')[-1])
                self.train_op['shared_conv_rpn_restore_iter'] = rpn_restore_iter
                print ('done')
            except:
                raise BaseException('Check your pretrained {:s}'.format(shared_conv_rpn_ckpt.model_checkpoint_path))
        else:
            self.train_op['shared_conv_rpn_restore_rpn_fraction_saver'] = self.shared_conv_rpn_restore_rpn_fraction_saver
            self.train_op['shared_conv_rpn_restore_detect_fraction_saver'] = self.shared_conv_rpn_restore_detect_fraction_saver
            try:
                # shared_conv网络detect参数恢复
                print('从detect网络 ', self.detect_output_dir, '恢复shared conv network 的detect部分')
                shared_conv_rpn_restore_detect_fraction_ckpt = tf.train.get_checkpoint_state(self.detect_output_dir)
                self.train_op['shared_conv_rpn_restore_detect_fraction_ckpt'] = shared_conv_rpn_restore_detect_fraction_ckpt
                print ('Restoring from {}...'.format(shared_conv_rpn_restore_detect_fraction_ckpt.model_checkpoint_path),)
                print ('done')
            except:
                raise BaseException('Check your pretrained {:s}'.
                                    format(shared_conv_rpn_restore_detect_fraction_ckpt.model_checkpoint_path))
            try:
                # shared_conv网络rpn参数恢复
                print('从rpn网络 ', self.rpn_output_dir, '恢复shared conv network 的rpn部分')
                shared_conv_rpn_restore_rpn_fraction_ckpt = tf.train.get_checkpoint_state(self.rpn_output_dir)
                self.train_op['shared_conv_rpn_restore_rpn_fraction_ckpt'] = shared_conv_rpn_restore_rpn_fraction_ckpt
                print ('Restoring from {}...'.format(shared_conv_rpn_restore_rpn_fraction_ckpt.model_checkpoint_path),)
                print ('done')
            except:
                raise BaseException('Check your pretrained {:s}'.
                                    format(shared_conv_rpn_restore_rpn_fraction_ckpt.model_checkpoint_path))


    def shared_conv_detect_train_model(self, shared_conv_network, restore=False):

        shared_conv_total_loss, shared_conv_detect_cross_entropy, shared_conv_detect_loss_box, \
        shared_conv_rpn_cross_entropy, shared_conv_rpn_loss_box = shared_conv_network.build_loss()
        self.train_op['shared_conv_total_loss'] = shared_conv_total_loss
        shared_conv_rpn_loss = shared_conv_rpn_cross_entropy + shared_conv_rpn_loss_box
        self.train_op['shared_conv_rpn_loss'] = shared_conv_rpn_loss
        self.train_op['shared_conv_rpn_cross_entropy'] = shared_conv_rpn_cross_entropy
        self.train_op['shared_conv_rpn_loss_box'] = shared_conv_rpn_loss_box
        shared_conv_detect_loss = shared_conv_detect_cross_entropy + shared_conv_detect_loss_box
        self.train_op['shared_conv_detect_loss'] = shared_conv_detect_loss
        self.train_op['shared_conv_detect_cross_entropy'] = shared_conv_detect_cross_entropy
        self.train_op['shared_conv_detect_loss_box'] = shared_conv_detect_loss_box

        # optimizer
        if cfg.ZLRM.TRAIN.SOLVER == 'Adam':
            opt = tf.train.AdamOptimizer(cfg.ZLRM.TRAIN.LEARNING_RATE)
        elif cfg.ZLRM.TRAIN.SOLVER == 'RMS':
            opt = tf.train.RMSPropOptimizer(cfg.ZLRM.TRAIN.LEARNING_RATE)
        else:
            shared_conv_detect_lr = tf.Variable(cfg.ZLRM.TRAIN.LEARNING_RATE, trainable=False)
            # lr = tf.Variable(0.0, trainable=False)
            momentum = cfg.ZLRM.TRAIN.MOMENTUM
            opt = tf.train.MomentumOptimizer(shared_conv_detect_lr, momentum)
            self.train_op['shared_conv_detect_lr'] = shared_conv_detect_lr

        shared_conv_detect_global_step = tf.Variable(0, trainable=False)
        self.train_op['shared_conv_detect_global_step'] = shared_conv_detect_global_step
        with_clip = True
        if with_clip:
            tvars = tf.trainable_variables()
            print('shared conv detect network 所有的可训练参数', tvars)
            tvars = [v for v in tvars if (v.name.split('_')[0] == 'transform4' or
                                          v.name.split('_')[0] == 'res4' or
                                          v.name.split('/')[0] == 'conv_new_1' or
                                          v.name.split('/')[0] == 'rfcn_cls' or
                                          v.name.split('/')[0] == 'rfcn_bbox'
                                          )]
            print('shared conv detect network 训练参数', tvars)
            grads, norm = tf.clip_by_global_norm(tf.gradients(shared_conv_detect_loss, tvars), 10.0)
            shared_conv_detect_train_op = opt.apply_gradients(zip(grads, tvars), global_step=shared_conv_detect_global_step)
        else:
            tvars = tf.trainable_variables()
            print('shared conv detect network 所有的可训练参数', tvars)
            tvars = [v for v in tvars if (v.name.split('_')[0] == 'transform4' or
                                          v.name.split('_')[0] == 'res4' or
                                          v.name.split('/')[0] == 'conv_new_1' or
                                          v.name.split('/')[0] == 'rfcn_cls' or
                                          v.name.split('/')[0] == 'rfcn_bbox'
                                          )]
            print('shared conv detect network 训练参数', tvars)
            shared_conv_detect_train_op = opt.minimize(shared_conv_detect_loss, global_step=shared_conv_detect_global_step,
                                                    var_list=tvars)
        self.train_op['shared_conv_detect_train_op'] = shared_conv_detect_train_op

        # intialize variables
        shared_conv_detect_init = tf.global_variables_initializer()
        self.train_op['shared_conv_detect_init'] = shared_conv_detect_init
        shared_conv_detect_restore_iter = 0
        self.train_op['shared_conv_detect_restore_iter'] = shared_conv_detect_restore_iter
        self.train_op['shared_conv_detect_restore_saver'] = self.shared_conv_detect_restore_saver
        if restore:
            try:
                # shared_conv_detect网络参数恢复
                print('从 ', self.shared_conv_detect_output_dir, '恢复detectn网络')
                shared_conv_detect_ckpt = tf.train.get_checkpoint_state(self.shared_conv_detect_output_dir)
                self.train_op['shared_conv_detect_ckpt'] = shared_conv_detect_ckpt
                print ('Restoring from {}...'.format(shared_conv_detect_ckpt.model_checkpoint_path),)
                stem = os.path.splitext(os.path.basename(shared_conv_detect_ckpt.model_checkpoint_path))[0]
                shared_conv_detect_restore_iter = int(stem.split('_')[-1])
                self.train_op['shared_conv_detect_restore_iter'] = shared_conv_detect_restore_iter
                print ('done')
            except:
                raise BaseException('Check your pretrained {:s}'.format(shared_conv_detect_ckpt.model_checkpoint_path))
        else:
            try:
                # detect网络参数恢复
                print('从 ', self.shared_conv_rpn_output_dir, '初次从shared_conv_rpn恢复detectn网络')
                shared_conv_detect_ckpt = tf.train.get_checkpoint_state(self.shared_conv_rpn_output_dir)
                self.train_op['shared_conv_detect_ckpt'] = shared_conv_detect_ckpt
                print ('Restoring from {}...'.format(shared_conv_detect_ckpt.model_checkpoint_path),)
            except:
                raise BaseException('Check your pretrained {:s}'.format(shared_conv_detect_ckpt.model_checkpoint_path))


    def train_model(self, rpn_network, detect_network, shared_conv_network,
                    rpn_sess, detect_sess, shared_conv_sess,
                    max_iters):
        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)
        if self.train_step == 'train_step_rpn':
            print('@@训练rpn网络@@')
            rpn_sess.run(self.train_op['rpn_init'])

            if self.pretrained_model is not None and not self.restore:
                try:
                    print('Loading rpn pretrained model '
                          'weights from {:s}'.format(self.pretrained_model))
                    with self.rpn_graph.as_default():
                        rpn_network.load(self.pretrained_model, rpn_sess, True)
                except:
                    raise BaseException('Check your pretrained model {:s}'.format(self.pretrained_model))
            if self.restore:
                self.train_op['rpn_restore_saver'].restore(rpn_sess, self.train_op['rpn_ckpt'].model_checkpoint_path)
            rpn_sess.run(self.train_op['rpn_global_step'].assign(self.train_op['rpn_restore_iter']))
            rpn_lr = self.train_op['rpn_lr']
            last_snapshot_iter = -1
            timer = Timer()
            for iter in range(self.train_op['rpn_restore_iter'], max_iters):
                timer.tic()

                # learning rate
                if iter != 0 and iter % cfg.ZLRM.TRAIN.STEPSIZE == 0:
                    rpn_sess.run(tf.assign(rpn_lr, rpn_lr.eval(session=rpn_sess) * cfg.ZLRM.TRAIN.GAMMA))

                # get one batch
                blobs = data_layer.forward()

                if (iter + 1) % (cfg.ZLRM.TRAIN.DISPLAY) == 0:
                    print('image: %s' % (blobs['im_name']), )

                feed_dict = {
                    rpn_network.data: blobs['data'],
                    rpn_network.im_info: blobs['im_info'],
                    rpn_network.gt_boxes: blobs['gt_boxes']
                }

                fetch_list = [self.train_op['rpn_cross_entropy'],
                              self.train_op['rpn_loss_box'],
                              self.train_op['rpn_train_op']]

                fetch_list += []
                rpn_loss_cls_value, rpn_loss_box_value, \
                _, = rpn_sess.run(fetches=fetch_list, feed_dict=feed_dict)

                _diff_time = timer.toc(average=False)

                if (iter) % (cfg.ZLRM.TRAIN.DISPLAY) == 0:
                    print('iter: %d / %d, total rpn loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, lr: %f' % \
                              (iter, max_iters, rpn_loss_cls_value + rpn_loss_box_value, \
                               rpn_loss_cls_value, rpn_loss_box_value, rpn_lr.eval(session=rpn_sess)))
                    print('speed: {:.3f}s / iter'.format(_diff_time))

                if (iter + 1) % cfg.ZLRM.TRAIN.SNAPSHOT_ITERS == 0:
                    last_snapshot_iter = iter
                    with self.rpn_graph.as_default():
                        self.snapshot(rpn_sess, rpn_network, self.train_op['rpn_restore_saver'], iter, self.rpn_output_dir)

            if last_snapshot_iter != iter:
                with self.rpn_graph.as_default():
                    self.snapshot(rpn_sess, rpn_network, self.train_op['rpn_restore_saver'], iter, self.rpn_output_dir)

        if self.train_step == 'train_step_detect':
            print('@@训练detect网络@@')
            rpn_sess.run(self.train_op['rpn_init'])
            detect_sess.run(self.train_op['detect_init'])
            self.train_op['rpn_restore_saver'].restore(rpn_sess, self.train_op['rpn_ckpt'].model_checkpoint_path)
            if self.pretrained_model is not None and not self.restore:
                try:
                    print('Loading detect pretrained model '
                          'weights from {:s}'.format(self.pretrained_model))
                    with self.detect_graph.as_default():
                        detect_network.load(self.pretrained_model, detect_sess, True)
                except:
                    raise BaseException('Check your pretrained model {:s}'.format(self.pretrained_model))
            if self.restore:
                self.train_op['detect_restore_saver'].restore(detect_sess,
                                                              self.train_op['detect_ckpt'].model_checkpoint_path)
            detect_sess.run(self.train_op['detect_global_step'].assign(self.train_op['detect_restore_iter']))
            detect_lr = self.train_op['detect_lr']
            last_snapshot_iter = -1
            timer = Timer()
            for iter in range(self.train_op['detect_restore_iter'], max_iters):
                timer.tic()

                # learning rate
                if iter != 0 and iter % cfg.ZLRM.TRAIN.STEPSIZE == 0:
                    detect_sess.run(tf.assign(detect_lr, detect_lr.eval(session=detect_sess) * cfg.ZLRM.TRAIN.GAMMA))

                # get one batch
                blobs = data_layer.forward()



                if (iter + 1) % (cfg.ZLRM.TRAIN.DISPLAY) == 0:
                    print('image: %s' % (blobs['im_name']), )

                # 取roi-data
                rpn_feed_dict = {
                    rpn_network.data: blobs['data'],
                    rpn_network.im_info: blobs['im_info'],
                    rpn_network.gt_boxes: blobs['gt_boxes']
                }

                rpn_fetch_list = [self.train_op['rpn_cross_entropy'],
                                  self.train_op['rpn_loss_box'],
                                  rpn_network.get_output('roi-data')]
                rpn_fetch_list += []
                rpn_loss_cls_value, rpn_loss_box_value, roi_data = rpn_sess.run(fetches=rpn_fetch_list, feed_dict=rpn_feed_dict)

                detect_feed_dict = {
                    detect_network.data: blobs['data'],
                    detect_network.im_info: blobs['im_info'],
                    detect_network.rois_p2: roi_data[0],
                    detect_network.rois_p3: roi_data[1],
                    detect_network.rois_p4: roi_data[2],
                    detect_network.rois_p5: roi_data[3],
                    detect_network.rois_p6: roi_data[4],
                    detect_network.labels: roi_data[5],
                    detect_network.bbox_targets: roi_data[6],
                    detect_network.bbox_inside_weights: roi_data[7],
                    detect_network.bbox_outside_weights: roi_data[8]
                }

                detect_fetch_list = [self.train_op['detect_cross_entropy'],
                              self.train_op['detect_loss_box'],
                              self.train_op['detect_train_op']]

                detect_fetch_list += []
                detect_loss_cls_value, detect_loss_box_value, \
                _, = detect_sess.run(fetches=detect_fetch_list, feed_dict=detect_feed_dict)

                _diff_time = timer.toc(average=False)

                if (iter) % (cfg.ZLRM.TRAIN.DISPLAY) == 0:
                    print('iter: %d / %d, total detect loss: %.4f, detect_loss_cls: %.4f, detect_loss_box: %.4f, '
                          'rpn_loss_cls: %.4f, rpn_loss_box: %.4f, lr: %f' % \
                              (iter, max_iters, detect_loss_cls_value + detect_loss_box_value, \
                               detect_loss_cls_value, detect_loss_box_value,
                               rpn_loss_cls_value, rpn_loss_box_value, detect_lr.eval(session=detect_sess)))
                    print('speed: {:.3f}s / iter'.format(_diff_time))

                if (iter + 1) % cfg.ZLRM.TRAIN.SNAPSHOT_ITERS == 0:
                    last_snapshot_iter = iter
                    with self.detect_graph.as_default():
                        self.snapshot(detect_sess, detect_network, self.train_op['detect_restore_saver'], iter, self.detect_output_dir)

            if last_snapshot_iter != iter:
                with self.detect_graph.as_default():
                    self.snapshot(detect_sess, detect_network, self.train_op['detect_restore_saver'], iter, self.detect_output_dir)

        if self.train_step == 'train_step_shared_conv_rpn':
            print('@@训练shared_conv_rpn网络@@')
            shared_conv_sess.run(self.train_op['shared_conv_rpn_init'])

            if self.restore:
                self.train_op['shared_conv_rpn_restore_saver']. \
                    restore(shared_conv_sess, self.train_op['shared_conv_rpn_ckpt'].model_checkpoint_path)
            else:
                self.train_op['shared_conv_rpn_restore_detect_fraction_saver'].\
                    restore(shared_conv_sess,
                            self.train_op['shared_conv_rpn_restore_detect_fraction_ckpt'].model_checkpoint_path)
                self.train_op['shared_conv_rpn_restore_rpn_fraction_saver']. \
                    restore(shared_conv_sess,
                            self.train_op['shared_conv_rpn_restore_rpn_fraction_ckpt'].model_checkpoint_path)

            shared_conv_sess.run(self.train_op['shared_conv_rpn_global_step'].assign(self.train_op['shared_conv_rpn_restore_iter']))
            shared_conv_rpn_lr = self.train_op['shared_conv_rpn_lr']
            last_snapshot_iter = -1
            timer = Timer()
            for iter in range(self.train_op['shared_conv_rpn_restore_iter'], max_iters):
                timer.tic()

                # learning rate
                if iter != 0 and iter % cfg.ZLRM.TRAIN.STEPSIZE == 0:
                    shared_conv_sess.run(tf.assign(shared_conv_rpn_lr, shared_conv_rpn_lr.eval(session=shared_conv_sess) * cfg.ZLRM.TRAIN.GAMMA))

                # get one batch
                blobs = data_layer.forward()

                if (iter + 1) % (cfg.ZLRM.TRAIN.DISPLAY) == 0:
                    print('image: %s' % (blobs['im_name']), )

                feed_dict = {
                    shared_conv_network.data: blobs['data'],
                    shared_conv_network.im_info: blobs['im_info'],
                    shared_conv_network.gt_boxes: blobs['gt_boxes']
                }

                fetch_list = [self.train_op['shared_conv_rpn_cross_entropy'],
                              self.train_op['shared_conv_rpn_loss_box'],
                              self.train_op['shared_conv_rpn_train_op']]

                fetch_list += []
                shared_conv_rpn_loss_cls_value, shared_conv_rpn_loss_box_value, \
                _, = shared_conv_sess.run(fetches=fetch_list, feed_dict=feed_dict)

                _diff_time = timer.toc(average=False)

                if (iter) % (cfg.ZLRM.TRAIN.DISPLAY) == 0:
                    print('iter: %d / %d, total shared conv rpn loss: %.4f, shared_conv_rpn_loss_cls: %.4f, '
                          'shared_conv_rpn_loss_box: %.4f, lr: %f' % \
                              (iter, max_iters, shared_conv_rpn_loss_cls_value + shared_conv_rpn_loss_box_value, \
                               shared_conv_rpn_loss_cls_value, shared_conv_rpn_loss_box_value,
                               shared_conv_rpn_lr.eval(session=shared_conv_sess)))
                    print('speed: {:.3f}s / iter'.format(_diff_time))

                if (iter + 1) % cfg.ZLRM.TRAIN.SNAPSHOT_ITERS == 0:
                    last_snapshot_iter = iter
                    with self.shared_conv_graph.as_default():
                        self.snapshot(shared_conv_sess, shared_conv_network,
                                      self.train_op['shared_conv_rpn_restore_saver'], iter, self.shared_conv_rpn_output_dir)

            if last_snapshot_iter != iter:
                with self.shared_conv_graph.as_default():
                    self.snapshot(shared_conv_sess, shared_conv_network,
                                  self.train_op['shared_conv_rpn_restore_saver'], iter, self.shared_conv_rpn_output_dir)

        if self.train_step == 'train_step_shared_conv_detect':
            print('@@训练shared_conv_detect网络@@')
            shared_conv_sess.run(self.train_op['shared_conv_detect_init'])

            self.train_op['shared_conv_detect_restore_saver']. \
                restore(shared_conv_sess, self.train_op['shared_conv_detect_ckpt'].model_checkpoint_path)

            shared_conv_sess.run(self.train_op['shared_conv_detect_global_step'].assign(self.train_op['shared_conv_detect_restore_iter']))
            shared_conv_detect_lr = self.train_op['shared_conv_detect_lr']
            last_snapshot_iter = -1
            timer = Timer()
            for iter in range(self.train_op['shared_conv_detect_restore_iter'], max_iters):
                timer.tic()

                # learning rate
                if iter != 0 and iter % cfg.ZLRM.TRAIN.STEPSIZE == 0:
                    shared_conv_sess.run(tf.assign(shared_conv_detect_lr,
                                                   shared_conv_detect_lr.eval(session=shared_conv_sess) * cfg.ZLRM.TRAIN.GAMMA))

                # get one batch
                blobs = data_layer.forward()

                if (iter + 1) % (cfg.ZLRM.TRAIN.DISPLAY) == 0:
                    print('image: %s' % (blobs['im_name']), )

                feed_dict = {
                    shared_conv_network.data: blobs['data'],
                    shared_conv_network.im_info: blobs['im_info'],
                    shared_conv_network.gt_boxes: blobs['gt_boxes']
                }

                fetch_list = [self.train_op['shared_conv_detect_cross_entropy'],
                              self.train_op['shared_conv_detect_loss_box'],
                              self.train_op['shared_conv_rpn_cross_entropy'],
                              self.train_op['shared_conv_rpn_loss_box'],
                              self.train_op['shared_conv_detect_train_op']]

                fetch_list += []
                shared_conv_detect_loss_cls_value, shared_conv_detect_loss_box_value, \
                shared_conv_rpn_loss_cls_value, shared_conv_rpn_loss_box_value,\
                _, = shared_conv_sess.run(fetches=fetch_list, feed_dict=feed_dict)

                _diff_time = timer.toc(average=False)

                if (iter) % (cfg.ZLRM.TRAIN.DISPLAY) == 0:
                    print('iter: %d / %d, total shared conv detect loss: %.4f, shared_conv_detect_loss_cls: %.4f, '
                          'shared_conv_detect_loss_box: %.4f, shared_conv_rpn_loss_cls: %.4f, '
                          'shared_conv_rpn_loss_box: %.4f, lr: %f' % \
                              (iter, max_iters, shared_conv_detect_loss_cls_value + shared_conv_detect_loss_box_value, \
                               shared_conv_detect_loss_cls_value, shared_conv_detect_loss_box_value,
                               shared_conv_rpn_loss_cls_value, shared_conv_rpn_loss_box_value,
                               shared_conv_detect_lr.eval(session=shared_conv_sess)))
                    print('speed: {:.3f}s / iter'.format(_diff_time))

                if (iter + 1) % cfg.ZLRM.TRAIN.SNAPSHOT_ITERS == 0:
                    last_snapshot_iter = iter
                    with self.shared_conv_graph.as_default():
                        self.snapshot(shared_conv_sess, shared_conv_network,
                                      self.train_op['shared_conv_detect_restore_saver'], iter, self.shared_conv_detect_output_dir)

            if last_snapshot_iter != iter:
                with self.shared_conv_graph.as_default():
                    self.snapshot(shared_conv_sess, shared_conv_network,
                                  self.train_op['shared_conv_detect_restore_saver'], iter, self.shared_conv_detect_output_dir)


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.ZLRM.TRAIN.USE_FLIPPED:
        print ('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print ('done')

    print ('Preparing training data...')
    if cfg.ZLRM.TRAIN.HAS_RPN:
        if cfg.ZLRM.IS_MULTISCALE:
            # TODO: fix multiscale training (single scale is already a good trade-off)
            print ('#### warning: multi-scale has not been tested.')
            print ('#### warning: using single scale by setting IS_MULTISCALE: False.')
            gdl_roidb.prepare_roidb(imdb)
        else:
            rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print ('done')

    return imdb.roidb


def get_data_layer(roidb, num_classes):
    """return a data layer."""
    if cfg.ZLRM.TRAIN.HAS_RPN:
        if cfg.ZLRM.IS_MULTISCALE:
            # obsolete
            # layer = GtDataLayer(roidb)
            raise BaseException("Calling caffe modules...")
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer

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


def train_net(rpn_network_name, detect_network_name, shared_conv_network_name, imdb, roidb,
              rpn_output_dir=None, detect_output_dir=None, shared_conv_rpn_output_dir=None, shared_conv_detect_output_dir=None,
              train_step='train_step_rpn', pretrained_model=None, max_iters=40000, restore=False):
    """Train a Fast R-CNN networks."""

    # 判断是否恢复rpn或detect的参数
    if train_step == 'train_step_rpn':
        if restore:
            rpn_restore = True
            detect_restore = False
            shared_conv_restore = False
        else:
            rpn_restore = False
            detect_restore = False
            shared_conv_restore = False
    elif train_step == 'train_step_detect':
        if restore:
            rpn_restore = True
            detect_restore = True
            shared_conv_restore = False
        else:
            rpn_restore = True
            detect_restore = False
            shared_conv_restore = False
    elif train_step == 'train_step_shared_conv_rpn':
        if restore:
            rpn_restore = False
            detect_restore = False
            shared_conv_restore = True
        else:
            rpn_restore = False
            detect_restore = False
            shared_conv_restore = False
    elif train_step == 'train_step_shared_conv_detect':
        if restore:
            rpn_restore = False
            detect_restore = False
            shared_conv_restore = True
        else:
            rpn_restore = False
            detect_restore = False
            shared_conv_restore = False
    else:
        raise KeyError('未知的op_step步骤')

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    # 构建图
    rpn_graph = tf.Graph()
    detect_graph = tf.Graph()
    shared_conv_graph = tf.Graph()

    with rpn_graph.as_default():
        rpn_network = get_network(rpn_network_name)
    with detect_graph.as_default():
        detect_network = get_network(detect_network_name)
    with shared_conv_graph.as_default():
        shared_conv_network = get_network(shared_conv_network_name)

    sw = SolverWrapper(imdb, roidb, rpn_graph, detect_graph, shared_conv_graph,
                       rpn_output_dir, detect_output_dir, shared_conv_rpn_output_dir, shared_conv_detect_output_dir,
                       train_step=train_step, restore=restore, pretrained_model=pretrained_model)
    with rpn_graph.as_default():
        sw.rpn_train_model(rpn_network, restore=rpn_restore)
    with detect_graph.as_default():
        sw.detect_train_model(detect_network, restore=detect_restore)
    with shared_conv_graph.as_default():
        if train_step == 'train_step_shared_conv_rpn':
            sw.shared_conv_rpn_train_model(shared_conv_network, restore=shared_conv_restore)
        elif train_step == 'train_step_shared_conv_detect':
            sw.shared_conv_detect_train_model(shared_conv_network, restore=shared_conv_restore)

    # 为构建的图分别建立会话
    rpn_sess = tf.Session(graph=rpn_graph)
    detect_sess = tf.Session(graph=detect_graph)
    shared_conv_sess = tf.Session(graph=shared_conv_graph)

    print('Solving...')
    sw.train_model(rpn_network, detect_network, shared_conv_network, rpn_sess, detect_sess, shared_conv_sess,
                   max_iters)
    print('done solving')

    # 关闭会话
    rpn_sess.close()
    detect_sess.close()
    shared_conv_sess.close()
