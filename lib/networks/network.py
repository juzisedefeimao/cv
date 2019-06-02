import numpy as np
import tensorflow as tf
from .netconfig import cfg
from lib.rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from lib.rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py
from lib.rpn_msr.proposal_target_layer_tf import proposal_target_layer as proposal_target_layer_py
from lib.Siammask_rpn_msr.anchor_target_layer import anchor_target_layer as siammase_anchor_target_layer_py
from lib.Siammask_rpn_msr.proposal_layer_tf import proposal_layer as siammase_proposal_layer_py
from lib.fpn_rpn_msr.fpn_anchor_target_layer import anchor_target_layer as fpn_anchor_target_layer_py
from lib.fpn_rpn_msr.fpn_proposal_layer import proposal_layer as fpn_proposal_layer_py
from lib.fpn_rpn_msr.fpn_proposal_target_layer import proposal_target_layer as fpn_proposal_target_layer_py
# from lib.psroi_pooling_layer import psroi_pooling_op as psroi_pooling_op



DEFAULT_PADDING = 'SAME'

def include_original(dec):
    """ Meta decorator, which make the original function callable (via f._original() )"""
    def meta_decorator(f):
        decorated = dec(f)
        decorated._original = f
        return decorated
    return meta_decorator

@include_original
def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=False, op_step=None):
        Resnet_50 = {'res2b_branch2a': 'res1_2_conv1', 'res2b_branch2c':'res1_2_conv3', 'res2b_branch2b':'res1_2_conv2',
                     'bn4c_branch2a':'res3_3_bn1', 'res2a_branch2a':'res1_1_conv1', 'bn4c_branch2c':'res3_3_bn3',
                     'res2a_branch2c':'res1_1_conv3', 'bn2a_branch2a':'res1_1_bn1', 'bn2a_branch2c':'res1_1_bn3',
                     'bn2a_branch2b':'res1_1_bn2', 'bn5b_branch2c':'res4_2_bn3', 'bn5b_branch2b':'res4_2_bn2',
                     'bn5b_branch2a':'res4_2_bn1', 'res4a_branch2b':'res3_1_conv2', 'res4a_branch2c':'res3_1_conv3',
                     'res4a_branch2a':'res3_1_conv1', 'bn3b_branch2a':'res2_2_bn1', 'bn3b_branch2c':'res2_2_bn3',
                     'bn3b_branch2b':'res2_2_bn2', 'res3b_branch2a':'res2_2_conv1', 'res3b_branch2b':'res2_2_conv2',
                     'res3b_branch2c':'res2_2_conv3', 'bn4b_branch2b':'res3_2_bn2', 'bn4b_branch2c':'res3_2_bn3',
                     'bn4b_branch2a':'res3_2_bn1', 'res5b_branch2b':'res4_2_conv2', 'res5b_branch2c':'res4_2_conv3',
                     'res5b_branch2a':'res4_2_conv1', 'res4a_branch1':'transform3_conv', 'bn4d_branch2a':'res3_4_bn1',
                     'res4c_branch2b':'res3_3_conv2', 'bn4d_branch2c':'res3_4_bn3', 'bn5a_branch1':'transform4_bn',
                     'bn3c_branch2b':'res2_3_bn2', 'res2a_branch2b':'res1_1_conv2', 'bn3c_branch2a':'res2_3_bn1',
                     'bn5a_branch2b':'res4_1_bn2', 'bn5a_branch2c':'res4_1_bn3', 'bn4c_branch2b':'res3_3_bn2',
                     'res4d_branch2a':'res3_4_conv1', 'res4d_branch2c':'res3_4_conv3', 'res4d_branch2b':'res3_4_conv2',
                     'bn4f_branch2c':'res3_6_bn3', 'res3a_branch1':'transform2_conv', 'bn3d_branch2c':'res2_4_bn3',
                     'bn3d_branch2b':'res2_4_bn2', 'bn3d_branch2a':'res2_4_bn1', 'bn4a_branch1':'transform3_bn',
                     'bn4e_branch2c':'res3_5_bn3', 'bn4e_branch2b':'res3_5_bn2', 'bn4e_branch2a':'res3_5_bn1',
                     'bn5c_branch2a':'res4_3_bn1', 'bn5c_branch2b':'res4_3_bn2', 'bn5c_branch2c':'res4_3_bn3',
                     'bn4f_branch2a':'res3_6_bn1', 'bn4a_branch2c':'res3_1_bn3', 'bn4a_branch2b':'res3_1_bn2',
                     'bn4a_branch2a':'res3_1_bn1', 'res4c_branch2a':'res3_3_conv1', 'bn4d_branch2b':'res3_4_bn2',
                     'res4c_branch2c':'res3_3_conv3', 'res4e_branch2b':'res3_5_conv2', 'res4e_branch2c':'res3_5_conv3',
                     'res4e_branch2a':'res3_5_conv1', 'conv1':'conv1', 'res2c_branch2b':'res1_3_conv2',
                     'res2c_branch2c':'res1_3_conv3', 'res2c_branch2a':'res1_3_conv1', 'res5c_branch2a':'res4_3_conv1',
                     'bn_conv1':'bn1', 'res5c_branch2b':'res4_3_conv2', 'res3a_branch2a':'res2_1_conv1',
                     'bn3a_branch2a':'res2_1_bn1', 'res3a_branch2c':'res2_1_conv3', 'bn3a_branch2c':'res2_1_bn3',
                     'res5a_branch2b':'res4_1_conv2', 'res2a_branch1':'transform1_conv', 'bn2c_branch2c':'res1_3_bn3',
                     'bn2c_branch2b':'res1_3_bn2', 'bn2c_branch2a':'res1_3_bn1', 'res3c_branch2c':'res2_3_conv3',
                     'res3c_branch2b':'res2_3_conv2', 'res3c_branch2a':'res2_3_conv1', 'res4b_branch2c':'res3_2_conv3',
                     'res4b_branch2b':'res3_2_conv2', 'res4b_branch2a':'res3_2_conv1',
                     'res4f_branch2c':'res3_6_conv3', 'res4f_branch2b':'res3_6_conv2', 'res4f_branch2a':'res3_6_conv1',
                     'bn3a_branch2b':'res2_1_bn2', 'res3a_branch2b':'res2_1_conv2', 'bn3c_branch2c':'res2_3_bn3',
                     'res5c_branch2c':'res4_3_conv3', 'bn3a_branch1':'transform2_bn', 'bn2a_branch1':'transform1_bn',
                     'bn5a_branch2a':'res4_1_bn1', 'bn4f_branch2b':'res3_6_bn2', 'bn2b_branch2a':'res1_2_bn1',
                     'bn2b_branch2b':'res1_2_bn2', 'bn2b_branch2c':'res1_2_bn3', 'res5a_branch2c':'res4_1_conv3',
                     'res5a_branch1':'transform4_conv', 'res5a_branch2a':'res4_1_conv1', 'res3d_branch2b':'res2_4_conv2',
                     'res3d_branch2c':'res2_4_conv3', 'res3d_branch2a':'res2_4_conv1'}
        if op_step == 'rpn':
            for (i, j) in Resnet_50.items():
                Resnet_50[i] = 'rpn-' + j
            print('npy初始化rpn网络')
        elif op_step == 'detect':
            for (i, j) in Resnet_50.items():
                Resnet_50[i] = 'detect-' + j
            print('npy初始化detect网络')
        elif op_step == 'cnn_encoder':
            for (i, j) in Resnet_50.items():
                Resnet_50[i] = 'cnn_encoder/' + j
            print('npy初始化cnn_encoder网络')

        data_dict = np.load(data_path, encoding='latin1').item()
        for key in data_dict:
            if key in Resnet_50:
                key_s = Resnet_50[key]

            with tf.variable_scope(key_s, reuse=True):
                for subkey in data_dict[key]:
                    # var = tf.get_variable(subkey)
                    # session.run(var.assign(data_dict[key][subkey]))
                    # print("assign pretrain model " + subkey + " to " + key)
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        print ("assign pretrain model "+subkey+ " to "+key)
                    except ValueError:
                        print ("ignore "+key)
                        if not ignore_missing:
                            print('不能忽视', key)
                            raise

    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, str):
                try:
                    layer = self.layers[layer]
                    print (layer)
                except KeyError:
                    print (self.layers.keys())
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print (self.layers.keys())
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')


    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, reuse=False, biased=True,relu=True,
             init_GAN_weights=False, spectral_norm=False, padding=DEFAULT_PADDING, dilation=[1,1,1,1], trainable=True):
        """ contribution by miraclebiu, and biased option"""
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding, dilations=dilation)
        with tf.variable_scope(name if reuse == False else name.split('/')[0]) as scope:


            if init_GAN_weights or relu:
                factor = 2.0
                uniform = False
            else:
                factor = 1.0
                uniform = True

            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=factor, mode='FAN_IN',
                                                                          uniform=uniform)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.ZLRM.TRAIN.WEIGHT_DECAY))

            # 是否使用spectral normalization（在对抗生产网络里使用）
            if spectral_norm:
                kernel = self.spectral_norm(kernel)
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = convolve(input, kernel)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(conv, biases)
            else:
                conv = convolve(input, kernel)
                if relu:
                    return tf.nn.relu(conv)
                return conv

    # @layer
    # def conv_dw_group(self, input, kernel, name):
    #     with tf.variable_scope(name) as scope:
    #         input_shape = input.get_shape()
    #         input_transpose = tf.transpose(input,perm=[1,2,3,0])
    #         input_reshape = tf.reshape(input_transpose, (1,input_shape[1],input_shape[2],-1))
    #         kernel_shape = kernel.get_shape()
    #         kernel_transpose = tf.transpose(kernel, perm=[1,2,3,0])
    #         kernel_reshape = tf.reshape(kernel_transpose, (kernel_shape[1], kernel_shape[2], -1, 1))
    #         conv_dw = tf.nn.depthwise_conv2d(input_reshape,kernel_reshape,[1,1,1,1], padding='VALID')
    #         conv_dw_shape = conv_dw.get_shape()
    #         conv_dw_reshape = tf.reshape(conv_dw,(conv_dw_shape[1],conv_dw_shape[2],input_shape[3], -1))
    #         conv_dw_transpose = tf.transpose(conv_dw_reshape,perm=[3,0,1,2])
    #
    #     return conv_dw_transpose

    @layer
    def conv_dw_group(self, input, name, if_split=False):
        with tf.variable_scope(name) as scope:
            if if_split:
                pass
            else:
                kernel_shape = input[1].get_shape()
                kernel_transpose = tf.transpose(input[1], perm=[1, 2, 3, 0])
                conv_dw = tf.nn.depthwise_conv2d(input[0], kernel_transpose, [1, 1, 1, 1], padding='VALID')
                conv_dw_shape = conv_dw.get_shape()
                conv_dw_reshape = tf.reshape(conv_dw, [conv_dw_shape[0], conv_dw_shape[1],
                                                         conv_dw_shape[2], kernel_shape[3], kernel_shape[0]])
                conv_dw_reshape_transpose = tf.transpose(conv_dw_reshape, perm=[0, 4, 1, 2, 3])
                conv_dw_reshape_transpose_reshape = tf.reshape(conv_dw_reshape_transpose, [-1, conv_dw_shape[1],
                                                                                           conv_dw_shape[2], kernel_shape[3]])

        return conv_dw_reshape_transpose_reshape

    @layer
    def upbilinear(self, input, name):
        up_h = tf.shape(input[1])[1]
        up_w = tf.shape(input[1])[2]
        return tf.image.resize_bilinear(input[0], [up_h, up_w], name=name)

    @layer
    def deconv(self, input, filter_size, kernel, stride=1, name='deconv', biased=True, relu=True,
                     padding=DEFAULT_PADDING, trainable=True):
        with tf.variable_scope(name) as scope:

            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=2, mode='FAN_AVG', uniform=False)
            deconv = tf.layers.conv2d_transpose(inputs=input, filters=filter_size, kernel_size=kernel,
                                               kernel_initializer=init_weights, strides=stride, padding=padding,
                                           use_bias=biased, trainable=trainable, name=scope.name)
            if relu:
                return tf.nn.relu(deconv)
            return deconv

    @layer
    def upconv(self, input, shape, c_o, ksize=4, stride = 2, name = 'upconv', biased=False, relu=True,
               spectral_norm=False, padding=DEFAULT_PADDING, trainable=True):
        """ up-conv"""
        self.validate_padding(padding)

        c_in = input.get_shape()[3].value
        in_shape = input.get_shape()
        if shape is None:
            h = ((in_shape[1]) * stride)
            w = ((in_shape[2]) * stride)
            new_shape = [in_shape[0], h, w, c_o]
        else:
            new_shape = [in_shape[0], shape[0], shape[1], c_o]
        output_shape = tf.stack(new_shape)
        # print(output_shape)

        filter_shape = [ksize, ksize, c_o, c_in]

        with tf.variable_scope(name) as scope:
            # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=2, mode='FAN_AVG', uniform=False)
            filters = self.make_var('weights', filter_shape, init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.ZLRM.TRAIN.WEIGHT_DECAY))
            # 是否使用spectral normalization（在对抗生产网络里使用）
            if spectral_norm:
                filters = self.spectral_norm(filters)
            deconv = tf.nn.conv2d_transpose(input, filters, output_shape =output_shape,
                                            strides=[1, stride, stride, 1], padding=DEFAULT_PADDING, name=scope.name)
            # coz de-conv losses shape info, use reshape to re-gain shape
            deconv = tf.reshape(deconv, new_shape)

            if biased:
                init_biases = tf.constant_initializer(0.0)
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                if relu:
                    bias = tf.nn.bias_add(deconv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(deconv, biases)
            else:
                if relu:
                    return tf.nn.relu(deconv)
                return deconv

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def sigmoid(self, input, name):
        return tf.nn.sigmoid(input, name=name)

    # @layer
    # def lrelu(self, input, name):
    #     input_shape = input.get_shape()
    #     lrelu = tf.py_func(self.lrelu_np, [input], [tf.float32])
    #     lrelu_tensor = tf.convert_to_tensor(lrelu, name=name)
    #     lrelu_tensor_reshape = tf.reshape(lrelu_tensor,input_shape, name=name)
    #     return lrelu_tensor_reshape

    @layer
    def lrelu(self, input, alpha=cfg.LRELU_DECAY, name=None):
        return tf.maximum(input, alpha * input, name=name)

    @layer
    def tanh(self, input, name):
        return tf.nn.tanh(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        if isinstance(input, tuple):
            input = input[0]
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def global_avg_pool(self, input, name):
        if isinstance(input, tuple):
            input = input[0]

        return tf.reduce_mean(input, axis=[1,2], name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        print(inputs)
        return tf.concat(inputs, axis=axis, name=name)
    @layer
    def conv_concat(self, input, name):
        x_shapes = input[0].get_shape()
        y_shapes = input[1].get_shape()

        return tf.concat([input[0], input[1] * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])],
                         axis=3, name=name)

    @layer
    def relation_concat(self, input, C_WAY, query_num = None, name='relation_concat'):
        with tf.variable_scope(name) as scope:
            sample_all = input[0]
            query = input[1]
            query_shape = query.get_shape()

            sample_split = tf.split(sample_all, num_or_size_splits=C_WAY, axis=0)
            relation_concat = []
            for split in sample_split:
                split_shape = split.get_shape()
                sample_one = tf.reduce_mean(split, axis=0)
                sample_one = tf.reshape(sample_one, (1, split_shape[1], split_shape[2], split_shape[3]))
                if query_num == None:
                    sample_transform = tf.tile(sample_one, [query_shape[0], 1, 1, 1])
                else:
                    constant = tf.constant([1,1,1], dtype=tf.int32)
                    tile_shape = tf.concat([query_num, constant], axis=0)
                    sample_transform = tf.tile(sample_one, tile_shape)
                relation_concat.append(tf.concat([query, sample_transform], axis=3))
            relation_concat = tf.stack(relation_concat, axis=1)

        return tf.reshape(relation_concat, (-1, query_shape[1], query_shape[2], query_shape[3] * 2))

    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True):
        with tf.variable_scope(name) as scope:
            print('kkk', input)
            input_shape = input.get_shape()
            if len(input_shape) == 4:
                input_reshape = tf.reshape(input, [-1, input_shape[1] * input_shape[2] * input_shape[3 ]])
            elif len(input_shape) == 2:
                input_reshape = tf.reshape(input, [-1, input_shape[1]])
            else:
                raise Exception('fc形状错误')
            input_reshape_shape = input_reshape.get_shape()


            # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.02)
            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=2, mode='FAN_AVG', uniform=False)
            init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [input_reshape_shape[1], num_out], init_weights, trainable, \
                                    regularizer=self.l2_regularizer(cfg.ZLRM.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(input_reshape, weights, biases, name=scope.name)
            print(fc)
            return fc

    @layer
    def gaussian_noise_layer(self, input, std=0.15, name=None):
        noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=std, dtype=tf.float32)
        return input + noise

    @layer
    def add(self,input,name):
        """contribution by miraclebiu"""
        return tf.add(input[0],input[1], name=name)

    @layer
    def batch_normalization(self,input,name,relu=True, trainable=False):
        """contribution by miraclebiu"""
        print(input)
        if relu:
            temp_layer=tf.layers.batch_normalization(input,scale=True,center=True,training=trainable,name=name)
            return tf.nn.relu(temp_layer)
        else:
            return tf.layers.batch_normalization(input,scale=True,center=True,training=trainable,name=name)

    @layer
    def negation(self, input, name):
        """ simply multiplies -1 to the tensor"""
        return tf.multiply(input, -1.0, name=name)

    @layer
    def dropout(self, input, keep_prob, name, training=True):
        return tf.layers.dropout(input, keep_prob, training=training, name=name)

    def l2_regularizer(self, weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight_decay')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
        return regularizer

    def smooth_l1_dist(self, deltas, sigma2=9.0, name='smooth_l1_dist'):
        with tf.name_scope(name=name) as scope:
            deltas_abs = tf.abs(deltas)
            smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0/sigma2), tf.float32)
            return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                        (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)

    @layer
    def proposal_target_layer(self, input, classes, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        with tf.variable_scope(name) as scope:
            # inputs: 'rpn_rois','gt_boxes', 'gt_ishard', 'dontcare_areas'
            rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights \
                = tf.py_func(proposal_target_layer_py,
                             [input[0], input[1], classes],
                             [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
            # rois <- (1 x H x W x A, 5) e.g. [0, x1, y1, x2, y2]
            # rois = tf.convert_to_tensor(rois, name='rois')
            rois = tf.reshape(rois, [-1, 5], name='rois')  # goes to roi_pooling
            labels = tf.convert_to_tensor(tf.cast(labels, tf.int32), name='labels')  # goes to FRCNN loss
            bbox_targets = tf.convert_to_tensor(bbox_targets, name='bbox_targets')  # goes to FRCNN loss
            bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name='bbox_inside_weights')
            bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name='bbox_outside_weights')

            self.layers['rois'] = rois

            return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    @layer
    def proposal_layer(self, input, cfg_key, _feat_stride, anchor_scales, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
            # input[0] shape is (1, H, W, Ax2)
            # rpn_rois <- (1 x H x W x A, 5) [0, x1, y1, x2, y2], scores
        rpn_rois, scores = tf.py_func(proposal_layer_py, \
                                     [input[0], input[1], input[2], cfg_key, _feat_stride, anchor_scales], \
                                     [tf.float32, tf.float32])
        rpn_rois = tf.reshape(rpn_rois, [-1, 5], name=name)
        scores = tf.reshape(scores, [-1, 1])
        return rpn_rois, scores

    @layer
    def anchor_target_layer(self, input, _feat_stride, anchor_scales, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:
            # 'rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info'
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
                tf.py_func(anchor_target_layer_py,
                           [input[0], input[1], input[2], _feat_stride, anchor_scales],
                           [tf.float32, tf.float32, tf.float32, tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32),
                                              name='rpn_labels')  # shape is (1 x H x W x A, 2)
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets,
                                                    name='rpn_bbox_targets')  # shape is (1 x H x W x A, 4)
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights,
                                                           name='rpn_bbox_inside_weights')  # shape is (1 x H x W x A, 4)
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights,
                                                            name='rpn_bbox_outside_weights')  # shape is (1 x H x W x A, 4)


            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    @layer
    def siammase_anchor_target_layer(self, input, _feat_stride, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:
            # 'rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info'
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
                tf.py_func(siammase_anchor_target_layer_py,
                           [input[0], input[1], input[2], _feat_stride],
                           [tf.float32, tf.float32, tf.float32, tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32),
                                              name='rpn_labels')  # shape is (batch x classes x H x W x A)
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets,
                                                    name='rpn_bbox_targets')  # shape is (batch x classes x H x W x A, 4)
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights,
                                                           name='rpn_bbox_inside_weights')  # shape is ((batch x classes x H x W x A, 4)
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights,
                                                            name='rpn_bbox_outside_weights')  # shape is ((batch x classes x H x W x A, 4)

            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    @layer
    def siammase_proposal_layer(self, input, cfg_key, _feat_stride, anchor_scales, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
            # input[0] shape is (1, H, W, Ax2)
            # rpn_rois <- (1 x H x W x A, 5) [0, x1, y1, x2, y2], scores
        rpn_rois, scores = tf.py_func(siammase_proposal_layer_py, \
                                      [input[0], input[1], input[2], cfg_key, _feat_stride], \
                                      [tf.float32, tf.float32])
        # for i in range(input[0].shape[0]):
        #     rpn_rois[i] = tf.reshape(rpn_rois[i], [-1, 5], name=name)
        #     scores[i] = tf.reshape(scores[i], [-1, 1])
        return rpn_rois, scores

    @layer
    def fpn_proposal_layer(self, input, _feat_strides, anchor_sizes, cfg_train_key, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
            # input[0] shape is (1, H, W, Ax2)
            # rpn_rois <- (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        if cfg_train_key == True:
            # 'rpn_cls_prob_reshape/P2', 'rpn_bbox_pred/P2',
            # 'rpn_cls_prob_reshape/P3', 'rpn_bbox_pred/P3',
            # 'rpn_cls_prob_reshape/P4', 'rpn_bbox_pred/P4',
            # 'rpn_cls_prob_reshape/P5', 'rpn_bbox_pred/P5',
            # 'im_info'
            with tf.variable_scope(name) as scope:
                return tf.reshape(tf.py_func(fpn_proposal_layer_py, \
                                             [input[0], input[1], \
                                              input[2], input[3], \
                                              input[4], input[5], \
                                              input[6], input[7], \
                                              input[8], input[9], \
                                              input[10], cfg_train_key, _feat_strides, anchor_sizes], \
                                             [tf.float32]), \
                                  [-1, 5], name='rpn_rois')

        with tf.variable_scope(name) as scope:
            rpn_rois_P2, rpn_rois_P3, rpn_rois_P4, rpn_rois_P5, rpn_rois_P6, rpn_rois = tf.py_func(fpn_proposal_layer_py, \
                                                                                                   [input[0], input[1], \
                                                                                                    input[2], input[3], \
                                                                                                    input[4], input[5], \
                                                                                                    input[6], input[7], \
                                                                                                    input[8], input[9], \
                                                                                                    input[10], cfg_train_key,
                                                                                                    _feat_strides,
                                                                                                    anchor_sizes], \
                                                                                                   [tf.float32,
                                                                                                    tf.float32,
                                                                                                    tf.float32,
                                                                                                    tf.float32,
                                                                                                    tf.float32,
                                                                                                    tf.float32])

            rpn_rois_P2 = tf.reshape(rpn_rois_P2, [-1, 5], name='rpn_rois_P2')  # shape is (1 x H(P) x W(P) x A(P), 5)
            rpn_rois_P3 = tf.reshape(rpn_rois_P3, [-1, 5], name='rpn_rois_P3')  # shape is (1 x H(P) x W(P) x A(P), 5)
            rpn_rois_P4 = tf.reshape(rpn_rois_P4, [-1, 5], name='rpn_rois_P4')  # shape is (1 x H(P) x W(P) x A(P), 5)
            rpn_rois_P5 = tf.reshape(rpn_rois_P5, [-1, 5], name='rpn_rois_P5')  # shape is (1 x H(P) x W(P) x A(P), 5)
            rpn_rois_P6 = tf.reshape(rpn_rois_P6, [-1, 5], name='rpn_rois_P6')  # shape is (1 x H(P) x W(P) x A(P), 5)
            rpn_rois = tf.reshape(rpn_rois, [-1, 5], name='rpn_rois')  # shape is (1 x H(P) x W(P) x A(P), 5)

            self.layers['rois'] = rpn_rois

            return rpn_rois_P2, rpn_rois_P3, rpn_rois_P4, rpn_rois_P5, rpn_rois_P6

    @layer
    def fpn_anchor_target_layer(self, input, _feat_strides, anchor_sizes, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:
            # 'rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info'
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
                tf.py_func(fpn_anchor_target_layer_py,
                           [input[0], input[1], input[2], input[3], input[4], input[5], input[6],
                            _feat_strides, anchor_sizes],
                           [tf.float32, tf.float32, tf.float32, tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32),
                                              name='rpn_labels')  # shape is (1 x H x W x A, 2)
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets,
                                                    name='rpn_bbox_targets')  # shape is (1 x H x W x A, 4)
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights,
                                                           name='rpn_bbox_inside_weights')  # shape is (1 x H x W x A, 4)
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights,
                                                            name='rpn_bbox_outside_weights')  # shape is (1 x H x W x A, 4)

            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


    @layer
    def fpn_proposal_target_layer(self, input, classes, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        with tf.variable_scope(name) as scope:
            # inputs: 'rpn_rois','gt_boxes', 'gt_ishard', 'dontcare_areas'
            rois_P2, rois_P3, rois_P4, rois_P5, rois_P6, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, rois \
                = tf.py_func(fpn_proposal_target_layer_py,
                             [input[0], input[1], classes],
                             [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                              tf.float32, tf.float32, tf.float32])
            # rois_Px <- (1 x H x W x A(x), 5) e.g. [0, x1, y1, x2, y2]
            # rois = tf.convert_to_tensor(rois, name='rois')
            rois = tf.reshape(rois, [-1, 5], name='rois')  # goes to roi_pooling
            rois_P2 = tf.reshape(rois_P2, [-1, 5], name='rois_P2')  # goes to roi_pooling
            rois_P3 = tf.reshape(rois_P3, [-1, 5], name='rois_P3')  # goes to roi_pooling
            rois_P4 = tf.reshape(rois_P4, [-1, 5], name='rois_P4')  # goes to roi_pooling
            rois_P5 = tf.reshape(rois_P5, [-1, 5], name='rois_P5')  # goes to roi_pooling
            rois_P6 = tf.reshape(rois_P6, [-1, 5], name='rois_P6')  # goes to roi_pooling
            labels = tf.convert_to_tensor(tf.cast(labels, tf.int32), name='labels')  # goes to FRCNN loss
            bbox_targets = tf.convert_to_tensor(bbox_targets, name='bbox_targets')  # goes to FRCNN loss
            bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name='bbox_inside_weights')
            bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name='bbox_outside_weights')

            self.layers['rois'] = rois

            return rois_P2, rois_P3, rois_P4, rois_P5, rois_P6, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, rois

    @layer
    def roi_segment(self, input, name):
        self.layers['roi-data/P2'] = input[0]
        self.layers['roi-data/P3'] = input[1]
        self.layers['roi-data/P4'] = input[2]
        self.layers['roi-data/P5'] = input[3]
        self.layers['roi-data/P6'] = input[4]


    @layer
    def psroi_pool(self, input, output_dim, group_size, spatial_scale, name):
        """contribution by miraclebiu"""
        # only use the first input
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        if isinstance(input[1], tuple):
            input[1] = input[1][0]
        with tf.variable_scope(name) as scope:
            rfcn = input[0]
            box_ind, bbox = self._normalize_bbox(rfcn, input[1], feat_stride=1.0/spatial_scale, name='rois2bbox')
            position_sensitive_boxes = []
            ymin, xmin, ymax, xmax = tf.unstack(bbox, axis=1)
            step_y = (ymax - ymin) / group_size
            step_x = (xmax - xmin) / group_size

            for bin_y in range(group_size):
                for bin_x in range(group_size):
                    box_coordinates = [ymin + bin_y * step_y,
                                       xmin + bin_x * step_x,
                                       ymin + (bin_y + 1) * step_y,
                                       xmin + (bin_x + 1) * step_x]
                    position_sensitive_boxes.append(tf.stack(box_coordinates, axis=1))

            # rfcn = tf.image.pad_to_bounding_box(rfcn, 0, 0, int(cfg.ZLRM.TRAIN.SCALES[0] * spatial_scale + group_size + 2), int(cfg.ZLRM.TRAIN.MAX_SIZE * spatial_scale+ group_size + 2))

            feature_split = tf.split(rfcn, num_or_size_splits=group_size * group_size, axis=3)
            # ttt = tf.stack(feature_split, axis=3)
            image_crops = []
            for (split, box) in zip(feature_split, position_sensitive_boxes):
                crop = tf.image.crop_and_resize(split, box, tf.to_int32(box_ind), [6, 6])
                crop = tf.reduce_mean(crop, axis=[1, 2])
                image_crops.append(crop)
            image_crops = tf.stack(image_crops, axis=1)
            psroipool = tf.reshape(image_crops, [-1, group_size, group_size, output_dim])
            print(psroipool)
        return psroipool, image_crops

    @layer
    def fpn_roi_pool(self, input, pooled_height, pooled_width, name):
        """contribution by miraclebiu"""
        # only use the first input
        if isinstance(input[0], tuple): # P2
            input[0] = input[0][0]

        if isinstance(input[1], tuple): # P3
            input[1] = input[1][0]

        if isinstance(input[2], tuple): # P4
            input[2] = input[2][0]

        if isinstance(input[3], tuple): # P5
            input[3] = input[3][0]

        if isinstance(input[4], tuple): # P6
            input[4] = input[4][0]

        with tf.variable_scope(name) as scope:
            roi_pool_P2 = self.roi_pool(input[0], input[5][0],
                                    pooled_height,
                                    pooled_width,
                                    1.0 / 4.0,
                                    name='roi_pool_P2')
            roi_pool_P3 = self.roi_pool(input[1], input[5][1],
                                    pooled_height,
                                    pooled_width,
                                    1.0 / 8.0,
                                    name='roi_pool_P3')
            roi_pool_P4 = self.roi_pool(input[2], input[5][2],
                                    pooled_height,
                                    pooled_width,
                                    1.0 / 16.0,
                                    name='roi_pool_P4')
            roi_pool_P5 = self.roi_pool(input[3], input[5][3],
                                    pooled_height,
                                    pooled_width,
                                    1.0 / 32.0,
                                    name='roi_pool_P5')
            roi_pool_P6 = self.roi_pool(input[4], input[5][4],
                                    pooled_height,
                                    pooled_width,
                                    1.0 / 64.0,
                                    name='roi_pool_P6')

            return tf.concat(axis=0, values=[roi_pool_P2, roi_pool_P3, roi_pool_P4,
                                             roi_pool_P5, roi_pool_P6], name='roi_pool_concat')

    def roi_pool(self, input, rois, pooled_height, pooled_width, spatial_scale, name):

        with tf.variable_scope(name) as scope:
            box_ind, bbox = self._normalize_bbox(input, rois, feat_stride=1.0 / spatial_scale, name='rois2bbox')

            # ttt = tf.stack(feature_split, axis=3)
            image_crops = tf.image.crop_and_resize(input, bbox, tf.to_int32(box_ind), [pooled_height, pooled_width])
            print('jjjjj', image_crops)
            # for box in bbox:
            #     crop = tf.image.crop_and_resize(input, box, tf.to_int32(box_ind), [pooled_height, pooled_width])
            #     image_crops.append(crop)
            # image_crops = tf.stack(image_crops, axis=1)
        return image_crops

    @layer
    def ps_pool(self, input, output_dim, group_size, name):
        """contribution by miraclebiu"""

        with tf.variable_scope(name) as scope:
            feature_split = tf.split(input, num_or_size_splits=group_size * group_size, axis=3)
            image_crops = []
            for split in feature_split:
                crop = tf.reduce_mean(split, axis=[1, 2])
                image_crops.append(crop)
            image_crops = tf.stack(image_crops, axis=1)
            pspool = tf.reshape(image_crops, [-1, group_size, group_size, output_dim])
            print(pspool)
        return pspool

    def _normalize_bbox(self, bottom, bbox, feat_stride ,name):
        with tf.variable_scope(name_or_scope=name):
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * feat_stride
            width = (tf.to_float(bottom_shape[2]) - 1) * feat_stride

            indexes, x1, y1, x2, y2 = tf.unstack(bbox, axis=1)
            x1 = x1 / width
            y1 = y1 / height
            x2 = x2 / width
            y2 = y2 / height
            # bboxs = tf.stack([y1, x1, y2, x2], axis=1)
            bboxes = tf.stop_gradient(tf.stack([y1, x1, y2, x2], 1))
            return indexes, bboxes

    @layer
    def reshape_layer(self, input, output_shape, name):
        return tf.reshape(input, output_shape, name=name)

    # @layer
    # def reshape_layer(self, input, d, name):
    #     input_shape = tf.shape(input)
    #     if name == 'rpn_cls_prob_reshape':
    #         #
    #         # transpose: (1, AxH, W, 2) -> (1, 2, AxH, W)
    #         # reshape: (1, 2xA, H, W)
    #         # transpose: -> (1, H, W, 2xA)
    #         return tf.transpose(tf.reshape(tf.transpose(input, [0, 3, 1, 2]),
    #                                        [input_shape[0],
    #                                         int(d),
    #                                         tf.cast(
    #                                             tf.cast(input_shape[1], tf.float32) / tf.cast(d, tf.float32) * tf.cast(
    #                                                 input_shape[3], tf.float32), tf.int32),
    #                                         input_shape[2]
    #                                         ]),
    #                             [0, 2, 3, 1], name=name)
    #     else:
    #         return tf.transpose(tf.reshape(tf.transpose(input, [0, 3, 1, 2]),
    #                                        [input_shape[0],
    #                                         int(d),
    #                                         tf.cast(tf.cast(input_shape[1], tf.float32) * (
    #                                                     tf.cast(input_shape[3], tf.float32) / tf.cast(d, tf.float32)),
    #                                                 tf.int32),
    #                                         input_shape[2]
    #                                         ]),
    #                             [0, 2, 3, 1], name=name)

    @layer
    def spatial_reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
        return tf.reshape(input, [input_shape[0], input_shape[1], -1, int(d)])

    @layer
    def spatial_softmax(self, input, name):
        input_shape = tf.shape(input)
        # d = input.get_shape()[-1]
        return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                          [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)

    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                              [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)
        else:
            return tf.nn.softmax(input, name=name)

    # =============================gan classifier=============================

    @layer
    def attention(self, input, channel, name):
        with tf.variable_scope(name):
            (
                self.feed(input)
                    .conv( 1, 1, channel // 8, 1, 1, name='conv_f', relu=False, trainable=True)
            )
            (
                self.feed(input)
                    .conv(1, 1, channel // 8, 1, 1, name='conv_g', relu=False, trainable=True)
            )
            (
                self.feed(input)
                    .conv(1, 1, channel, 1, 1, name='conv_h', relu=False, trainable=True)
            )
            f = self.layers['conv_f']
            g = self.layers['conv_g']
            h = self.layers['conv_h']

            # N = h * w
            s = tf.matmul(self.hw_flatten(g), self.hw_flatten(f), transpose_b=True) # # [bs, N, N]

            beta = tf.nn.softmax(s, axis=-1)  # attention map

            o = tf.matmul(beta, self.hw_flatten(h)) # [bs, N, C]
            gamma = tf.get_variable(name="gamma", shape=[1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=input.shape) # [bs, h, w, C]
            x = gamma * o + input

        return x

    def hw_flatten(self, input):
        return tf.reshape(input, shape=[input.shape[0], -1, input.shape[-1]])

    def spectral_norm(self, w, iteration=1):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

        u_hat = u
        v_hat = None
        for i in range(iteration):
            """
            power iteration
            Usually iteration = 1 will be enough
            """
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = self.l2_norm(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = self.l2_norm(u_)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        w_norm = w / sigma

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_norm, w_shape)

        return w_norm

    def l2_norm(self, v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    def flatten(self, input):
        return tf.layers.flatten(input)

    @layer
    def flatten_layer(self, input, name):
        return tf.layers.flatten(input, name=name)
