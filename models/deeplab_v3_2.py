import tensorflow as tf
from utils.loss import *
from utils.metrics import *
import warnings
warnings.filterwarnings('ignore')
slim = tf.contrib.slim

_DEFAULT_MULTI_GRID = [1, 1, 1]

def update_argparser(parser):
    parser.set_defaults(
        train_steps=120000,
        learning_rate=((63000, 80000, 100000), (0.0001, 0.00005, 0.00001, 0.000001)),
        save_checkpoints_steps=5000,
    )

class ResNet_segmentation(object):
    """
    Original ResNet-101 ('resnet_v1_101.ckpt')
    """
    def __init__(self, inputs, num_classes, phase, encoder_name):
        if encoder_name not in ['res101', 'res50']:
            print('encoder_name ERROR!')
            print("Please input: res101, res50")
            sys.exit(-1)
        self.encoder_name = encoder_name
        self.inputs = inputs
        self.num_classes = num_classes
        self.channel_axis = 3
        self.phase = phase # train (True) or test (False), for BN layers in the decoder
        self.build_network()
    
    def build_network(self):
        self.encoding = self.build_encoder()
        self.outputs = self.build_decoder(self.encoding)
        
    def build_encoder(self):
        scope_name = 'resnet_v1_101' if self.encoder_name == 'res101' else 'resnet_v1_50'
        with tf.variable_scope(scope_name) as scope:
            outputs = self._start_block('conv1')
            with tf.variable_scope('block1') as scope:
                outputs = self._bottleneck_resblock(outputs, 256, 'unit_1',	identity_connection=False)
                outputs = self._bottleneck_resblock(outputs, 256, 'unit_2')
                outputs = self._bottleneck_resblock(outputs, 256, 'unit_3')
            with tf.variable_scope('block2') as scope:
                outputs = self._bottleneck_resblock(outputs, 512, 'unit_1',	half_size=True, identity_connection=False)
                for i in range(2, 5):
                    outputs = self._bottleneck_resblock(outputs, 512, 'unit_%d' % i)
            with tf.variable_scope('block3') as scope:
                outputs = self._bottleneck_resblock(outputs, 1024, 'unit_1', identity_connection=False)
                for i in range(2, 24):
                    outputs = self._bottleneck_resblock(outputs, 1024, 'unit_%d' % i)
            with tf.variable_scope('block4') as scope:
                outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_1', identity_connection=False)
                outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_2')
                outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_3')
                return outputs
    
    def build_decoder(self, encoding):
        with tf.variable_scope('decoder') as scope:
            net = self._ASPP(encoding, "ASPP_layer", depth=256)
            net = slim.conv2d(net, self.num_classes, [1, 1], activation_fn=None,
                              normalizer_fn=None, scope='logits')

            size = tf.shape(self.inputs)[1:3]
            # resize the output logits to match the labels dimensions
            outputs = tf.image.resize_bilinear(net, size)
            return outputs

	# blocks
    def _start_block(self, name):
        """Gets root_block_fn for beta variant.
        ResNet-v1 beta variant modifies the first original 7x7 convolution to three
        3x3 convolutions.
        Args:
        net: A tensor of size [batch, height, width, channels], input to the model.
        Returns:
        A tensor after three 3x3 convolutions.
        """
        net = self.conv2d_same(self.inputs, 64, 3, stride=2, scope='conv1_1')
        net = self.conv2d_same(net, 64, 3, stride=1, scope='conv1_2')
        net = self.conv2d_same(net, 128, 3, stride=1, scope='conv1_3')
        net = self._max_pool2d(net, 3, 2, name='pool1')
        return net

    def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               unit_rate=1,
               rate=1,
               outputs_collections=None,
               scope=None):
        """Bottleneck residual unit variant with BN after convolutions.
        This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
        its definition. Note that we use here the bottleneck variant which has an
        extra bottleneck layer.
        When putting together two consecutive ResNet blocks that use this unit, one
        should use stride = 2 in the last unit of the first block.
        Args:
            inputs: A tensor of size [batch, height, width, channels].
            depth: The depth of the ResNet unit output.
            depth_bottleneck: The depth of the bottleneck layers.
            stride: The ResNet unit's stride. Determines the amount of downsampling of
            the units output compared to its input.
            unit_rate: An integer, unit rate for atrous convolution.
            rate: An integer, rate for atrous convolution.
            outputs_collections: Collection to add the ResNet unit output.
            scope: Optional variable_scope.
        Returns:
            The ResNet unit's output.
        """
        with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
            depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
            if depth == depth_in:
                shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
            else:
                shortcut = slim.conv2d(
                inputs,
                depth,
                [1, 1],
                stride=stride,
                activation_fn=None,
                scope='shortcut')

            residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                                scope='conv1')
            residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                                rate=rate*unit_rate, scope='conv2')
            residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                                activation_fn=None, scope='conv3')
            output = tf.nn.relu(shortcut + residual)

            return output

    def _bottleneck_resblock(self, x, num_o, name, half_size=False, identity_connection=True):
        first_s = 2 if half_size else 1
        assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
        # branch1
        if not identity_connection:
            o_b1 = self._conv2d(x, 1, num_o, first_s, name='%s/bottleneck_v1/shortcut' % name)
            o_b1 = self._batch_norm(o_b1, name='%s/bottleneck_v1/shortcut' % name, is_training=False, activation_fn=None)
        else:
            o_b1 = x
        # branch2
        o_b2a = self._conv2d(x, 1, num_o / 4, first_s, name='%s/bottleneck_v1/conv1' % name)
        o_b2a = self._batch_norm(o_b2a, name='%s/bottleneck_v1/conv1' % name, is_training=False, activation_fn=tf.nn.relu)

        o_b2b = self._conv2d(o_b2a, 3, num_o / 4, 1, name='%s/bottleneck_v1/conv2' % name)
        o_b2b = self._batch_norm(o_b2b, name='%s/bottleneck_v1/conv2' % name, is_training=False, activation_fn=tf.nn.relu)

        o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='%s/bottleneck_v1/conv3' % name)
        o_b2c = self._batch_norm(o_b2c, name='%s/bottleneck_v1/conv3' % name, is_training=False, activation_fn=None)
        # add
        outputs = self._add([o_b1,o_b2c], name='%s/bottleneck_v1/add' % name)
        # relu
        outputs = self._relu(outputs, name='%s/bottleneck_v1/relu' % name)
        return outputs

    def _dilated_bottle_resblock(self, x, num_o, dilation_factor, name, identity_connection=True):
        assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
        # branch1
        if not identity_connection:
            o_b1 = self._conv2d(x, 1, num_o, 1, name='%s/bottleneck_v1/shortcut' % name)
            o_b1 = self._batch_norm(o_b1, name='%s/bottleneck_v1/shortcut' % name, is_training=False, activation_fn=None)
        else:
            o_b1 = x
        # branch2
        o_b2a = self._conv2d(x, 1, num_o / 4, 1, name='%s/bottleneck_v1/conv1' % name)
        o_b2a = self._batch_norm(o_b2a, name='%s/bottleneck_v1/conv1' % name, is_training=False, activation_fn=tf.nn.relu)

        o_b2b = self._dilated_conv2d(o_b2a, 3, num_o / 4, dilation_factor, name='%s/bottleneck_v1/conv2' % name)
        o_b2b = self._batch_norm(o_b2b, name='%s/bottleneck_v1/conv2' % name, is_training=False, activation_fn=tf.nn.relu)

        o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='%s/bottleneck_v1/conv3' % name)
        o_b2c = self._batch_norm(o_b2c, name='%s/bottleneck_v1/conv3' % name, is_training=False, activation_fn=None)
        # add
        outputs = self._add([o_b1,o_b2c], name='%s/bottleneck_v1/add' % name)
        # relu
        outputs = self._relu(outputs, name='%s/bottleneck_v1/relu' % name)
        return outputs

    def _ASPP(net, scope, depth=256):
        """
        ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
        (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
        :param net: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
        :param scope: scope name of the aspp layer
        :return: network layer with aspp applyed to it.
        """

        with tf.variable_scope(scope):
            feature_map_size = tf.shape(net)

            # apply global average pooling
            image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keepdims=True)
            image_level_features = slim.conv2d(image_level_features, depth, [1, 1], scope="image_level_conv_1x1",
                                            activation_fn=None)
            image_level_features = tf.image.resize_bilinear(image_level_features, (feature_map_size[1], feature_map_size[2]))

            at_pool1x1 = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_0", activation_fn=None)

            at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1", rate=6, activation_fn=None)

            at_pool3x3_2 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_2", rate=12, activation_fn=None)

            at_pool3x3_3 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_3", rate=18, activation_fn=None)

            net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3,
                            name="concat")
            net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)
            return net

    # layers
    def _conv2d(self, x, kernel_size, num_o, stride, name, biased=False):
        """
        Conv2d without BN or relu.
        """
        num_x = x.shape[self.channel_axis].value
        with tf.variable_scope(name) as scope:
            w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
            s = [1, stride, stride, 1]
            o = tf.nn.conv2d(x, w, s, padding='SAME')
            if biased:
                b = tf.get_variable('biases', shape=[num_o])
                o = tf.nn.bias_add(o, b)
            return o

    def conv2d_same(self, inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
        """Strided 2-D convolution with 'SAME' padding.
        When stride > 1, then we do explicit zero-padding, followed by conv2d with
        'VALID' padding.
        Note that
            net = conv2d_same(inputs, num_outputs, 3, stride=stride)
        is equivalent to
            net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
            net = subsample(net, factor=stride)
        whereas
            net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')
        is different when the input's height or width is even, which is why we add the
        current function. For more details, see ResnetUtilsTest.testConv2DSameEven().
        Args:
            inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
            num_outputs: An integer, the number of output filters.
            kernel_size: An int with the kernel_size of the filters.
            stride: An integer, the output stride.
            rate: An integer, rate for atrous convolution.
            scope: Scope.
        Returns:
            output: A 4-D tensor of size [batch, height_out, width_out, channels] with
            the convolution output.
        """
        if stride == 1:
            return tf.layers.conv2d(inputs, num_outputs, kernel_size, strides=(1,1), dilation_rate=(rate,rate),padding='SAME', name=scope)
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            inputs = tf.pad(inputs,[[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
            return tf.layers.conv2d(inputs, num_outputs, kernel_size, strides=(stride,stride), dilation_rate=(rate,rate), padding='VALID', name=scope)

    def _dilated_conv2d(self, x, kernel_size, num_o, dilation_factor, name, biased=False):
        """
        Dilated conv2d without BN or relu.
        """
        num_x = x.shape[self.channel_axis].value
        with tf.variable_scope(name) as scope:
            w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
            o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
            if biased:
                b = tf.get_variable('biases', shape=[num_o])
                o = tf.nn.bias_add(o, b)
            return o

    def _relu(self, x, name):
        return tf.nn.relu(x, name=name)

    def _add(self, x_l, name):
        return tf.add_n(x_l, name=name)

    def _max_pool2d(self, x, kernel_size, stride, name):
        k = [1, kernel_size, kernel_size, 1]
        s = [1, stride, stride, 1]
        return tf.nn.max_pool(x, k, s, padding='SAME', name=name)
        
    def _batch_norm(self, x, name, is_training, activation_fn, trainable=True):
        # For a small batch size, it is better to keep 
        # the statistics of the BN layers (running means and variances) frozen, 
        # and to not update the values provided by the pre-trained model by setting is_training=False.
        # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
        # if they are presented in var_list of the optimiser definition.
        # Set trainable = False to remove them from trainable_variables.
        with tf.variable_scope(name+'/BatchNorm') as scope:
            o = tf.contrib.layers.batch_norm(x,scale=True,activation_fn=activation_fn,is_training=is_training,trainable=trainable,scope=scope)
            return o


def model_fn(features, labels, mode, params):
    ''' Model function'''

    if mode == tf.estimator.ModeKeys.TRAIN:
        train = True
    else:
        train = False
    
    img_input = tf.reshape(features, [-1, params.crop_size, params.crop_size, 3])

    # Create network
    raw_output = deeplab_v3(img_input, params, train)

    predictions = tf.argmax(raw_output, axis=-1)

    # Setup the estimator according to the phase (Train, eval)
    reduced_loss = None
    train_op = None
    eval_metric_ops = {}

    # compute loss(train and eval)
    loss = softmax_sparse_crossentropy_ignoring_last_label(labels,raw_output)

    # L2 regularization
    l2_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    # Loss function
    reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

    # evaluation metric
    miou, update_op = mIOU(raw_output,labels,params.num_classes,img_input)

    # configure training
    if mode == tf.estimator.ModeKeys.TRAIN:
        # piecewise learning rate scheduler
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.piecewise_constant(global_step, params.learning_rate[0], params.learning_rate[1])

        # SGD + momentum optimizer
        optimizer = tf.train.MomentumOptimizer(learning_rate,momentum = 0.9)
        train_op = optimizer.minimize(reduced_loss, global_step=tf.train.get_or_create_global_step())

    if mode  == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'miou': (miou, update_op)
        }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=reduced_loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs=None,
    )