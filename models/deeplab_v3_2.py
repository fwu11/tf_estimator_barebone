# TODO verify the model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
from utils.loss import *
from utils.metrics import *
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.layers.python.layers import initializers
import warnings
warnings.filterwarnings('ignore')


def update_argparser(parser):
    parser.set_defaults(
        train_steps=100000,
        learning_rate=((60000, 80000), (0.0001, 0.00001,0.000001)),
        save_checkpoints_steps=200,
    )

'''
def update_argparser(parser):
    parser.set_defaults(
        train_steps=40000,
        learning_rate=((20000,30000), (0.0001, 0.00001,0.000001)),
        save_checkpoints_steps=1000,
    )
'''

class ResNet_segmentation(object):
    """
    Original ResNet-101 ('resnet_v1_101.ckpt')
    """
    def __init__(self, inputs, args, phase, output_stride, encoder_name):
        if encoder_name not in ['res101', 'res50']:
            print('encoder_name ERROR!')
            print("Please input: res101, res50")
            sys.exit(-1)
        self.encoder_name = encoder_name
        self.inputs = inputs
        self.num_classes = args.num_classes
        self.phase = phase # train (True) or test (False), for BN layers in the decoder
        self.multi_grid = args.multi_grid
        self.output_stride = output_stride
        # The current_stride variable keeps track of the effective stride of the
        # activations. This allows us to invoke atrous convolution whenever applying
        # the next residual unit would result in the activations having stride larger
        # than the target output_stride.
        self.current_stride = 1
        self.rate = 1  # The atrous convolution rate parameter.
        self.build_network()
    
    def build_network(self):
        self.encoding = self.build_encoder()
        self.outputs = self.build_decoder(self.encoding)
        
    def build_encoder(self, outputs_collections = None):
        scope_name = 'resnet_v1_101' if self.encoder_name == 'res101' else 'resnet_v1_50'
        with tf.variable_scope(scope_name):
            if self.output_stride is not None:
                if self.output_stride % 4 != 0:
                    raise ValueError('The output_stride needs to be a multiple of 4.')
                self.output_stride /= 4

            net = self._start_block()
            net = tf.layers.max_pooling2d(net, pool_size = 3, strides=2, padding='same', name='pool1')

            # block 1
            with tf.variable_scope('block1'):	
                for i in range(1, 3):
                    with tf.variable_scope('unit_%d' % i):
                        net = self._bottleneck_resblock(net, 256, 64, stride=1, unit_rate=1)
                with tf.variable_scope('unit_3'):
                    net = self._bottleneck_resblock(net, 256, 64, stride=2, unit_rate=1)
            # block 2
            with tf.variable_scope('block2'):
                for i in range(1, 4):
                    with tf.variable_scope('unit_%d' % i):
                        net = self._bottleneck_resblock(net, 512, 128, stride=1, unit_rate=1)
                with tf.variable_scope('unit_4'):
                    net = self._bottleneck_resblock(net, 512, 128, stride=2, unit_rate=1)

            # block 3
            with tf.variable_scope('block3'):
                for i in range(1, 23):
                    with tf.variable_scope('unit_%d' % i):
                        net = self._bottleneck_resblock(net, 1024, 256, stride=1, unit_rate=1)
                with tf.variable_scope('unit_23'):
                    net = self._bottleneck_resblock(net, 1024, 256, stride=2, unit_rate=1)

            # block 4
            with tf.variable_scope('block4'):
                for i in range(1, 4):
                    with tf.variable_scope('unit_%d' % i):
                        net = self._bottleneck_resblock(net, 2048, 512, stride=1, unit_rate = self.multi_grid[i-1])
            
            if self.output_stride is not None and self.current_stride != self.output_stride:
                raise ValueError('The target output_stride cannot be reached.')
            return net
    
    def build_decoder(self, encoding):
        with tf.variable_scope('decoder'):
            net = self._ASPP(encoding, "ASPP_layer", depth=256)
            net = self._conv2d(net, self.num_classes, 1, activation_fn=None, use_batch_norm=False, name='logits')

            size = tf.shape(self.inputs)[1:3]
            # resize the output logits to match the labels dimensions
            outputs = tf.image.resize_bilinear(net, size)
            return outputs

	# blocks
    def _start_block(self):
        """Gets root_block_fn for beta variant.
        ResNet-v1 beta variant modifies the first original 7x7 convolution to three
        3x3 convolutions.
        Args:
        net: A tensor of size [batch, height, width, channels], input to the model.
        Returns:
        A tensor after three 3x3 convolutions.
        """
        net = self._conv2d_same(self.inputs, 64, 3, stride=2, scope='conv1_1')
        net = self._conv2d_same(net, 64, 3, stride=1, scope='conv1_2')
        net = self._conv2d_same(net, 128, 3, stride=1, scope='conv1_3')
        return net

    def _bottleneck_resblock(self, net, depth, depth_bottleneck, stride, unit_rate):
        """Wrap up the bottleneck function
        """
        # If we have reached the target output_stride, then we need to employ
        # atrous convolution with stride=1 and multiply the atrous rate by the
        # current unit's stride for use in subsequent layers.
        if self.output_stride is not None and self.current_stride > self.output_stride:
            raise ValueError('The target output_stride cannot be reached.')
        if self.output_stride is not None and self.current_stride == self.output_stride:
            net = self.bottleneck(net, depth, depth_bottleneck, stride=1, unit_rate=unit_rate, rate=self.rate)
            self.rate *= stride
        else:
            net = self.bottleneck(net, depth, depth_bottleneck, stride, unit_rate, rate=1)
            self.current_stride *= stride
        
        return net

    def bottleneck(self,
                    inputs,
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
            depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
            if depth == depth_in:
                shortcut = self._subsample(inputs, stride, 'shortcut')
            else:
                shortcut = self._conv2d(
                inputs,
                depth,
                1,
                stride,
                activation_fn = None,
                name='shortcut')

            residual = self._conv2d(inputs, depth_bottleneck, 1, stride=1, name='conv1')

            residual = self._conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate*unit_rate, scope='conv2')
            
            residual = self._conv2d(residual, depth, 1, stride=1, activation_fn=None, name='conv3')
                                
            output = tf.nn.relu(shortcut + residual)

            return output


    def _ASPP(self, net, scope, depth=256):
        """
        ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
        when output stride = 8, rates are doubled
        (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
        :param net: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
        :param scope: scope name of the aspp layer
        :return: network layer with aspp applyed to it.
        """
        # get the true output_stride
        self.output_stride *= 4

        if self.output_stride == 16:
            rates = [6,12,18]
        elif self.output_stride == 8:
            rates = [12,24,36]

        with tf.variable_scope(scope):
            feature_map_size = tf.shape(net)

            # apply global average pooling
            image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keepdims=True)
            image_level_features = self._conv2d(image_level_features, depth, 1, activation_fn=None, fine_tune_batch_norm = True, name="image_level_conv_1x1")

            image_level_features = tf.image.resize_bilinear(image_level_features, (feature_map_size[1], feature_map_size[2]))

            at_pool1x1 = self._conv2d(net, depth, 1, activation_fn=None, fine_tune_batch_norm = True, name="conv_1x1_0")

            at_pool3x3_1 = self._conv2d(net, depth, 3, rate=rates[0], activation_fn=None, fine_tune_batch_norm = True, name="conv_3x3_1")

            at_pool3x3_2 = self._conv2d(net, depth, 3, rate=rates[1], activation_fn=None, fine_tune_batch_norm = True, name="conv_3x3_2")

            at_pool3x3_3 = self._conv2d(net, depth, 3, rate=rates[2], activation_fn=None, fine_tune_batch_norm = True, name="conv_3x3_3")

            net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3, name="concat")

            net = self._conv2d(net, depth, 1, activation_fn=None, fine_tune_batch_norm = True, name="conv_1x1_output")
            net = tf.layers.dropout(net,rate=0.1,training=self.phase, name="dropout")
            
            return net

    def _conv2d_same(self, inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
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
            return self._conv2d(inputs,num_outputs,kernel_size,1,rate,'same',name=scope)
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            inputs = tf.pad(inputs,[[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
            return self._conv2d(inputs,num_outputs,kernel_size,stride,rate,'valid',name=scope) 
    
    def _conv2d(self, 
			net,
			num_o,
			kernel_size,  
			stride=1,
			rate=1,
			padding='SAME',			
			weight_decay=0.0001,
			activation_fn=tf.nn.relu,
			use_batch_norm=True,
            fine_tune_batch_norm = False,
			name = None):
	
        """
        Conv2d + BN + relu.
        """
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': self.phase and fine_tune_batch_norm,
            'trainable': True,
            'fused': True,  # Use fused batch norm if possible.
        }
        
        net = tf.contrib.layers.conv2d(net,
                                num_o,
                                kernel_size,
                                stride,
                                padding = padding,
                                rate = rate,
                                activation_fn = activation_fn,
                                normalizer_fn = tf.contrib.layers.batch_norm if use_batch_norm else None,
                                normalizer_params = batch_norm_params,
                                weights_initializer = initializers.variance_scaling_initializer(),
                                weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
                                scope = name)
        return net

    def _subsample(self, inputs, factor, scope=None):
        """Subsamples the input along the spatial dimensions.
        Args:
        inputs: A `Tensor` of size [batch, height_in, width_in, channels].
        factor: The subsampling factor.
        scope: Optional variable_scope.
        Returns:
        output: A `Tensor` of size [batch, height_out, width_out, channels] with the
            input, either intact (if factor == 1) or subsampled (if factor > 1).
        """
        if factor == 1:
            return inputs
        else:
            return tf.layers.max_pooling2d(inputs, pool_size = 1, strides=factor, padding='same', name=scope)

def model_fn(features, labels, mode, params):
    ''' Model function'''

    if mode == tf.estimator.ModeKeys.TRAIN:
        train = True
        output_stride = params.train_output_stride
    else:
        train = False
        output_stride = params.eval_output_stride
    
    img_input = tf.reshape(features, [-1, params.crop_size, params.crop_size, 3])

    # Create network
    net = ResNet_segmentation(img_input, params, train, output_stride, 'res101')

    # predictions
    raw_output  = net.outputs

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
        # comment out next two lines if batch norm is frozen
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
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