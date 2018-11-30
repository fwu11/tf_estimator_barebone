"""Resnet v1 model variants.
Code branched out from slim/nets/resnet_v1.py, and please refer to it for
more details.
The original version ResNets-v1 were proposed by:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

from models import resnet_utils
from utils.metrics import *
from utils.loss import *
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


@slim.add_arg_scope
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

        return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


def root_block_fn_for_beta_variant(net):
    """Gets root_block_fn for beta variant.
    ResNet-v1 beta variant modifies the first original 7x7 convolution to three
    3x3 convolutions.
    Args:
    net: A tensor of size [batch, height, width, channels], input to the model.
    Returns:
    A tensor after three 3x3 convolutions.
    """
    net = resnet_utils.conv2d_same(net, 64, 3, stride=2, scope='conv1_1')
    net = resnet_utils.conv2d_same(net, 64, 3, stride=1, scope='conv1_2')
    net = resnet_utils.conv2d_same(net, 128, 3, stride=1, scope='conv1_3')

    return net


def resnet_v1_beta(inputs,
                   blocks,
                   num_classes=None,
                   is_training=None,
                   global_pool=True,
                   output_stride=None,
                   root_block_fn=None,
                   scope=None):
    """Generator for v1 ResNet models (beta variant).
    This function generates a family of modified ResNet v1 models. In particular,
    the first original 7x7 convolution is replaced with three 3x3 convolutions.
    See the resnet_v1_*() methods for specific model instantiations, obtained by
    selecting different block instantiations that produce ResNets of various
    depths.
    The code is modified from slim/nets/resnet_v1.py, and please refer to it for
    more details.
    Args:
        inputs: A tensor of size [batch, height_in, width_in, channels].
        blocks: A list of length equal to the number of ResNet blocks. Each element
        is a resnet_utils.Block object describing the units in the block.
        num_classes: Number of predicted classes for classification tasks. If None
        we return the features before the logit layer.
        is_training: Enable/disable is_training for batch normalization.
        global_pool: If True, we perform global average pooling before computing the
        logits. Set to True for image classification, False for dense prediction.
        output_stride: If None, then the output will be computed at the nominal
        network stride. If output_stride is not None, it specifies the requested
        ratio of input to output spatial resolution.
        root_block_fn: The function consisting of convolution operations applied to
        the root input. If root_block_fn is None, use the original setting of
        RseNet-v1, which is simply one convolution with 7x7 kernel and stride=2.
        reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
        scope: Optional variable_scope.
    Returns:
        net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
        If global_pool is False, then height_out and width_out are reduced by a
        factor of output_stride compared to the respective height_in and width_in,
        else both height_out and width_out equal one. If num_classes is None, then
        net is the output of the last ResNet block, potentially after global
        average pooling. If num_classes is not None, net contains the pre-softmax
        activations.
        end_points: A dictionary from components of the network to the corresponding
        activation.
    Raises:
        ValueError: If the target output_stride is not valid.
    """
    if root_block_fn is None:
        root_block_fn = functools.partial(resnet_utils.conv2d_same,
                                      num_outputs=64,
                                      kernel_size=7,
                                      stride=2,
                                      scope='conv1')
    with tf.variable_scope(scope, 'resnet_v1', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                         resnet_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
            if is_training is not None:
                #arg_scope = tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm], is_training=False)
                arg_scope = slim.arg_scope([slim.batch_norm], is_training=False)
            else:
                arg_scope = slim.arg_scope([])
            with arg_scope:
                net = inputs
                if output_stride is not None:
                    if output_stride % 4 != 0:
                        raise ValueError('The output_stride needs to be a multiple of 4.')
                    output_stride /= 4
                net = root_block_fn(net)
                net = slim.max_pool2d(net, 3, stride=2, padding='SAME', scope='pool1')
                net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)

                if global_pool:
                    # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keepdims=True)
                if num_classes is not None:
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                    normalizer_fn=None, scope='logit')
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if num_classes is not None:
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points


def resnet_v1_beta_block(scope, base_depth, num_units, stride):
    """Helper function for creating a resnet_v1 beta variant bottleneck block.
    Args:
        scope: The scope of the block.
        base_depth: The depth of the bottleneck layer for each unit.
        num_units: The number of units in the block.
        stride: The stride of the block, implemented as a stride in the last unit.
        All other units have stride=1.
    Returns:
        A resnet_v1 bottleneck block.
    """
    return resnet_utils.Block(scope, bottleneck, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1,
        'unit_rate': 1
    }] * (num_units - 1) + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride,
        'unit_rate': 1
    }])

def resnet_v1_101_beta(inputs,
                       num_classes=None,
                       is_training=None,
                       global_pool=False,
                       output_stride=None,
                       multi_grid=None,
                       scope='resnet_v1_101'):
    """Resnet v1 101 beta variant.
    This variant modifies the first convolution layer of ResNet-v1-101. In
    particular, it changes the original one 7x7 convolution to three 3x3
    convolutions.
    Args:
        inputs: A tensor of size [batch, height_in, width_in, channels].
        num_classes: Number of predicted classes for classification tasks. If None
        we return the features before the logit layer.
        is_training: Enable/disable is_training for batch normalization.
        global_pool: If True, we perform global average pooling before computing the
        logits. Set to True for image classification, False for dense prediction.
        output_stride: If None, then the output will be computed at the nominal
        network stride. If output_stride is not None, it specifies the requested
        ratio of input to output spatial resolution.
        multi_grid: Employ a hierarchy of different atrous rates within network.
        reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
        scope: Optional variable_scope.
    Returns:
        net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
        If global_pool is False, then height_out and width_out are reduced by a
        factor of output_stride compared to the respective height_in and width_in,
        else both height_out and width_out equal one. If num_classes is None, then
        net is the output of the last ResNet block, potentially after global
        average pooling. If num_classes is not None, net contains the pre-softmax
        activations.
        end_points: A dictionary from components of the network to the corresponding
        activation.
    Raises:
        ValueError: if multi_grid is not None and does not have length = 3.
    """
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID
    else:
        if len(multi_grid) != 3:
            raise ValueError('Expect multi_grid to have length 3.')

    blocks = [
        resnet_v1_beta_block(
            'block1', base_depth=64, num_units=3, stride=2),
        resnet_v1_beta_block(
            'block2', base_depth=128, num_units=4, stride=2),
        resnet_v1_beta_block(
            'block3', base_depth=256, num_units=23, stride=2),
        resnet_utils.Block('block4', bottleneck, [
            {'depth': 2048,
            'depth_bottleneck': 512,
            'stride': 1,
            'unit_rate': rate} for rate in multi_grid]),
    ]
    return resnet_v1_beta(
        inputs,
        blocks=blocks,
        num_classes=num_classes,
        is_training=is_training,
        global_pool=global_pool,
        output_stride=output_stride,
        root_block_fn=functools.partial(root_block_fn_for_beta_variant),
        scope=scope)

@slim.add_arg_scope
def atrous_spatial_pyramid_pooling(net, scope, depth=256):
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


def deeplab_v3(inputs, args, is_training):

    # inputs has shape - Original: [batch, 513, 513, 3]
    with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training)):
        _, end_points = resnet_v1_101_beta(inputs,
                               args.num_classes,
                               is_training=is_training,
                               global_pool=False,
                               output_stride=args.output_stride,
                               multi_grid=args.multi_grid)

        with tf.variable_scope("DeepLab_v3"):

            # get block 4 feature outputs
            net = end_points[args.resnet_model + '/block4']

            net = atrous_spatial_pyramid_pooling(net, "ASPP_layer", depth=256)

            net = slim.conv2d(net, args.num_classes, [1, 1], activation_fn=None,
                              normalizer_fn=None, scope='logits')

            size = tf.shape(inputs)[1:3]
            # resize the output logits to match the labels dimensions
            net = tf.image.resize_bilinear(net, size)
            return net

def prepare_label(input_batch, new_size, num_classes, one_hot=True):
    """Resize masks and perform one-hot encoding.
    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.
      num_classes: number of classes to predict (including background).
      one_hot: whether perform one-hot encoding.
    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    with tf.name_scope('label_encode'):
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
    return input_batch

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

        # learning rate scheduler
        '''
        global_step = tf.train.get_or_create_global_step()
        starter_learning_rate = params.starting_learning_rate
        end_learning_rate = 0.0001
        decay_steps = params.train_steps
        learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                            decay_steps, end_learning_rate,
                                            power=0.9)
        '''

        # SGD + momentum optimizer
        optimizer = tf.train.MomentumOptimizer(learning_rate,momentum = 0.9)
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
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