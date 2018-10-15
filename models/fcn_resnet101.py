import numpy as np
import tensorflow as tf
from tensorflow import layers
from utils.loss import *
from utils.metrics import *

def fcn_head(input_tensor,filters,num_class,is_train):
    eps = 1e-5
    x = layers.conv2d(inputs = input_tensor,filters = filters,kernel_size = (3,3),padding = 'same',use_bias = False)
    x = layers.batch_normalization(inputs = x, momentum=0.9,epsilon = eps,training = is_train)
    x = tf.nn.relu(features = x)
    x = layers.dropout(inputs = x,rate =0.1,training = is_train)
    x = layers.conv2d(inputs = x, filters = num_class,kernel_size = (1,1), use_bias = False)
    return x

def identity_block(input_tensor, filters, is_train):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''

    eps = 1e-5
    num_filter1, num_filter2, num_filter3 = filters

    x = layers.conv2d(inputs = input_tensor,filters = num_filter1, kernel_size = (1, 1), name = 'conv1', use_bias = False)
    x = layers.batch_normalization(inputs = x, momentum=0.9,epsilon = eps, name = 'bn1',training = is_train)
    x = tf.nn.relu(features = x, name = 'relu1')

    x = layers.conv2d(inputs = x, filters = num_filter2, kernel_size = (3, 3),padding = 'same', name = 'conv2', use_bias=False)
    x = layers.batch_normalization(inputs = x, momentum=0.9, epsilon = eps, name = 'bn2',training = is_train)
    x = tf.nn.relu(features = x, name = 'relu2')

    x = layers.conv2d(inputs = x, filters = num_filter3, kernel_size = (1, 1), name = 'conv3', use_bias=False)
    x = layers.batch_normalization(inputs = x, momentum=0.9, epsilon = eps, name = 'bn3',training = is_train)
    
    x = x + input_tensor
    x = tf.nn.relu(features = x, name = 'relu4')
    return x

# Atrous-Convolution version of residual blocks
def dilated_identity_block(input_tensor, filters, dilation,is_train):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1e-5
    num_filter1, num_filter2, num_filter3 = filters


    x = layers.conv2d(inputs = input_tensor, filters = num_filter1, kernel_size = (1, 1), name='conv1', use_bias = False)
    x = layers.batch_normalization(inputs = x, momentum=0.9,epsilon = eps, name = 'bn1',training = is_train)
    x = tf.nn.relu(features = x, name = 'relu1')

    x = layers.conv2d(inputs = x, filters = num_filter2, kernel_size = (3, 3), padding = 'same', dilation_rate = dilation, name='conv2', use_bias=False)
    x = layers.batch_normalization(inputs = x, momentum=0.9,epsilon = eps, name = 'bn2',training = is_train)
    x = tf.nn.relu(features = x, name = 'relu2')

    x = layers.conv2d(inputs = x, filters = num_filter3, kernel_size = (1, 1), name= 'conv3', use_bias = False)
    x = layers.batch_normalization(inputs = x, momentum=0.9,epsilon = eps, name = 'bn3',training = is_train)

    x = x + input_tensor
    x = tf.nn.relu(features = x, name = 'relu4')
    return x

def dilated_conv_block(input_tensor,filters, dilation,is_train):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1e-5
    num_filter1, num_filter2, num_filter3 = filters

    x = layers.conv2d(inputs = input_tensor, filters = num_filter1, kernel_size = (1, 1), name = 'conv1', use_bias = False)
    x = layers.batch_normalization(inputs = x, epsilon = eps, momentum=0.9, name= 'bn1',training = is_train)
    x = tf.nn.relu(features = x, name = 'relu1')

    x = layers.conv2d(inputs = x, filters = num_filter2, kernel_size = (3, 3), padding = 'same', dilation_rate = dilation, name= 'conv2',use_bias = False)
    x = layers.batch_normalization(inputs = x, epsilon = eps, momentum=0.9,  name= 'bn2',training = is_train)
    x = tf.nn.relu(features = x, name = 'relu2')

    x = layers.conv2d(inputs = x, filters = num_filter3, kernel_size = (1, 1), name = 'conv3',use_bias=False)
    x = layers.batch_normalization(inputs = x, epsilon = eps, momentum=0.9, name = 'bn3',training = is_train)

    shortcut = layers.conv2d(inputs = input_tensor, filters = num_filter3, kernel_size = (1, 1), name = 'conv4',use_bias=False)
    shortcut = layers.batch_normalization(inputs = shortcut, epsilon = eps, momentum=0.9, name = 'bn4',training = is_train)

    x = x + shortcut
    x = tf.nn.relu(features = x, name = 'relu4')
    return x

def conv_block(input_tensor, filters, stride,is_train):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''

    eps = 1e-5
    num_filter1, num_filter2, num_filter3 = filters

    x = layers.conv2d(inputs = input_tensor, filters = num_filter1, kernel_size = (1, 1), name = 'conv1', use_bias = False)
    x = layers.batch_normalization(inputs = x, epsilon = eps, momentum=0.9, name =  'bn1',training = is_train)
    x = tf.nn.relu(features = x, name = 'relu1')

    x = layers.conv2d(inputs = x, filters = num_filter2, kernel_size = (3, 3),padding = 'same', strides= stride, name = 'conv2', use_bias=False)
    x = layers.batch_normalization(inputs = x, epsilon=eps, momentum=0.9, name = 'bn2',training = is_train)
    x = tf.nn.relu(features = x, name = 'relu2')

    x = layers.conv2d(inputs = x, filters = num_filter3, kernel_size = (1, 1), name =  'conv3', use_bias=False)
    x = layers.batch_normalization(inputs = x, epsilon=eps, momentum=0.9,  name = 'bn3',training = is_train)

    shortcut = layers.conv2d(inputs = input_tensor, filters = num_filter3, kernel_size = (1, 1), strides= stride, name= 'conv4', use_bias=False)
    shortcut = layers.batch_normalization(inputs = shortcut, epsilon=eps, momentum=0.9, name= 'bn4',training = is_train)

    x = x + shortcut
    x = tf.nn.relu(features = x, name = 'relu4')
    return x

def conv_layer(input_tensor,is_train):
    eps = 1e-5
    x = layers.conv2d(inputs = input_tensor,filters = 64, kernel_size = (3, 3), strides=(2, 2), padding='same', use_bias=False, name='conv1')
    x = layers.batch_normalization(inputs = x, momentum = 0.9, epsilon=eps, name='bn2',training = is_train)
    x = tf.nn.relu(features = x, name = 'relu3')
    
    x = layers.conv2d(inputs = x, filters = 64, kernel_size = (3, 3), strides=(1, 1), padding='same', use_bias=False, name='conv4')
    x = layers.batch_normalization(inputs = x, momentum = 0.9, epsilon=eps, name='bn5',training = is_train)
    x = tf.nn.relu(features = x, name = 'relu6')
    
    x = layers.conv2d(inputs = x, filters = 128, kernel_size = (3, 3), strides=(1, 1), padding='same', use_bias=False,  name='conv7')
    return x


def model_fn(features, labels, mode, params):
    ''' Model function for fcn_resnet101_models'''

    if mode == tf.estimator.ModeKeys.TRAIN:
        train = True
    else:
        train = False

    eps = 1e-5
    
    img_input = tf.reshape(features, [-1, params["crop_size"], params["crop_size"], 3])

    # stage 0
    with tf.variable_scope('stage0'):
        conv_1 = conv_layer(img_input,is_train = train)    
        conv_1 = layers.batch_normalization(inputs = conv_1, epsilon = eps, momentum=0.9, name='bn8',training = train)
        conv_1 = tf.nn.relu(features = conv_1, name = 'relu9')
        conv_1 = layers.max_pooling2d (inputs = conv_1, pool_size = (3, 3), strides=(2, 2), padding='same', name='pool10')

    # stage 1
    with tf.variable_scope('stage1'):
        layer_1 = conv_block(conv_1, [64, 64, 256], stride=(1, 1),is_train = train)
        for i in range(1,3):
            with tf.variable_scope('layer{}'.format(i)):
                layer_1 = identity_block(layer_1, [64, 64, 256], is_train = train)


    # stage 2
    with tf.variable_scope('stage2'):
        layer_2 = conv_block(layer_1, [128, 128, 512], stride=(2, 2),is_train = train)
        for i in range(1,4):
            with tf.variable_scope('layer{}'.format(i)):
                layer_2 = identity_block(layer_2, [128, 128, 512], is_train = train)

    # stage 3
    with tf.variable_scope('stage3'):
        layer_3 = dilated_conv_block(layer_2, [256, 256, 1024], dilation = (1,1),is_train = train)
        for i in range(1,23):
            with tf.variable_scope('layer{}'.format(i)):
                layer_3 = dilated_identity_block(layer_3, [256, 256, 1024], dilation = (2,2),is_train = train)

    # stage 4
    with tf.variable_scope('stage4'):
        layer_4 = dilated_conv_block(layer_3, [512, 512, 2048], dilation = (2,2),is_train = train)
        for i in range(1,3):
            with tf.variable_scope('layer{}'.format(i)):
                layer_4 = dilated_identity_block(layer_4, [512, 512, 2048], dilation = (4,4),is_train = train)
   
    x = fcn_head(layer_4,512,21,is_train = train)
    logits = tf.image.resize_bilinear(images = x, size = [params["crop_size"],params["crop_size"]])

    auxout = fcn_head(layer_3,256,21,is_train = train)
    aux_logits = tf.image.resize_bilinear(images = auxout,size = [params["crop_size"],params["crop_size"]])

    predictions = tf.argmax(logits, axis=-1)


    # Setup the estimator according to the phase (Train, eval)
    loss = None
    train_op = None
    eval_metric_ops = {}

    # compute loss(train and eval)
    loss1 = softmax_sparse_crossentropy_ignoring_last_label(labels,logits)
    loss2 = softmax_sparse_crossentropy_ignoring_last_label(labels,aux_logits)
    loss = loss1 + 0.5 * loss2

    # make loss available to TensorBoard in both TRAIN and EVAL modes
    tf.summary.scalar('cross_entropy', loss)    
    
    # evaluation metric
    miou, update_op = mIOU(logits,labels,classes=params["num_classes"])


    # configure training
    if mode == tf.estimator.ModeKeys.TRAIN:
        # learning rate scheduler
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.001
        end_learning_rate = 0
        decay_steps = params["train_epoch"] * params["num_training_examples"]
        learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                            decay_steps, end_learning_rate,
                                            power=0.9)
        # SGD + momentum optimizer
        optimizer = tf.train.MomentumOptimizer(learning_rate,momentum = 0.9)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode  == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'miou': (miou, update_op)
        }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs=None,
    )
