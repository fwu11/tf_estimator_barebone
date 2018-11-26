import tensorflow as tf
from utils.loss import *
from utils.metrics import *
import warnings
warnings.filterwarnings('ignore')

"""
This script defines the segmentation network.
The encoding part is a pre-trained ResNet. This script supports several settings (you need to specify in main.py):
	Original ResNet-101 ('resnet_v1_101.ckpt')

You may find the download links in README.
To use the pre-trained models, the name of each layer is the same as that in .ckpy file.
"""
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
                outputs = self._dilated_bottle_resblock(outputs, 1024, 2, 'unit_1',	identity_connection=False)
                for i in range(2, 24):
                    outputs = self._dilated_bottle_resblock(outputs, 1024, 2, 'unit_%d' % i)
            with tf.variable_scope('block4') as scope:
                outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_1', identity_connection=False)
                outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_2')
                outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_3')
                return outputs
    
    def build_decoder(self, encoding):
        with tf.variable_scope('decoder') as scope:
            outputs = self._ASPP(encoding, self.num_classes, [6, 12, 18, 24])
            return outputs

	# blocks
    def _start_block(self, name):
        outputs = self._conv2d(self.inputs, 7, 64, 2, name=name)
        outputs = self._batch_norm(outputs, name=name, is_training=False, activation_fn=tf.nn.relu)
        outputs = self._max_pool2d(outputs, 3, 2, name='pool1')
        return outputs

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

    def _ASPP(self, x, num_o, dilations):
        o = []
        for i, d in enumerate(dilations):
            o.append(self._dilated_conv2d(x, 3, num_o, d, name='aspp/conv%d' % (i+1), biased=True))
        return self._add(o, name='aspp/add')

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
        
    def _batch_norm(self, x, name, is_training, activation_fn, trainable=False):
        # For a small batch size, it is better to keep 
        # the statistics of the BN layers (running means and variances) frozen, 
        # and to not update the values provided by the pre-trained model by setting is_training=False.
        # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
        # if they are presented in var_list of the optimiser definition.
        # Set trainable = False to remove them from trainable_variables.
        with tf.variable_scope(name+'/BatchNorm') as scope:
            o = tf.contrib.layers.batch_norm(x,scale=True,activation_fn=activation_fn,is_training=is_training,trainable=trainable,scope=scope)
            return o

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
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
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
    
    img_input = tf.reshape(features, [-1, params["crop_size"], params["crop_size"], 3])

    # Create network
    net = ResNet_segmentation(img_input, params["num_classes"], train, 'res101')
    # Variables that load from pre-trained model.
    restore_var = [v for v in tf.global_variables() if 'resnet_v1' in v.name]
    # Trainable Variables
    all_trainable = tf.trainable_variables()
    # Fine-tune part
    encoder_trainable = [v for v in all_trainable if 'resnet_v1' in v.name] # lr * 1.0
    # Decoder part
    decoder_trainable = [v for v in all_trainable if 'decoder' in v.name]
    
    decoder_w_trainable = [v for v in decoder_trainable if 'weights' in v.name or 'gamma' in v.name] # lr * 10.0
    decoder_b_trainable = [v for v in decoder_trainable if 'biases' in v.name or 'beta' in v.name] # lr * 20.0
    # Check
    assert(len(all_trainable) == len(decoder_trainable) + len(encoder_trainable))
    assert(len(decoder_trainable) == len(decoder_w_trainable) + len(decoder_b_trainable))

    # predictions
    raw_output  = net.outputs

    # Output size
    output_shape = tf.shape(raw_output)
    output_size = (output_shape[1], output_shape[2])

    predictions = tf.argmax(raw_output, axis=-1)

    # Setup the estimator according to the phase (Train, eval)
    reduced_loss = None
    train_op = None
    eval_metric_ops = {}

    # Groud Truth: ignoring all labels greater or equal than n_classes
    label_proc = prepare_label(labels, output_size, num_classes=21, one_hot=False)
    raw_gt = tf.reshape(label_proc, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, 21 - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    raw_prediction = tf.reshape(raw_output, [-1, 21])
    prediction = tf.gather(raw_prediction, indices)

    # compute loss(train and eval)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    # L2 regularization
    l2_losses = [0.0005 * tf.nn.l2_loss(v) for v in all_trainable if 'weights' in v.name]
    # Loss function
    reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)


    # evaluation metric
    miou, update_op = mIOU(raw_output,labels,params["num_classes"],img_input)


    # configure training
    if mode == tf.estimator.ModeKeys.TRAIN:
        # learning rate scheduler
        global_step = tf.train.get_or_create_global_step()
        starter_learning_rate = 5e-4
        #starter_learning_rate = 0.001
        end_learning_rate = 0
        decay_steps = params["train_epoch"] * params["num_training_examples"]
        learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                            decay_steps, end_learning_rate,
                                            power=0.9)

        # We have several optimizers here in order to handle the different lr_mult
        # which is a kind of parameters in Caffe. This controls the actual lr for each
        # layer.
        opt_encoder = tf.train.MomentumOptimizer(learning_rate, momentum = 0.9)
        opt_decoder_w = tf.train.MomentumOptimizer(learning_rate * 10.0, momentum = 0.9)
        opt_decoder_b = tf.train.MomentumOptimizer(learning_rate * 20.0, momentum = 0.9)
        # To make sure each layer gets updated by different lr's, we do not use 'minimize' here.
        # Instead, we separate the steps compute_grads+update_params.
        # Compute grads
        grads = tf.gradients(reduced_loss, encoder_trainable + decoder_w_trainable + decoder_b_trainable)
        grads_encoder = grads[:len(encoder_trainable)]
        grads_decoder_w = grads[len(encoder_trainable) : (len(encoder_trainable) + len(decoder_w_trainable))]
        grads_decoder_b = grads[(len(encoder_trainable) + len(decoder_w_trainable)):]
        # Update params
        train_op_conv = opt_encoder.apply_gradients(zip(grads_encoder, encoder_trainable), global_step=tf.train.get_global_step())
        train_op_fc_w = opt_decoder_w.apply_gradients(zip(grads_decoder_w, decoder_w_trainable), global_step=tf.train.get_global_step())
        train_op_fc_b = opt_decoder_b.apply_gradients(zip(grads_decoder_b, decoder_b_trainable), global_step=tf.train.get_global_step())
        # Finally, get the train_op!
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for collecting moving_mean and moving_variance
        with tf.control_dependencies(update_ops):
            train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)

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