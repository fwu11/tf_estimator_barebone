import tensorflow as tf

def softmax_sparse_crossentropy_ignoring_last_label(y_true,y_pred):
    y_pred = tf.reshape(y_pred,(-1,y_pred.shape.as_list()[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = tf.one_hot(tf.to_int32(tf.reshape(y_true,[-1])),y_pred.shape.as_list()[-1]+1)
    unpacked = tf.unstack(y_true,axis=-1)
    y_true = tf.stack(unpacked[:-1],axis = -1)

    cross_entropy = -tf.reduce_sum(y_true * log_softmax, axis = 1)
    loss = tf.reduce_mean(cross_entropy)

    return loss

'''
# Groud Truth: ignoring all labels greater or equal than n_classes
label_proc = prepare_label(self.label_batch, output_size, num_classes=self.conf.num_classes, one_hot=False)
raw_gt = tf.reshape(label_proc, [-1,])
indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, self.conf.num_classes - 1)), 1)
gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
raw_prediction = tf.reshape(raw_output, [-1, self.conf.num_classes])
prediction = tf.gather(raw_prediction, indices)

# Pixel-wise softmax_cross_entropy loss
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
# L2 regularization
l2_losses = [self.conf.weight_decay * tf.nn.l2_loss(v) for v in all_trainable if 'weights' in v.name]
# Loss function
self.reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)
'''