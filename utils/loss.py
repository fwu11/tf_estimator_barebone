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
