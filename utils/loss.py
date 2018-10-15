import tensorflow as tf
import tensorflow.keras.backend as K 

def softmax_sparse_crossentropy_ignoring_last_label(y_true,y_pred):
    y_pred = K.reshape(y_pred,(-1,K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true,axis=-1)
    y_true = tf.stack(unpacked[:-1],axis = -1)

    cross_entropy = -K.sum(y_true * log_softmax, axis = 1)
    loss = K.mean(cross_entropy)

    return loss