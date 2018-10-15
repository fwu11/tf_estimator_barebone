import tensorflow as tf

def mIOU(pred,label,classes):
    prediction = tf.reshape(pred, [-1,])
    gt = tf.reshape(label, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(gt, classes - 1)), 1)  # ignore all labels >= num_classes
    gt = tf.cast(tf.gather(gt, indices), tf.int32)
    prediction = tf.gather(prediction, indices)
    mIoU, update_op = tf.metrics.mean_iou(labels = gt, predictions = prediction, num_classes = classes)
    
    return mIoU, update_op


