import tensorflow as tf

'''
def mIOU(pred,label,classes):
    prediction = tf.reshape(pred, [-1,])
    gt = tf.reshape(label, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(gt, classes - 1)), 1)  # ignore all labels >= num_classes
    gt = tf.cast(tf.gather(gt, indices), tf.int32)
    prediction = tf.gather(prediction, indices)

    if not tf.contrib.distribute.has_distribution_strategy():
        mIoU, update_op = tf.metrics.mean_iou(labels = gt, predictions = prediction, num_classes = classes)
    else:
        # Metrics in the old version not compatible with distribution strategies during
        # training. This does not affect the overall performance of the model.
        mIoU = tf.constant(0)
        update_op = tf.no_op()
    return mIoU, update_op
'''
def mIOU(raw_output,label,classes,image_batch):
    # predictions
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output = tf.argmax(raw_output, axis=3)
    pred = tf.expand_dims(raw_output, axis=3)
    pred = tf.reshape(pred, [-1,])
    # labels
    gt = tf.reshape(label, [-1,])
    # Ignoring all labels greater than or equal to n_classes.
    temp = tf.less_equal(gt, classes - 1)
    weights = tf.cast(temp, tf.int32)

    # fix for tf 1.3.0
    gt = tf.where(temp, gt, tf.cast(temp, tf.uint8))

    # mIoU
    if not tf.contrib.distribute.has_distribution_strategy():
        mIoU, update_op = tf.metrics.mean_iou(labels = gt, predictions = pred, num_classes = classes, weights=weights)
    else:
        # Metrics in the old version not compatible with distribution strategies during
        # training. This does not affect the overall performance of the model.
        mIoU = tf.constant(0)
        update_op = tf.no_op()

    return mIoU, update_op


