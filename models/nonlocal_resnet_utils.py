# implementation of the "non-local" block
# this "non-local" block is added to the 2D ConvNet

import tensorflow as tf
slim = tf.contrib.slim

def _nonlocal_block(net, dim_inner, embed = True, softmax = False, maxpool = 2, scope = None):
    '''
    Args:
        input: Input into the block. Tensor with shape (B,H,W,C)
        dim_inner: Number of bottleneck channels.
        embed: Whether or not use the "embedding"
        softmax: Whether or not to use the softmax operation which makes it
               equivalent to soft-attention.
        maxpool: How large of a max-pooling to use to help reduce
               the computational burden. Default is 2, use False for none.
        scope: An optional scope for all created variables.
    Returns:
        A spatial non-local block.
    '''
    with tf.variable_scope(scope, 'nonlocal', values = [net]) as sc:
        with slim.arg_scope([slim.con2d], normalizer_fn = None):
            if embed:
                theta =  slim.conv2d(net, dim_inner, [1, 1], activation_fn=None, scope='theta')
                phi = slim.conv2d(net, dim_inner, [1, 1], activation_fn=None, scope='phi')
            else:
                theta = net
                phi = net
            g = slim.conv2d(net, dim_inner, [1, 1], activation_fn=None, scope='g')
        
        # subsampling after phi and g (max pooling)
        if maxpool is not False and maxpool > 1:
            phi = slim.max_pool2d(phi, [maxpool, maxpool], stride=maxpool, scope='pool_phi')
            g_orig = g = slim.max_pool2d(g, [maxpool, maxpool], stride=maxpool, scope='pool_g')
        
        # reshape (B,H,W,C) to (B,HW,C)
        theta_flat = tf.reshape(theta, [tf.shape(theta)[0], -1, tf.shape(theta)[-1]])
        phi_flat = tf.reshape(phi, [tf.shape(phi)[0], -1, tf.shape(phi)[-1]])
        g_flat = tf.reshape(g, [tf.shape(g)[0], -1, tf.shape(g)[-1]])

        theta_flat.set_shape([theta.shape[0], theta.shape[1] * theta.shape[2] if None not in theta.shape[1:3] else None, theta.shape[-1]])
        phi_flat.set_shape([phi.shape[0], phi.shape[1] * phi.shape[2] if None not in phi.shape[1:3] else None, phi.shape[-1]])
        g_flat.set_shape([g.shape[0], g.shape[1] * g.shape[2] if None not in g.shape[1:3] else None, g.shape[-1]])

        # Compute f(a, b) -> (B,HW,HW)
        f = tf.matmul(theta_flat, tf.transpose(phi_flat, [0, 2, 1]))
        if softmax:
            f = tf.nn.softmax(f)
        else:
            # replacing softmax with scaling by 1/N, N is the number of positions in x
            f = f / tf.cast(tf.shape(f)[-1], tf.float32)

        # Compute f * g ("self-attention") -> (B,HW,C)
        fg = tf.matmul(f, g_flat)
        # (B,HW,C) -> (B,H,W,C)
        fg = tf.reshape(fg, tf.shape(g_orig))

        # Go back up to the original depth, add residual, zero-init.
        # batch normalization after Wz
        with slim.arg_scope([slim.batch_norm], param_initializers={'gamma': tf.zeros_initializer()}):
            fg = slim.conv2d(fg, net.shape[-1], [1, 1], activation_fn=None,normalizer_fn=slim.batch_norm, scope='fg')
        
        net = net + fg

        return slim.utils.collect_named_outputs(None, sc.name, net)
