# DenseNet network implemented with tensorflow
# taehun kim
# taehoon1018@postech.ac.kr

import tensorflow as tf

# bottleneck module
def bottleneck(input, filters, dropout_rate, training, scope):
    with tf.variable_scope(scope):
        with tf.variable_scope('conv1'):
            input = tf.layers.batch_normalization(input, training=training)
            input = tf.nn.relu(input)
            input = tf.layers.conv2d(input, filters=filters * 4, kernel_size=[1, 1], strides=[1, 1], padding='SAME')
            input = tf.layers.dropout(input, rate=dropout_rate, training=training)

        with tf.variable_scope('conv2'):
            input = tf.layers.batch_normalization(input, training=training)
            input = tf.nn.relu(input)
            input = tf.layers.conv2d(input, filters=filters, kernel_size=[3, 3], strides=[1, 1], padding='SAME')
            input = tf.layers.dropout(input, rate=dropout_rate, training=training)

    return input

# transition module
def transition(input, filters, compression_factor, dropout_rate, training, scope):
    with tf.variable_scope(scope):
        input = tf.layers.batch_normalization(input, training=training)
        input = tf.nn.relu(input)
        input = tf.layers.conv2d(input, filters=int(filters * compression_factor), kernel_size=[1, 1])
        input = tf.layers.dropout(input, rate=dropout_rate, training=training)
        input = tf.layers.average_pooling2d(input, pool_size=[2, 2], strides=2, padding='VALID')

        return input

# dense block module
def dense_block(input, layers, filters, dropout_rate, training, scope):
    with tf.variable_scope(scope):
        layer_stack = []

        for layer in range(layers):
            if layer == 0:
                layer_stack = input
            bottleneck_ = bottleneck(layer_stack, filters=filters, dropout_rate=dropout_rate, training=training,
                                     scope='bottleneck_' + str(layer))
            layer_stack = tf.concat([layer_stack, bottleneck_], axis=-1)

        # return bottleneck_
        return layer_stack

#
def model(input, blocks, layers, filters, classes, compression_factor, dropout_rate, init_subsample, training):
    if init_subsample is True:
        input = tf.layers.conv2d(input, filters=filters * 2, kernel_size=[7, 7], strides=[2, 2], padding='SAME')
        input = tf.layers.batch_normalization(input, training=training)
        input = tf.nn.relu(input)
        input = tf.layers.max_pooling2d(input, pool_size=[3, 3], strides=[2, 2], padding='SAME')
    else:
        input = tf.layers.conv2d(input, filters=filters * 2, kernel_size=[3, 3], strides=[1, 1], padding='SAME')


    if len(layers) != blocks:
        raise ValueError('Expected list, but passed', type(layers), 'instead.')
    for block in range(blocks):
        input = dense_block(input, layers[block], filters, dropout_rate, training, scope='dense_block_' + str(block))
        if block != blocks - 1:
            input = transition(input, input.shape[-1], compression_factor, dropout_rate, training, scope='transition_' + str(block))

    input = tf.layers.batch_normalization(input, training=training)
    input = tf.nn.relu(input)
    input = tf.layers.average_pooling2d(input, pool_size=input.shape[1:3], strides=[1, 1])
    input = tf.layers.flatten(input)
    input = tf.layers.dense(input, units=classes)

    return input