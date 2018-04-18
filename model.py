# -*- coding: utf-8 -*-


import tensorflow.contrib.slim as slim

import tensorflow as tf

from tensorflow.contrib.slim import batch_norm

class OrdinaryCNNModel():
    """CNN model.
    """
    def __init__(self, config, images, labels, is_training):
        self.config = config
        self.images = images
        self.labels = labels
        self.is_training = is_training


    def build_graph(self):
        """Build graph use tf
        """
        images, labels = self.images, self.labels

        # tf.summary.image("images", images, max_outputs=self.config.batch_size)

        # images = (images - 0.5) * 2

        is_training = self.is_training

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                             weights_regularizer=tf.contrib.layers.l2_regularizer(self.config.weight_decay),
                             weights_initializer=tf.contrib.layers.xavier_initializer(),
                             biases_regularizer=tf.contrib.layers.l2_regularizer(self.config.bias_decay),
                             biases_initializer=tf.zeros_initializer(),
                             activation_fn=tf.nn.relu):
            net = images
            # build layers
            # conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME')
            net = slim.conv2d(net, 512, [3, 3], padding=self.config.padding_way, scope='conv1_1')
            # net = batch_norm(net, is_training=is_training)
            net = slim.layers.dropout(net, is_training=is_training)

            net = slim.conv2d(net, 512, [3, 3], padding=self.config.padding_way, scope='conv1_2')
            # net = batch_norm(net, is_training=is_training)
            tf.summary.histogram('conv1_output', net)
            net = slim.layers.dropout(net, is_training=is_training)

            # max_pool2d(inputs, kernel_size, stride=2, padding='VALID')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            net = slim.conv2d(net, 512, [3, 3], padding=self.config.padding_way, scope='conv2_1')
            # net = batch_norm(net, is_training=is_training)
            net = slim.layers.dropout(net, is_training=is_training)

            net = slim.conv2d(net, 256, [3, 3], padding=self.config.padding_way, scope='conv2_2')
            # net = batch_norm(net, is_training=is_training)
            tf.summary.histogram('conv2_output', net)
            net = slim.layers.dropout(net, is_training=is_training)

            net = slim.max_pool2d(net, [2, 2], scope='pool2')

            net = slim.conv2d(net, 256, [3, 3], padding=self.config.padding_way, scope='conv3_1')
            # net = batch_norm(net, is_training=is_training)
            net = slim.layers.dropout(net, is_training=is_training)

            net = slim.conv2d(net, 128, [3, 3], padding=self.config.padding_way, scope='conv3_2')
            net = batch_norm(net, is_training=is_training)
            tf.summary.histogram('conv3_output', net)

            net = slim.max_pool2d(net, [2, 2], scope='pool3')

            net = slim.conv2d(net, 128, [3, 3], padding=self.config.padding_way, scope='conv4_1')
            # net = batch_norm(net, is_training=is_training)
            # net = slim.layers.dropout(net, is_training=is_training)
            #
            # net = slim.conv2d(net, 64, [3, 3], padding=self.config.padding_way, scope='conv4_2')
            # tf.summary.histogram('conv4_output', net)

            net = slim.max_pool2d(net, [2, 2], scope='pool4')

            net = slim.flatten(net, scope='flatten')

            net = slim.fully_connected(net, 256, scope='fc1')
            # tf.summary.histogram('fc1_output', net)
            net = slim.fully_connected(net, self.config.landmark_num * 2, activation_fn=None, scope='fc2')
            tf.summary.histogram('fc2_output', net)

        self.output = net

        self.loss = tf.losses.mean_squared_error(labels, self.output)
        tf.summary.scalar('loss', self.loss)

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.cost = tf.add(tf.reduce_sum(self.loss), tf.reduce_sum(regularization_losses))


    def build(self):
        self.build_graph()

