# -*- coding: utf-8 -*-


import tensorflow.contrib.slim as slim

from tensorpack import (ModelDesc, get_current_tower_context,
                        regularize_cost_from_collection, get_global_step_var)
import tensorflow as tf

from dataset import get_dataset

class OrdinaryCNNModel(ModelDesc):
    """CNN model.
    """
    def __init__(self, config):
        self.config = config

    def inputs(self):
        return [tf.placeholder(tf.float32, [None, self.config.width, self.config.height, self.config.channels], 'images'),
                tf.placeholder(tf.float32, [None, self.config.target_length], 'labels')]

    def build_graph(self, images, labels):
        """Build graph use tf
        """
        ctx = get_current_tower_context()
        is_training = ctx.is_training

        # images =

        with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected],
                             weights_regularizer=tf.contrib.layers.l2_regularizer(self.config.weight_decay),
                             weights_initializer=tf.contrib.layers.xavier_initializer(),
                             biases_regularizer=tf.contrib.layers.l2_regularizer(self.config.bias_decay),
                             biases_initializer=tf.zeros_initializer(),
                             activation_fn=tf.nn.relu):
            net = images
            # build layers
            # conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME')
            net = slim.layers.conv2d(net, 256, [3, 3], padding='VALID', scope='conv1_1')
            net = slim.layers.dropout(net, is_training=is_training)

            net = slim.layers.conv2d(net, 256, [3, 3], padding='VALID', scope='conv1_2')
            tf.summary.histogram('conv1_output', net)
            net = slim.layers.dropout(net, is_training=is_training)

            # max_pool2d(inputs, kernel_size, stride=2, padding='VALID')
            net = slim.layers.max_pool2d(net, [2, 2], scope='pool1')

            net = slim.layers.conv2d(net, 128, [3, 3], padding='VALID', scope='conv2_1')
            net = slim.layers.dropout(net, is_training=is_training)

            net = slim.layers.conv2d(net, 128, [3, 3], padding='VALID', scope='conv2_2')
            tf.summary.histogram('conv2_output', net)
            net = slim.layers.dropout(net, is_training=is_training)

            net = slim.layers.max_pool2d(net, [2, 2], scope='pool2')

            net = slim.layers.conv2d(net, 64, [3, 3], padding='VALID', scope='conv3_1')
            net = slim.layers.dropout(net, is_training=is_training)

            net = slim.layers.conv2d(net, 64, [3, 3], padding='VALID', scope='conv3_2')
            tf.summary.histogram('conv3_output', net)

            net = slim.flatten(net, scope='flatten')
            net = slim.layers.max_pool2d(net, [2, 2], scope='pool3')

            net = slim.layers.fully_connected(net, 128, scope='fc1')
            tf.summary.histogram('fc1_output', net)
            net = slim.layers.fully_connected(net, 50, scope='fc2')
            tf.summary.histogram('fc2_output', net)

        # define loss
        logits = tf.identity(net, name='logits_export')
        self.loss = tf.losses.mean_squared_error(labels, logits)
        tf.summary.scalar('loss', self.loss)

        self.cost = self.loss + regularize_cost_from_collection()

    def optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=468 * 10,
            decay_rate=0.3, staircase=True, name='learning_rate')
        return tf.train.AdamOptimizer(lr)

def get_data():
    return get_dataset()
