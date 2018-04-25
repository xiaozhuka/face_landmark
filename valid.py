# -*- coding: utf-8 -*-
import tensorflow as tf

from config import ModelConfig

from dataset import get_dataset_from_h5, random_augmentation
from model import OrdinaryCNNModel
import datetime
import time
import os

from dataset import *

import numpy as np
from cv2 import resize, imread

tf.logging.set_verbosity(tf.logging.INFO)

####################################################################
#####################  model config#################################
####################################################################
model_config = ModelConfig()
model_config.summary_path = r'E:\python_vanilla\log_dir\test_2018_4_25_2'
# model_config.summary_path = r'/home/cuishi/zcq/python_vanilla/log_dir/test'
if not os.path.isdir(model_config.summary_path):
    os.mkdir(model_config.summary_path)
model_config.width = 40
model_config.height = 40
model_config.channels = 3
model_config.min_save_name = os.path.join(model_config.summary_path, 'min_loss.model')
model_config.batch_size = 64+32
model_config.landmark_num = 21
model_config.weight_decay = 1e-3
model_config.bias_decay = 1e-3
model_config.train_path = r'E:\data\ImageList_facepose_25pointsSELECTED_train_ex8_RESIZE50_noFlip_noShuffle.h5'
model_config.test_path = r'E:\data\ImageList_facepose_25pointsSELECTED_test_exMODI8_RESIZE50_noFlip_noShuffle.h5'
# model_config.train_path = r'/home/cuishi/zcq/data/ImageList_facepose_25pointsSELECTED_train_ex8_RESIZE50_noFlip_noShuffle.h5'
# model_config.test_path = r'/home/cuishi/zcq/data/ImageList_facepose_25pointsSELECTED_test_exMODI8_RESIZE50_noFlip_noShuffle.h5'
model_config.data_paths = ['data', 'landmarks']
model_config.max_epoch = 500
model_config.shuffle_buffer_size = 256
model_config.clip_grad = False
model_config.summary_frequency = 100
model_config.padding_way = 'SAME'
model_config.learning_rate = 1e-3
model_config.valid_dataset = r"E:\fld_result\facepose"
model_config.train_ratio = 0.9
model_config.restore = False
model_config.bn = False
model_config.loss_amp = 10.0


##############################################################
####################  sess config  ###########################
##############################################################
sess_config = tf.ConfigProto()
sess_config.allow_soft_placement = True
sess_config.gpu_options.allow_growth = True
sess_config.gpu_options.allocator_type = "BFC"
# sess_config.log_device_placement = True


# with tf.device('/gpu:2'):
##################################################
############## build model #######################
##################################################
images = tf.placeholder(tf.float32, [None, model_config.width, model_config.height, model_config.channels], 'images')
labels = tf.placeholder(tf.float32, [None, 2*model_config.landmark_num], 'labels')
# labels = tf.placeholder(tf.float32, [None, model_config.width, model_config.height, model_config.landmark_num], 'labels')
is_training = tf.Variable(True, dtype=tf.bool)
model = OrdinaryCNNModel(model_config, images, labels, is_training)
model.build()
loss = model.cost
face_feature_result = model.output
# ----------------------------------------------------
# learning rate
global_step = tf.Variable(
    initial_value=0,
    name="global_step",
    trainable=False,
    collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

# -------------------------------------------------------
# is_training op
assing_is_training_true_op = tf.assign(is_training, True)
assing_is_training_false_op = tf.assign(is_training, False)


# Validation
saver_restore = tf.train.Saver()
with tf.Session() as sess:

    saver_restore.restore(sess, model_config.min_save_name)

    sess.run(assing_is_training_false_op)
    for single_img in os.listdir(model_config.valid_dataset):
        if not single_img.endswith('.jpg'):
            continue
        if not single_img.startswith('dlib_'):
            continue
        img_ori = imread(os.path.join(model_config.valid_dataset, single_img))
        img_ori_shape = img_ori.shape
        img_resized = img_ori.copy()
        img_resized = resize(img_resized, (model_config.height, model_config.width))
        img_feed = img_feed[None, ...]
        val_result = sess.run([face_feature_result], feed_dict={images: img_feed})

        landmark = val_result[0]

        for t in range(int(model_config.landmark_num)):
            cv2.circle(img_ori, (int(round(img_ori_shape[1]*landmark[0, 2*t])),
                                 int(round(img_ori_shape[0]*landmark[0, 2*t+1]))), 3, (0, 0, 255), -1)
        imwrite(os.path.join(model_config.summary_path, single_img), img_ori.astype(np.uint8))

        for t in range(int(model_config.landmark_num)):
            cv2.circle(img_resized, (int(round(model_config.width*landmark[0, 2*t])),
                                     int(round(model_config.height*landmark[0, 2*t+1]))), 1, (0, 0, 255), -1)
        imwrite(os.path.join(model_config.summary_path, 'resized_'+single_img), img_ori.astype(np.uint8))