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
min_test_cost = 1e4
# ONLY_VALID = True

####################################################################
#####################  model config#################################
####################################################################
model_config = ModelConfig()
model_config.summary_path = r'E:\python_vanilla\log_dir\test_2018_4_18_2'
# model_config.summary_path = r'/home/cuishi/zcq/python_vanilla/log_dir/test'
if not os.path.isdir(model_config.summary_path):
    os.mkdir(model_config.summary_path)
model_config.width = 40
model_config.height = 40
model_config.channels = 3
model_config.min_save_name = os.path.join(model_config.summary_path, 'min_loss.model')
model_config.batch_size = 64
model_config.landmark_num = 21
model_config.weight_decay = 1e-3
model_config.bias_decay = 1e-3
model_config.train_path = r'E:\data\ImageList_facepose_25pointsSELECTED_train_ex8_RESIZE50_noFlip_noShuffle.h5'
model_config.test_path = r'E:\data\ImageList_facepose_25pointsSELECTED_test_exMODI8_RESIZE50_noFlip_noShuffle.h5'
# model_config.train_path = r'/home/cuishi/zcq/data/ImageList_facepose_25pointsSELECTED_train_ex8_RESIZE50_noFlip_noShuffle.h5'
# model_config.test_path = r'/home/cuishi/zcq/data/ImageList_facepose_25pointsSELECTED_test_exMODI8_RESIZE50_noFlip_noShuffle.h5'
model_config.data_paths = ['data', 'landmarks']
model_config.max_epoch = 15
model_config.shuffle_buffer_size = 512 + 256
model_config.clip_grad = False
model_config.summary_frequency = 100
model_config.padding_way = 'SAME'
model_config.learning_rate = 1e-3
model_config.valid_dataset = r"E:\fld_result\facepose"
model_config.train_ratio = 0.8
model_config.restore = False


##############################################################
####################  sess config  ###########################
##############################################################
sess_config = tf.ConfigProto()
sess_config.allow_soft_placement = True
sess_config.gpu_options.allow_growth = True
sess_config.gpu_options.allocator_type = "BFC"
# sess_config.log_device_placement = True


###########################################################
################## get dataset ############################
###########################################################
# train_ds, test_ds, info = get_dataset_from_h5(model_config)
train_ds, test_ds, info = get_dataset_from_file(model_config)
tf.logging.info(info)
train_iterator = train_ds.make_initializable_iterator()
test_iterator = test_ds.make_initializable_iterator()
train_next_element = train_iterator.get_next()
test_next_element = test_iterator.get_next()


##################################################
############## build model #######################
##################################################
images = tf.placeholder(tf.float32, [None, model_config.width, model_config.height, model_config.channels], 'images')
labels = tf.placeholder(tf.float32, [None, 2*model_config.landmark_num], 'labels')
is_training = tf.Variable(True, dtype=tf.bool)
model = OrdinaryCNNModel(model_config, images, labels, is_training)
model.build()
loss = model.cost
face_feature_result = model.output
# test_loss = tf.Variable(initial_value=0,
#                         name='test_loss',
#                         trainable=False,
#                         dtype=tf.float32)


# ----------------------------------------------------
# learning rate
global_step = tf.Variable(
    initial_value=0,
    name="global_step",
    trainable=False,
    collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
boundaries = [x for x in np.array([int(2000),
                                     int(21000)],
                                    dtype=np.int32)]
staged_lr = [x for x in [0.005, 1e-3, 1e-4]]
learning_rate = tf.train.piecewise_constant(global_step,
                                              boundaries, staged_lr)
# learning_rate = tf.train.exponential_decay(model_config.learning_rate,
#                                            global_step=global_step,
#                                            decay_steps=100,decay_rate=0.9)
tf.summary.scalar('learning_rate', learning_rate)

# ---------------------------------------------------- #
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.8, use_nesterov=True)
grads = optimizer.compute_gradients(loss=loss)
for i, (g, v) in enumerate(grads):
    if g is not None:
        if model_config.clip_grad:
            g = tf.clip_by_norm(g, model_config.clip_grad)
        grads[i] = (g, v)  # clip gradients
        tf.summary.histogram('gradient/' + v.name[:-2], g)

updates_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(updates_ops):
    train_op = optimizer.apply_gradients(grads, global_step=global_step)
merged_summary_op = tf.summary.merge_all()


# -------------------------------------------------------
# is_training op
assing_is_training_true_op = tf.assign(is_training, True)
assing_is_training_false_op = tf.assign(is_training, False)



saver = tf.train.Saver()
total_step = 0
with tf.Session(config=sess_config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    if model_config.restore:
        # if os.path.isfile(model_config.min_save_name):
        saver.restore(sess, model_config.min_save_name)
        # else:
        #     print("Model configuration min_save_name: %s don't exists, training from beginning."%(model_config.min_save_name))

    summary_writer = tf.summary.FileWriter(model_config.summary_path, sess.graph)

    for epoch_idx in range(model_config.max_epoch):
        sess.run(train_iterator.initializer)
        sess.run(test_iterator.initializer)
        starttime = datetime.datetime.now()

        tf.logging.info(
            "Starting train epoch: %d at time: %s" % (
                epoch_idx+1, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

        while True:
            total_step += 1
            try:
                imgs, labs = sess.run(train_next_element)
                # if imgs.shape[0] != model_config.batch_size:
                #     continue

                if (total_step > 1) and (total_step % model_config.summary_frequency == 0):
                    _, loss_, summary_str = sess.run(
                        [train_op, loss, merged_summary_op],
                        feed_dict={images: imgs,
                                   labels: labs})

                    summary_writer.add_summary(summary_str, total_step)
                else:
                    _, loss_ = sess.run([train_op, loss],
                                                 feed_dict={images: imgs,
                                                            labels: labs})
            except tf.errors.OutOfRangeError:
                endtime = datetime.datetime.now()
                tf.logging.info(
                    "Epoch %d takes %d seconds" % (
                        epoch_idx+1, (endtime - starttime).seconds))
                sess.run(assing_is_training_false_op)
                tmp_test_loss = []

                tf.logging.info(
                    "Starting validation epoch: %d at time: %s" % (
                        epoch_idx+1, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

                # test
                while True:
                    try:
                        imgs, labs = sess.run(test_next_element)

                        feed_dict = {images: imgs,
                                     labels: labs}
                        _tmp_test_loss = sess.run([loss], feed_dict=feed_dict)[0]
                        # t_l = tf.get_variable(name="test_loss")
                        tmp_test_loss.append(_tmp_test_loss)

                    except tf.errors.OutOfRangeError:
                        tmp_test_cost = np.mean(tmp_test_loss)
                        tf.logging.info('Total test loss: %f'%(tmp_test_cost))
                        if tmp_test_cost < min_test_cost:
                            min_test_cost = tmp_test_cost
                            # if this model is less than before, save it
                            tf.logging.info('Save model to %s with minimum loss: %f' % (model_config.min_save_name, tmp_test_cost))
                            saver.save(sess, model_config.min_save_name)
                        break
                break


# Validation
saver_restore = tf.train.Saver()
with tf.Session() as sess:

    saver_restore.restore(sess, model_config.min_save_name)
    for single_img in os.listdir(model_config.valid_dataset):
        if not single_img.endswith('.jpg'):
            continue
        if not single_img.startswith('dlib_'):
            continue
        img_ori = imread(os.path.join(model_config.valid_dataset, single_img))
        img_ori_shape = img_ori.shape
        img_resized = resize(img_ori, (model_config.height, model_config.width))
        img_feed = img_resized.copy()
        img_feed = (img_feed - np.mean(img_feed)) / np.std(img_feed)
        img_feed = img_feed[None, ...]
        val_result = sess.run([face_feature_result], feed_dict={images: img_feed})

        landmark = val_result[0]

        # print(landmark)
        for t in range(int(model_config.landmark_num)):
            cv2.circle(img_ori, (int(img_ori_shape[0] * landmark[0, 2*t]), int(img_ori_shape[1] *landmark[0, 2*t+1])), 2, (255, 0, 0), -1)
        imwrite(os.path.join(model_config.summary_path, single_img), img_ori.astype(np.uint8))


with open(os.path.join(model_config.summary_path, 'min_loss.txt'), 'w') as f:
    f.write('Min loss in test set: %f.'%min_test_cost)

with open(os.path.join(model_config.summary_path, 'model_config.txt'), 'w') as f:
    for (k, v) in model_config.__dict__.items():
        f.write("%s : %s\n"%(str(k), str(v)))
