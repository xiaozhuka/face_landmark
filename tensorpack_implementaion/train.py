# -*- coding: utf-8 -*-

from model import OrdinaryCNNModel, get_data
from tensorpack.utils import logger
from config import ModelConfig
from tensorpack.dataflow import BatchData
from tensorpack import (TrainConfig, ModelSaver, MinSaver,
                        InferenceRunner, ScalarStats, Trainer)
import tensorflow as tf

log_dir = r'E:\python_vanilla\log_dir'
logger.set_logger_dir(log_dir, action='d')

model_config = ModelConfig()
model_config.width = 40
model_config.height = 40
model_config.channels = 1
model_config.min_save_name = 'min_loss.model'
model_config.batch_size = 16
model_config.target_length = 50
model_config.weight_decay = 0.1
model_config.bias_decay = 1e-2

cnn_model = OrdinaryCNNModel(model_config)
tf.reset_default_graph()

ds_train, ds_test = get_data()
ds_train = BatchData(ds_train, model_config.batch_size)
ds_test = BatchData(ds_test, 16, remainder=True)

logger.auto_set_dir()

train_config = TrainConfig(
    model=cnn_model,
    dataflow=ds_train,
    callbacks=[
        ModelSaver(),
        MinSaver('loss', filename=model_config.min_save_name),
        InferenceRunner(
            ds_test,
            ScalarStats(['loss'])
            ),
    ],
    max_epoch=200,
)

Trainer(train_config).train()

