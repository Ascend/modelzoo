
from npu_bridge.npu_init import *
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import cfg
from network import East
from losses import quad_loss
from data_generator import gen

import tensorflow as tf
#import tensorflow.python.keras as keras
#from tensorflow.python.keras import backend as K
from keras import backend as K
from npu_bridge.npu_init import *

sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess = tf.Session(config=sess_config)
K.set_session(sess)

def npu_keras_optimizer(opt):
    npu_opt = KerasDistributeOptimizer(opt)
    return npu_opt
east = East()
east_network = east.east_network()
east_network.summary()
east_network.compile(loss=quad_loss, optimizer=Adam(lr=cfg.lr, decay=cfg.decay))
if (cfg.load_weights and os.path.exists(cfg.saved_model_weights_file_path)):
    east_network.load_weights(cfg.saved_model_weights_file_path)
east_network.fit_generator(generator=gen(), steps_per_epoch=cfg.steps_per_epoch, epochs=cfg.epoch_num, validation_data=gen(is_val=True), validation_steps=cfg.validation_steps, verbose=1, initial_epoch=cfg.initial_epoch, callbacks=[EarlyStopping(patience=cfg.patience, verbose=1), ModelCheckpoint(filepath=cfg.model_weights_path, save_best_only=True, save_weights_only=True, verbose=1)])
east_network.save(cfg.saved_model_file_path)
east_network.save_weights(cfg.saved_model_weights_file_path)
sess.close()
