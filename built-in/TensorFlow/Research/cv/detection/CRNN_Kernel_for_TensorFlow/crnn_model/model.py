from crnn_model import crnn_net
import tensorflow as tf
from data_provider import shadownet_data_feed_pipline
from data_provider import tf_io_pipline_fast_tools
from config import global_config

from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer


CFG = global_config.cfg



class Model(object):
  def __init__(self,config,cmd_args):
    self.config = config
    self.cmd_args = cmd_args
  

  def get_estimator_model_func(self, features, labels, mode, params=None):
    
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    if is_training:
      return self.train_fn(features,labels,mode)
    else:
      return self.evaluate_fn(features,labels,mode)


  def train_fn(self,features,labels,mode):
    """

    """
    train_images = features
    train_labels = labels
    shadownet = crnn_net.ShadowNet(
        phase='train',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )

    # set up decoder
    decoder = tf_io_pipline_fast_tools.CrnnFeatureReader(
        char_dict_path=self.cmd_args.char_dict_path,
        ord_map_dict_path=self.cmd_args.ord_map_dict_path
    )

    # compute loss and seq distance
    train_inference_ret, train_ctc_loss = shadownet.compute_loss(
        inputdata=train_images,
        labels=train_labels,
        name='shadow_net',
        reuse=False
    )
    loss = tf.identity(train_ctc_loss, name='loss')

    # train_decoded, train_log_prob = tf.nn.ctc_greedy_decoder(
    #     train_inference_ret,
    #     CFG.ARCH.SEQ_LENGTH * np.ones(CFG.TRAIN.BATCH_SIZE),
    #     merge_repeated=False
    # )
    #
    #
    # train_sequence_dist = tf.reduce_mean(
    #     tf.edit_distance(tf.cast(train_decoded[0], tf.int32), train_labels),
    #     name='train_edit_distance'
    # )

    # set learning rate
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.polynomial_decay(
        learning_rate=CFG.TRAIN.LEARNING_RATE,
        global_step=global_step,
        decay_steps=CFG.TRAIN.STEPS,
        end_learning_rate=0.000001,
        power=CFG.TRAIN.LR_DECAY_RATE
    )

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    opt = NPUDistributedOptimizer(optimizer)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      gate_gradients = tf.train.Optimizer.GATE_NONE
      grads_and_vars = opt.compute_gradients(train_ctc_loss, gate_gradients=gate_gradients)
      train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=train_ctc_loss, train_op=train_op)


  def evaluate_fn(self, features, labels,mode):
    # prepare dataset

    val_images = features
    val_labels = labels

    # declare crnn net
    shadownet_val = crnn_net.ShadowNet(
        phase='test',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )

    # set up decoder
    decoder = tf_io_pipline_fast_tools.CrnnFeatureReader(
        char_dict_path=self.cmd_args.char_dict_path,
        ord_map_dict_path=self.cmd_args.ord_map_dict_path)

    # compute loss and seq distance
    val_inference_ret, val_ctc_loss = shadownet_val.compute_loss(
        inputdata=val_images,
        labels=val_labels,
        name='shadow_net',
        reuse=True)
    loss = tf.identity(val_ctc_loss, name='loss')
    val_decoded, val_log_prob = tf.nn.ctc_greedy_decoder(
        val_inference_ret,
        CFG.ARCH.SEQ_LENGTH * np.ones(CFG.TRAIN.BATCH_SIZE),
        merge_repeated=False)

    # val_sequence_dist = tf.reduce_mean(
    #     tf.edit_distance(tf.cast(val_decoded[0], tf.int32), val_labels),
    #     name='val_edit_distance'
    # )
    val_labels_str = decoder.sparse_tensor_to_str(val_labels)
    val_predictions = decoder.sparse_tensor_to_str(val_decoded[0])
    avg_val_accuracy = evaluation_tools.compute_accuracy_ops(val_labels_str, val_predictions)

    return tf.estimator.EstimatorSpec(mode,loss=val_ctc_loss,eval_metrics=avg_val_accuracy)

  def get_train_inputs(self):
    train_images, train_labels, train_images_paths = train_dataset.inputs(
            batch_size=self.cmd_args.batch_size)
    return train_images,train_labels

  def get_eval_inputs(self):
    val_images, val_labels, val_images_paths = val_dataset.inputs(
            batch_size=self.cmd_args.eval_batch_size)
    return val_images, val_labels


