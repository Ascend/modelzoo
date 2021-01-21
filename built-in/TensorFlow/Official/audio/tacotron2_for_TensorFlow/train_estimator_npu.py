# -*-coding:utf8-*-

import os
import tensorflow as tf
import sample_generator
from networks import encoder, decoder
import hparams
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
import glob
from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager
from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_config import DumpConfig


ks = tf.keras
tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_integer('num_gpus', default=1, help='none')
tf.flags.DEFINE_string('local_path', None, 'local path')
tf.flags.DEFINE_string('train_folder', None, 'train data folder')
tf.flags.DEFINE_integer('epoch', 0, 'epoch')
tf.flags.DEFINE_integer('steps_per_epoch', 0, 'steps per epoch')
tf.flags.DEFINE_integer('save_checkpoints_steps', 500, 'save checkpoint every n steps')
tf.flags.DEFINE_integer('log_every_n_steps', 100, 'log every n steps')
tf.flags.DEFINE_integer('num_warmup_steps', None, 'warm up steps')
tf.flags.DEFINE_integer('shuffle_buffer', 0, 'shuffle buffer size')
tf.flags.DEFINE_integer('start_decay', 10000, 'learning rate start decay step')
tf.flags.DEFINE_integer('decay_steps', 40000, 'learning rate decay steps')
tf.flags.DEFINE_integer('vocab_size', 148, 'length of vocabulary')
tf.flags.DEFINE_integer('max_text_len', 188, 'max length of text sequence')
tf.flags.DEFINE_integer('max_mel_len', 870, 'max length of mel spect')
tf.flags.DEFINE_integer('multi_npu', 0, 'whether use multiple gpu for training')
tf.flags.DEFINE_integer('rank_size', 1, 'number of npus')
tf.flags.DEFINE_integer('rank_id', 0, 'npu device id')
tf.flags.DEFINE_float('lr', 1e-4, 'learning rate')
tf.flags.DEFINE_float('decay_rate', 0.5, 'learning rate decay rate')

work_flags = tf.flags.FLAGS


def model_fn_builder(hyper_params, vocab_size):

    def model_fn(features, labels, mode, params):
        # prepare inputs
        text_seq = features['text']
        text_seq = tf.slice(text_seq, [0, 0], [32, 50])
        mask = features['mask']
        mels = labels['mels']
        mels = tf.slice(mels, [0, 0, 0], [32, 100, 80])
        gate = labels['gate']
        gate = tf.slice(gate, [0, 0], [32, 100])

        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.constant(value=work_flags.lr, shape=[], dtype=tf.float32)
        # need to build another learning rate decay schedule
        learning_rate = tf.train.exponential_decay(
            learning_rate,
            global_step - work_flags.start_decay,  # lr = 1e-3 at step 50k
            work_flags.decay_steps,
            work_flags.decay_rate,  # lr = 1e-5 around step 310k
            name='lr_exponential_decay')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        if work_flags.multi_npu == 1:
            # loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000,
            #                                                        decr_every_n_nan_or_inf=2, decr_ratio=0.5)
            optimizer = NPUDistributedOptimizer(optimizer)
            # optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager, is_distributed=True)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # get output of feature extraction net
            encoded = encoder(text_seq, training=True)
            mel_output, bef_mel_output, done_output, decoder_state, LTSM, step \
                = decoder(mels, encoded, training=True)
            # get train losses
            mel_loss = tf.losses.mean_squared_error(labels=mels, predictions=mel_output) + \
                tf.losses.mean_squared_error(labels=mels, predictions=bef_mel_output)
            gate_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=done_output, labels=gate))
            train_loss = mel_loss + gate_loss
            # create train op
            train_op = optimizer.minimize(loss=train_loss, global_step=global_step)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=train_loss,
                train_op=train_op
            )
        else:
            raise ValueError("Only TRAIN mode are supported: %s" % mode)
        return output_spec

    return model_fn


def input_fn_builder(folder, hyper_params):
    gen_fn = sample_generator.create_generator(data_path=folder)

    def input_fn():

        def parser(record1, record2, record3, record4):
            return {'text': record1, 'mask': record2}, {'mels': record3, 'gate': record4}

        dataset = tf.data.Dataset.from_generator(
            generator=gen_fn,
            output_types=(tf.float32, tf.bool, tf.float32, tf.int32),
            output_shapes=(
                tf.TensorShape([work_flags.max_text_len, ]),
                tf.TensorShape([work_flags.max_text_len, ]),
                tf.TensorShape([work_flags.max_mel_len, hyper_params.n_mel_channels]),
                tf.TensorShape([work_flags.max_mel_len, ])
            )
        )
        # dataset = dataset.map(parser)
        # dataset = dataset.batch(hyper_params.batch_size)
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                lambda record1, record2, record3, record4: parser(record1, record2, record3, record4),
                batch_size=hyper_params.batch_size,
                num_parallel_batches=4,
                drop_remainder=True
            )
        )
        if work_flags.multi_npu == 1:
            dataset = dataset.shard(work_flags.rank_size, work_flags.rank_id)
        return dataset

    return input_fn


if __name__ == '__main__':
    if not os.path.exists(work_flags.local_path):
        os.makedirs(work_flags.local_path)

    # build model_fn
    h_param = hparams.create_hparams()
    model_fn = model_fn_builder(hyper_params=h_param, vocab_size=work_flags.vocab_size)

    # prepare estimator
    _local_checkpoint_path = os.path.join(work_flags.local_path, 'checkpoint/')
    dump_config = DumpConfig(enable_dump=True, dump_path="/home/t00495118/tacotron2/output", dump_step="0|5|10",
                             dump_mode="all")

    estimator_config = tf.ConfigProto(
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)
    run_config = NPURunConfig(
        hcom_parallel=True,
        # dump_config=dump_config if work_flags.rank_id == 0 else None,
        precision_mode="allow_mix_precision",
        enable_data_pre_proc=True,
        save_checkpoints_steps=work_flags.save_checkpoints_steps,
        session_config=estimator_config,
        model_dir=_local_checkpoint_path,
        iterations_per_loop=1,
        keep_checkpoint_max=5,
        log_step_count_steps=work_flags.log_every_n_steps
    )
    feature_predictor = NPUEstimator(
        model_fn=model_fn, config=run_config)

    # build input function
    input_fn = input_fn_builder(folder=work_flags.train_folder, hyper_params=h_param)

    # prepare hooks

    # train model
    feature_predictor.train(
        input_fn=input_fn, steps=work_flags.epoch * work_flags.steps_per_epoch
    )
