# -*-coding:utf8-*-

import os
import tensorflow as tf
import sample_generator
from networks import encoder, decoder
import hparams
import glob


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
tf.flags.DEFINE_float('lr', 1e-4, 'learning rate')
tf.flags.DEFINE_float('decay_rate', 0.5, 'learning rate decay rate')

work_flags = tf.flags.FLAGS


def model_fn_builder(hyper_params, vocab_size):

    def model_fn(features, labels, mode, params):
        # prepare inputs
        text_seq = features['text']
        mask = features['mask']
        mels = labels['mels']
        gate = labels['gate']

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
            gate_label = tf.one_hot(record4, depth=2)
            return {'text': record1, 'mask': record2}, {'mels': record3, 'gate': gate_label}

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
        return dataset

    return input_fn


if __name__ == '__main__':
    # build model_fn
    h_param = hparams.create_hparams()
    model_fn = model_fn_builder(hyper_params=h_param, vocab_size=work_flags.vocab_size)

    # prepare estimator
    _local_checkpoint_path = os.path.join(work_flags.local_path, 'checkpoint/')
    distribution = tf.contrib.distribute.MirroredStrategy(
        num_gpus=work_flags.num_gpus) if work_flags.num_gpus > 1 else None
    run_config = tf.estimator.RunConfig(model_dir=_local_checkpoint_path,
                                        save_checkpoints_steps=work_flags.save_checkpoints_steps,
                                        log_step_count_steps=work_flags.log_every_n_steps,
                                        keep_checkpoint_max=5,
                                        train_distribute=distribution)
    feature_predictor = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=_local_checkpoint_path, config=run_config)

    # build input function
    input_fn = input_fn_builder(folder=work_flags.train_folder, hyper_params=h_param)

    # prepare hooks

    # train model
    feature_predictor.train(
        input_fn=input_fn, steps=work_flags.epoch * work_flags.steps_per_epoch
    )
