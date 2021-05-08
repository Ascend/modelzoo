# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
## ==============================================================================
import os
import tensorflow as tf

from config import trans_config as config
from dataloader import data_provider
from datasets import dataset_factory
from nets import nets_factory


class EstimatorImpl:
    def __init__(self, env):
        self.env = env

    def model_fn(self, features, labels, mode, params):
        print("In EstimatorImpl, num_classes is %d" % config.num_classes)
        num_classes = config.num_classes

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        if mode == tf.estimator.ModeKeys.TRAIN:
            network_fn = nets_factory.get_network_fn(
                self.env.FLAGS.model_name,
                num_classes=(num_classes - self.env.FLAGS.labels_offset),
                weight_decay=self.env.FLAGS.weight_decay,
                is_training=True)

            logits = self.env.calc_logits(network_fn, features)
            loss, total_loss = self.env.calc_loss(logits, labels)

            #### accuracy ####
            predictions = tf.argmax(logits, 1)
            accuracy_ops = tf.metrics.accuracy(tf.argmax(labels, 1), predictions)
            tf.identity(accuracy_ops[1], name='train_accuracy')
            #### accuracy ####

            tf.identity(total_loss, 'train_loss')

            global_step = tf.train.get_or_create_global_step()
            train_op = self.env.create_train_op(global_step, summaries, total_loss)

            estimator_spec = tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.TRAIN, loss=total_loss, train_op=train_op)
            # restore ckpt for finetune
            if config.restore_path:
                print("----------begin finetune--------------")
                variables_to_restore = tf.contrib.slim.get_variables_to_restore(
                    exclude=self.env.FLAGS.restore_exclude)
                tf.train.init_from_checkpoint(
                    self.env.FLAGS.restore_path,
                    {v.name.split(':')[0]: v for v in variables_to_restore})

        elif mode == tf.estimator.ModeKeys.EVAL:
            network_fn = nets_factory.get_network_fn(
                self.env.FLAGS.model_name,
                num_classes=(num_classes - self.env.FLAGS.labels_offset),
                weight_decay=self.env.FLAGS.weight_decay,
                is_training=False)

            logits = self.env.calc_logits(network_fn, features)
            loss, total_loss = self.env.calc_loss(logits, labels)
            predictions = tf.argmax(logits, 1)
            accuracy_ops = tf.metrics.accuracy(tf.argmax(labels, 1), predictions)
            tf.identity(accuracy_ops[1], name='eval_accuracy')
            estimator_spec = tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=total_loss, eval_metric_ops={'accuracy': accuracy_ops})

        return estimator_spec

    def main(self):
        logdir = self.env.create_logdir()

        from logger import LogSessionRunHook

        config = {
            'num_training_samples': self.env.num_samples,
            # for 1p, just per loop print, for 8p, print each epoch
            'display_every': 1,
            'log_name': 'train_log.log',
            'log_dir': logdir,
            'global_batch_size': self.env.FLAGS.batch_size * int(os.getenv('RANK_SIZE')),
            'iterations_per_loop': self.env.FLAGS.iterations_per_loop if self.env.FLAGS.iterations_per_loop is not None else self.env.calc_steps_per_epoch()
        }

        hooks = [LogSessionRunHook(config, warmup_steps=self.env.FLAGS.warmup_epochs * self.env.calc_steps_per_epoch())]

        #################################################################
        from npu_bridge.estimator.npu.npu_config import NPURunConfig
        from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
        from npu_bridge.estimator.npu.npu_config import DumpConfig
        
        self.estimator_config = tf.ConfigProto(
            inter_op_parallelism_threads=10,
            intra_op_parallelism_threads=10,
            allow_soft_placement=True)

        self.estimator_config.gpu_options.allow_growth = True

        gpu_thread_count = 2
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = str(gpu_thread_count)
        os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

        #dump_config = DumpConfig(
        #    enable_dump= "True" == os.getenv("FLAG_ENABLE_DUMP"), 
        #    dump_path=os.getenv("DUMP_PATH"),
        #    dump_step=os.getenv("DUMP_STEP"),
        #    dump_mode=os.getenv("DUMP_MODE"))
        if self.env.FLAGS.over_dump == "True":
            print("NPU overflow dump is enabled")
            from npu_bridge.npu_init import DumpConfig
            dump_config = DumpConfig(
                enable_dump_debug=True, dump_path=self.env.FLAGS.over_dump_path, dump_debug_mode="all")

            run_config = NPURunConfig(
                dump_config=dump_config,
                hcom_parallel=True,
                precision_mode="allow_mix_precision",
                enable_data_pre_proc=True,
                save_checkpoints_steps=self.env.calc_steps_per_epoch(),
                session_config=self.estimator_config,
                model_dir=logdir,
                iterations_per_loop=config['iterations_per_loop'],
                keep_checkpoint_max=5)
        else:
            print("NPU overflow dump is disabled")
            run_config = NPURunConfig(
                hcom_parallel=True,
                precision_mode="allow_mix_precision",
                enable_data_pre_proc=True,
                save_checkpoints_steps=self.env.calc_steps_per_epoch(),
                session_config=self.estimator_config,
                model_dir=logdir,
                iterations_per_loop=config['iterations_per_loop'],
                keep_checkpoint_max=5)

        classifier =NPUEstimator(
            model_fn= self.model_fn, 
            config= run_config
            )
        ###################################################################

        classifier.train(
            input_fn=self.train_data,
            max_steps=self.env.FLAGS.max_number_of_steps,
            hooks=hooks,
        )

    def train_data(self):
        dataset = dataset_factory.get_dataset(self.env.FLAGS.dataset_name, 'train', self.env.FLAGS.dataset_dir)

        preprocessing_name = self.env.FLAGS.preprocessing_name or self.env.FLAGS.model_name
        _, ds = data_provider.get_data(dataset, self.env.FLAGS.batch_size,
            dataset.num_classes, self.env.FLAGS.labels_offset, True,
            preprocessing_name, self.env.FLAGS.use_grayscale)

        return ds

    def eval_data(self):
        dataset = dataset_factory.get_dataset(self.env.FLAGS.dataset_name, 'validation', self.env.FLAGS.dataset_dir)

        preprocessing_name = self.env.FLAGS.preprocessing_name or self.env.FLAGS.model_name
        _, ds = data_provider.get_data(dataset, self.env.FLAGS.batch_size,
            dataset.num_classes, self.env.FLAGS.labels_offset, False,
            preprocessing_name, self.env.FLAGS.use_grayscale)

        return ds
