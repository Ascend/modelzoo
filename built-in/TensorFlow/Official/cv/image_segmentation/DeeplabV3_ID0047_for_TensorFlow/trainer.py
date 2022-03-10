# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import tensorflow as tf
import math
import time

from utils  import train_helper
from utils.train_helper import stage
from utils.logger import rank0log

from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator


FLAGS = tf.app.flags.FLAGS


class Trainer(object):
    def __init__(self, data, model, logger):

        self.data = data
        self.model = model
        self.logger = logger
        self.print_logger = self.logger.logger
        self.classifier, self.training_hook = self.get_npu_classifier()


    def get_npu_classifier(self):
        session_config = tf.ConfigProto(
           inter_op_parallelism_threads=10,
           intra_op_parallelism_threads=10,
           allow_soft_placement=True)

        num_training_samples =self.data.num_training_samples 
        nsteps_per_epoch = num_training_samples//FLAGS.train_batch_size//FLAGS.rank


        run_config = NPURunConfig(hcom_parallel=True, precision_mode="allow_mix_precision", enable_data_pre_proc=True,
                                  save_checkpoints_steps = FLAGS.save_checkpoints_steps,
                                  session_config=session_config, model_dir = FLAGS.train_logdir,
                                  iterations_per_loop=FLAGS.iterations_per_loop,
                                  keep_checkpoint_max=FLAGS.max_to_save,
                                  enable_small_channel=1)


        classifier =NPUEstimator(
            model_fn= self.model.get_estimator_model_func, 
            config= run_config
      	  )
      
        training_hooks = [train_helper.PrefillStagingAreasHook()]
        training_hooks.append(self.logger)

        return classifier, training_hooks

    def train(self):
        print ('training steps: %d' % FLAGS.training_number_of_steps)
        self.classifier.train( input_fn=lambda:self.data.get_train_input_fn(),
                               max_steps = FLAGS.training_number_of_steps,
                               hooks = self.training_hook
                              )

    def evaluate(self):
        rank0log(self.print_logger, "Evaluating")
        time.sleep(5)  # a little extra margin...
        try:
            ckpts = train_helper.sort_and_load_ckpts(FLAGS.checkpoint_dir)
            print("=========ckpt==========")
            print(ckpts)
            print("=========ckpt==========")
            latest_ckpt =[ckpts[-1]]
            for i, c in enumerate(latest_ckpt):
                eval_result = self.classifier.evaluate(
                    input_fn=lambda: self.data.get_eval_input_fn(),
                    checkpoint_path=c['path'])
                for key in eval_result.keys():
                    rank0log(self.print_logger, '{} :  {:3.5f}  '.format(key, eval_result[key]))
                rank0log(self.print_logger, 'checkpoint: {}  '.format(c['path'] ))

        except KeyboardInterrupt:
            self.print_logger.error("Keyboard interrupt")

