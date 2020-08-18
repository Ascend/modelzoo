import tensorflow as tf
import math
import time
import os
from . import train_helper
from .train_helper import stage
from .logger import rank0log


class Trainer(object):
    def __init__(self, session, config, data, model, logger):
        self.sess = session
        self.config = config
        self.data = data
        self.model = model
        self.logger = logger
        self.print_logger = self.logger.logger
        self.all_preds = []
        self.all_targets = []
        if self.config['accelerator'] == 'gpu':
            self.classifier, self.training_hook = self.get_gpu_classifier()
        else:
            self.classifier, self.training_hook = self.get_npu_classifier()

    def get_gpu_classifier(self):
        classifier = tf.estimator.Estimator(
            model_fn=self.model.get_estimator_model_func,
            model_dir=self.config['log_dir'],
            config = tf.estimator.RunConfig(
                    session_config=self.sess.estimator_config,
                    save_checkpoints_steps=self.config['save_checkpoints_steps'] if self.config['do_checkpoint'] else None,
                    keep_checkpoint_max=None
                     )
            )

        training_hooks = []
        #training_hooks = [train_helper.PrefillStagingAreasHook()]
        training_hooks.append(self.logger)

        # add for MultiGPU
        import horovod.tensorflow as hvd
        training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))

        return classifier, training_hooks

    def get_npu_classifier(self):
        from npu_bridge.estimator.npu.npu_config import NPURunConfig
        from npu_bridge.estimator.npu.npu_estimator import NPUEstimator

        run_config = NPURunConfig(
            hcom_parallel=True,
            precision_mode="allow_mix_precision",
            enable_data_pre_proc=True,
            save_checkpoints_steps=112590,
            session_config=self.sess.estimator_config,
            model_dir=self.config['log_dir'],
            #iterations_per_loop=self.config['iterations_per_loop'],
            iterations_per_loop=self.config['nsteps_per_epoch'],
            #iterations_per_loop=100,
            keep_checkpoint_max=5)

        classifier =NPUEstimator(
            model_fn= self.model.get_estimator_model_func, 
            config= run_config
      	  )
      
        training_hooks = []
        #training_hooks = [train_helper.PrefillStagingAreasHook()]
        training_hooks.append(self.logger)

        return classifier, training_hooks

    def train(self):
        print ('training steps: %d' % self.config['nstep'])
        self.classifier.train( input_fn=lambda:self.data.get_train_input_fn(),
                             #  max_steps = self.config['max_train_steps'],
                               max_steps = self.config['nstep'],
                               #steps = 100,
                               hooks = self.training_hook
                              )

    def evaluate(self):
        rank0log(self.print_logger, "Evaluating")
        rank0log(self.print_logger, "Validation dataset size: {}".format(self.config['num_evaluating_samples'] ))
        time.sleep(5)  # a little extra margin...

        do_eval = False
        import horovod.tensorflow as hvd
        if hvd.rank() == 0:
            do_eval = True

        if do_eval == False:
            print("Skip ", hvd.rank())
            return

        try:
            ckpts = train_helper.sort_and_load_ckpts(self.config['log_dir'])
            print("=========ckpt==========")
            print(ckpts)
            print("=========ckpt==========")
            for i, c in enumerate(ckpts):
                #if i < len(ckpts) - 1:
                #    if i % self.config['eval_interval'] != 0:
                #        continue
                eval_result = self.classifier.evaluate(
                    input_fn=lambda: self.data.get_eval_input_fn(),
                    checkpoint_path=c['path'])
                c['epoch'] = math.ceil(c['step'] / (self.config['num_training_samples']/ (self.config['batch_size'])))
                c['top1'] = eval_result['val-top1acc']
                c['top5'] = eval_result['val-top5acc']
                c['loss'] = eval_result['loss']

            rank0log(self.print_logger, ' step  epoch  top1    top5     loss   checkpoint_time(UTC)')
            for i, c in enumerate(ckpts):
                if 'top1' not in c:
                    continue
                rank0log(self.print_logger,'{:5d}  {:5.1f}  {:5.3f}  {:6.2f}  {:6.2f}  {time}'
                         .format(c['step'],
                                 c['epoch'],
                                 c['top1'] * 100,
                                 c['top5'] * 100,
                                 c['loss'],
                                 time=time.strftime('%Y-%m-%d %H:%M:%S', 
                                    time.localtime(c['mtime']))))
            rank0log(self.print_logger, "Finished evaluation")
        except KeyboardInterrupt:
            self.print_logger.error("Keyboard interrupt")

    def train_and_evaluate(self):
        epochs_between_evals = self.config['epochs_between_evals']

        for i in range(self.config['num_epochs'] // epochs_between_evals):

            rank0log(self.print_logger, "Starting a training cycle")

            self.classifier.train(input_fn=lambda:self.data.get_train_input_fn(),
                            steps = self.config['nsteps_per_epoch']*epochs_between_evals,
                        #steps=300,
                            hooks = self.training_hook )

            #saver = tf.train.Saver(tf.global_variables())
            #saver.save(self.sess, self.config['log_dir'], global_step=tf.train.get_global_step())

            do_eval = False
            
            if self.config['accelerator'] == 'gpu':
                import horovod.tensorflow as hvd
                if hvd.rank() == 0:
                    do_eval = True
            else:
                #if int(os.getenv('DEVICE_ID')) == 0:
                #    do_eval = True
                do_eval = True
 
            if do_eval:
                rank0log(self.print_logger, "Starting to evaluate")
                rank0log(self.print_logger, "Validation dataset size: {}".format(self.config['num_evaluating_samples'] ))
                time.sleep(5)  # a little extra margin...

                ckpts = train_helper.sort_and_load_ckpts(self.config['log_dir'])
                c = ckpts[-1]
                eval_result = self.classifier.evaluate(
                    input_fn=lambda: self.data.get_eval_input_fn(),
                    checkpoint_path=c['path'])

                #c['epoch'] = math.ceil(c['step'] / (self.config['num_training_samples']/ (self.config['batch_size'] * hvd.size())))
                c['epoch'] = math.ceil(c['step'] / (self.config['num_training_samples'] / (self.config['batch_size'] * self.config['rank_size'])))
                c['top1'] = eval_result['val-top1acc']
                c['top5'] = eval_result['val-top5acc']
                c['loss'] = eval_result['loss']

                rank0log(self.print_logger, ' step  epoch  top1    top5     loss   checkpoint_time(UTC)')

                rank0log(self.print_logger,'{:5d}  {:5.1f}  {:5.3f}  {:6.2f}  {:6.2f}  {time}'
                        .format(c['step'],
                                c['epoch'],
                                c['top1'] * 100,
                                c['top5'] * 100,
                                c['loss'],
                                time=time.strftime('%Y-%m-%d %H:%M:%S',
                                    time.localtime(c['mtime']))))

