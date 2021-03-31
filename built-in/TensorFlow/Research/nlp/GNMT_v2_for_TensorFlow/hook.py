import tensorflow as tf
class _LogSessionRunHook(tf.train.SessionRunHook):
    def before_run(self, run_context):
        return tf.estimator.SessionRunArgs(
                fetches=['overflow_status_reduce_all:0', 'loss_scale:0', 'learning_rate/ToFloat:0', 'learning_rate/Exp:0'])

    def after_run(self, run_context, run_values):
        print('ToFloat=', run_values.results[2], ' Exp=', run_values.results[3], flush=True)
        if not run_values.results[0]:
            print('Find overflow in this step, skip apply gradients, loss scale value=%d' % run_values.results[1], flush=True)
        else:
            print('Apply gradients, loss scale value=%d' % run_values.results[1], flush=True)
