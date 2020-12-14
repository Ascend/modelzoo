from mindspore.train.callback import Callback


class EvaluateCallBack(Callback):
    def __init__(self, model, eval_dataset, per_print_time=1000):
        super(EvaluateCallBack, self).__init__()
        self.model = model
        self.per_print_time = per_print_time
        self.eval_dataset = eval_dataset

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        if cb_params.cur_step_num % self.per_print_time == 0:
            result = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            print('cur epoch {}, cur_step {}, top1 accuracy {}, top5 accuracy {}.'.format(cb_params.cur_epoch_num,
                                                                                          cb_params.cur_step_num,
                                                                                          result['top_1_accuracy'],
                                                                                          result['top_5_accuracy']))

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        result = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
        print('cur epoch {}, cur_step {}, top1 accuracy {}, top5 accuracy {}.'.format(cb_params.cur_epoch_num,
                                                                                      cb_params.cur_step_num,
                                                                                      result['top_1_accuracy'],
                                                                                      result['top_5_accuracy']))
