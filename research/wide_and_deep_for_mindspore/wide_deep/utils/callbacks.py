from mindspore.train.callback import Callback


def add_write(file_path, out_str):
    with open(file_path, 'a+', encoding="utf-8") as file_out:
        file_out.write(out_str + "\n")


class LossCallBack(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF terminating training.

    Note:
        If per_print_times is 0 do not print loss.

    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, config, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.config = config

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        wide_loss, deep_loss = cb_params.net_outputs[0].asnumpy(), \
                               cb_params.net_outputs[1].asnumpy()
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        cur_num = cb_params.cur_step_num

        if self._per_print_times != 0 and cur_num % self._per_print_times == 0:
            loss_file = open(self.config.loss_file_name, "a+")
            loss_file.write(
                "epoch: %s step: %s, wide_loss is %s, deep_loss is %s" %
                (cb_params.cur_epoch_num, cur_step_in_epoch, wide_loss,
                 deep_loss))
            loss_file.write("\n")
            loss_file.close()
            print("epoch: %s step: %s, wide_loss is %s, deep_loss is %s" % (
                cb_params.cur_epoch_num, cur_step_in_epoch, wide_loss,
                deep_loss))


class EvalCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss is NAN or INF terminating training.
    Note:
        If per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, model, eval_dataset, auc_metric, config, print_per_step=1):
        super(EvalCallBack, self).__init__()
        if not isinstance(print_per_step, int) or print_per_step < 0:
            raise ValueError("print_step must be int and >= 0.")
        self.print_per_step = print_per_step
        self.model = model  #
        self.eval_dataset = eval_dataset  # eval_dataset
        self.aucMetric = auc_metric
        self.aucMetric.clear()
        self.config = config

    #
    def epoch_end(self, run_context):
        self.aucMetric.clear()
        out = self.model.eval(self.eval_dataset)
        out_str = "=====" * 5 + "EvalCallBack model.eval(): {}".format(
            out.values())
        print(out_str)
        add_write(self.config.eval_file_name, out_str)
    #
