from mindspore.nn.metrics import Metric
from sklearn.metrics import roc_auc_score


class AUCMetric(Metric):
    def __init__(self):
        super(AUCMetric, self).__init__()

    def clear(self):
        """Clear the internal evaluation result."""
        self.true_labels = []
        self.pred_probs = []

    def update(self, *inputs):  # inputs
        all_predict = inputs[1].asnumpy()  # predict
        all_label = inputs[2].asnumpy()  # label
        self.true_labels.extend(all_label.flatten().tolist())
        self.pred_probs.extend(all_predict.flatten().tolist())

    def eval(self):
        if len(self.true_labels) != len(self.pred_probs):
            raise RuntimeError(
                'true_labels.size() is not equal to pred_probs.size()')

        auc = roc_auc_score(self.true_labels, self.pred_probs)
        print("=====" * 20 + " auc_metric  end ")
        print("=====" * 20 + " auc: {}".format(auc))
        return auc

