# 通过加载已经训练好的pb模型，执行推理
import numpy as np
import time
import os
import sys
import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.estimator import npu_ops

from transforms import NormalizeImages, OneHotLabels, apply_transforms, PadXYZ, RandomCrop3D, \
    RandomHorizontalFlip, RandomGammaCorrection, RandomVerticalFlip, RandomBrightnessCorrection, CenterCrop, \
    apply_test_transforms, Cast

from losses import make_loss, eval_dice, total_dice

# 用户自定义模型路径、输入、输出
model_path = './unet3d.pb'
input_tensor_name = 'input:0'
output_tensor_name = 'Softmax:0'

class Classifier(object):

    def __init__(self):
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True

        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")

        custom_op.parameter_map["graph_run_mode"].i = 0

        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

        self.graph = self.__load_model(model_path)
        self.input_tensor = self.graph.get_tensor_by_name(input_tensor_name)
        self.output_tensor = self.graph.get_tensor_by_name(output_tensor_name)

        self.sess = tf.Session(config=config, graph=self.graph)

    def __load_model(self, model_file):
        with tf.gfile.GFile(model_file, "rb") as gf:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(gf.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")

        return graph

    def do_infer(self, batch_data):
        out = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: batch_data})
        return out

def cross_validation(x: np.ndarray, fold_idx: int, n_folds: int):
    if fold_idx < 0 or fold_idx >= n_folds:
        raise ValueError('Fold index has to be [0, n_folds). Received index {} for {} folds'.format(fold_idx, n_folds))

    _folders = np.array_split(x, n_folds)

    return np.concatenate(_folders[:fold_idx] + _folders[fold_idx + 1:]), _folders[fold_idx]

class Dataset:
    def __init__(self, data_dir, batch_size=1, fold_idx=0, n_folds=5, seed=0, pipeline_factor=1, params=None):
        self._folders = np.array([os.path.join(data_dir, path) for path in os.listdir(data_dir)])
        self._train, self._eval = cross_validation(self._folders, fold_idx=fold_idx, n_folds=n_folds)
        self._pipeline_factor = pipeline_factor
        self._data_dir = data_dir
        self.params = params

        self._batch_size = batch_size
        self._seed = seed

        self._xshape = (240, 240, 155, 4)
        self._yshape = (240, 240, 155)

    def parse(self, serialized):
        features = {
            'X': tf.io.FixedLenFeature([], tf.string),
            'Y': tf.io.FixedLenFeature([], tf.string),
            'mean': tf.io.FixedLenFeature([4], tf.float32),
            'stdev': tf.io.FixedLenFeature([4], tf.float32)
        }

        parsed_example = tf.io.parse_single_example(serialized=serialized,
                                                    features=features)

        x = tf.io.decode_raw(parsed_example['X'], tf.uint8)
        x = tf.cast(tf.reshape(x, self._xshape), tf.uint8)
        y = tf.io.decode_raw(parsed_example['Y'], tf.uint8)
        y = tf.cast(tf.reshape(y, self._yshape), tf.uint8)

        mean = parsed_example['mean']
        stdev = parsed_example['stdev']

        return x, y, mean, stdev

    def eval_fn(self):
        ds = tf.data.TFRecordDataset(filenames=self._eval)
        assert len(self._eval) > 0, "Evaluation data not found. Did you specify --fold flag?"

        ds = ds.cache()
        ds = ds.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        transforms = [
            CenterCrop((224, 224, 155)),
            Cast(dtype=tf.float32),
            NormalizeImages(),
            OneHotLabels(n_classes=4),
            PadXYZ()
        ]

        ds = ds.map(map_func=lambda x, y, mean, stdev: apply_transforms(x, y, mean, stdev, transforms=transforms),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size=self._batch_size,
                      drop_remainder=True)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds


if __name__ == '__main__':

    filepath = sys.argv[1]

    classifier = Classifier()

    dataset = Dataset(data_dir=filepath, batch_size=2, fold_idx=0, n_folds=5)
    ds = dataset.eval_fn()
    iter = ds.make_initializable_iterator()

    ds_sess = tf.Session()
    ds_sess.run(iter.initializer)
    next_element = iter.get_next()
    
    class_1 = []
    class_2 = []
    class_3 = []
    total = []
    weights = []

    i = 0
    while True:
        try:
            input = ds_sess.run(next_element)
            batch_data = input[0]
            batch_labels = input[1]

            batch_logits = classifier.do_infer(batch_data)

            logits = batch_logits
            labels = batch_labels

            labels = tf.cast(labels, tf.float32)
            labels = labels[..., 1:]
            logits = logits[..., 1:]

            eval_acc = eval_dice(y_true=labels, y_pred=tf.round(logits))
            total_eval_acc = total_dice(tf.round(logits), labels)

            eval_acc_1, eval_acc_2, eval_acc_3 = ds_sess.run(eval_acc)
            total_eval_acc = ds_sess.run(total_eval_acc)

            class_1.append(eval_acc_1)
            class_2.append(eval_acc_2)
            class_3.append(eval_acc_3)
            total.append(total_eval_acc)
            weights.append(1)

            TumorCore_avg = np.average(class_1, weights=weights)
            PeritumoralEdema_avg = np.average(class_2, weights=weights)
            EnhancingTumor_avg = np.average(class_3, weights=weights)
            WholeTumor_avg = np.average(total, weights=weights)

            MeanDice_mean = np.average([TumorCore_avg, PeritumoralEdema_avg, EnhancingTumor_avg])

            i += 1
        except tf.errors.OutOfRangeError:
            print("###Total result:", "TumorCore: ", TumorCore_avg, \
                                      "PeritumoralEdema: ", PeritumoralEdema_avg, \
                                      "EnhancingTumor: ", EnhancingTumor_avg, \
                                      "MeanDice: ", MeanDice_mean, \
                                      "WholeTumor: ", WholeTumor_avg)
            break

    ds_sess.close()
    classifier.sess.close()

