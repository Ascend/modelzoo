import argparse
import time
import numpy as np

from mindspore import Tensor, context
import mindspore.common.dtype as mstype

from mindspore.train.serialization import load_checkpoint, load_param_into_net, save_checkpoint, _get_merged_param_data
from mindspore.dataset import GeneratorDataset

from src.util import CTCLabelConverter, AverageMeter
from src.config import Config_CNNCTC
from src.dataset import IIIT_Generator_batch
from src.CNNCTC.model import CNNCTC_Model

config = Config_CNNCTC()
CHARACTER = config.CHARACTER

NUM_CLASS = config.NUM_CLASS
HIDDEN_SIZE = config.HIDDEN_SIZE
FINAL_FEATURE_WIDTH = config.FINAL_FEATURE_WIDTH

TEST_BATCH_SIZE = config.TEST_BATCH_SIZE
TEST_DATASET_SIZE = config.TEST_DATASET_SIZE

CKPT_PATH = config.CKPT_PATH

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False,
                    save_graphs_path=".", enable_auto_mixed_precision=False)


def test_dataset_creator():
    ds = GeneratorDataset(IIIT_Generator_batch, ['img', 'label_indices', 'text', 'sequence_length', 'label_str'])
    ds.set_dataset_size(int(TEST_DATASET_SIZE // TEST_BATCH_SIZE))
    return ds


def test():
    ds = test_dataset_creator()

    net = CNNCTC_Model(NUM_CLASS, HIDDEN_SIZE, FINAL_FEATURE_WIDTH)
    net._phase = 'train'  # to reduce transdata time

    ckpt_path = CKPT_PATH

    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    print('parameters loaded! from: ', ckpt_path)

    converter = CTCLabelConverter(CHARACTER)

    model_run_time = AverageMeter()
    npu_to_cpu_time = AverageMeter()
    postprocess_time = AverageMeter()

    count = 0
    correct_count = 0
    for data in ds.create_tuple_iterator():
        img, label_indices, text, sequence_length, length = data

        img_tensor = Tensor(img, mstype.float32)

        model_run_begin = time.time()
        model_predict = net(img_tensor)
        model_run_end = time.time()
        model_run_time.update(model_run_end - model_run_begin)

        npu_to_cpu_begin = time.time()
        model_predict = np.squeeze(model_predict.asnumpy())
        npu_to_cpu_end = time.time()
        npu_to_cpu_time.update(npu_to_cpu_end - npu_to_cpu_begin)

        postprocess_begin = time.time()
        preds_size = np.array([model_predict.shape[1]] * TEST_BATCH_SIZE)
        preds_index = np.argmax(model_predict, 2)
        preds_index = np.reshape(preds_index, [-1])
        preds_str = converter.decode(preds_index, preds_size)
        postprocess_end = time.time()
        postprocess_time.update(postprocess_end - postprocess_begin)

        label_str = converter.reverse_encode(text, length)

        if count == 0:
            model_run_time.reset()
            npu_to_cpu_time.reset()
            postprocess_time.reset()
        else:
            print('---------model run time--------', model_run_time.avg)
            print('---------npu_to_cpu run time--------', npu_to_cpu_time.avg)
            print('---------postprocess run time--------', postprocess_time.avg)

        print("Prediction samples: \n", preds_str[:5])
        print("Ground truth: \n", label_str[:5])
        for pred, label in zip(preds_str, label_str):
            if pred == label:
                correct_count += 1
            count += 1

    print('accuracy: ', correct_count / count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FasterRcnn training")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Pretrain file path.")
    args_opt = parser.parse_args()
    if args_opt.ckpt_path != "":
        CKPT_PATH = args_opt.ckpt_path
    test()
