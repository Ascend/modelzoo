import math
import argparse
import mindspore.nn as nn
from mindspore import context
from mindspore.communication.management import init
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.model import Model, ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.dataset import train_dataset_creator
from src.config import config
from src.ETSNET.etsnet import ETSNet
from src.ETSNET.dice_loss import DiceLoss
from src.network_define import WithLossCell, TrainOneStepCell, LossCallBack

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--run_distribute', default=False, action='store_true',
                    help='Run distribute, default is false.')
parser.add_argument('--pre_trained', type=str, default='', help='Pretrain file path.')
parser.add_argument('--device_id', type=int, default=0, help='Device id, default is 0.')
parser.add_argument('--device_num', type=int, default=1, help='Use device nums, default is 1.')
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args.device_id)


def lr_generator(start_lr, lr_scale, total_iters):
    lrs = [start_lr * (lr_scale ** math.floor(cur_iter * 1.0 / (total_iters / 3))) for cur_iter in range(total_iters)]
    return lrs

def train():
    if args.run_distribute:
        context.set_auto_parallel_context(device_num=args.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          mirror_mean=True, parameter_broadcast=True)
        init()

    # dataset/network/criterion/optim
    ds = train_dataset_creator()
    step_size = ds.get_dataset_size()
    print('Create dataset done!')

    config.INFERENCE = False
    net = ETSNet(config)
    net = net.set_train()
    param_dict = load_checkpoint(args.pre_trained)
    load_param_into_net(net, param_dict)
    print('Load Pretrained parameters done!')

    criterion = DiceLoss(batch_size=config.TRAIN_BATCH_SIZE)

    lrs = lr_generator(start_lr=1e-3, lr_scale=0.1, total_iters=config.TRAIN_TOTAL_ITER)
    opt = nn.SGD(params=net.trainable_params(), learning_rate=lrs, momentum=0.99, weight_decay=5e-4)

    # warp model
    net = WithLossCell(net, criterion)
    if args.run_distribute:
        net = TrainOneStepCell(net, opt, reduce_flag=True, mean=True, degree=args.device_num)
    else:
        net = TrainOneStepCell(net, opt)

    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossCallBack(per_print_times=20)
    # set and apply parameters of check point
    ckpoint_cf = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=2)
    ckpoint_cb = ModelCheckpoint(prefix="ETSNet", config=ckpoint_cf, directory=config.TRAIN_MODEL_SAVE_PATH)

    model = Model(net)
    model.train(config.TRAIN_REPEAT_NUM, ds, dataset_sink_mode=False, callbacks=[time_cb, loss_cb, ckpoint_cb])


if __name__ == '__main__':
    train()
