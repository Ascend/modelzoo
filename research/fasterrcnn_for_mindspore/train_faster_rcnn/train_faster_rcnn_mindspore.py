import os
import mindspore as ms
import mindspore.dataset as de
import os.path as osp
import numpy as np
import itertools
from minddata_faster_rcnn.dependency.datasets import get_dataset
from minddata_faster_rcnn.dependency.datasets import build_sampler
from minddata_faster_rcnn.dependency_test.datasets import get_dataset as get_dataset_test
from minddata_faster_rcnn.dependency_test.datasets import build_sampler as build_sampler_test
import itertools
import mindspore._c_dataengine as deMap
import datetime
import time
import math
from minddata_faster_rcnn.transform import rescale_column, resize_column, impad_to_multiple_column, imnormalize_column, flip_column, transpose_column, random_crop_column, photo_crop_column, expand_column
from minddata_faster_rcnn.transform import resize_column_test
from mindspore import context
from mindspore import Tensor
from mindspore.train.model import Model
from mindspore import nn 
from mindspore.nn.optim.momentum import Momentum
from mindspore.ops import functional as F, composite as C
import mindspore.ops.operations as P
from mindspore.nn import Cell
from mindspore import Parameter, ParameterTuple
from faster_rcnn_network.config import Config_Faster_Rcnn 
from faster_rcnn_network.faster_rcnn_r50 import Faster_Rcnn_Resnet50 
from faster_rcnn_network.lr_schedule import dynamic_lr 
from mindspore.nn import SGD
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import Callback
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.parallel_utils import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from minddata_faster_rcnn.coco import *
from minddata_faster_rcnn.coco_utils import *
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import time


context.set_context(mode=context.GRAPH_MODE, device_target="Davinci", save_graphs=True, reserve_class_name_in_scope=False)


de_sampler_len = 0
dataset_len = 20
epoch_count = 12
time_stamp_init = False
time_stamp_first = 0


def get_ms_timestamp():
    t = time.time()
    return int(round(t * 1000)) 


class LossCallBack(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF terminating training.

    Note:
        If per_print_times is 0 do not print loss.

    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.count = 0
        self.rpn_loss_sum = 0
        self.rcnn_loss_sum = 0
        self.rpn_cls_loss_sum = 0
        self.rpn_reg_loss_sum = 0
        self.rcnn_cls_loss_sum = 0
        self.rcnn_reg_loss_sum = 0
        
        global time_stamp_init, time_stamp_first
        if not time_stamp_init:
            time_stamp_first = get_ms_timestamp()
            time_stamp_init = True

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        rpn_loss = cb_params.net_outputs[0].asnumpy()
        rcnn_loss = cb_params.net_outputs[1].asnumpy()
        rpn_cls_loss = cb_params.net_outputs[2].asnumpy()
        
        rpn_reg_loss = cb_params.net_outputs[3].asnumpy()
        rcnn_cls_loss = cb_params.net_outputs[4].asnumpy()
        rcnn_reg_loss = cb_params.net_outputs[5].asnumpy()
        
        self.count += 1
        self.rpn_loss_sum += float(rpn_loss)
        self.rcnn_loss_sum += float(rcnn_loss)
        self.rpn_cls_loss_sum += float(rpn_cls_loss)
        self.rpn_reg_loss_sum += float(rpn_reg_loss)
        self.rcnn_cls_loss_sum += float(rcnn_cls_loss)
        self.rcnn_reg_loss_sum += float(rcnn_reg_loss)

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        cur_num = cb_params.cur_step_num

        if self.count >= 1:
            global time_stamp_first
            time_stamp_current = get_ms_timestamp()

            rpn_loss = self.rpn_loss_sum/self.count
            rcnn_loss = self.rcnn_loss_sum/self.count
            rpn_cls_loss = self.rpn_cls_loss_sum/self.count

            rpn_reg_loss = self.rpn_reg_loss_sum/self.count
            rcnn_cls_loss = self.rcnn_cls_loss_sum/self.count
            rcnn_reg_loss = self.rcnn_reg_loss_sum/self.count

            total_loss = rpn_loss + rcnn_loss
        
            loss_file = open("./loss.log", "a+")
            loss_file.write("%lu epoch: %s step: %s ,rpn_loss is %.5f, rcnn_loss is %.5f, %.5f, %.5f, %.5f, %.5f, %.5f" %
                            (time_stamp_current - time_stamp_first, cb_params.cur_epoch_num, cur_step_in_epoch,
                             rpn_loss, rcnn_loss, rpn_cls_loss, rpn_reg_loss,
                             rcnn_cls_loss, rcnn_reg_loss, total_loss))
            loss_file.write("\n")
            loss_file.close()

            self.count = 0
            self.rpn_loss_sum = 0 
            self.rcnn_loss_sum = 0
            self.rpn_cls_loss_sum = 0
            self.rpn_reg_loss_sum = 0
            self.rcnn_cls_loss_sum = 0
            self.rcnn_reg_loss_sum = 0


class WithEvalCell(Cell):
    """
    Add loss for model. Return loss, outputs and label to calculate the metrics.

    Args:
        network (Cell): The network Cell.
        loss_fn (Cell): The loss Cell.
    """

    def __init__(self, network):
        super(WithEvalCell, self).__init__(auto_prefix=False)
        self._network = network

    def construct(self, x, img_shape, gt_bboxe, gt_label, gt_num):
        outputs = self._network(x, img_shape, gt_bboxe, gt_label, gt_num)

        return outputs, outputs, outputs


class WithLossCell(Cell):
    """
    Wrap the network with loss function to compute loss.

    Args:
        backbone (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.

    Examples:
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        >>> loss_net = WithLossCell(net, loss_fn)
    """
    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, x, img_shape, gt_bboxe, gt_label, gt_num):
        """
        Compute loss based on the wrapped loss cell.

        Args:
            data (Tensor): Tensor data to train.
            label (Tensor): Tensor label data.

        Returns:
            Tensor, compute result.
        """
        loss1, loss2, loss3, loss4, loss5, loss6  = self._backbone(x, img_shape, gt_bboxe, gt_label, gt_num)
        return self._loss_fn(loss1, loss2, loss3, loss4, loss5, loss6) 

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, return backbone network.
        """
        return self._backbone


class TrainOneStepCell(Cell):
    """
    Network training package class.

    Append an optimizer to the training network after that the construct function
    can be called to create the backward graph.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default value is 1.0.

    Examples:
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> loss_net = WithLossCell(net, loss_fn)
        >>> train_net = TrainOneStepCell(loss_net, optim)
    """
    def __init__(self, network, optimizer, sens=1.0, reduce_flag=False, mean=True, degree=None):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.backbone = network._backbone
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation('grad',
                                    get_by_list=True,
                                    sens_param=True)
        self.sens = Tensor((np.ones((1,)) * sens).astype(np.float16))
        self.reduce_flag = reduce_flag
        if reduce_flag:
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, x, img_shape, gt_bboxe, gt_label, gt_num):
        weights = self.weights
        loss1, loss2, loss3, loss4, loss5, loss6 = self.backbone(x, img_shape, gt_bboxe, gt_label, gt_num)
        grads = self.grad(self.network, weights)(x, img_shape, gt_bboxe, gt_label, gt_num, self.sens)
        if self.reduce_flag:
            grads = self.grad_reducer(grads)
        return F.depend(loss1, self.optimizer(grads)), loss2, loss3, loss4, loss5, loss6


class LossNet(nn.Cell):
    def __init__(self):
        super(LossNet, self).__init__()
    def construct(self, x1, x2, x3, x4, x5, x6):
        return x1 + x2


def bbox2result_1image(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    
    if bboxes.shape[0] == 0:
        return [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        return [bboxes[labels == i, :] for i in range(num_classes - 1)]


def compose_map(keep_ratio, flip, size_divisor, input_1, input_2, input_3, input_4, input_5):
    flip = True if np.random.rand() < 0.5 else False   
    crop = True if np.random.rand() < 0.2 else False   
    photo = True if np.random.rand() < 0.5 else False   
    expand = True if np.random.rand() < 1.0 else False   
    input_data = input_1, input_2, input_3, input_4, input_5

    if expand:
        input_data = expand_column(*input_data)
    if keep_ratio:
        input_data = rescale_column(*input_data)
    else:
        input_data = resize_column(*input_data)
    if photo:
        input_data = photo_crop_column(*input_data)
    input_data = imnormalize_column(*input_data)
    if flip:
        input_data = flip_column(*input_data)

    input_data = transpose_column(*input_data)
    return input_data


def compose_map_test(keep_ratio, flip, size_divisor, input_1, input_2, input_3, input_4, input_5):
    flip = False if np.random.rand() < 0.5 else False   
    input_data = input_1, input_2, input_3, input_4, input_5
    if keep_ratio:
        input_data = rescale_column(*input_data)
    else:
        input_data = resize_column_test(*input_data)
    input_data = imnormalize_column(*input_data)
    if flip:
        input_data = flip_column(*input_data)

    input_data = transpose_column(*input_data)
    return input_data


def faster_rcnn_train_dataset():
    dataset_type = 'CocoDataset'
    import os
    data_root = os.getenv("TDT_DATASET")
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    data = dict(
        imgs_per_gpu=2,
        workers_per_gpu=2,
        train=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            img_scale=(720, 1280),
            img_norm_cfg=img_norm_cfg,
            size_divisor=32,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=True,
            with_label=True),
        val=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_val2017.json',
            img_prefix=data_root + 'val2017/',
            img_scale=(720, 1280),
            img_norm_cfg=img_norm_cfg,
            size_divisor=32,
            flip_ratio=0,
            with_mask=False,
            with_crowd=True,
            with_label=True),
        test=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_val2017.json',
            img_prefix=data_root + 'val2017/',
            img_scale=(720, 1280),
            img_norm_cfg=img_norm_cfg,
            size_divisor=32,
            flip_ratio=0,
            with_mask=False,
            with_label=False,
            test_mode=True))

    train_dataset = get_dataset(data['val'])
    de_sampler = build_sampler(train_dataset,
                               data['imgs_per_gpu'],
                               data['workers_per_gpu'],
                               dist=False)

    dataset = de.GeneratorDataset(train_dataset, ['img', 'img_shape', 'gt_bboxe', 'gt_label', 'gt_num'])

    batch_size = 2
    global de_sampler_len
    de_sampler_len = len(train_dataset)
    dataset.set_dataset_size(dataset_len*2)

    keep_ratio = False
    flip = False
    size_divisor = 32

    colName = ['img', 'img_shape', 'gt_bboxe', 'gt_label', 'gt_num']
    compose_func = (lambda input_1, input_2, input_3, input_4, input_5: compose_map(keep_ratio, flip, size_divisor, \
                    input_1, input_2, input_3, input_4, input_5))
    dataset = dataset.map(input_columns=colName, output_columns=colName, operations=compose_func)
    dataset = dataset.shuffle(buffer_size=(de_sampler_len//16))   #Dataset Shuffle

    dataset = dataset.batch(batch_size=2)
    dataset = dataset.repeat(count=epoch_count*de_sampler_len//(dataset_len*2)) 
    dataset.de_sampler_len =  de_sampler_len

    generator_num_iters = math.ceil(de_sampler_len/batch_size)
    dataset.map_model = 4
 
    dataset.channel_name = 'FasterRCNN'

    return dataset


def faster_rcnn_test_dataset():
    dataset_type = 'CocoDataset'
    import os
    data_root = os.getenv("TDT_DATASET")
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    data = dict(
        imgs_per_gpu=2,
        workers_per_gpu=2,
        train=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            img_scale=(720, 1280),
            img_norm_cfg=img_norm_cfg,
            size_divisor=32,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=True,
            with_label=True),
        val=dict(
            type=dataset_type,
            ann_file =data_root + 'annotations/instances_val2017.json',
            img_prefix=data_root + 'val2017/',
            img_scale=(720, 1280),
            img_norm_cfg=img_norm_cfg,
            size_divisor=32,
            flip_ratio=0,
            with_mask=False,
            with_crowd=True,
            with_label=True),
        test=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_val2017.json',
            img_prefix=data_root + 'val2017/',
            img_scale=(720, 1280),
            img_norm_cfg=img_norm_cfg,
            size_divisor=32,
            flip_ratio=0,
            with_mask=False,
            with_label=False,
            test_mode=True))

    # create dataset and sampler
    train_dataset = get_dataset_test(data['test'])
    de_sampler = build_sampler_test(train_dataset,
                               data['imgs_per_gpu'],
                               data['workers_per_gpu'],
                               dist=False)

    batch_size = 1
    dataset = de.GeneratorDataset(train_dataset, ['img', 'img_shape', 'gt_bboxe', 'gt_label', 'gt_num'])
    generator_num_iters = math.ceil(len(train_dataset)/batch_size)
    dataset.set_dataset_size(generator_num_iters)
 
    keep_ratio = False
    flip = False
    size_divisor = 32

    colName = ['img', 'img_shape', 'gt_bboxe', 'gt_label', 'gt_num']
    compose_func = (lambda input_1, input_2, input_3, input_4, input_5: compose_map_test(keep_ratio, flip, size_divisor, \
                    input_1, input_2, input_3, input_4, input_5))
    dataset = dataset.map(input_columns=colName, output_columns=colName, operations=compose_func)
 
    dataset = dataset.batch(batch_size=1)
    dataset = dataset.repeat(count=epoch_count)

    dataset.map_model = 4 

    dataset.channel_name = 'FasterRCNN'

    return dataset


def faster_rcnn_test_dataset_feed():
    dataset_type = 'CocoDataset'
    import os
    data_root = os.getenv("TDT_DATASET")
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    data = dict(
        imgs_per_gpu=2,
        workers_per_gpu=2,
        train=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            img_scale=(720, 1280),
            img_norm_cfg=img_norm_cfg,
            size_divisor=32,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=True,
            with_label=True),
        val=dict(
            type=dataset_type,
            ann_file =data_root + 'annotations/instances_val2017.json',
            img_prefix=data_root + 'val2017/',
            img_scale=(720, 1280),
            img_norm_cfg=img_norm_cfg,
            size_divisor=32,
            flip_ratio=0,
            with_mask=False,
            with_crowd=True,
            with_label=True),
        test=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_val2017.json',
            img_prefix=data_root + 'val2017/',
            img_scale=(720, 1280),
            img_norm_cfg=img_norm_cfg,
            size_divisor=32,
            flip_ratio=0,
            with_mask=False,
            with_label=False,
            test_mode=True))

    # create dataset and sampler
    train_dataset = get_dataset_test(data['test'])
    de_sampler = build_sampler_test(train_dataset,
                               data['imgs_per_gpu'],
                               data['workers_per_gpu'],
                               dist=False)

    batch_size = 1
    dataset = de.GeneratorDataset(train_dataset, ['img', 'img_shape', 'gt_bboxe', 'gt_label', 'gt_num'])
    generator_num_iters = math.ceil(len(train_dataset)/batch_size)
    dataset.set_dataset_size(generator_num_iters)
 
    keep_ratio = False
    flip = False
    size_divisor = 32

    colName = ['img', 'img_shape', 'gt_bboxe', 'gt_label', 'gt_num']
    compose_func = (lambda input_1, input_2, input_3, input_4, input_5: compose_map_test(keep_ratio, flip, size_divisor, \
                    input_1, input_2, input_3, input_4, input_5))
    dataset = dataset.map(input_columns=colName, output_columns=colName, operations=compose_func)
 
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.repeat(count=1)

    dataset.map_model = 0  

    dataset.channel_name = 'FasterRCNN'

    return dataset


def faster_rcnn_train_dataset_parallel():
    dataset_type = 'CocoDataset'
    import os
    data_root = os.getenv("TDT_DATASET") 
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    data = dict(
        imgs_per_gpu=2,
        workers_per_gpu=2,
        train=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            img_scale=(720, 1280),
            img_norm_cfg=img_norm_cfg,
            size_divisor=32,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=True,
            with_label=True),
        test=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_val2017.json',
            img_prefix=data_root + 'val2017/',
            img_scale=(720, 1280),
            img_norm_cfg=img_norm_cfg,
            size_divisor=32,
            flip_ratio=0,
            with_mask=False,
            with_label=False,
            test_mode=True))

    de.config.set_prefetch_size(4)
    # create dataset and sampler
    train_dataset = get_dataset(data['train'])
    de_sampler = build_sampler(train_dataset,
                               data['imgs_per_gpu'],
                               data['workers_per_gpu'],
                               world_size=int(os.getenv("RANK_SIZE")),
                               rank=int(os.getenv("RANK_ID")),
                               dist=True)

    dataset = de.GeneratorDataset(train_dataset, ['img', 'img_shape', 'gt_bboxe', 'gt_label', 'gt_num'],
                                  sampler=de_sampler)
 
    global de_sampler_len
    de_sampler_len = len(de_sampler)    
    dataset.set_dataset_size(dataset_len*2)

    # img_transform Operation
    keep_ratio = False
    flip = False
    size_divisor = 32

    colName = ['img', 'img_shape', 'gt_bboxe', 'gt_label', 'gt_num']
    compose_func = (lambda input_1, input_2, input_3, input_4, input_5: compose_map(keep_ratio, flip, size_divisor, \
                    input_1, input_2, input_3, input_4, input_5))
    dataset = dataset.map(input_columns=colName, output_columns=colName, operations=compose_func)
    dataset = dataset.shuffle(buffer_size=(len(de_sampler)//16)) 
 
    dataset = dataset.batch(batch_size=2)
    dataset = dataset.repeat(count=epoch_count*de_sampler_len//(dataset_len*2)) 
    dataset.de_sampler_len = de_sampler_len

    batch_size = 2
    dataset.map_model = 4
    dataset.channel_name = 'FasterRCNN'

    return dataset


def net_train_mode(network):
    net = network
    net = net.set_train(True) 

    return net


def net_test_mode(network):
    net = network 
    net = net.set_train(False) 
    
    return net


def test_train_faster_rcnn():
    ds = faster_rcnn_train_dataset()
    epoch_size = epoch_count*de_sampler_len//(dataset_len*2)

    config = Config_Faster_Rcnn()
    net = Faster_Rcnn_Resnet50()
    net = net_train_mode(net)

    load_path = os.getenv("LOAD_CHECKPOINT_PATH")
    assert isinstance(load_path, str)
    param_dict = load_checkpoint(load_path)
    for item in list(param_dict.keys()):
       if not item.startswith('backbone'):
           param_dict.pop(item)
    load_param_into_net(net, param_dict)

    loss = LossNet()
    config.base_lr = 0.005 
    lr = Tensor(dynamic_lr(config), ms.float32)

    opt = SGD(params=net.trainable_params(), learning_rate=lr, momentum=0.91, weight_decay=0.0001, loss_scale = 1.0)
    net = WithLossCell(net, loss)
    net = TrainOneStepCell(net, opt, sens = 1.0)

    callback = LossCallBack()
    ckptconfig = CheckpointConfig(save_checkpoint_steps = 1, keep_checkpoint_max = 35)
    ckpoint_cb = ModelCheckpoint(prefix='faster_rcnn', directory="checkpoint", config=ckptconfig)

    model = Model(net)
    model._train_network = net
    model.train(epoch_size, ds, callbacks=[callback, ckpoint_cb])


def test_faster_rcnn_mAP():
    from datetime import datetime
    ds = faster_rcnn_test_dataset_feed() 

    epoch_size = 1
    count = 1
    batch_size = 1
    max_num = 100
    num_classes = 81

    config = Config_Faster_Rcnn()
    net = Faster_Rcnn_Resnet50()
    net = net_test_mode(net)   
 
    load_path = os.getenv("LOAD_CHECKPOINT_PATH")    
    assert(isinstance(load_path, str))
    
    param_dict = load_checkpoint(load_path)
    load_param_into_net(net, param_dict)
    
    data_root = os.getenv("TDT_DATASET")
    assert(isinstance(data_root, str))    
    ann_file = data_root + 'annotations/instances_val2017.json'
    dataset_coco = COCO(ann_file)
    outputs = []
    iter = 0
    net_time_total = 0

    for val in ds.create_dict_iterator():
        iter += 1
        img_data = val['img']
        img_metas = val['img_shape']
        gt_bboxes = val['gt_bboxe']
        gt_labels = val['gt_label']
        gt_num = val['gt_num']

        # run net
        output = net(Tensor(img_data), Tensor(img_metas), Tensor(gt_bboxes), Tensor(gt_labels), Tensor(gt_num))

        # output
        all_bbox = output[0]
        all_label = output[1]
        all_mask = output[2]

        for j in range(config.test_batch_size):
            all_bbox_squee = np.squeeze(all_bbox.asnumpy())
            all_label_squee = np.squeeze(all_label.asnumpy())
            all_mask_squee = np.squeeze(all_mask.asnumpy())

            all_bboxes_tmp = []
            all_labels_tmp = []
            all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
            all_labels_tmp_mask = all_label_squee[all_mask_squee]

            if all_bboxes_tmp_mask.shape[0] > max_num:
                inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
                inds = inds[:max_num]
                all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
                all_labels_tmp_mask = all_labels_tmp_mask[inds]

            outputs_tmp = bbox2result_1image(all_bboxes_tmp_mask, all_labels_tmp_mask, num_classes)

            outputs.append(outputs_tmp)

    eval_types = ["bbox"]
    result_files = results2json(dataset_coco, outputs, "./results.pkl")
    
    coco_eval(result_files, eval_types, dataset_coco, single_result=True)


def test_train_faster_rcnn_parallel():
    ds = faster_rcnn_train_dataset_parallel()
    epoch_size = epoch_count*de_sampler_len//(dataset_len*2)

    config = Config_Faster_Rcnn()
    net = Faster_Rcnn_Resnet50()
    net = net_train_mode(net)

    load_path = os.getenv("LOAD_CHECKPOINT_PATH")
    assert isinstance(load_path, str)
    param_dict = load_checkpoint(load_path)
    for item in list(param_dict.keys()):
       if not item.startswith('backbone'):
           param_dict.pop(item)
    load_param_into_net(net, param_dict)

    device_num = int(os.getenv("RANK_SIZE"))
    loss = LossNet()
    lr = Tensor(dynamic_lr(config, rank_size = device_num), ms.float32)

    opt = SGD(params=net.trainable_params(), learning_rate=lr, momentum=0.91, weight_decay=0.0001, loss_scale=1.0)
    from mindspore.communication.management import init, release    
    init()    
    
    net = WithLossCell(net, loss)
    net = TrainOneStepCell(net, opt, sens=1.0, reduce_flag=True, mean=True, degree=device_num)

    callback = LossCallBack()
    ckptconfig = CheckpointConfig(save_checkpoint_steps = 10000, keep_checkpoint_max = 100)
    ckpoint_cb = ModelCheckpoint(prefix='faster_rcnn', directory="checkpoint", config=ckptconfig)
   
    context.set_auto_parallel_context(parallel_mode = ParallelMode.DATA_PARALLEL, device_num = device_num) 
    model = Model(net) 
    model._train_network = net
    model.train(epoch_size, ds, callbacks=[callback, ckpoint_cb])


def test_faster_rcnn():
    ds_train = faster_rcnn_train_dataset_parallel()
    ds = faster_rcnn_test_dataset_feed()

    epoch_size = de_sampler_len//(dataset_len*2)
    max_num = 100
    num_classes = 81

    config = Config_Faster_Rcnn()

    net = Faster_Rcnn_Resnet50()
    net_test = net_test_mode(net)
    net_train = net_train_mode(net_test)

    load_path = os.getenv("LOAD_CHECKPOINT_PATH")
    assert isinstance(load_path, str)
    param_dict = load_checkpoint(load_path)
    for item in list(param_dict.keys()):
       if not item.startswith('backbone'):
           param_dict.pop(item)
    load_param_into_net(net_train, param_dict)

    device_num = int(os.getenv("RANK_SIZE"))
    loss = LossNet()
    lr = Tensor(dynamic_lr(config, rank_size = device_num), ms.float32)

    opt = SGD(params=net_train.trainable_params(), learning_rate=lr, momentum=0.91, weight_decay=0.0001, loss_scale=1.0)
    from mindspore.communication.management import init, release
    init()

    net_train = WithLossCell(net_train, loss)
    net_train = TrainOneStepCell(net_train, opt, sens=1.0, reduce_flag=True, mean=True, degree=device_num)

    data_root = os.getenv("TDT_DATASET")
    assert(isinstance(data_root, str))
    dataset_coco = COCO(data_root + 'annotations/instances_val2017.json')
    coco_metric = COCOMetric(dataset_coco, max_num=max_num, num_classes=num_classes, bbox_json_dir="./results.pkl")
   
    callback = LossCallBack()
    ckptconfig = CheckpointConfig(save_checkpoint_steps = 10000, keep_checkpoint_max = 100)
    ckpoint_cb = ModelCheckpoint(prefix='faster_rcnn', directory="checkpoint", config=ckptconfig)

    context.set_auto_parallel_context(parallel_mode = ParallelMode.DATA_PARALLEL, device_num = device_num) 
    model = Model(net_train, eval_network = net_test, metrics={'Precision/mAP@.50IOU': coco_metric})
    for i in range(epoch_count):
        model.train(epoch_size, ds_train, callbacks=[callback, ckpoint_cb])
        print("FasterRcnn Train epoch:{}".format(i))

    outputs = []
    iter = 0
    net_time_total = 0

    for val in ds.create_dict_iterator():
        iter += 1
        img_data = val['img']
        img_metas = val['img_shape']
        gt_bboxes = val['gt_bboxe']
        gt_labels = val['gt_label']
        gt_num = val['gt_num']

        # run net
        net_test.set_train(False)
        output = net_test(Tensor(img_data), Tensor(img_metas), Tensor(gt_bboxes), Tensor(gt_labels), Tensor(gt_num))

        # output
        all_bbox = output[0]
        all_label = output[1]
        all_mask = output[2]

        for j in range(config.test_batch_size):
            all_bbox_squee = np.squeeze(all_bbox.asnumpy())
            all_label_squee = np.squeeze(all_label.asnumpy())
            all_mask_squee = np.squeeze(all_mask.asnumpy())

            all_bboxes_tmp = []
            all_labels_tmp = []
            all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
            all_labels_tmp_mask = all_label_squee[all_mask_squee]

            if all_bboxes_tmp_mask.shape[0] > max_num:
                inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
                inds = inds[:max_num]
                all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
                all_labels_tmp_mask = all_labels_tmp_mask[inds]

            outputs_tmp = bbox2result_1image(all_bboxes_tmp_mask, all_labels_tmp_mask, num_classes)

            outputs.append(outputs_tmp)

    eval_types = ["bbox"]
    result_files = results2json(dataset_coco, outputs, "./results.pkl")

    coco_eval(result_files, eval_types, dataset_coco, single_result=True)


