# Copyright 2020 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import numpy as np
import time
from mindspore import Tensor
import mindspore.nn as nn
from mindspore.nn import Conv2d, ReLU, MaxPool2d
import mindspore.ops.operations as Op
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import get_group_size
from mindspore.ops import operations as P
import mindspore as ms
from mindspore.context import ParallelMode
from mindspore.common.initializer import initializer
from mindspore.common import dtype as mstype
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore import context
context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
#selfCat = Op.Concat(axis = 1)

time_stamp_init = False
time_stamp_first = 0
loadvgg = 1

            
class Depend_network(nn.Cell):
    def __init__(self,network):
        super(Depend_network,self).__init__()
        self.network = network
        
    def construct(self, *args):
        loss,heatmaps_loss, pafs_loss = self.network(*args)
        return loss
        
class TrainingWrapper(nn.Cell):
    def __init__(self, network, optimizer, sens=1):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.depend_network = Depend_network(network)
        # self.weights = ms.ParameterTuple(network.trainable_params())
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation( get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.print = P.Print()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            #if mean.get_device_num_is_set():
            # if mean:
                #degree = context.get_auto_parallel_context("device_num")
            # else:
            degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, *args):
        weights = self.weights
        #print('lr',self.optimizer.get_lr())
        loss,heatmaps_loss, pafs_loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        #grads = self.grad(self.network, weights)(*args, sens)
        grads = self.grad(self.depend_network, weights)(*args, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        #return F.depend(loss, self.optimizer(grads))
        # for grad in grads:
            # self.print(grad)
        loss = F.depend(loss, self.optimizer(grads))
        return loss,heatmaps_loss, pafs_loss

class OpenPoseNet(nn.Cell):
    insize = 368
    def __init__(self, vggpath = ''):
        super(OpenPoseNet, self).__init__()
        self.base = Base_model()
        self.stage_1 = Stage_1()
        self.stage_2 = Stage_x()
        self.stage_3 = Stage_x()
        self.stage_4 = Stage_x()
        self.stage_5 = Stage_x()
        self.stage_6 = Stage_x()
        self.shape = P.Shape()
        self.cat = Op.Concat(axis = 1)
        self.print = P.Print()
        # for m in self.modules():
            # if isinstance(m, Conv2d):
                # init.constant_(m.bias, 0)
        if loadvgg and vggpath:
            param_dict = load_checkpoint(vggpath)
            param_dict_new = {}
            trans_name = 'base.vgg_base.'
            for key, values in param_dict.items():
              
                #print('key:',key,self.shape(values))
                if key.startswith('moments.'):
                    continue
                elif key.startswith('network.'):
                    param_dict_new[trans_name+key[17:]] = values
                # else:
                    # param_dict_new[key] = values
            #print(param_dict_new)
            load_param_into_net(self.base.vgg_base, param_dict_new)

    def construct(self, x):
        heatmaps = []
        pafs = []
        feature_map = self.base(x)
        h1, h2 = self.stage_1(feature_map)
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_2(self.cat((h1, h2, feature_map)))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_3(self.cat((h1, h2, feature_map)))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_4(self.cat((h1, h2, feature_map)))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_5(self.cat((h1, h2, feature_map)))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_6(self.cat((h1, h2, feature_map)))
        pafs.append(h1)
        heatmaps.append(h2)
        return pafs, heatmaps        

class Vgg(nn.Cell):
    def __init__(self, cfg, batch_norm=False):
        # Important: When choose vgg, batch_size should <=64, otherwise will cause unknown error
        super(Vgg, self).__init__()
        self.layers = self._make_layer(cfg, batch_norm=batch_norm)

    def construct(self, x):
        x = self.layers(x)
        return x

    def _make_layer(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                # layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, pad_mode = 'same')]
            else:
             
                conv2d = Conv2d(in_channels=in_channels,
                                out_channels=v,
                                kernel_size=3,
                                stride = 1,
                                # padding=1,
                                pad_mode='same',
                                has_bias=True)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
                else:
                    layers += [conv2d, nn.ReLU()]
                in_channels = v
        return nn.SequentialCell(layers)
        
class VGG_Base(nn.Cell):
    def __init__(self):
        super(VGG_Base, self).__init__()
        self.conv1_1 = Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, pad_mode='same',has_bias = True)
        self.conv1_2 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad_mode='same',has_bias = True)
        
        self.conv2_1 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, pad_mode='same',has_bias = True)
        self.conv2_2 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,pad_mode='same',has_bias = True)

        self.conv3_1 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, pad_mode='same',has_bias = True)
        self.conv3_2 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, pad_mode='same',has_bias = True)
        self.conv3_3 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, pad_mode='same',has_bias = True)
        self.conv3_4 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, pad_mode='same',has_bias = True)
      
        self.conv4_1 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, pad_mode='same',has_bias = True)
        self.conv4_2 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, pad_mode='same',has_bias = True)
        self.relu = ReLU()
        self.max_pooling_2d = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def construct(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.max_pooling_2d(x)
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.max_pooling_2d(x)
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.relu(self.conv3_4(x))
        x = self.max_pooling_2d(x)
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        return x

class VGG_Base_MS(nn.Cell):
    def __init__(self):
        super(VGG_Base_MS, self).__init__()
        self.Layer1_1 = Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, pad_mode='same',has_bias = True)
        self.Layer1_2 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad_mode='same',has_bias = True)
        
        self.Layer2_1 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, pad_mode='same',has_bias = True)
        self.Layer2_2 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,pad_mode='same',has_bias = True)

        self.Layer3_1 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, pad_mode='same',has_bias = True)
        self.Layer3_2 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, pad_mode='same',has_bias = True)
        self.Layer3_3 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, pad_mode='same',has_bias = True)
        self.Layer3_4 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, pad_mode='same',has_bias = True)
      
        self.Layer4_1 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, pad_mode='same',has_bias = True)
        self.Layer4_2 = Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, pad_mode='same',has_bias = True)
        self.relu = ReLU()
        self.max_pooling_2d = nn.MaxPool2d(kernel_size=2, stride=2)

    def construct(self, x):
        x = self.relu(self.Layer1_1(x))
        x = self.relu(self.Layer1_2(x))
        x = self.max_pooling_2d(x)
        x = self.relu(self.Layer2_1(x))
        x = self.relu(self.Layer2_2(x))
        x = self.max_pooling_2d(x)
        x = self.relu(self.Layer3_1(x))
        x = self.relu(self.Layer3_2(x))
        x = self.relu(self.Layer3_3(x))
        x = self.relu(self.Layer3_4(x))
        x = self.max_pooling_2d(x)
        x = self.relu(self.Layer4_1(x))
        x = self.relu(self.Layer4_2(x))
        return x

class Base_model(nn.Cell):
    def __init__(self):
        super(Base_model, self).__init__()
        #cfgs_zh = {'16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512]}
        cfgs_zh = {'19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512]}
        #cfgs_zh = {'16': [64, 64,128, 128, 256, 256, 256, 512, 512, 512]}
        self.vgg_base = Vgg(cfgs_zh['19'], batch_norm = False)
        #self.vgg_base = VGG_Base()
        #self.vgg_base = VGG_Base_MS()
        
        self.conv4_3_CPM = Conv2d(in_channels=512, out_channels=256,  kernel_size = 3, stride = 1, pad_mode='same', has_bias = True)
        self.conv4_4_CPM = Conv2d(in_channels=256, out_channels=128,  kernel_size = 3, stride = 1, pad_mode='same', has_bias = True)
        self.relu = ReLU()
    def construct(self, x):
        x = self.vgg_base(x)
        x = self.relu(self.conv4_3_CPM(x))
        x = self.relu(self.conv4_4_CPM(x))
        return x
  
class Stage_1(nn.Cell):
    def __init__(self):
        super(Stage_1, self).__init__()
       
        self.conv1_CPM_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad_mode='same', has_bias = True)
        self.conv2_CPM_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad_mode='same', has_bias = True)
        self.conv3_CPM_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad_mode='same', has_bias = True)
      
        self.conv4_CPM_L1 = Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, pad_mode='same', has_bias = True)
       
        self.conv5_CPM_L1 = Conv2d(in_channels=512, out_channels=38, kernel_size=1, stride=1, pad_mode='same', has_bias = True)
       
        self.conv1_CPM_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad_mode='same', has_bias = True)
        self.conv2_CPM_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad_mode='same', has_bias = True)
        self.conv3_CPM_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad_mode='same', has_bias = True)
      
        self.conv4_CPM_L2 = Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, pad_mode='same', has_bias = True)
        self.conv5_CPM_L2 = Conv2d(in_channels=512, out_channels=19, kernel_size=1, stride=1, pad_mode='same', has_bias = True)
        self.relu = ReLU()
        
    def construct(self, x):
        h1 = self.relu(self.conv1_CPM_L1(x)) # branch1
        h1 = self.relu(self.conv2_CPM_L1(h1))
        h1 = self.relu(self.conv3_CPM_L1(h1))
        h1 = self.relu(self.conv4_CPM_L1(h1))
        h1 = self.conv5_CPM_L1(h1)
        h2 = self.relu(self.conv1_CPM_L2(x)) # branch2
        h2 = self.relu(self.conv2_CPM_L2(h2))
        h2 = self.relu(self.conv3_CPM_L2(h2))
        h2 = self.relu(self.conv4_CPM_L2(h2))
        h2 = self.conv5_CPM_L2(h2)
        return h1, h2
    
class Stage_x(nn.Cell):
    def __init__(self):
        super(Stage_x, self).__init__()
        self.conv1_L1 = Conv2d(in_channels = 185, out_channels = 128, kernel_size = 7, stride = 1, pad_mode='same', has_bias = True)
       
        self.conv2_L1 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, pad_mode='same', has_bias = True)
        self.conv3_L1 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, pad_mode='same', has_bias = True)
        self.conv4_L1 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, pad_mode='same', has_bias = True)
        self.conv5_L1 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, pad_mode='same', has_bias = True)
       
        self.conv6_L1 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 1, stride = 1, pad_mode='same', has_bias = True)
       
        self.conv7_L1 = Conv2d(in_channels = 128, out_channels = 38, kernel_size = 1, stride = 1, pad_mode='same', has_bias = True)
       
        self.conv1_L2 = Conv2d(in_channels = 185, out_channels = 128, kernel_size = 7, stride = 1, pad_mode='same', has_bias = True)
        
        self.conv2_L2 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, pad_mode='same', has_bias = True)
        self.conv3_L2 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, pad_mode='same', has_bias = True)
        self.conv4_L2 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, pad_mode='same', has_bias = True)
        self.conv5_L2 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, pad_mode='same', has_bias = True)
  
        self.conv6_L2 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 1, stride = 1, pad_mode='same', has_bias = True)
        self.conv7_L2 = Conv2d(in_channels = 128, out_channels = 19, kernel_size = 1, stride = 1, pad_mode='same', has_bias = True)
        self.relu = ReLU()
        
    def construct(self, x):
        h1 = self.relu(self.conv1_L1(x)) # branch1
        h1 = self.relu(self.conv2_L1(h1))
        h1 = self.relu(self.conv3_L1(h1))
        h1 = self.relu(self.conv4_L1(h1))
        h1 = self.relu(self.conv5_L1(h1))
        h1 = self.relu(self.conv6_L1(h1))
        h1 = self.conv7_L1(h1)
        h2 = self.relu(self.conv1_L2(x)) # branch2
        h2 = self.relu(self.conv2_L2(h2))
        h2 = self.relu(self.conv3_L2(h2))
        h2 = self.relu(self.conv4_L2(h2))
        h2 = self.relu(self.conv5_L2(h2))
        h2 = self.relu(self.conv6_L2(h2))
        h2 = self.conv7_L2(h2)
        return h1, h2
