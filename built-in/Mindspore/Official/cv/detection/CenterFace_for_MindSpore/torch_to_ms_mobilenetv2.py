# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
mobilenet model transform: torch => mindspore
"""
import torch
import os
import argparse
import numpy as np

from mindspore.train.serialization import load_checkpoint, save_checkpoint
from mindspore import Tensor

def load_model(model_path):
    state_dict_ = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.find("num_batches_tracked") != -1:
            continue
        elif k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    return state_dict

def load_model_ms(model_path, args=None):
    state_dict_useless = ['global_step', 'learning_rate',
                          'beta1_power', 'beta2_power']
    if os.path.isfile(model_path):
        param_dict = load_checkpoint(model_path)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key in state_dict_useless or key.startswith('moments.') or key.startswith('moment1.') or key.startswith('moment2.'):
                continue
            elif key.startswith('centerface_network.'): #useless, since the start name is "network.backbone."
                param_dict_new[key[19:]] = values
            else:
                param_dict_new[key] = values
    else:
        assert FileNotFoundError('{} not exists or not a pre-trained file'.format(model_path))
        exit(1)
    return param_dict_new

def name_map(pt, ckpt):
    out = {}
    for name in ckpt:
        # conv + bn
        pt_name = name

        pt_name = pt_name.replace('network.backbone.', '')
        # backbone
        pt_name = pt_name.replace('need_fp1', 'feature_1')
        pt_name = pt_name.replace('need_fp2', 'feature_2')
        pt_name = pt_name.replace('need_fp3', 'feature_4')
        pt_name = pt_name.replace('need_fp4', 'feature_6')
        pt_name = pt_name.replace('.features', '')
        pt_name = pt_name.replace('.moving_mean', '.running_mean')
        pt_name = pt_name.replace('.moving_variance', '.running_var')
        pt_name = pt_name.replace('.gamma', '.weight')
        pt_name = pt_name.replace('.beta', '.bias')
        # fpn
        pt_name = pt_name.replace('.up1', '.up_0')
        pt_name = pt_name.replace('.up2', '.up_1')
        pt_name = pt_name.replace('.up3', '.up_2')

        # heads
        pt_name = pt_name.replace('hm_head.0.', 'hm.')
        pt_name = pt_name.replace('wh_head.', 'wh.')
        pt_name = pt_name.replace('off_head.', 'hm_offset.')
        pt_name = pt_name.replace('kps_head.', 'landmarks.')

        pt_name = pt_name.replace('network.head.fc.', 'classifier.1.')

        out[pt_name] = name
    return out

def pt_to_ckpt(pt, ckpt, out_ckpt):
    state_dict_torch = load_model(pt)
    state_dict_ms = load_model_ms(ckpt)
    name_relate = name_map(state_dict_torch, state_dict_ms)
    new_params_list = []

    for key in state_dict_torch:
        param_dict = {}
        parameter = state_dict_torch[key]
        parameter = parameter.numpy()

        # depwise conv pytorch[cout, 1, k , k] -> ms[1, cin, k , k], cin = cout
        if state_dict_ms[name_relate[key]].data.shape != parameter.shape:
            parameter = parameter.transpose(1, 0, 2, 3)
            print('ms=', state_dict_ms[name_relate[key]].data.shape, 'pytorch=', parameter.shape, 'name=',key)


        param_dict['name'] = name_relate[key]
        param_dict['data'] = Tensor(parameter)
        new_params_list.append(param_dict)

    save_checkpoint(new_params_list, out_ckpt)
    return state_dict_ms

if __name__ == "__main__":
    '''
    centerFace need a pretrained mobilenet_v2 model:
        mobilenet_v2_key.ckpt is a model with all value zero, we need the key/cell/module name for this model.
        you must first use this script to convert your mobilenet_v2 pytorch model to mindspore model as a pretrain model.
        The key/cell/module name must as follow, otherwise you need to modify "name_map" function:
            --mindspore: as the same as mobilenet_v2_key.ckpt
            --pytorch: same as official pytorch model(e.g., official mobilenet_v2-b0353104.pth)
    '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ckpt_fn', type=str, default='/model_path/mobilenet_v2_key.ckpt',
                        help='ckpt for user to get cell/module name')
    parser.add_argument('--pt_fn', type=str, default='/model_path/mobilenet_v2-b0353104.pth',
                        help='checkpoint filename to convert')
    parser.add_argument('--out_ckpt_fn', type=str, default='/model_path/mobilenet_v2-b0353104.ckpt',
                        help='convert output ckpt path')

    args = parser.parse_args()
    print('args:', args)

    # beta <=> bias, gamma <=> weight
    pt_to_ckpt(args.pt_fn, args.ckpt_fn, args.out_ckpt_fn)
