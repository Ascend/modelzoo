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
"""Face Recognition eval."""
import os
import re
import numpy as np
from PIL import Image
import argparse
import torch
import torchvision.transforms as trans
from tqdm import tqdm
import warnings

from mindspore import context
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.reid import SphereNet

warnings.filterwarnings('ignore')
devid = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=True, device_id=devid)


def inclass_likehood(ims_info, type='cos'):
    obj_feas = {}
    likehoods = []
    for name, _, fea in ims_info:
        if re.split('_\d\d\d\d', name)[0] not in obj_feas:
            obj_feas[re.split('_\d\d\d\d', name)[0]] = []
        obj_feas[re.split('_\d\d\d\d', name)[0]].append(fea)
    for _, feas in tqdm(obj_feas.items()):
        feas = np.array(feas)
        if type == 'cos':
            likehood_mat = np.dot(feas, np.transpose(feas)).tolist()
            for row in likehood_mat:
                likehoods += row
        else:
            for fea in feas.tolist():
                likehoods += np.sum(-(fea - feas) ** 2, axis=1).tolist()

    likehoods = np.array(likehoods)
    return likehoods


def btclass_likehood(ims_info, fm_range=1000, type='cos'):
    likehoods = []
    count = 0
    for name1, _, fea1 in tqdm(ims_info):
        count += 1
        frame_id1, obj_id1 = re.split('_\d\d\d\d', name1)[0], name1.split('_')[-1]
        fea1 = np.array(fea1)
        for name2, _, fea2 in ims_info:
            frame_id2, obj_id2 = re.split('_\d\d\d\d', name2)[0], name2.split('_')[-1]
            if frame_id1 == frame_id2:
                continue
            fea2 = np.array(fea2)
            if type == 'cos':
                likehoods.append(np.sum(fea1 * fea2))
            else:
                likehoods.append(np.sum(-(fea1 - fea2) ** 2))

    likehoods = np.array(likehoods)
    return likehoods


def tar_at_far(inlikehoods, btlikehoods):
    test_point = [0.5, 0.3, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    tar_far = []
    for point in test_point:
        thre = btlikehoods[int(btlikehoods.size * point)]
        n_ta = np.sum(inlikehoods > thre)
        tar_far.append((point, float(n_ta) / inlikehoods.size, thre))

    return tar_far


def load_images(paths, batch_size=128):
    ll = []
    transform = trans.Compose([
        trans.Resize((96, 64)),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    for i in range(len(paths)):
        im = Image.open(paths[i])
        ts = transform(im)
        ll.append(ts)
        if len(ll) == batch_size:
            yield torch.stack(ll, dim=0)
            ll.clear()
    if len(ll) > 0:
        yield torch.stack(ll, dim=0)


def main(args):
    model_path = args.pretrained
    result_file = model_path.replace('.ckpt', '.txt')
    if os.path.exists(result_file):
        os.remove(result_file)

    with open(result_file, 'a+') as result_fw:
        result_fw.write(model_path + '\n')

        network = SphereNet(num_layers=12, feature_dim=128, shape=(96, 64))
        if os.path.isfile(model_path):
            param_dict = load_checkpoint(model_path)
            param_dict_new = {}
            for key, values in param_dict.items():
                if key.startswith('moments.'):
                    continue
                elif key.startswith('model.'):
                    param_dict_new[key[6:]] = values
                else:
                    param_dict_new[key] = values
            load_param_into_net(network, param_dict_new)
            print('-----------------------load model success-----------------------')
        else:
            print('-----------------------load model failed -----------------------')

        network.add_flags_recursive(fp16=True)
        network.set_train(False)

        root_path = args.eval_dir
        root_file_list = os.listdir(root_path)
        ims_info = []
        for sub_path in root_file_list:
            for im_path in os.listdir(os.path.join(root_path, sub_path)):
                ims_info.append((im_path.split('.')[0], os.path.join(root_path, sub_path, im_path)))

        paths = [path for name, path in ims_info]
        names = [name for name, path in ims_info]
        print("exact feature...")

        l_t = []
        for batch in load_images(paths):
            batch = batch.numpy().astype(np.float32)
            batch = Tensor(batch)
            fea = network(batch)
            l_t.append(fea.asnumpy().astype(np.float16))
        feas = np.concatenate(l_t, axis=0)
        ims_info = list(zip(names, paths, feas.tolist()))

        print("exact inclass likehood...")
        inlikehoods = inclass_likehood(ims_info)
        inlikehoods[::-1].sort()

        print("exact btclass likehood...")
        btlikehoods = btclass_likehood(ims_info)
        btlikehoods[::-1].sort()
        tar_far = tar_at_far(inlikehoods, btlikehoods)

        for far, tar, thre in tar_far:
            print('---{}: {}@{}'.format(far, tar, thre))

        for far, tar, thre in tar_far:
            result_fw.write('{}: {}@{} \n'.format(far, tar, thre))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reid test')
    parser.add_argument('--pretrained', type=str, default='', help='pretrained model to load')
    parser.add_argument('--eval_dir', type=str, default='', help='eval image dir, e.g. /home/test')

    args = parser.parse_args()
    print(args)

    main(args)
