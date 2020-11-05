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
import time
import cv2
import math
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Sampler
from PIL import Image
from pprint import pformat
import numpy as np

from mindspore import Tensor, context
devid = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=devid)
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import config_inference
from src.backbone.resnet import get_backbone
from src.logging import get_logger


class TxtDataset(object):
    def __init__(self, root_all, filenames, transform=None):
        super(TxtDataset, self).__init__()
        self.imgs = []
        self.labels = []
        for root, filename in zip(root_all, filenames):
            fin = open(filename, "r")
            for line in fin:
                self.imgs.append(os.path.join(root, line.strip().split(" ")[0]))
                self.labels.append(line.strip())
            fin.close()
        self.transform = transform

    def __getitem__(self, index):
        try:
            img = cv2.cvtColor(cv2.imread(self.imgs[index]), cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        except:
            print(self.imgs[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, index

    def __len__(self):
        return len(self.imgs)

    def get_all_labels(self):
        return self.labels

class DistributedSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_replicas = 1
        self.rank = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

def get_dataloader(img_predix_all, img_list_all, batch_size, img_transforms):
    dataset = TxtDataset(img_predix_all, img_list_all, transform=img_transforms)
    sampler = DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size, 
                    sampler=sampler, 
                    shuffle=False,
                    pin_memory=False,
                    num_workers=8,
                    drop_last=False,
                    )
    return dataloader, len(dataset)

def generate_test_pair(jk_list, zj_list):
    file_paths = [jk_list, zj_list]
    jk_dict = {}
    zj_dict = {}
    jk_zj_dict_list = [jk_dict, zj_dict]
    for path, x_dict in zip(file_paths, jk_zj_dict_list):
        with open(path,'r') as fr:
            for line in fr:
                label = line.strip().split(' ')[1]
                tmp = x_dict.get(label,[])
                tmp.append(line.strip())
                x_dict[label] = tmp
    zj2jk_pairs = []
    for key in jk_dict:
        jk_file_list = jk_dict[key]
        zj_file_list = zj_dict[key]
        for zj_file in zj_file_list:
            zj2jk_pairs.append([zj_file, jk_file_list])
    return zj2jk_pairs

def check_minmax(data, min=0.99, max=1.01):
    _min = data.min()
    _max = data.max()
    if torch.isnan(_min) or torch.isnan(_max):
        args.logger.info('ERROR, nan happened, please check if used fp16 or other error')
        raise Exception
    elif _min < min or _max > max:
        args.logger.info('ERROR, min or max is out if range, range=[{}, {}], minmax=[{}, {}]'.format(min, max, _min, _max))
        raise Exception

def get_model(args):
    net = get_backbone(args)
    if args.fp16:
        net.add_flags_recursive(fp16=True)

    if args.weight.endswith('.ckpt'):
        param_dict = load_checkpoint(args.weight)
        param_dict_new = {}
        for key, value in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = value
            else:
                param_dict_new[key] = value
        load_param_into_net(net, param_dict_new)
        args.logger.info('INFO, ------------- load model success--------------')
    else:
        args.logger.info('ERROR, not supprot file:{}, please check weight in config.py'.format(args.weight))
        return 0
    net.set_train(False)
    return net

def cal_topk(idx, zj2jk_pairs, test_embedding_tot, dis_embedding_tot):
    args.logger.info('start idx:{} subprocess...'.format(idx))
    correct = torch.Tensor([0] * 2)
    tot = torch.Tensor([0])

    zj, jk_all = zj2jk_pairs[idx]
    zj_embedding = test_embedding_tot[zj]
    jk_all_embedding = torch.cat([test_embedding_tot[jk].unsqueeze(0) for jk in jk_all], dim=0)
    args.logger.info('INFO, calculate top1 acc index:{}, zj_embedding shape:{}'.format(idx, zj_embedding.shape))
    args.logger.info('INFO, calculate top1 acc index:{}, jk_all_embedding shape:{}'.format(idx, jk_all_embedding.shape))

    test_time = time.time()
    top100_jk2zj = torch.mm(zj_embedding.unsqueeze(0), dis_embedding_tot).topk(100)[0].squeeze(0)
    top100_zj2jk = torch.mm(jk_all_embedding, dis_embedding_tot).topk(100)[0]
    test_time_used = time.time() - test_time
    args.logger.info('INFO, calculate top1 acc index:{}, torch.mm().top(100) time used:{:.2f}s'.format(idx, test_time_used))

    tot[0] = len(jk_all)
    for i, jk in enumerate(jk_all):
        jk_embedding = test_embedding_tot[jk]
        similarity = torch.dot(jk_embedding, zj_embedding)
        if similarity > top100_jk2zj[0]:
            correct[0] += 1
        if similarity > top100_zj2jk[i, 0]:
            correct[1] += 1
    return correct, tot

@torch.no_grad()
def main(args):
    if not os.path.exists(args.test_dir):
        args.logger.info('ERROR, test_dir is not exists, please set test_dir in config.py.')
        exit(0)
    all_start_time = time.time()

    net = get_model(args)
    compile_time_used = time.time() - all_start_time
    args.logger.info('INFO, graph compile finished, time used:{:.2f}s, start calculate img embedding'.format(compile_time_used))

    img_transforms = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    #for test images
    args.logger.info('INFO, start step1, calculate test img embedding, weight file = {}'.format(args.weight))
    step1_start_time = time.time()

    dataloader, img_tot = get_dataloader(args.test_img_predix, args.test_img_list, args.test_batch_size, img_transforms)
    args.logger.info('INFO, dataloader total test img:{}, total test batch:{}'.format(img_tot, len(dataloader)))
    test_embedding_tot_tensor = torch.zeros(img_tot, args.emb_size)
    test_img_labels = dataloader.dataset.get_all_labels()
    for img, idxs in dataloader:
        img = img.numpy()
        out = net(Tensor(img)).asnumpy().astype(np.float32)
        out = torch.from_numpy(out)
        embeddings = F.normalize(out)
        for batch in range(embeddings.size(0)):
            test_embedding_tot_tensor[idxs[batch]] = embeddings[batch]
    try:
        check_minmax(torch.norm(test_embedding_tot_tensor, p=2, dim=1).detach().cpu())
    except:
        return 0

    test_embedding_tot = {}
    for idx, label in enumerate(test_img_labels):
        test_embedding_tot[label] = test_embedding_tot_tensor[idx]

    step2_start_time = time.time()
    step1_time_used = step2_start_time - step1_start_time
    args.logger.info('INFO, step1 finished, time used:{:.2f}s, start step2, calculate dis img embedding'.format(step1_time_used))

    # for dis images
    dataloader, img_tot = get_dataloader(args.dis_img_predix, args.dis_img_list, args.dis_batch_size, img_transforms)
    dis_embedding_tot = torch.zeros(img_tot, args.emb_size)
    total_batch = len(dataloader)
    args.logger.info('INFO, dataloader total dis img:{}, total dis batch:{}'.format(img_tot, total_batch))
    start_time = time.time()
    img_per_gpu = int(math.ceil(1.0 * img_tot / args.world_size))
    delta_num = img_per_gpu * args.world_size - img_tot
    start_idx = img_per_gpu * args.local_rank - max(0, args.local_rank - (args.world_size - delta_num))
    for idx, (img, _) in enumerate(dataloader):
        img = img.numpy()
        out = net(Tensor(img)).asnumpy().astype(np.float32)
        out = torch.from_numpy(out)
        embeddings = F.normalize(out)
        dis_embedding_tot[start_idx:(start_idx + embeddings.size(0))] = embeddings
        start_idx += embeddings.size(0)
        if args.local_rank % 8 == 0 and idx % args.log_interval == 0 and idx > 0:
            speed = 1.0 * (args.dis_batch_size * args.log_interval * args.world_size) / (time.time() - start_time)
            time_left = (total_batch - idx - 1) * args.dis_batch_size *args.world_size / speed
            args.logger.info('INFO, processed [{}/{}], speed: {:.2f} img/s, left:{:.2f}s'.format(idx, total_batch, speed, time_left))
            start_time = time.time()
    try:
        check_minmax(torch.norm(dis_embedding_tot, p=2, dim=1).detach().cpu())
    except:
        return 0

    step3_start_time = time.time()
    step2_time_used = step3_start_time - step2_start_time
    args.logger.info('INFO, step2 finished, time used:{:.2f}s, start step3, calculate top1 acc'.format(step2_time_used))

    # clear npu memory
    dataloader = None
    img = None
    net = None

    dis_embedding_tot = dis_embedding_tot.permute(1, 0)
    args.logger.info('INFO, calculate top1 acc dis_embedding_tot shape:{}'.format(dis_embedding_tot.shape))

    # find best match
    assert len(args.test_img_list) % 2 == 0
    task_num = int(len(args.test_img_list) / 2)
    correct = torch.Tensor([0] * (2 * task_num))
    tot = torch.Tensor([0] * task_num)

    for i in range(int(len(args.test_img_list) / 2)):
        jk_list = args.test_img_list[2 * i]
        zj_list = args.test_img_list[2 * i + 1]
        zj2jk_pairs = sorted(generate_test_pair(jk_list, zj_list))
        sampler = DistributedSampler(zj2jk_pairs)
        args.logger.info('INFO, calculate top1 acc sampler len:{}'.format(len(sampler)))
        for idx in sampler:
            out1, out2 = cal_topk(idx, zj2jk_pairs, test_embedding_tot, dis_embedding_tot)
            correct[2 * i] += out1[0]
            correct[2 * i + 1] += out1[1]
            tot[i] += out2[0]

    args.logger.info('local_rank={},tot={},correct={}'.format(args.local_rank, tot, correct))

    step3_time_used = time.time() - step3_start_time
    args.logger.info('INFO, step3 finished, time used:{:.2f}s'.format(step3_time_used))
    args.logger.info('weight:{}'.format(args.weight))
    tot = tot.detach().cpu().numpy()
    correct = correct.detach().cpu().numpy()
    avg_accs = []
    for i in range(int(len(args.test_img_list) / 2)):
        test_set_name = 'test_dataset'
        zj2jk_acc = correct[2 * i] / tot[i]
        jk2zj_acc = correct[2 * i + 1] / tot[i]
        avg_acc = (zj2jk_acc + jk2zj_acc) / 2
        results = '[{}]: zj2jk={:.4f}, jk2zj={:.4f}, avg={:.4f}'.format(test_set_name, zj2jk_acc, jk2zj_acc, avg_acc)
        args.logger.info(results)
    args.logger.info('INFO, tot time used: {:.2f}s'.format(time.time() - all_start_time))


if __name__ == '__main__':
    args = config_inference
    args.test_img_predix = [args.test_dir,
                            args.test_dir]
    args.test_img_list = [os.path.join(args.test_dir, 'lists/jk_list.txt'),
                          os.path.join(args.test_dir, 'lists/zj_list.txt')]
    args.dis_img_predix = [args.test_dir, ]
    args.dis_img_list = [os.path.join(args.test_dir, 'lists/dis_list.txt'), ]

    log_path = os.path.join(args.ckpt_path, 'logs')
    args.logger = get_logger(log_path, args.local_rank)

    args.logger.info('Config\n\n%s\n' % pformat(args))

    main(args)