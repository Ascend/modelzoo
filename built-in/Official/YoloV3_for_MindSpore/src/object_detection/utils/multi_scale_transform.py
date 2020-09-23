import numpy as np
import random

from src.object_detection.yolo_v3.transforms import preprocess_fn


class MultiScaleTrans(object):
    def __init__(self, config, device_num):
        self.config = config
        self.seed = 0
        self.size_list = []
        self.resize_rate = config.resize_rate
        self.dataset_size = config.dataset_size
        self.size_dict = {}
        self.seed_num = int(1e6)
        self.seed_list = self.generate_seed_list(seed_num=self.seed_num)
        self.resize_count_num = int(np.ceil(self.dataset_size / self.resize_rate))
        self.device_num = device_num

    def generate_seed_list(self, init_seed=1234, seed_num=int(1e6), seed_range=(1, 1000)):
        seed_list = []
        random.seed(init_seed)
        for i in range(seed_num):
            seed = random.randint(seed_range[0], seed_range[1])
            seed_list.append(seed)
            # random.seed(seed)
        return seed_list

    def __call__(self, imgs, annos, batchInfo):
        epoch_num = batchInfo.get_epoch_num()
        size_idx = int(batchInfo.get_batch_num() / self.resize_rate)
        seed_key = self.seed_list[(epoch_num * self.resize_count_num + size_idx) % self.seed_num]
        ret_imgs = []
        ret_annos = []

        if self.size_dict.get(seed_key, None) is None:
            seed = random.seed(seed_key)
            new_size = random.choice(self.config.multi_scale)
            self.size_dict[seed] = new_size
        else:
            seed = seed_key

        input_size = self.size_dict[seed]
        for img, anno in zip(imgs, annos):
            img, anno = preprocess_fn(img, anno, self.config, input_size, self.device_num)
            ret_imgs.append(img.transpose(2, 0, 1).copy())
            ret_annos.append(anno)
        return np.array(ret_imgs), np.array(ret_annos)
