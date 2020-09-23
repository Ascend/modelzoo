import os
import cv2
import numpy as np

import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as c_transforms
import mindspore.dataset.transforms.py_transforms as py_transforms
from mindspore.ops import operations as P


class SegDatasetGen(object):
    def __init__(self, 
                 data_root='',
                 data_lst='',
                 batch_size=32,
                 crop_size=512,
                 image_mean=[127.0, 113.0, 104.0],
                 image_std=[58.8, 58.8, 58.8],
                 max_scale=2.0,
                 min_scale=0.5,
                 ignore_label=255,
                 num_classes=21,
                 num_parallel_calls=4,
                 shard_num=None,
                 shard_id=None,
                 rand_seed=412051):

        self.data_root = data_root
        self.data_lst = data_lst
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.num_parallel_calls = num_parallel_calls
        self.shard_num = shard_num
        self.shard_id = shard_id

        np.random.seed(rand_seed)
        assert(max_scale > min_scale)

        with open(self.data_lst) as f:
            lines = f.readlines()
        lines = [l.strip().split(' ')  for l in lines]
        self.samples = [[os.path.join(self.data_root, i[0]), 
                        os.path.join(self.data_root, i[1])]
                            for i in lines]
    
    def generator_function(self):
        np.random.shuffle(self.samples)
        for i in self.samples:
            img = cv2.imread(i[0])
            msk = cv2.imread(i[1], cv2.IMREAD_GRAYSCALE)
            yield (img, msk)

    def preprocess_(self, image, label):
        image_out = image
        label_out = label
        
        sc = np.random.uniform(self.min_scale, self.max_scale)
        new_h, new_w = int(sc * image_out.shape[0]), int(sc * image_out.shape[1])
        image_out = cv2.resize(image_out, (new_w, new_h), 
            interpolation=cv2.INTER_CUBIC)
        label_out = cv2.resize(label_out, (new_w, new_h), 
            interpolation=cv2.INTER_NEAREST)
        
        image_out = (image_out - self.image_mean) / self.image_std
        
        h_, w_ = max(new_h, self.crop_size), max(new_w, self.crop_size)
        pad_h, pad_w = h_ - new_h, w_ - new_w
        if pad_h > 0 or pad_w > 0:
            image_out = cv2.copyMakeBorder(image_out, 0, pad_h, 0, pad_w, 
                cv2.BORDER_CONSTANT, value=0)
            label_out = cv2.copyMakeBorder(label_out, 0, pad_h, 0, pad_w, 
                cv2.BORDER_CONSTANT, value=self.ignore_label)
        offset_h = np.random.randint(0, h_ - self.crop_size + 1)
        offset_w = np.random.randint(0, w_ - self.crop_size + 1)
        image_out = image_out[offset_h:offset_h+self.crop_size, 
            offset_w:offset_w+self.crop_size, :]
        label_out = label_out[offset_h:offset_h+self.crop_size, 
            offset_w:offset_w+self.crop_size]
        
        if np.random.uniform(0.0, 1.0) > 0.5:
            image_out = image_out[:,::-1,:]
            label_out = label_out[:,::-1]
        
        image_out = image_out.transpose((2,0,1))
        image_out = image_out.copy()
        label_out = label_out.copy()

        return image_out, label_out
    
    def get_dataset(self, repeat=1):
        dataset_ = de.GeneratorDataset(self.generator_function, ['data', 'label'])#,
            #num_shards=self.shard_num, shard_id=self.shard_id)
        dataset_.set_dataset_size(len(self.samples))
        dataset_ = dataset_.map(input_columns=["data", "label"], 
            output_columns=["data", "label"], operations=self.preprocess_, 
            num_parallel_workers=self.num_parallel_calls)
        dataset_ = dataset_.batch(self.batch_size, drop_remainder=True)
        dataset_ = dataset_.repeat(repeat)
        return dataset_
