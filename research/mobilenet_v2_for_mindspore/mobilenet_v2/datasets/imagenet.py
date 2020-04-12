import os
import numpy as np
import math
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
from mindspore import dtype as mstype

import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.transforms.vision.c_transforms as V_C


class DistributedSampler(object):
    def __init__(self, dataset, rank, group_size, shuffle=True, seed=0):
        self.dataset = dataset
        self.rank = rank
        self.group_size = group_size
        self.dataset_length = len(self.dataset)
        self.num_samples  = int(math.ceil(self.dataset_length * 1.0 / self.group_size))
        self.total_size = self.num_samples * self.group_size
        self.shuffle = shuffle
        self.seed = seed
 
    def __iter__(self):
        if self.shuffle:
            self.seed = (self.seed + 1) & 0xffffffff
            np.random.seed(self.seed)
            indices = np.random.permutation(self.dataset_length).tolist()
        else:
            indices = list(range(len(self.dataset.classes)))
 
        indices += indices[:(self.total_size - len(indices))]
        indices = indices[self.rank::self.group_size]
        return iter(indices)
 
    def __len__(self):
        return self.num_samples


class TxtDataset(object):
    def __init__(self, root, txt_name):
        super(TxtDataset, self).__init__()
        self.imgs = []
        self.labels = []
        fin = open(txt_name, "r")
        for line in fin:
            img_name, label = line.strip().split(' ')
            self.imgs.append(os.path.join(root, img_name))
            self.labels.append(int(label))
        fin.close()
 
    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        return img, self.labels[index]
 
    def __len__(self):
        return len(self.imgs)
 
 
def imagenet_dataset(data_dir, image_size, per_batch_size, max_epoch, rank, group_size, 
                     mode='train', 
                     input_mode='folder',
                     root='',
                     num_parallel_workers=None,
                     shuffle=None,
                     sampler=None,
                     class_indexing=None,
                     drop_remainder=True,
                     transform=None,
                     target_transform=None):
    """
    A function that returns a dataset for imagenet. The mode of input dataset could be "folder" or "txt". 
    If it is "folder", all images within one folder have the same label. If it is "txt", all paths of images
    are written into a textfile.
 
    Args:
        data_dir (str): Path to the root directory that contains the dataset for "input_mode="folder"". 
            Or path of the textfile that contains every image's path of the dataset.
        image_size (str): Size of the input images.
        per_batch_size (int): the batch size of evey step during training.
        max_epoch (int): the number of epochs.
        rank (int): The shard ID within num_shards (default=None).
        group_size (int): Number of shards that the dataset should be divided
            into (default=None).
        mode (str): "train" or others. Default: " train".
        input_mode (str): The form of the input dataset. "folder" or "txt". Default: "folder".
        root (str): the images path for "input_mode="txt"". Default: " ".
        num_parallel_workers (int): Number of workers to read the data. Default: None.
        shuffle (bool): Whether or not to perform shuffle on the dataset
            (default=None, performs shuffle).
        sampler (Sampler): Object used to choose samples from the dataset. Default: None.
        class_indexing (dict): A str-to-int mapping from folder name to index
            (default=None, the folder names will be sorted
            alphabetically and each class will be given a
            unique index starting from 0).
 
    Examples:
        >>> from mindvision.common.datasets.imagenet import imagenet_dataset
        >>> # path to imagefolder directory. This directory needs to contain sub-directories which contain the images
        >>> dataset_dir = "/path/to/imagefolder_directory"
        >>> de_dataset = imagenet_dataset(train_data_dir, image_size=[224, 244],
        >>>                               per_batch_size=64, max_epoch=100,
        >>>                               rank=0, group_size=4)
        >>> # Path of the textfile that contains every image's path of the dataset.
        >>> dataset_dir = "/path/to/dataset/images/train.txt"
        >>> images_dir = "/path/to/dataset/images"
        >>> de_dataset = imagenet_dataset(train_data_dir, image_size=[224, 244],
        >>>                               per_batch_size=64, max_epoch=100,
        >>>                               rank=0, group_size=4,
        >>>                               input_mode="txt", root=images_dir)
    """
 
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
 
    if transform is None:
        if mode == 'train':
            transform_img = [
                V_C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0)),
                V_C.RandomHorizontalFlip(prob=0.5),
                V_C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4),
                V_C.Normalize(mean=mean, std=std),
                V_C.HWC2CHW()
            ]
        else:
            transform_img = [
                V_C.Decode(),
                V_C.Resize((256, 256)),
                V_C.CenterCrop(image_size),
                V_C.Normalize(mean=mean, std=std),
                V_C.HWC2CHW()
            ]
    else:
        transform_img = transform
 
    if target_transform is None:
        transform_label = [C.TypeCast(mstype.int32)]
    else:
        transform_label = target_transform
 
    if input_mode == 'folder':
        de_dataset = de.ImageFolderDatasetV2(data_dir, num_parallel_workers=num_parallel_workers,
                                             shuffle=shuffle, sampler=sampler, class_indexing=class_indexing,
                                             num_shards=group_size, shard_id=rank)
    else:
        dataset = TxtDataset(root, data_dir)
        sampler = DistributedSampler(dataset, rank, group_size, shuffle=shuffle)
        de_dataset = de.GeneratorDataset(dataset, ["image", "label"], sampler=sampler)
        de_dataset.set_dataset_size(len(sampler))
    
    de_dataset = de_dataset.map(input_columns="image", operations=transform_img)
    de_dataset = de_dataset.map(input_columns="label", operations=transform_label)
 
    columns_to_project = ["image", "label"]
    de_dataset = de_dataset.project(columns=columns_to_project)
 
    de_dataset = de_dataset.batch(per_batch_size, drop_remainder=drop_remainder)
    de_dataset = de_dataset.repeat(max_epoch)   
    
    return de_dataset