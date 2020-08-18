import lmdb
import math
import numpy as np
import six
import pickle
import sys
from PIL import Image

from .util import CTCLabelConverter
from .config import Config_CNNCTC

from mindspore.communication.management import init, get_rank, get_group_size

config = Config_CNNCTC()
CHARACTER = config.CHARACTER

TRAIN_DATASET_PATH = config.TRAIN_DATASET_PATH
TRAIN_DATASET_INDEX_PATH = config.TRAIN_DATASET_INDEX_PATH
TRAIN_BATCH_SIZE = config.TRAIN_BATCH_SIZE

TEST_DATASET_PATH = config.TEST_DATASET_PATH
TEST_BATCH_SIZE = config.TEST_BATCH_SIZE

FINAL_FEATURE_WIDTH = config.FINAL_FEATURE_WIDTH
IMG_H = config.IMG_H
IMG_W = config.IMG_W


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.max_size = max_size
        self.PAD_type = PAD_type

    def __call__(self, img):
        # toTensor
        img = np.array(img, dtype=np.float32)
        img = img.transpose([2, 0, 1])
        img = img.astype(np.float)
        img = np.true_divide(img, 255)
        # normalize
        img = np.subtract(img, 0.5)
        img = np.true_divide(img, 0.5)

        c, h, w = img.shape
        Pad_img = np.zeros(shape=self.max_size, dtype=np.float32)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = np.tile(np.expand_dims(img[:, :, w - 1], 2), (1, 1, self.max_size[2] - w))

        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100):
        self.imgH = imgH
        self.imgW = imgW

    def __call__(self, images):

        resized_max_w = self.imgW
        input_channel = 3
        transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

        resized_images = []
        for image in images:
            w, h = image.size
            ratio = w / float(h)
            if math.ceil(self.imgH * ratio) > self.imgW:
                resized_w = self.imgW
            else:
                resized_w = math.ceil(self.imgH * ratio)

            resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
            resized_images.append(transform(resized_image))

        image_tensors = np.concatenate([np.expand_dims(t, 0) for t in resized_images], 0)

        return image_tensors


def get_img_from_lmdb(env, index):
    with env.begin(write=False) as txn:
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key).decode('utf-8')
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        try:
            img = Image.open(buf).convert('RGB')  # for color image

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            img = Image.new('RGB', (IMG_W, IMG_H))
            label = '[dummy_label]'

    label = label.lower()

    return img, label


def ST_MJ_Generator_batch_fixed_length():
    align_collector = AlignCollate()

    converter = CTCLabelConverter(CHARACTER)

    env = lmdb.open(TRAIN_DATASET_PATH, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    if not env:
        print('cannot create lmdb from %s' % (TRAIN_DATASET_PATH))
        sys.exit(0)

    with open(TRAIN_DATASET_INDEX_PATH, 'rb') as f:
        st_mj_filtered_index_list = pickle.load(f)

    print(f'num of samples in ST_MJ dataset: {len(st_mj_filtered_index_list)}')

    cnt = 0

    img_ret = []
    text_ret = []

    for index in st_mj_filtered_index_list:

        img, label = get_img_from_lmdb(env, index)

        img_ret.append(img)
        text_ret.append(label)

        if len(img_ret) == TRAIN_BATCH_SIZE:
            img_ret = align_collector(img_ret)
            text_ret, length = converter.encode(text_ret)

            label_indices = []
            for i in range(len(length)):
                for j in range(length[i]):
                    label_indices.append((i, j))
            label_indices = np.array(label_indices, np.int64)
            sequence_length = np.array([FINAL_FEATURE_WIDTH] * TRAIN_BATCH_SIZE, dtype=np.int32)
            text_ret = text_ret.astype(np.int32)

            yield img_ret, label_indices, text_ret, sequence_length
            
            cnt += 1
            img_ret = []
            text_ret = []


def ST_MJ_Generator_batch_fixed_length_para():
    align_collector = AlignCollate()

    converter = CTCLabelConverter(CHARACTER)

    env = lmdb.open(TRAIN_DATASET_PATH, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    if not env:
        print('cannot create lmdb from %s' % (TRAIN_DATASET_PATH))
        sys.exit(0)

    with open(TRAIN_DATASET_INDEX_PATH, 'rb') as f:
        st_mj_filtered_index_list = pickle.load(f)

    print(f'num of samples in ST_MJ dataset: {len(st_mj_filtered_index_list)}')

    cnt = 0

    index_ret = []
    img_ret = []
    text_ret = []
    rank_id = get_rank()
    rank_size = get_group_size()

    for index in st_mj_filtered_index_list:

        index_ret.append(index)

        if len(index_ret) == TRAIN_BATCH_SIZE:
            if cnt % rank_size == rank_id:

                for index_ in index_ret:
                    img, label = get_img_from_lmdb(env, index_)

                    img_ret.append(img)
                    text_ret.append(label)

                img_ret = align_collector(img_ret)
                text_ret, length = converter.encode(text_ret)

                label_indices = []
                for i in range(len(length)):
                    for j in range(length[i]):
                        label_indices.append((i, j))
                label_indices = np.array(label_indices, np.int64)
                sequence_length = np.array([FINAL_FEATURE_WIDTH] * TRAIN_BATCH_SIZE, dtype=np.int32)
                text_ret = text_ret.astype(np.int32)
                yield img_ret, label_indices, text_ret, sequence_length

            cnt += 1
            index_ret = []
            img_ret = []
            text_ret = []


def IIIT_Generator_batch():
    max_len = int((26 + 1) // 2)

    align_collector = AlignCollate()

    converter = CTCLabelConverter(CHARACTER)

    env = lmdb.open(TEST_DATASET_PATH, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    if not env:
        print('cannot create lmdb from %s' % (TEST_DATASET_PATH))
        sys.exit(0)

    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()))
        nSamples = nSamples

        # Filtering
        filtered_index_list = []
        for index in range(nSamples):
            index += 1  # lmdb starts with 1
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')

            if len(label) > max_len:
                continue

            illegal_sample = False
            for char_item in label.lower():
                if char_item not in CHARACTER:
                    illegal_sample = True
                    break
            if illegal_sample:
                continue

            filtered_index_list.append(index)

    img_ret = []
    text_ret = []

    print(f'num of samples in IIIT dataset: {len(filtered_index_list)}')

    for index in filtered_index_list:

        img, label = get_img_from_lmdb(env, index)

        img_ret.append(img)
        text_ret.append(label)

        if len(img_ret) == TEST_BATCH_SIZE:
            img_ret = align_collector(img_ret)
            text_ret, length = converter.encode(text_ret)

            label_indices = []
            for i in range(len(length)):
                for j in range(length[i]):
                    label_indices.append((i, j))
            label_indices = np.array(label_indices, np.int64)
            sequence_length = np.array([26] * TEST_BATCH_SIZE, dtype=np.int32)
            text_ret = text_ret.astype(np.int32)

            yield img_ret, label_indices, text_ret, sequence_length, length

            img_ret = []
            text_ret = []
