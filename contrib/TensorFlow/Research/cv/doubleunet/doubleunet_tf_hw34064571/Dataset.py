import time
import numpy as np
# import matplotlib
# matplotlib.use('Tkagg')
# import matplotlib.pyplot as plt
import h5py
import os
import threading
import queue
import numpy as np
import random
import cv2 as cv
from math import ceil


# from libtiff import TIFF


def save_h5(path, images, labels):
    print('saving', path)
    file = h5py.File(name=path, mode='w')
    file['images'] = images
    file['labels'] = labels


def load_h5(path):
    print('loading', path)
    file = h5py.File(name=path, mode='r')
    return file['images'], file['labels']


def adjust_image(image):
    def _clip(image):
        return np.clip(image, 0, 255).astype(np.uint8)

    def RGB2HSV(image):
        return cv.cvtColor(image, cv.COLOR_RGB2HSV)

    def HSV2RGB(image):
        return cv.cvtColor(image, cv.COLOR_HSV2RGB)

    def adjust_contrast(image, factor=2 * (random.random() - 0.5)):
        mean = image.mean(axis=0).mean(axis=0)
        return _clip((image - mean) * factor + mean)

    def adjust_hue(image, delta=2 * (random.random() - 0.5)):
        image[0] = np.mod(image[0] + delta * 180, 180)
        return image

    def adjust_saturation(image, factor=2 * (random.random() - 0.5)):
        image[1] = np.clip(image[1] * factor, 0, 255)
        return image

    def adjust_brightness(image, delta=2 * (random.random() - 0.5)):
        return _clip(image + delta * 255)

    image1 = adjust_contrast(image)
    image2 = adjust_brightness(image)
    image = RGB2HSV(image)
    image3 = HSV2RGB(adjust_saturation(image))
    image4 = HSV2RGB(adjust_hue(image))
    return image1, image2, image3, image4


class VOC2012:
    def __init__(self, root_path='/sda3/dongxu/LAB/CVC-ClinicDB',
                 aug_path='/sda3/dongxu/LAB/CVC-ClinicDB',
                 image_size=(256, 320),
                 resize_method='resize', checkpaths=False):
        '''
        Create a VOC2012 object
        This function will set all paths needed, do not set them mannully expect you have
        changed the dictionary structure
        Args:
            root_path:the Pascal VOC 2012 folder path
            aug_path:The augmentation dataset path. If you don't want to use it, ignore
            image_size:resize images and labels into this size
            resize_method:'resize' or 'pad', if pad, images and labels will be paded into 500x500
                        and the parameter image_size will not be used
        '''
        self.root_path = root_path
        self.resize_method = resize_method
        if resize_method != 'resize' and resize_method != 'pad':
            print('Unknown resize method:', resize_method)
            exit()
        if root_path[len(root_path) - 1] != '/' and root_path[len(root_path) - 1] != '\\':
            self.root_path += '/'
        self.train_list_path = self.root_path + 'train.txt'
        self.val_list_path = self.root_path + 'test.txt'
        self.image_path = self.root_path + 'Original/'
        self.label_path = self.root_path + 'Ground_Truth/'
        self.aug_path = aug_path
        if aug_path[len(aug_path) - 1] != '/' and aug_path[len(aug_path) - 1] != '\\':
            self.aug_path += '/'
        self.image_size = image_size
        if checkpaths:
            self.check_paths()

    def check_paths(self):
        '''
        check all paths and display the status of paths
        '''
        if not (os.path.exists(self.root_path) and os.path.isdir(self.root_path)):
            print('Warning: Dictionary', self.root_path, ' does not exist')
        if not (os.path.exists(self.train_list_path) and os.path.isfile(self.train_list_path)):
            print('Warning: Training list file', self.train_list_path, 'does not exist')
        if not (os.path.exists(self.val_list_path) and os.path.isfile(self.val_list_path)):
            print('Warning: Validation list file', self.val_list_path, 'does not exist')
        if not (os.path.exists(self.image_path) and os.path.isdir(self.image_path)):
            print('Warning: Dictionary', self.image_path, 'does not exist')
        if not (os.path.exists(self.label_path) and os.path.isdir(self.label_path)):
            print('Warning: Dictionary', self.label_path, 'does not exist')
        if not (os.path.exists(self.aug_path) and os.path.isdir(self.aug_path)):
            print('Warning: Dictionary', self.aug_path, 'does not exist')

    def read_train_list(self):
        '''
        Read the filenames of training images and labels into self.train_list
        '''
        self.train_list = []
        f = open(self.train_list_path, 'r')
        line = None
        while 1:
            line = f.readline().replace('\n', '')
            if line is None or len(line) == 0:
                break
            self.train_list.append(line)
        f.close()

    def read_val_list(self):
        '''
        Read the filenames of validation images and labels into self.val_list
        '''
        self.val_list = []
        f = open(self.val_list_path, 'r')
        line = None
        while 1:
            line = f.readline().replace('\n', '')
            if line is None or len(line) == 0:
                break
            self.val_list.append(line)
        f.close()

    def read_train_images(self):
        '''
        Read training images into self.train_images
        If you haven't called self.read_train_list(), it will call first
        After reading images, it will resize them
        '''
        self.train_images = []
        if hasattr(self, 'train_list') == False:
            self.read_train_list()
        for filename in self.train_list:
            # image = cv.imread(self.image_path + filename + '.jpg')
            image = TIFF.open(self.image_path + filename, mode='r').read_image()
            image = image[14:280, 43:355]
            if self.resize_method == 'resize':
                image = cv.resize(image, self.image_size)
            elif self.resize_method == 'pad':
                height = np.shape(image)[0]
                width = np.shape(image)[1]
                image = cv.copyMakeBorder(image, 0, 500 - height, 0, 500 - width, cv.BORDER_CONSTANT, value=0)
            image1, image2, image3, image4 = adjust_image(image)
            self.train_images.append(image)
            self.train_images.append(image1)
            self.train_images.append(image2)
            self.train_images.append(image3)
            self.train_images.append(image4)
            if len(self.train_images) % 100 == 0:
                print('Reading train images', len(self.train_images), '/', len(self.train_list) * 5)

    def read_train_labels(self):
        '''
        Read training labels into self.train_labels
        If you haven't called self.read_train_list(), it will call first
        After reading labels, it will resize them
        Note:image[image > 20] = 0 will remove all white borders in original labels
        '''
        self.train_labels = []
        if hasattr(self, 'train_list') == False:
            self.read_train_list()
        for filename in self.train_list:
            # image = cv.imread(self.label_path + filename + '.png')
            # image = np.array(Image.open(self.label_path + filename + '.png'))
            image = TIFF.open(self.label_path + filename, mode='r').read_image()
            image = image[14:280, 43:355]
            if self.resize_method == 'resize':
                image = cv.resize(image, self.image_size, interpolation=cv.INTER_NEAREST)
            elif self.resize_method == 'pad':
                height = np.shape(image)[0]
                width = np.shape(image)[1]
                image = cv.copyMakeBorder(image, 0, 500 - height, 0, 500 - width, cv.BORDER_CONSTANT, value=0)
            for i in range(5):
                self.train_labels.append(image)
            if len(self.train_labels) % 100 == 0:
                print('Reading train labels', len(self.train_labels), '/', len(self.train_list) * 5)

    def read_val_images(self):
        '''
           Read validation images into self.val_images
           If you haven't called self.read_val_list(), it will call first
           After reading images, it will resize them
        '''
        self.val_images = []
        if hasattr(self, 'val_list') == False:
            self.read_val_list()
        for filename in self.val_list:
            # image = cv.imread(self.image_path + filename + '.jpg')
            image = TIFF.open(self.image_path + filename, mode='r').read_image()
            image = image[14:280, 43:355]
            if self.resize_method == 'resize':
                image = cv.resize(image, self.image_size)
            elif self.resize_method == 'pad':
                height = np.shape(image)[0]
                width = np.shape(image)[1]
                image = cv.copyMakeBorder(image, 0, 500 - height, 0, 500 - width, cv.BORDER_CONSTANT, value=0)
            self.val_images.append(image)
            if len(self.val_images) % 100 == 0:
                print('Reading val images', len(self.val_images), '/', len(self.val_list) * 5)

    def read_val_labels(self):
        '''
           Read validation labels into self.val_labels
           If you haven't called self.read_val_list(), it will call first
           After reading labels, it will resize them
           Note:image[image > 100] = 0 will remove all white borders in original labels
        '''
        self.val_labels = []
        if hasattr(self, 'val_list') == False:
            self.read_val_list()
        for filename in self.val_list:
            image = TIFF.open(self.label_path + filename, mode='r').read_image()
            image = image[14:280, 43:355]
            if self.resize_method == 'resize':
                image = cv.resize(image, self.image_size, interpolation=cv.INTER_NEAREST)
            elif self.resize_method == 'pad':
                height = np.shape(image)[0]
                width = np.shape(image)[1]
                image = cv.copyMakeBorder(image, 0, 500 - height, 0, 500 - width, cv.BORDER_CONSTANT, value=0)
            self.val_labels.append(image)
            if len(self.val_labels) % 100 == 0:
                print('Reading val labels', len(self.val_labels), '/', len(self.val_list) * 5)

    def save_train_data(self, path='./voc2012_train.h5'):
        '''
        save training images and labels into path in the form of .h5
        Args:
            path:The path you want to save train data into.It must be xxx.h5
        '''
        save_h5(path, self.train_images, self.train_labels)

    def save_val_data(self, path='./voc2012_val.h5'):
        '''
        save validation images and labels into path in the form of .h5
        Args:
            path:The path you want to save train data into.It must be xxx.h5
        '''
        save_h5(path, self.val_images, self.val_labels)

    def read_all_data_and_save(self, train_data_save_path='./process_data/voc2011_train0.8.h5',
                               val_data_save_path='./process_data/voc2011_val0.8.h5'):
        '''
        Read training and validation data and save them into two .h5 files.
        Args:
            train_data_save_path:The path you want to save training data into.
            val_data_save_path:The path you want to save validation data into.
        '''
        self.read_train_images()
        self.read_train_labels()
        self.read_val_images()
        self.read_val_labels()
        np.random.seed(1049)
        np.random.shuffle(self.train_images)
        np.random.seed(1049)
        np.random.shuffle(self.train_labels)

        np.random.seed(646)
        np.random.shuffle(self.val_images)
        np.random.seed(646)
        np.random.shuffle(self.val_labels)
        self.save_train_data(train_data_save_path)
        self.save_val_data(val_data_save_path)

    def load_all_data(self, train_data_load_path='./voc2012_train.h5', val_data_load_path='./voc2012_val.h5'):
        '''
        Load training and validation data from .h5 files
        Args:
            train_data_load_path:The training data .h5 file path.
            val_data_load_path:The validation data .h5 file path.
        '''
        self.load_train_data(train_data_load_path)
        self.load_val_data(val_data_load_path)
        self.train_labels = np.array(self.train_labels)
        print('convert to numpy complete')
        self.train_images = np.array(self.train_images)
        print('convert to numpy complete')

    def data_shuffle(self):
        seed = np.int(time.time())
        np.random.seed(seed)
        np.random.shuffle(self.train_images)
        np.random.seed(seed)
        np.random.shuffle(self.train_labels)

    def load_train_data(self, path='./voc2012_train.h5'):
        '''
        Load training data from .h5 files
        Args:
            train_data_load_path:The training data .h5 file path.
        '''
        self.train_images, self.train_labels = load_h5(path)

    def load_val_data(self, path='./voc2012_val.h5'):
        '''
        Load validation data from .h5 files
        Args:
            val_data_load_path:The validation data .h5 file path.
        '''
        self.val_images, self.val_labels = load_h5(path)

    def get_batch_train(self, batch_size):
        '''
        Get a batch data from training data.
        It maintains an internal location variable and get from start to end gradually.
        When it comes into the end, it returns to the start.
        Args:
            batch_size:The number of images or labels returns at a time.
        Return:
            batch_images:A batch of images with shape:[batch_size, image_size, image_size, 3]
            batch_labels:A batch of labels with shape:[batch_size, image_size, image_size]
        '''
        if hasattr(self, 'train_location') == False:
            self.train_location = 0
        end = min(self.train_location + batch_size, len(self.train_images))
        start = self.train_location
        batch_images = self.train_images[start:end]
        batch_labels = self.train_labels[start:end]
        self.train_location = (self.train_location + batch_size) % len(self.train_images)
        if end - start != batch_size:
            batch_images = np.concatenate([batch_images, self.train_images[0:self.train_location]], axis=0)
            batch_labels = np.concatenate([batch_labels, self.train_labels[0:self.train_location]], axis=0)

        return batch_images, batch_labels

    def get_batch_val(self, batch_size):
        '''
        Get a batch data from validation data.
        It maintains an internal location variable and get from start to end gradually.
        When it comes into the end, it returns to the start.
        Args:
            batch_size:The number of images or labels returns at a time.
        Return:
            batch_images:A batch of images with shape:[batch_size, image_size, image_size, 3]
            batch_labels:A batch of labels with shape:[batch_size, image_size, image_size]
        '''
        if hasattr(self, 'val_location') == False:
            self.val_location = 0
        end = min(self.val_location + batch_size, len(self.val_images))
        start = self.val_location
        batch_images = self.val_images[start:end]
        batch_labels = self.val_labels[start:end]
        self.val_location = (self.val_location + batch_size) % len(self.val_images)
        if end - start != batch_size:
            batch_images = np.concatenate([batch_images, self.val_images[0:self.val_location]], axis=0)
            batch_labels = np.concatenate([batch_labels, self.val_labels[0:self.val_location]], axis=0)
        return batch_images, batch_labels

    def get_batch_aug(self, batch_size):
        '''
        Get a batch data from augmentation data.
        It maintains an internal location variable and get from start to end gradually.
        When it comes into the end, it returns to the start.
        Args:
           batch_size:The number of images or labels returns at a time.
        Return:
           batch_images:A batch of images with shape:[batch_size, image_size, image_size, 3]
           batch_labels:A batch of labels with shape:[batch_size, image_size, image_size]
        '''
        if hasattr(self, 'aug_location') == False:
            self.aug_location = 0
        end = min(self.aug_location + batch_size, len(self.aug_images))
        start = self.aug_location
        batch_images = self.aug_images[start:end]
        batch_labels = self.aug_labels[start:end]
        self.aug_location = (self.aug_location + batch_size) % len(self.aug_images)
        if end - start != batch_size:
            batch_images = np.concatenate([batch_images, self.aug_images[0:self.aug_location]], axis=0)
            batch_labels = np.concatenate([batch_labels, self.aug_labels[0:self.aug_location]], axis=0)

        return batch_images, batch_labels

    def add_batch_aug_queue(self, batch_size, max_queue_size):
        if hasattr(self, 'aug_queue') == False:
            self.aug_queue = queue.Queue(maxsize=max_queue_size)
        while 1:
            image_batch, label_batch = self.get_batch_aug(batch_size)
            image_batch, label_batch = self.random_resize(image_batch, label_batch)
            self.aug_queue.put([image_batch, label_batch])

    def start_batch_aug_queue(self, batch_size, max_queue_size=30):
        if hasattr(self, 'aug_queue') == False:
            queue_thread = threading.Thread(target=self.add_batch_aug_queue, args=(batch_size, max_queue_size))
            queue_thread.start()

    def get_batch_aug_fast(self, batch_size, max_queue_size=30):
        '''
        A fast function for get augmentation batch.Use another thread to get batch and put into a queue.
        :param batch_size: batch size
        :param max_queue_size: the max capacity of the queue
        :return: An image batch with shape [batch_size, height, width, 3]
                and a label batch with shape [batch_size, height, width, 1]
        '''
        # create queue thread
        if hasattr(self, 'aug_queue') == False:
            queue_thread = threading.Thread(target=self.add_batch_aug_queue, args=(batch_size, max_queue_size))
            queue_thread.start()
        while hasattr(self, 'aug_queue') == False:
            time.sleep(0.1)
        image_batch, label_batch = self.aug_queue.get()
        return image_batch, label_batch

    def random_resize(self, image_batch, label_batch, random_blur=True):
        '''
        resize the batch data randomly
        :param image_batch: shape [batch_size, height, width, 3]
        :param label_batch: shape [batch_size, height, width, 1]
        :param random_blur:If true, blur the image randomly with Gaussian Blur method
        :return:
        '''
        new_image_batch = []
        new_label_batch = []
        batch_shape = np.shape(image_batch)
        a = random.random() / 2 + 0.5  # (0,1) -> (0, 1.5)->(0.5, 2)
        b = random.random() / 2 + 0.5  # (0,1) -> (0, 1.5)->(0.5, 2)
        batch_size = batch_shape[0]
        new_height = int(a * batch_shape[1])
        new_width = int(b * batch_shape[2])
        for i in range(batch_size):
            image = image_batch[i]
            if random_blur:
                radius = int(random.randrange(0, 3)) * 2 + 1
                image = cv.GaussianBlur(image, (radius, radius), random.randrange(0, 3))
            new_image_batch.append(cv.resize(image, (new_height, new_width)))
            new_label_batch.append(cv.resize(label_batch[i], (new_height, new_width), interpolation=cv.INTER_NEAREST))
        return new_image_batch, new_label_batch

    def index_to_rgb(self, index):
        '''
        Find the rgb color with the class index
        :param index:
        :return: A list like [1, 2, 3]
        '''
        color_dict = {0: [0, 0, 0], 1: [128, 0, 0], 2: [0, 128, 0], 3: [128, 128, 0], 4: [0, 0, 128], 5: [128, 0, 128],
                      6: [0, 128, 128], 7: [128, 128, 128], 8: [64, 0, 0], 9: [192, 0, 0], 10: [64, 128, 0],
                      11: [192, 128, 0], 12: [64, 0, 128], 13: [192, 0, 128], 14: [64, 128, 128], 15: [192, 128, 128],
                      16: [0, 64, 0], 17: [128, 64, 0], 18: [0, 192, 0], 19: [128, 192, 0], 20: [0, 64, 128]}
        return color_dict[index]

    def gray_to_rgb(self, image):
        '''
        Convert the gray image(mask image) to a rgb image
        :param image: gray image, with shape [height, width]
        :return: rgb image, with shape [height, width, 3]
        '''
        height = np.shape(image)[0]
        width = np.shape(image)[1]
        result = np.zeros([height, width, 3], dtype='uint8')
        for h in range(height):
            for w in range(width):
                result[h][w] = self.index_to_rgb(image[h][w])
        return result

    def get_one_class_label(self, label, class_id):
        new_label = label
        new_label[new_label != class_id] = 0
        return new_label


if __name__ == '__main__':
    voc2012 = VOC2012(root_path='/sda3/dongxu/LAB/CVC-ClinicDB', image_size=(320, 256))
    voc2012.read_all_data_and_save(train_data_save_path='/sda3/dongxu/LAB/double_unet/dataset/cvcdb_train.h5',
                                   val_data_save_path='/sda3/dongxu/LAB/double_unet/dataset/cvcdb_test.h5')
