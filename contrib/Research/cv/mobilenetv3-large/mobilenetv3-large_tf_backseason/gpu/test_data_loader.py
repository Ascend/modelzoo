import tensorflow as tf
from datasets import dataset_factory
from dataloader import data_provider

preprocessing_name = 'inception_v2'
batch_size = 256
dataset_name = 'imagenet'
dataset_dir = '/data/Datasets/imagenet_TF'
labels_offset = 0
use_grayscale = False
enable_hvd = False
data_loader_mode = 'splited'

dataset = dataset_factory.get_dataset(dataset_name, 'train', dataset_dir)
iterator, ds = data_provider.get_data(dataset, batch_size,
															 dataset.num_classes, labels_offset, True,
															 preprocessing_name, use_grayscale,
															 None, enable_hvd,
															 data_loader_mode=data_loader_mode)
images, labels = iterator.get_next()

sess = tf.compat.v1.Session()
sess.run(iterator.initializer)
sess.run(tf.compat.v1.global_variables_initializer())
sess.run(tf.compat.v1.local_variables_initializer())

for i in range(100000):
	print(f'i: {i}')
	sess.run([images])

