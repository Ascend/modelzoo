import tensorflow  as tf

def _parse_read(example_proto):
	#read the image and labels from tfrecord
	#return：image,label 
    features = {"image": tf.FixedLenFeature([], tf.string, default_value=""),
                "height": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "width": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "channels": tf.FixedLenFeature([1], tf.int64, default_value=[3]),
                "colorspace": tf.FixedLenFeature([], tf.string, default_value=""),
                "img_format": tf.FixedLenFeature([], tf.string, default_value=""),
                "label": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "bbox_xmin": tf.VarLenFeature(tf.float32),
                "bbox_xmax": tf.VarLenFeature(tf.float32),
                "bbox_ymin": tf.VarLenFeature(tf.float32),
                "bbox_ymax": tf.VarLenFeature(tf.float32),
                "text": tf.FixedLenFeature([], tf.string, default_value=""),
                "filename": tf.FixedLenFeature([], tf.string, default_value="")
               }
    parsed_features = tf.parse_single_example(example_proto, features)
    label = parsed_features["label"]
    images = tf.image.decode_jpeg(parsed_features["image"], channels=3)
    images = tf.cast(images, tf.float32)
    images = tf.image.resize_images(images,[299,299])
    return images, label

def train_parse_augmentation(image, label):
	#dataset aygmentation
	#return: image label
    #image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_flip_up_down(image)
    #image = tf.image.random_brightness(image,max_delta=0.5)
    #image = tf.image.random_contrast(image,1, 5)
    #imaeg = tf.image.random_saturation(image,1,5)
    image /= 127.5
    image -= 1.0
    return image, label


def test_parse_augmentation(images, labels):
    images /= 127.5
    images -= 1.0
    return images, labels



def get_train_data(tf_data_path,batch_size,epoch):
    dataset = tf.data.TFRecordDataset(tf_data_path)
    dataset = dataset.map(_parse_read, num_parallel_calls=2)
    dataset = dataset.map(train_parse_augmentation)
    dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.repeat(epoch)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch = iterator.get_next()
    return images_batch, labels_batch


def get_test_data(tf_data_path,batch_size):
    dataset = tf.data.TFRecordDataset(tf_data_path)
    dataset = dataset.map(_parse_read, num_parallel_calls=2)
    dataset = dataset.map(test_parse_augmentation)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch = iterator.get_next()
    return images_batch, labels_batch
