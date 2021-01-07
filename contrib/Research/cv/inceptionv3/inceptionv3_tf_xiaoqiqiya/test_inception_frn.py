# -*- coding: UTF-8 -*-
import tensorflow as tf
import os
import numpy as np
import sys
import inceptionv3_frn.inception_v3 as v3frn
import inceptionv3_frn.inception_utils as v3frn_utils
import tensorflow.contrib.slim as slim
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from preprocess import preprocess_for_eval
v3frn_arg_scope = v3frn_utils.inception_arg_scope




flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_path", None,
    "The config json file corresponding to the pre-trained RESNET model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "data_path", None,
    "The config json file corresponding to the pre-trained RESNET model. ")


def _parse_read(example_proto):
    features = {"image": tf.FixedLenFeature([], tf.string, default_value=""),
                "height": tf.FixedLenFeature([], tf.int64, default_value=[0]),
                "width": tf.FixedLenFeature([], tf.int64, default_value=[0]),
                "channels": tf.FixedLenFeature([], tf.int64, default_value=[3]),
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
    images = tf.image.decode_jpeg(parsed_features["image"])
    h = tf.cast(parsed_features['height'], tf.int64)
    w = tf.cast(parsed_features['width'], tf.int64)
    c = tf.cast(parsed_features['channels'], tf.int64)
    images = tf.reshape(images, [h, w, 3])
    images = tf.cast(images, tf.float32)
    images = images/255.
    images1 = preprocess_for_eval(images,299,299,0.80)
    images2 = preprocess_for_eval(images,299,299,0.85)
    images3 = preprocess_for_eval(images,299,299, 0.9)
    images4 = preprocess_for_eval(images,299,299,0.95)
    images5 = preprocess_for_eval(images,299,299,0.925)
    return images1,images2,images3,images4,images5,label


def evaluation(logits, labels):
    print("---------------------------------------------------",logits)
    logits = logits[0:batch_size,:]+logits[batch_size:2*batch_size,:]+logits[2*batch_size:3*batch_size,:]+logits[3*batch_size:4*batch_size,:]+logits[4*batch_size:5*batch_size,:]
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels,1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy',accuracy)
        correct5 = tf.nn.in_top_k(logits, labels,5)
        correct5 = tf.cast(correct5, tf.float16)
        accuracy5 = tf.reduce_mean(correct5)
    return  [accuracy,accuracy5]



def tf_data_list(tf_data_path):
    filepath = tf_data_path
    tf_data_list = []
    file_list = os.listdir(filepath)
    for i in file_list:
        tf_data_list.append(os.path.join(filepath,i))
    print("-----------------------------------------------------")
    print(tf_data_list)
    return tf_data_list


batch_size = 20
epochs = 1
image_num = 50000




dataset = tf.data.TFRecordDataset(tf_data_list(FLAGS.data_path))
dataset = dataset.map(_parse_read,num_parallel_calls=2)
dataset = dataset.repeat(1)
dataset = dataset.batch(batch_size, drop_remainder=True)
iterator = dataset.make_one_shot_iterator()
images_batch, images_batch1,images_batch2,images_batch3,images_batch4,labels_batch = iterator.get_next()
images_batchs = tf.concat([images_batch, images_batch1, images_batch2,images_batch3,images_batch4], axis=0)
print(images_batch, labels_batch)



os.environ["CUDA_VISIBLE_DEVICES"] = "1"
with tf.device("/gpu:1"):
    inputx = tf.placeholder(tf.float32, shape=[batch_size*5, 299, 299, 3], name="inputx")
    inputy = tf.placeholder(tf.int64, name="inputy")
    with slim.arg_scope(v3frn_arg_scope()):
        out,_ = v3frn.inception_v3(inputx,1001,False,create_aux_logits=False)
    test_acc = evaluation(out ,inputy)

    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True  # 在昇腾AI处理器执行训练
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 关闭remap开关
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    print("-----------------------------restore---------------------")
    saver.restore(sess, FLAGS.model_path)
    print("-----------------------------restore end---------------------")
    print("test start....")
    try:
        total = []
        total5 = []
        print("test start....")
        for step in range(int(image_num / batch_size)):
            x_in, y_in = sess.run([images_batchs, labels_batch])
            y_in = np.squeeze(y_in, 1)+1
            tra_acc= sess.run([test_acc], feed_dict={inputx: x_in, inputy: y_in})[0]
            total.append(tra_acc[0])
            total5.append(tra_acc[1])
            print(step)
        print("inception with frn  eval   Top-1 acc=%.4f    Top-5 acc=%.4f"%(np.mean(total),np.mean(total5)))    
    except tf.errors.OutOfRangeError:
        print('epoch limit reached')
    finally:
        print("test Done")
        sess.close()