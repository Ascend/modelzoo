import tensorflow as tf
import os
import numpy as np
import sys
import inceptionv3_frn.inception_v3 as v3frn
import inceptionv3_frn.inception_utils as v3frn_utils
from preprocess  import preprocess_for_train
import tensorflow.contrib.slim as slim
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from math import *
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

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

flags.DEFINE_string(
    "output_path", None,
    "The config json file corresponding to the pre-trained RESNET model. ")


flags.DEFINE_integer(
    "image_num", 50000,
    "The config json file corresponding to the pre-trained RESNET model. ")

flags.DEFINE_integer(
    "class_num", 1000,
    "The config json file corresponding to the pre-trained RESNET model. ")

flags.DEFINE_integer(
	"batch_size", 64,
    "The config json file corresponding to the pre-trained RESNET model. ")

flags.DEFINE_integer(
    "epoch", 10,
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
    images = images/255.0
    images = preprocess_for_train(images, 299, 299, None)
    return images, label



def training_op( log,label,lrs):
    # loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=log, labels=label, name='entropy_per_example')
    loss = tf.reduce_mean(cross_entropy, name='loss')
    correct = tf.nn.in_top_k(log, label, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, "float"))
    optimizer = tf.train.GradientDescentOptimizer(lrs)
    # optimizer = tf.train.MomentumOptimizer(lrs, 0.9)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        op = optimizer.minimize(loss)
    return op,loss,accuracy

def isin(a,s):
    for i in a:
        if i in s:
            return True
    return False


def tf_data_list(tf_data_path):
    filepath = tf_data_path
    tf_data_list = []
    file_list = os.listdir(filepath)
    for i in file_list:
        tf_data_list.append(os.path.join(filepath,i))
    print("-----------------------------------------------------")
    print(tf_data_list)
    return tf_data_list

def cosine_decay(global_step):
    decay_steps = 400000
    alpha = 0.0001
    learning_rate = 0.001
    global_step = min(global_step, decay_steps)
    cosine_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    decayed_learning_rate = learning_rate * decayed
    if global_step >decay_steps :
        return  0.00001
    return decayed_learning_rate

batch_size = FLAGS.batch_size
epochs = FLAGS.epoch
image_num = FLAGS.image_num

dataset = tf.data.TFRecordDataset(tf_data_list(FLAGS.data_path))
dataset = dataset.map(_parse_read,num_parallel_calls=2)
dataset = dataset.shuffle(batch_size*10)
dataset = dataset.repeat(epochs)
dataset = dataset.batch(batch_size, drop_remainder=True)
iterator = dataset.make_one_shot_iterator()
images_batch, labels_batch = iterator.get_next()
print(images_batch, labels_batch)



inputx = tf.placeholder(tf.float32, shape=[batch_size, 299, 299, 3], name="inputx")
inputy = tf.placeholder(tf.int64, name="inputy")
lrs = tf.placeholder(tf.float32, name="lrs")
with slim.arg_scope(v3frn_arg_scope()):
    out,_ = v3frn.inception_v3(inputx,1001,False)
train_op, train_loss, train_val = training_op(out,inputy,lrs)
config = tf.ConfigProto(allow_soft_placement=True)
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True  # 在昇腾AI处理器执行训练
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 关闭remap开关
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
print("-----------------------------restore---------------------")
variables_to_resotre = [var for var in tf.trainable_variables() if not isin(["beta","gamma","tau"],var.name) ]
saver = tf.train.Saver(variables_to_resotre)
saver.restore(sess,FLAGS. model_path)
print("-----------------------------restore end---------------------")
saver_train =  tf.train.Saver(max_to_keep=50)
print ("Training start....")
try:
    global_step = 0
    for epoch in range(epochs):
        for step in range(int(image_num/batch_size)):
            x_in,y_in = sess.run([images_batch,labels_batch])
            y_in = np.squeeze(y_in,1)
            y_in = y_in + 1
            global_step = global_step + 1
            _,tra_loss, tra_acc= sess.run([train_op,train_loss,train_val],feed_dict={inputx: x_in ,inputy: y_in,lrs:cosine_decay(global_step)})
            if (step+1)%10==0:
              print('Epoch %d, step %d,lr %.5f, train loss = %.4f, train accuracy = %.2f%%' %(epoch+1,step+1,cosine_decay(global_step),tra_loss, tra_acc * 100.0))
            if (step + 1) % 10 == 0:
                checkpoint_path = os.path.join(FLAGS.output_path, "inception_model.ckpt")
                saver_train.save(sess, checkpoint_path)
except tf.errors.OutOfRangeError:
    print('epoch limit reached')
finally:
    print("Training Done")
    sess.close()