from __future__ import print_function
######Modify for NPU begin#######
from npu_bridge.npu_init import *
######Modify for NPU end#######
import os, time, cv2, sys, math

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os, sys
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.client import device_lib
import pywt
import helpers
import utils_ as utils
from utils_ import get_model
from pywt import wavedec2
import matplotlib.pyplot as plt
config = tf.ConfigProto(log_device_placement=False)

def str2bool(v):
    if (v.lower() in ('yes', 'true', 't', 'y', '1')):
        return True
    elif (v.lower() in ('no', 'false', 'f', 'n', '0')):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
config.gpu_options.allow_growth = True

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
parser.add_argument('--save', type=int, default=4, help='Interval for saving weights')
parser.add_argument('--gpu', type=str, default='0', help='Choose GPU device to be used')
parser.add_argument('--mode', type=str, default='train', help='Select "train", "test", or "predict" mode. \
    Note that for prediction mode you have to specify an image to run the model on.')
parser.add_argument('--checkpoint', type=str, default='checkpoint', help='Checkpoint folder.')
parser.add_argument('--class_balancing', type=str2bool, default=False, help='Whether to use median frequency class weights to balance the classes in the loss')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default='lashan', help='Dataset you are using.')
parser.add_argument('--load_data', type=str2bool, default=True, help='Dataset loading type.')
parser.add_argument('--act', type=str2bool, default=True, help='True if sigmoid or false for softmax')
parser.add_argument('--crop_height', type=int, default=224, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=224, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=8, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=200, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=True, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=True, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change.')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle.')
parser.add_argument('--model', type=str, default='dunet', help='The model you are using. Currently supports:\
    encoder-decoder, deepUNet,attentionNet, deep, UNet')
######Modify for NPU begin#######
parser.add_argument('--loss_scale', type=int, default=0, help='The loss scale you are usring. 0: dynamic loss scale; >= 1: fix loss scale.')
######Modify for NPU end#######

args = parser.parse_args()

gpu = str(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

def load_divided(dataset_dir):
    train_input_names = []
    train_output_names = []
    val_input_names = []
    val_output_names = []
    test_input_names = []
    test_output_names = []
    cwd = os.getcwd()
    for file in os.listdir((dataset_dir + '/train')):
        train_input_names.append(((((cwd + '/') + dataset_dir) + '/train/') + file))
    for file in os.listdir((dataset_dir + '/train_labels')):
        train_output_names.append(((((cwd + '/') + dataset_dir) + '/train_labels/') + file))
    for file in os.listdir((dataset_dir + '/val')):
        val_input_names.append(((((cwd + '/') + dataset_dir) + '/val/') + file))
    for file in os.listdir((dataset_dir + '/val_labels')):
        val_output_names.append(((((cwd + '/') + dataset_dir) + '/val_labels/') + file))
    if (args.mode == 'test' or 'predict'):
        for file in os.listdir((dataset_dir + '/test')):
            test_input_names.append(((((cwd + '/') + dataset_dir) + '/test/') + file))
        for file in os.listdir((dataset_dir + '/test_labels')):
            test_output_names.append(((((cwd + '/') + dataset_dir) + '/test_labels/') + file))
    return (train_input_names, train_output_names, val_input_names, val_output_names, test_input_names, test_output_names)

def load_nondivided(dataset_dir):
    train_input_names = []
    train_output_names = []
    val_input_names = []
    val_output_names = []
    test_input_names = []
    test_output_names = []
    all_files = []
    cwd = os.getcwd()
    for file in os.listdir((dataset_dir + '/sat')):
        all_files.append(file)
    selected_val = random.sample(all_files, (len(all_files) // 18))
    train_input_names = [((((cwd + '/') + dataset_dir) + '/sat/') + name) for name in all_files if (name not in selected_val)]
    train_output_names = [((((cwd + '/') + dataset_dir) + '/gt/') + name) for name in all_files if (name not in selected_val)]
    val_input_names = [((((cwd + '/') + dataset_dir) + '/sat/') + name) for name in selected_val]
    val_output_names = [((((cwd + '/') + dataset_dir) + '/gt/') + name) for name in selected_val]
    print('Training data length : {} and validation data length: {}'.format(len(train_input_names), len(val_input_names)))
    return (train_input_names, train_output_names, val_input_names, val_output_names, test_input_names, test_output_names)

def prepare_data(dataset_dir=args.dataset, type=args.load_data):
    if type:
        (train_input_names, train_output_names, val_input_names, val_output_names, test_input_names, test_output_names) = load_divided(dataset_dir)
    else:
        (train_input_names, train_output_names, val_input_names, val_output_names, test_input_names, test_output_names) = load_nondivided(dataset_dir)
    print('Training data length : {} and validation data length: {}'.format(len(train_input_names), len(val_input_names)))
    (train_input_names.sort(), train_output_names.sort(), val_input_names.sort(), val_output_names.sort(), test_input_names.sort(), test_output_names.sort())
    return (train_input_names, train_output_names, val_input_names, val_output_names, test_input_names, test_output_names)

def check_available_gpus():
    local_devices = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_devices if (x.device_type == 'GPU')]
    gpu_num = len(gpu_names)
    print('{0} GPUs are detected : {1}'.format(gpu_num, gpu_names))
    ######Modify for NPU begin#######
    # return gpu_names
    return '1'
    ######Modify for NPU end#######

def get_names(gpus):
    n_gpus = len(gpus)
    if (n_gpus == 1):
        return (gpus[0], gpus[0], gpus[0], gpus[0])
    if (n_gpus == 2):
        return (gpus[0], gpus[0], gpus[1], gpus[1])
    if (n_gpus == 3):
        return (gpus[0], gpus[1], gpus[2], gpus[0])
    if (n_gpus == 4):
        return (gpus[0], gpus[1], gpus[2], gpus[3])

def load_image(path):
    # print("path==================>", path)
    image = cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)
    (h, w) = (args.crop_height, args.crop_width)
    image = cv2.resize(image, (h, w))
    return image

def load_image_gray(path):
    img = cv2.imread(path, -1)

    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (256, 256))
    return image

def data_augmentation(input_image, output_image):
    dice = random.random()
    (crop_width, crop_height) = (args.crop_width, args.crop_height)
    if (dice >= 0.15):
        (crop_height, crop_width) = (input_image.shape[1], input_image.shape[0])
    (input_image, output_image) = utils.random_crop(input_image, output_image, args.crop_height, args.crop_width)
    if (args.h_flip and random.randint(0, 1)):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if (args.v_flip and random.randint(0, 1)):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if args.brightness:
        factor = random.uniform(((- 1) * args.brightness), args.brightness)
        table = np.array([(((i / 255.0) ** factor) * 255) for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if args.rotation:
        angle = random.uniform(((- 1) * args.rotation), args.rotation)
    if args.rotation:
        M = cv2.getRotationMatrix2D(((input_image.shape[1] // 2), (input_image.shape[0] // 2)), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)
    return (input_image, output_image)

def download_checkpoints(model_name):
    subprocess.check_output(['python', 'get_pretrained_checkpoints.py', ('--model=' + model_name)])

(class_names_list, label_values) = helpers.get_label_info(os.path.join(args.dataset, 'class_dict.csv'))
class_names_string = ''
for class_name in class_names_list:
    if (not (class_name == class_names_list[(- 1)])):
        class_names_string = ((class_names_string + class_name) + ', ')
    else:
        class_names_string = (class_names_string + class_name)

num_classes = len(label_values)
gpus = check_available_gpus()

######Modify for NPU begin#######
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_fp32_to_fp16")

#算子溢出检测
# custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("./")
# custom_op.parameter_map["enable_dump_debug"].b = True
# custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
######Modify for NPU end#######

sess = tf.Session(config=config)
model = args.model
print('Preparing the model ...', model)

input = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='inputs')
output = tf.placeholder(tf.float32, shape=[None, None, None, num_classes], name='output')

keep_prob = tf.placeholder(tf.float32)

######Modify for NPU begin#######
input_A = tf.split(input, int(len(gpus)))
output_A = tf.split(output, int(len(gpus)))
######Modify for NPU end#######

print('Loading the data ...')
(train_input_names, train_output_names, val_input_names, val_output_names, test_input_names, test_output_names) = prepare_data()

if (model == 'sunet'):
    aux_output = tf.placeholder(tf.float32, shape=[None, None, None, num_classes], name='aux')
network = None
init_fn = None

if (model != 'ssunet'):
    network = get_model(model, input, num_classes, keep_prob, args.gpu)
else:
    (network, _) = get_model(model, input, num_classes, keep_prob, args.gpu)
act = args.act
loss = None
loss_l = []
if args.class_balancing:
    print('Computing class weights for', args.dataset, '...')
    class_weights = utils.compute_class_weights(labels_dir=(args.dataset + '/train_labels'), label_values=label_values)
    if act:
        for gpu_id in range(len(gpus)):
            with tf.device('/cpu:0'):
                with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                    network = get_model(model, input_A[gpu_id], num_classes, keep_prob, args.gpu)
                    unweighted_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=network, labels=output_A[gpu_id])
                    loss_l.append((unweighted_loss * class_weights))
    else:
        unweighted_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=output))
        loss = tf.reduce_mean((unweighted_loss * class_weights))
else:
    if act:
        for gpu_id in range(len(gpus)):
            with tf.device('/cpu:0'):
                with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                    network = get_model(model, input_A[gpu_id], num_classes, keep_prob, args.gpu)
                    _loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=network, labels=output_A[gpu_id])
                    loss_l.append(_loss)
    else:
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=output))
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=output)
        loss_l.append(loss)

loss = tf.reduce_mean(tf.concat(loss_l, axis=0))

######Modify for NPU begin#######
# opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss, var_list=[var for var in tf.trainable_variables()],colocate_gradients_with_ops=True)
opt = tf.train.AdamOptimizer(learning_rate=0.0001)
if args.loss_scale == 0:
     loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000,
                                                            decr_every_n_nan_or_inf=2, decr_ratio=0.5)
elif args.loss_scale >= 1:
     loss_scale_manager = FixedLossScaleManager(loss_scale=args.loss_scale)

opt = NPULossScaleOptimizer(opt, loss_scale_manager)
opt = opt.minimize(loss, var_list=[var for var in tf.trainable_variables()], colocate_gradients_with_ops=True)
######Modify for NPU end#######

saver = tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

# If a pre-trained ResNet is required, load the weights.
# This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())
utils.count_params()
if (init_fn is not None):
    init_fn(sess)

# Load a previous checkpoint if desired
check = args.checkpoint
if (not os.path.isdir(check)):
    os.makedirs(check)
model_checkpoint_name = (((((check + '/latest_model_') + args.model) + '_') + args.dataset) + '.ckpt')

if (args.continue_training or (not (args.mode == 'train'))):
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)

avg_scores_per_epoch = []

if (args.mode == 'train'):
    print('\n***** Begin training *****')
    print('Dataset -->', args.dataset)
    print('Model -->', args.model)
    print('Act sigmoid -->', args.act)
    print('Crop Height -->', args.crop_height)
    print('Crop Width -->', args.crop_width)
    print('Num Epochs -->', args.num_epochs)
    print('Batch Size -->', args.batch_size)
    print('Number Iterations -->', np.floor((len(train_input_names) / args.batch_size)))
    print('Num Classes -->', num_classes)
    print('Data Augmentation:')
    print('\tVertical Flip -->', args.v_flip)
    print('\tHorizontal Flip -->', args.h_flip)
    print('\tBrightness Alteration -->', args.brightness)
    print('\tRotation -->', args.rotation)
    print('')

    avg_loss_per_epoch = []
    # Which validation images do we want
    val_indices = []
    num_vals = min(args.num_val_images, len(val_input_names))
    # Set random seed to make sure models are validated on the same validation images.
    # So you can compare the results of different models more intuitively.
    random.seed(16)
    val_indices = random.sample(range(0, len(val_input_names)), num_vals)

    ######Modify for NPU begin#######
    total_st = time.time()
    FPS_list = []
    accuracy_list = []
    ######Modify for NPU end#######
    # Do the training here
    for epoch in range(0, args.num_epochs):
        cnt = 0
        # Equivalent to shuffling
        id_list = np.random.permutation(len(train_input_names))
        num_iters = int(np.floor((len(id_list) / args.batch_size)))
        
        st = time.time()
        train_time = []
        current_losses = []

        for i in range(num_iters):
            input_image_batch = []
            output_image_batch = []

            # Collect a batch of images
            for j in range(args.batch_size):
                index = ((i * args.batch_size) + j)
                id = id_list[index]
                # print("train_input_names[id]=========>", train_input_names[id])
                # print("crop_height=========>", args.crop_height)
                # print("crop_width=========>", args.crop_width)
                input_image = load_image(train_input_names[id])[:args.crop_height, :args.crop_width]
                output_image = load_image(train_output_names[id])[:args.crop_height, :args.crop_width]

                ######Modify for NPU begin#######
                with tf.device('/cpu:0'):
                #     (input_image, output_image) = data_augmentation(input_image, output_image)
                #     input_image = (np.float32(input_image) / 255.0)
                #     output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))
                (input_image, output_image) = data_augmentation(input_image, output_image)
                #Prep the data. Make sure the labels are in one-hot format
                input_image = (np.float32(input_image) / 255.0)
                output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))
                ######Modify for NPU end#######

                input_image_batch.append(np.expand_dims(input_image, axis=0))
                output_image_batch.append(np.expand_dims(output_image, axis=0))

            if (args.batch_size == 1):
                input_image_batch = input_image_batch[0]
                output_image_batch = output_image_batch[0]
            else:
                input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
                output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))

            # Do the training
            (_, current) = sess.run([opt, loss], feed_dict={input: input_image_batch, output: output_image_batch, keep_prob: 0.5})
            current_losses.append(current)

            cnt = (cnt + args.batch_size)
            ######Modify for NPU begin#######
            if ((cnt % args.batch_size) == 0):
                pass
            else:
                train_time.append(time.time()-st)
            ######Modify for NPU end#######
            if ((cnt % arg.batch_size) == 0)：
                string_print = ('Epoch = %03d Count = %03d Current_Loss = %.4f Time = %.2f' % (epoch, cnt, current, (time.time() - st)))
                utils.LOG(string_print)
                st = time.time()

        mean_loss = np.mean(current_losses)
        avg_loss_per_epoch.append(mean_loss)

        # Create directories if needed
        ######Modify for NPU begin#######
        #if (not os.path.isdir(('%s/%04d' % (check, epoch)))):
        #    os.makedirs(('%s/%04d' % (check, epoch)))
        
        #saver.save(sess, model_checkpoint_name)
        if ((val_indices != 0) and (epoch != 0) and ((epoch % 100) == 0)):
            saver.save(sess, model_checkpoint_name)
            #saver.save(sess, ('%s/%04d/model.ckpt' % (check, epoch)))
        ######Modify for NPU end#######

        target = open(('%s/%04d/val_scores.csv' % (check, epoch)), 'w')
        target.write('val_name, avg_accuracy, precision, recall, f1 score, mean iou \n')
        target.write(class_names_string)

        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []

        # Do the validation on a small set of validation images
        for ind in val_indices:
            input_image = np.float32(load_image(val_input_names[ind])[:args.crop_height, :args.crop_width])
            input_image = (np.expand_dims(input_image, axis=0) / 255.0)

            gt = load_image(val_output_names[ind])[:args.crop_height, :args.crop_width]
            gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

            file_name = utils.filepath_to_name(val_input_names[ind])
            input_l = []
            ######Modify for NPU begin#######
            # for _ in range(len(gpus)):
            #     input_l.append(input_image)
            #     input_l = np.squeeze(np.stack(input_l, axis=1))
            # if (len(gpus) == 1):
            #     input_l = np.expand_dims(input_l, axis=0)
            input_l.append(input_image)
            input_l = np.squeeze(np.stack(input_l, axis=1))
            input_l = np.expand_dims(input_l, axis=0)
            ######Modify for NPU end#######
            if (model != 'ssunet'):
                output_image = sess.run(network, feed_dict={input: input_l, keep_prob: 1.0})
            else:
                (output_image, aux_out) = sess.run(network, feed_dict={input: input_image, keep_prob: 1.0})
                output_image = np.array(aux_out[0, :, :, :])
                output_image = helpers.reverse_one_hot(output_image)
                out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
                cv2.imwrite(('%s/%04d/%s_aux_pred.png' % (check, epoch, file_name)), cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

            output_image = np.array(output_image[0, :, :, :])
            output_image = helpers.reverse_one_hot(output_image)
            out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
            
            (accuracy, class_accuracies, prec, rec, f1, iou) = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)
            target.write(('\n %s, %f, %f, %f, %f, %f \n' % (file_name, accuracy, prec, rec, f1, iou)))
            
            for item in class_accuracies:
                target.write((', %f' % item))
            target.write('\n')
            
            scores_list.append(accuracy)
            class_scores_list.append(class_accuracies)
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)
            iou_list.append(iou)
            
            gt = helpers.colour_code_segmentation(gt, label_values)
            
            im = cv2.imread(val_input_names[ind], (- 1))
            file_name = os.path.basename(val_input_names[ind])
            file_name = os.path.splitext(file_name)[0]
            cv2.imwrite(('%s/%04d/%s_pred.png' % (check, epoch, file_name)), cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
            cv2.imwrite(('%s/%04d/%s_gt.png' % (check, epoch, file_name)), cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))
            cv2.imwrite(('%s/%04d/%s_img.png' % (check, epoch, file_name)), im)

        avg_score = np.mean(scores_list)
        class_avg_scores = np.mean(class_scores_list, axis=0)
        avg_scores_per_epoch.append(avg_score)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_iou = np.mean(iou_list)
        ######Modify for NPU begin#######
        accuracy_list.append(avg_score)
        ######Modify for NPU end#######

        target.write('\n\n')
        target.write(('%s, %s, %s, %s, %s' % ('avg_score', 'avg_precision', 'avg_recall', 'avg_f1', 'avg_iou')))
        target.write(('\n %f, %f, %f, %f, %f' % (avg_score, avg_precision, avg_recall, avg_f1, avg_iou)))
        target.write('\n\n')
        for (index, item) in enumerate(class_avg_scores):
            target.write(('%s = %f' % (class_names_list[index], item)))
            target.write('\n')
        target.close()

        ######Modify for NPU begin#######
        FPS_list.append((cnt - arg.batch_size) / np.sum(train_time))
        print(('Average FPS for epoch # %04d = %.4f' % (epoch, (cnt - arg.batch_size) / np.sum(train_time)))))
        print(('Average accuracy for epoch # %04d = %.4f' % (epoch, avg_score)))
        ######Modify for NPU end#######
        '''
        print(('Model name %s' % args.model))
        print(('Average per class validation accuracies for epoch # %04d:' % epoch))
        for (index, item) in enumerate(class_avg_scores):
            print(('%s = %f' % (class_names_list[index], item)))
        remain_time = (epoch_time * ((args.num_epochs - 1) - epoch))
        (m, s) = divmod(remain_time, 60)
        (h, m) = divmod(m, 60)
        if (s != 0):
            train_time = ('Remaining training time = %d hours %d minutes %d seconds\n' % (h, m, s))
        else:
            train_time = 'Remaining training time : Training completed.\n'
        utils.LOG(train_time)
        scores_list = []
        '''

    print('\nFinal training duration：%.4f' % (time.time() - total_st))
    print('Final accuracy：%.4f' % accuracy_list[-1])
    print('Final performance FPS: %.4f' % np.mean(FPS_list))

    '''
    fig = plt.figure(figsize=(11, 8))

    ax1 = fig.add_subplot(111)
    ax1.plot(range(args.num_epochs), avg_scores_per_epoch)
    ax1.set_title('Average validation accuracy vs epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Avg. val. accuracy')
    plt.savefig('accuracy_vs_epochs.png')
    plt.clf()

    ax1 = fig.add_subplot(111)
    ax1.plot(range(args.num_epochs), avg_loss_per_epoch)
    ax1.set_title('Average loss vs epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Current loss')
    plt.savefig('loss_vs_epochs.png')
    '''
elif (args.mode == 'test'):
    print('\n***** Begin testing *****')
    print('Model -->', args.model)
    print('Dataset -->', args.dataset)
    print('Act Sigmoid -->', args.act)
    print('Crop Height -->', args.crop_height)
    print('Crop Width -->', args.crop_width)
    print('Num Classes -->', num_classes)
    print('')

    # Create directories if needed
    if (not os.path.isdir(('%s' % 'Val'))):
        os.makedirs(('%s' % 'Val'))

    target = open(('%s/val_scores.csv' % 'Val'), 'w')
    target.write(('val_name, avg_accuracy, precision, recall, f1 score, mean iou %s\n' % class_names_string))

    scores_list = []
    class_scores_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    iou_list = []
    run_times_list = []

    # Run testing on ALL test images
    for ind in range(len(val_input_names)):
        sys.stdout.write(('\rRunning test image %d / %d' % ((ind + 1), len(val_input_names))))
        sys.stdout.flush()

        input_image = (np.expand_dims(np.float32(load_image(val_input_names[ind])[:args.crop_height, :args.crop_width]), axis=0) / 255.0)
        gt = load_image(val_output_names[ind])[:args.crop_height, :args.crop_width]
        gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

        st = time.time()
        output_image = sess.run(network, feed_dict={input: input_image})

        run_times_list.append((time.time() - st))
        output_image = np.array(output_image[0, :, :, :])

        output_image = helpers.reverse_one_hot(output_image)

        out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

        file_name = utils.filepath_to_name(val_input_names[ind])
        gt = helpers.colour_code_segmentation(gt, label_values)
        im = cv2.imread(val_input_names[ind])
        cv2.imwrite(('%s/%s_pred.png' % ('Val', file_name)), cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
        cv2.imwrite(('%s/%s_img.png' % ('Val', file_name)), im)
        cv2.imwrite(('%s/%s_gt.png' % ('Val', file_name)), cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))
elif (args.mode == 'predict'):
    if (args.image is None):
        ValueError('You must pass an image path when using prediction mode.')
    print('\n***** Begin prediction *****')
    print('Dataset -->', args.dataset)
    print('Model -->', args.model)
    print('Crop Height -->', args.crop_height)
    print('Crop Width -->', args.crop_width)
    print('Num Classes -->', num_classes)
    print('Image -->', args.image)
    print('')

    ######Modify for NPU#######
    # 创建predict结果的存储路径，否则无法保存predict结果。
    if (not os.path.isdir(('%s' % 'Test'))):
        os.makedirs(('%s' % 'Test'))
    ######Modify for NPU#######

    for ind in range(len(test_input_names)):
        sys.stdout.write(('\rRunning prediction image %d / %d' % ((ind + 1), len(test_input_names))))
        sys.stdout.flush()

        ######Modify for NPU#######
        input_image = (np.expand_dims(np.float32(load_image(test_input_names[ind])[:args.crop_height, :args.crop_width]), axis=0) / 255.0)
        #input_image = (np.expand_dims(np.float32(load_image(args.image)[:args.crop_height, :args.crop_width]), axis=0) / 255.0)
        ######Modify for NPU#######

        st = time.time()
        output_image = sess.run(network, feed_dict={input: input_image})

        run_time = (time.time() - st)

        output_image = np.array(output_image[0, :, :, :])
        output_image = helpers.reverse_one_hot(output_image)
        ######Modify for NPU#######
        # out_vis_image = helpers.colour_code_segmentation(output_image, class_dict)
        out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
        ######Modify for NPU#######

        file_name = utils.filepath_to_name(test_input_names[ind])
        print('=============file_name,', file_name)
        cv2.imwrite(('%s/%s_pred.png' % ('Test', file_name)), cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
        print('=============%s/%s_pred.png,' % ('Test', file_name))
else:
    ValueError('Invalid mode selected.')

######Modify for NPU#######
sess.close()
######Modify for NPU#######
