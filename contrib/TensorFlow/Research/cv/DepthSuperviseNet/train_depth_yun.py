"""
Function: Depth Regression Net for Face Anti-spoofing
Author: AJ
Date: 2020.12.28
alias python='/home/ajliu/liuajian/tf_py37/bin/python3'
https://github.com git@github.com:liuajian/FAS_ModelZoo_v4.git
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import argparse, sys, os, time
import util.utils as utils
from util.dataset import Dataset
import numpy as np

import moxing as mox
import npu_bridge
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
config = tf.compat.v1.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

def main(args):
    #### Set GPU options ###
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
    # config.gpu_options.allow_growth = True

    ## Setting Parameters ###
    color_mean_div = []
    depth_mean_div = []
    alpha_beta_gamma = []
    lr_decay_epochs = []
    color_mean_div += (float(i) for i in args.color_mean)
    depth_mean_div += (float(i) for i in args.depth_mean)
    alpha_beta_gamma += (float(i) for i in args.alpha_beta_gamma)
    lr_decay_epochs += (int(i) for i in args.lr_decay_epochs)
    label_dim = 2
    color_size = (args.color_image_size, args.color_image_size)
    depth_size = (args.depth_image_size, args.depth_image_size)
    tf.compat.v1.set_random_seed(args.seed)
    ### Make folders of logs and models ###
    subdir = args.subdir  ### subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    model_dir = os.path.join(args.train_url, 'models', args.protocol, subdir)
    outputs_dir = os.path.join(args.train_url, 'outputs', subdir)
    scores_dir = os.path.join(args.train_url, 'scores', args.protocol, subdir)
    for phase in args.phases:
        utils.make_if_not_exist(os.path.join(outputs_dir, phase))
    utils.make_if_not_exist(model_dir)
    utils.make_if_not_exist(scores_dir)
    utils.write_arguments_to_file(args, os.path.join(scores_dir, 'result.txt'))
    ### Load data from different domain ###
    train_data = utils.load_oulu_npu(args.data_url, protocol=args.protocol, mode=args.phases[0])
    train_list, label_list, domain_list = utils.get_sframe_paths_labels(train_data, args.phases[0], ratio=args.ratio)
    epoch_size = int(len(train_list) / args.batch_size)
    dataset_train = Dataset(args, train_list, label_list, mode=args.phases[0])
    print('Load Train Data: num_images={}/num_ID={}/epoch_size={}'.format(len(train_list), len(label_list), epoch_size))
    ### Create a new Graph and set it as the default ###
    # with tf.Graph().as_default():
    ### Repeatedly running this block with the different graph will generate same value
    global_step = tf.Variable(0, trainable=False, name='global_step')
    ### Construction  placeholder
    isTraining_p = tf.compat.v1.placeholder(tf.bool, name='isTraining')
    color_batch_p = \
        tf.compat.v1.placeholder(tf.float32, shape=(None, color_size[0], color_size[1], 3), name='color')
    depth_label_batch_p = \
        tf.compat.v1.placeholder(tf.float32, shape=(None, depth_size[0], depth_size[1], 3), name='depth_label')
    label_batch_P = tf.compat.v1.placeholder(tf.int32, shape=(None,), name='label')
    domain_batch_p = tf.compat.v1.placeholder(tf.string, shape=(None,), name='domain')
    learning_rate_p = tf.compat.v1.placeholder(tf.float32, name='learning_rate')
    learning_rate = tf.compat.v1.train.exponential_decay(learning_rate=learning_rate_p, global_step=global_step,
        decay_steps=args.max_nrof_epochs * epoch_size, decay_rate=1, staircase=True)
    ### Build the inference graph ###
    import util.FaceMapNet_BAS as FaceMapNet_BAS
    # logits, embeddings, depth_map, accuracy, total_loss, bin_cla_loss, depth_loss, triplet_loss,fraction,pairwise_dist
    model_list = FaceMapNet_BAS.build_Multi_Adversarial_Loss(color_batch_p, depth_label_batch_p,
        label_batch_P, domain_batch_p, label_dim, alpha_beta_gamma, depth_size=depth_size[0],
        triplet_strategy=args.triplet_strategy, margin=args.triplet_margin, isTraining=True)
    ### Create a saver and train_op, and save the last three models
    saver, saver_restore, trainable_list, restore_trainable_list = utils.get_saver_tf()
    train_op = utils.get_train_op(model_list[4], trainable_list, global_step, args.optimizer, learning_rate)
    with tf.Session(config=config) as sess:
        sess.run(tf.group(tf.compat.v1.global_variables_initializer()))
        ### Start running operations on the Graph.
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        if args.pretrained_model and (args.pretrained_model != '.'):
            print('Restoring pretrained_model')
            ckpt = tf.train.get_checkpoint_state(args.pretrained_model)
            saver.restore(sess, ckpt.model_checkpoint_path)
        print('Running training')
        fid_train = open(os.path.join(scores_dir, 'Train_loss.txt'), 'w')
        for epoch in range(args.check_point, args.max_nrof_epochs):
            current_lr = utils.get_lr(args.lr, lr_decay_epochs, epoch, lr_decay=args.lr_decay_factor)
            train_Model(sess, epoch, dataset_train, epoch_size, color_mean_div, depth_mean_div, current_lr,
                learning_rate_p, isTraining_p, domain_batch_p, color_batch_p, depth_label_batch_p, label_batch_P,
                model_list, train_op, outputs_dir, args.phases[0], fid_train)
            ### Save model ###
            utils.save_variables_and_metagraph(sess, saver, model_dir, subdir, epoch)
            mox.file.copy_parallel(model_dir,   os.path.join('obs://ajian3/Jobs', 'models', args.protocol, subdir))
            mox.file.copy_parallel(outputs_dir, os.path.join('obs://ajian3/Jobs', 'outputs', subdir))
            mox.file.copy_parallel(scores_dir,  os.path.join('obs://ajian3/Jobs', 'scores', args.protocol, subdir))
            print("copy data successfully")
        ### End sess ###
        fid_train.close()
        sess.close()

def train_Model(sess, epoch, dataset_train, epoch_size, color_mean_div, depth_mean_div, current_lr,
                learning_rate_p, isTraining_p, domain_batch_p, color_batch_p, depth_label_batch_p, label_batch_P,
                model_list, train_op, outputs_dir, phase, fid):
    ### Training loop ###
    batch_number = 0
    interval_save = 1000
    while batch_number < epoch_size:
        st = time.time()
        data_list_ = sess.run(list(dataset_train.nextit))
        # print(epoch, batch_number, len(data_list_[1]))
        feed_dict = {learning_rate_p: current_lr, isTraining_p: True, domain_batch_p: data_list_[0],
                     color_batch_p: data_list_[2], depth_label_batch_p: data_list_[4], label_batch_P: data_list_[5]}
        tensor_list = [train_op] + list(model_list)
        _, logits_, _, depth_map_, accuracy_, total_loss_, bin_cla_loss_, depth_loss_, triplet_loss_, fraction_, _ = \
            sess.run(tensor_list, feed_dict=feed_dict)
        if batch_number % interval_save == 0:
            ### Generate intermediate results
            utils.write_ScoreImages(os.path.join(outputs_dir, phase),
                                    data_list_[2], data_list_[3], data_list_[4],
                                    data_list_[5], data_list_[0], data_list_[1],
            color_mean_div, depth_mean_div, batch_number, logits_, depth_map_, accuracy_, fid)
        duration = time.time() - st
        print('FAS: Epoch: [%d][%d/%d]\tTime %.3f\t[C:%.3f D:%.3f T:%.3f(%.3f)]\t[Loss: %.3f]\tAcc %2.3f\t''Lr %2.5f'
        %(epoch, batch_number + 1, epoch_size, duration, bin_cla_loss_, depth_loss_, triplet_loss_, fraction_,
        total_loss_, accuracy_, current_lr))
        if batch_number % 100 == 0:
            fid.write('FAS: Epoch: [%d][%d/%d]\tTime %.3f\t[C: %.3f D: %.3f T: %.3f]\t[Loss: %.3f]\tAcc %2.3f\t''Lr %2.5f'
            %(epoch, batch_number + 1, epoch_size, duration, bin_cla_loss_, depth_loss_, triplet_loss_, total_loss_,
            accuracy_, current_lr))
        batch_number += 1
    return True

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--train_url", type=str, default='/home/work/modelarts/outputs/Jobs/')
    parser.add_argument("--data_url",  type=str, default='/home/work/modelarts/inputs/Oulu-Train/')

    parser.add_argument("--subdir", type=str, default='001')
    parser.add_argument('--data_name', type=str, default='oulu')
    parser.add_argument("--protocol", type=str, default='oulu_protocal_2')
    parser.add_argument("--data_augment", type=list, default=[180, 0, 1, 1, 0],
                        help='[0]:max_angle [1]:RANDOM_FLIP [2]:RANDOM_CROP [3]:RANDOM_COLOR [4]:is_std')
    parser.add_argument('--net_name', type=str, default='facemap_tf', help='resnet_tf, facemap_tf')
    parser.add_argument("--optimizer", type=str, default='ADAM')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--color_image_size", type=int, default=256)
    parser.add_argument("--depth_image_size", type=int, default=32)
    parser.add_argument("--check_point", type=int, default=0)
    parser.add_argument("--max_nrof_epochs", type=int, default=60)
    parser.add_argument("--lr_decay_epochs", type=list, default=[20, 30, 40])
    parser.add_argument("--color_mean", type=list, default=[0.0, 127.5])
    parser.add_argument("--depth_mean", type=list, default=[0.0, 255.0])
    parser.add_argument("--ratio", type=int, default=3)
    parser.add_argument("--disorder_para", type=list, default=[8, 0.2, 0.02], help='[0]:alpha [1]:beta [2]:gamma')
    parser.add_argument("--alpha_beta_gamma", type=list, default=[1.0, 0, 1.0])
    parser.add_argument("--triplet_strategy", type=str, default='batch_hard', help='batch_all, batch_hard')
    parser.add_argument("--triplet_margin", type=float, default=1.0)
    parser.add_argument("--lr_decay_factor", type=float, default=0.1)
    parser.add_argument("--pretrained_model", type=str, default='.')
    parser.add_argument("--phases", type=list, default=['train', 'dev', 'test'])
    parser.add_argument("--seed", type=int, default=6)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


