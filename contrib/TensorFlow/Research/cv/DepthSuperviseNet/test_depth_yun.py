"""
Function: Depth Regression Net for Face Anti-spoofing
Author: AJ
Date: 2020.12.28
alias python='/home/ajliu/liuajian/tf_py37/bin/python3'
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import util.FaceMapNet_BAS as FaceMapNet_BAS
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

    ### Setting Parameters ###
    color_mean_div = []
    depth_mean_div = []
    alpha_beta_gamma = []
    color_mean_div += (float(i) for i in args.color_mean)
    depth_mean_div += (float(i) for i in args.depth_mean)
    alpha_beta_gamma += (float(i) for i in args.alpha_beta_gamma)
    label_dim = 2
    color_size = (args.color_image_size, args.color_image_size)
    depth_size = (args.depth_image_size, args.depth_image_size)
    ### Make folders of logs and models ###
    subdir = args.subdir
    model_dir = os.path.join(args.train_url, 'models', args.protocol, subdir)
    outputs_dir = os.path.join(args.train_url, 'outputs', subdir)
    scores_dir = os.path.join(args.train_url, 'scores', args.protocol, subdir)
    mox.file.copy_parallel(os.path.join('obs://ajian3/Jobs', 'models', args.protocol, subdir), model_dir)
    mox.file.copy_parallel(os.path.join('obs://ajian3/Jobs', 'outputs', subdir), outputs_dir)
    mox.file.copy_parallel(os.path.join('obs://ajian3/Jobs', 'scores', args.protocol, subdir), scores_dir)
    print("OBS2yun data successfully")

    ### Load data from different domain ###
    dev_data = utils.load_oulu_npu(args.data_url, protocol=args.protocol, mode=args.phases[1])
    test_data = utils.load_oulu_npu(args.data_url, protocol=args.protocol, mode=args.phases[2])
    dev_image, dev_label, dev_domain = utils.get_sframe_paths_labels(dev_data, args.phases[1], num='one')
    test_image, test_label, test_domain = utils.get_sframe_paths_labels(test_data, args.phases[2], num='one')
    print('Valid: num_images={}/num_ID={}'.format(len(dev_image), len(dev_label)))
    print('Test: num_images={}/num_ID={}'.format(len(test_image), len(test_label)))
    ### Create a new Graph and set it as the default ###
    # with tf.Graph().as_default():
    ### Repeatedly running this block with the different graph will generate same value
    tf.compat.v1.set_random_seed(args.seed)
    ### Construction  placeholder
    isTraining_p = tf.compat.v1.placeholder(tf.bool, name='isTraining')
    color_batch_p = \
        tf.compat.v1.placeholder(tf.float32, shape=(1, color_size[0], color_size[1], 3), name='color')
    depth_label_batch_p = \
        tf.compat.v1.placeholder(tf.float32, shape=(1, depth_size[0], depth_size[1], 3), name='depth_label')
    label_batch_P = tf.compat.v1.placeholder(tf.int32, shape=(1, ), name='label')
    domain_batch_p = tf.compat.v1.placeholder(tf.string, shape=(1, ), name='domain')
    ### Build the inference graph ###
    #logits, embeddings, depth_map, accuracy, total_loss, bin_cla_loss, depth_loss, triplet_loss,fraction,pairwise_dist
    model_list = FaceMapNet_BAS.build_Multi_Adversarial_Loss(color_batch_p, depth_label_batch_p,
        label_batch_P, domain_batch_p, label_dim, alpha_beta_gamma, depth_size=depth_size[0], isTraining=False)
    ### Create a saver and train_op, and save the last three models
    saver, _, _, _ = utils.get_saver_tf()
    # with tf.Session(config=config) as sess:
    sess = tf.Session(config=config)
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord, sess=sess)

    def evaluate_score(trainedModal):
        saver.restore(sess, trainedModal)
        for phase in args.phases[1:]:
            print('ceshi2:', phase)
            if phase == 'dev':
                dataset = Dataset(args, dev_image, dev_label, mode=args.phases[1])
                epoch_size = len(dev_image)
                fid = open(os.path.join(scores_dir, 'Dev_scores.txt'), 'w')
            elif phase == 'test':
                dataset = Dataset(args, test_image, test_label, mode=args.phases[2])
                epoch_size = len(test_image)
                fid = open(os.path.join(scores_dir, 'Test_scores.txt'), 'w')
            test_Model(sess, dataset, epoch_size, color_mean_div, depth_mean_div,
                isTraining_p, domain_batch_p, color_batch_p, depth_label_batch_p, label_batch_P,
                model_list, outputs_dir, phase, fid)
            fid.close()

    def evaluate_metric(iter_now, min_value):
        for score_ind in range(1, 3):
            if score_ind == 1: score_type = 'logit_score'
            else: score_type = 'depth_score'
            ### <score_ind==1:prob_score score_ind==2:exp_score>
            Dev_best_thre, TP, FN, FP, TN, test_EER, APCER, NPCER, ACER, Recall2, Recall3, Recall4 = \
                utils.performance(args.phases, scores_dir, score_ind)
            test_metric = ACER
            if test_metric * 100 < min_value:
                min_value = test_metric * 100
                ### Write results ###
                result_txt = os.path.join(scores_dir, 'result.txt')
                if not os.path.exists(result_txt):
                    lines = []
                else:
                    fid = open(result_txt, 'r')
                    lines = fid.readlines()
                    fid.close()
                str_line = \
                '%s thre %.4f\tTP %d\tFN %d\tFP %d\tTN %d\tAPCER %.4f\tNPCER %.4f\tACER %.4f\tTPR@-2 %.4f\t' \
                'TPR@-3 %.4f\tTPR@-4 %.4f'%(score_type, Dev_best_thre, TP, FN, FP, TN, APCER*100, NPCER*100, ACER*100,
                Recall2 * 100, Recall3 * 100, Recall4 * 100)
                fid = open(result_txt, 'w')
                line_new = str_line + '\tmodal_iter %d\n'%(iter_now)
                lines.append(line_new)
                fid.writelines(lines)
                fid.close()
        return min_value

    def offline_eval():
        all_path_ckpt = os.path.join(model_dir, 'checkpoint')
        min_value = 100
        while True:
            if not os.path.exists(all_path_ckpt):continue
            else:break
        iter_before = start_iteration
        fid = open(all_path_ckpt, 'r')
        lines = fid.readlines()
        fid.close()
        if len(lines) < 2:
            print('Hasn\'t enough model!')
            return 0
        for i in range(1, len(lines)):
            iter_now = int(lines[i].split('-')[-1][:-2])
            if iter_now - iter_before >= interval_iteration:
                trainedModal = \
                    os.path.join(model_dir, 'model-{}.ckpt-{}'.format(subdir, iter_now))
                print('load model:', trainedModal)
                evaluate_score(trainedModal)
                min_value = evaluate_metric(iter_now, min_value)
                iter_before = iter_now
                print('end one iteration!')
                mox.file.copy_parallel(outputs_dir, os.path.join('obs://ajian3/Jobs', 'outputs', subdir))
                mox.file.copy_parallel(scores_dir,  os.path.join('obs://ajian3/Jobs', 'scores', args.protocol, subdir))
                print("yun2OBS data successfully")

    print('Start Testing')
    start_iteration = 0
    interval_iteration = 1
    offline_eval()
    sess.close()

def test_Model(sess, dataset, epoch_size, color_mean_div, depth_mean_div, isTraining_p, domain_batch_p, color_batch_p,
               depth_label_batch_p, label_batch_P, model_list, outputs_dir, phase, fid):
    print('Running forward pass on evaluate set')
    batch_number = 0
    while batch_number < epoch_size:
        data_list_ = sess.run(list(dataset.nextit))
        feed_dict = {isTraining_p: False, domain_batch_p: data_list_[0], color_batch_p: data_list_[2],
                     depth_label_batch_p: data_list_[4], label_batch_P: data_list_[5]}
        tensor_list = list(model_list)
        logits_, _, depth_map_, accuracy_, total_loss_, bin_cla_loss_, depth_loss_, triplet_loss_, fraction_, _ = \
            sess.run(tensor_list, feed_dict=feed_dict)
        ### Generate intermediate results
        utils.write_ScoreImages(os.path.join(outputs_dir, phase),
                                data_list_[2], data_list_[3], data_list_[4],
                                data_list_[5], data_list_[0], data_list_[1],
                                color_mean_div, depth_mean_div, batch_number, logits_, depth_map_, accuracy_, fid)
        batch_number += 1
    return True

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--train_url", type=str, default='/home/work/modelarts/outputs/Jobs/')
    parser.add_argument("--data_url",  type=str, default='/home/work/modelarts/inputs/Oulu-Test/')
    parser.add_argument("--subdir", type=str, default='001')
    parser.add_argument('--data_name', type=str, default='oulu')
    parser.add_argument("--protocol", type=str, default='oulu_protocal_2')
    parser.add_argument("--data_augment", type=list, default=[0, 0, 0, 0, 0],
                        help='[0]:max_angle [1]:RANDOM_FLIP [2]:RANDOM_CROP [3]:RANDOM_COLOR [4]:is_std')
    parser.add_argument('--net_name', type=str, default='facemap_tf', help='resnet_tf, facemap_tf')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--color_image_size", type=int, default=256)
    parser.add_argument("--depth_image_size", type=int, default=32)
    parser.add_argument("--max_nrof_epochs", type=int, default=1)
    parser.add_argument("--color_mean", type=list, default=[0.0, 127.5])
    parser.add_argument("--depth_mean", type=list, default=[0.0, 255.0])
    parser.add_argument("--disorder_para", type=list, default=[8, 0.2, 0.02], help='[0]:alpha [1]:beta [2]:gamma')
    parser.add_argument("--alpha_beta_gamma", type=list, default=[1.0, 0, 1.0])
    parser.add_argument("--phases", type=list, default=['train', 'dev', 'test'])
    parser.add_argument("--seed", type=int, default=6)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


