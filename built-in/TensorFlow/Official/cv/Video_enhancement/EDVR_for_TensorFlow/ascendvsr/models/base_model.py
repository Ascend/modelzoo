import os
import glob
import numpy as np
import imageio
import time
import json
import re
import tensorflow as tf
# from skimage import io as sio
# from skimage import transform as strans

from ascendcv.runner.solver import build_solver
from ascendcv.runner.hccl_broadcast import broadcast_global_variables
from ascendcv.utils.dataloader import PrefetchGenerator
from ascendvsr.minibatch import Minibatch


class VSR(object):

    def __init__(self,
                 model_name,
                 scale,
                 num_frames,
                 data_dir,
                 set_file,
                 batch_size,
                 raw_size,
                 in_size,
                 output_dir,
                 solver,
                 is_train,
                 device,
                 is_distributed,
                 checkpoint,
                 cfg):
        self.model_name = model_name
        self.scale = scale
        self.num_frames = num_frames
        self.data_dir = data_dir
        self.set_file = set_file
        self.batch_size = batch_size
        self.raw_size = raw_size
        self.in_size = in_size
        self.output_dir = output_dir
        self.solver = solver
        self.is_train = is_train
        self.device = device
        self.is_distributed = is_distributed
        self.checkpoint = checkpoint
        self.cfg = cfg

    def build(self):
        b, h, w = self.batch_size, self.in_size[0], self.in_size[1]

        if self.cfg.model.input_format_dimension == 5:
            self.LR = tf.placeholder(tf.float32, shape=[b, self.num_frames, h, w, 3], name='L_input')
        elif self.cfg.model.input_format_dimension == 4:
            if b is None or b < 0:
                if self.is_train:
                    raise ValueError('batchsize cannot be None or less then 0 during training.')
                self.LR = tf.placeholder(tf.float32, shape=[None, h, w, 3], name='L_input')
            else:
                self.LR = tf.placeholder(tf.float32, shape=[b*self.num_frames, h, w, 3], name='L_input')
        else:
            raise ValueError(f'Input format dimension only support 4 or 5, '
                             f'but got {self.cfg.model.input_format_dimension}')

        self.SR = self.build_generator(self.LR)
        if self.is_train:
            self.HR = tf.placeholder(tf.float32, shape=[b, h * 4, w * 4, 3], name='H_truth')
            self.loss = self.caculate_loss(self.SR, self.HR)

    def caculate_loss(self, SR, HR, *kwargs):
        raise NotImplementedError

    def build_generator(self, x):
        raise NotImplementedError

    def save(self, sess, step):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.saver.save(sess, os.path.join(self.output_dir, self.model_name), global_step=step)

    def load(self, sess):
        regex = re.compile('[A-Za-z.]*-([0-9]*).?[A-Za-z0-9]*$')
        print(" [*] Reading SR checkpoints...")

        saver = tf.train.Saver()
        recover_step = 0
        if os.path.exists(self.checkpoint + '.meta'):
            saver.restore(sess, self.checkpoint)
            print(" [*] Reading checkpoints...{} Success".format(self.checkpoint))
            b, = regex.search(self.checkpoint).groups()
            if b is not None and b != '':
                recover_step = int(b) + 1
        else:
            ckpt = tf.train.get_checkpoint_state(self.output_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                saver.restore(sess, os.path.join(self.output_dir, ckpt_name))
                print(" [*] Reading checkpoints...{}/{} Success".format(self.output_dir, ckpt_name))
                b, = regex.search(ckpt_name).groups()
                if b is not None and b != '':
                    recover_step = int(b) + 1
            else:
                print(" [*] Reading checkpoints... ERROR")
                raise ValueError
        return recover_step
    
    def load_from_pb(self, sess):
        try:
            with open(self.checkpoint, 'rb') as f:
                graph_def = GraphDef()
                graph_def.ParseFromString(f.read())

                # When reading some .pb, encounter the following problem:
                # ValueError: Input 0 of node generator_1/dense/Assign was passed float from generator/dense/u:0 incompatible with expected float_ref.
                # Solution: https://github.com/Byronnar/tensorflow-serving-yolov3/issues/77
                with sess.graph.as_default():
                    # fix nodes
                    for node in graph_def.node:
                        if node.op == 'RefSwitch':
                            node.op = 'Switch'
                            for index in range(len(node.input)):
                                if 'moving_' in node.input[index]:
                                    node.input[index] = node.input[index] + '/read'
                        elif node.op == 'AssignSub':
                            node.op = 'Sub'
                            if 'use_locking' in node.attr: del node.attr['use_locking']
                        elif node.op == 'AssignAdd':
                            node.op = 'Add'
                            if 'use_locking' in node.attr: del node.attr['use_locking']
                        elif node.op == 'Assign':
                            node.op = 'Identity'
                            if 'use_locking' in node.attr: del node.attr['use_locking']
                            if 'validate_shape' in node.attr: del node.attr['validate_shape']
                            if len(node.input) == 2:
                                node.input[0] = node.input[1]
                                del node.input[1]

                return graph_def

        except DecodeError:
            meta_graph = tf.saved_model.loader.load(
                sess, [tf.saved_model.tag_constants.SERVING], os.path.dirname(pb_file))
            return meta_graph.graph_def
    
    def train(self, sess_cfg):
        self.minibatch = Minibatch(
            data_dir=self.data_dir, set_file=self.set_file, batch_size=self.batch_size, num_frames=self.num_frames,
            scale=self.scale, in_size=self.in_size)
        self.loader = PrefetchGenerator(self.minibatch, self.cfg.data.num_threads, self.cfg.data.train_data_queue_size)
        self.build()

        vars_all = tf.trainable_variables()

        solver = build_solver(self.solver, self.device, self.is_distributed)

        train_op = solver.opt.minimize(self.loss, var_list=vars_all)

        # Init variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess = tf.Session(config=sess_cfg)
        sess.run(init_op)
        
        recover_step = 0
        if self.cfg.continue_training:
            recover_step = self.load(sess)
        
        npu_distributed = self.cfg.device == 'npu' and self.is_distributed
        if npu_distributed:
            bcast_op = broadcast_global_variables(self.cfg.root_rank, os.environ['DEVICE_ID'])
            sess.run(bcast_op)

        if not (npu_distributed and int(os.environ['DEVICE_ID']) != self.cfg.root_rank):
            tf.io.write_graph(sess.graph_def, self.output_dir, 'train_graph.pbtxt')
            print(f'[INFO] Start training. Log device: {self.cfg.root_rank}')

        self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)

        ave_loss = None
        ave_time = None
        for it in range(recover_step, solver.total_step):
            input_lr, input_hr = next(self.loader)

            st_time = time.time()

            _, cur_lr, loss_v = sess.run(
                [train_op, solver.lr_schedule.lr, self.loss],
                feed_dict={self.LR: input_lr, self.HR: input_hr, solver.lr_schedule.lr: solver.update_lr()})

            onece_time = time.time() - st_time
            if ave_time is None or it < 20:
                ave_time = onece_time
            else:
                ave_time = ave_time * 0.995 + onece_time * 0.005
            ave_time = onece_time

            ave_loss = ave_loss * 0.995 + loss_v * 0.005 if ave_loss is not None else loss_v

            if (it + 1) % self.solver.print_interval == 0 and \
                not (npu_distributed and int(os.environ['DEVICE_ID']) != self.cfg.root_rank):
                fps = self.batch_size / ave_time * self.cfg.rank_size
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        'Step:{}, lr:{:.8f}, loss:{:.08f}, time:{:.2f}ms, fps:{:.2f}'.format((it + 1), cur_lr, ave_loss, ave_time * 1000, fps))

            if (it + 1) % self.solver.checkpoint_interval == 0 and \
                not (npu_distributed and int(os.environ['DEVICE_ID']) != self.cfg.root_rank):
                self.save(sess, (it + 1))
    
    def process_input_image(self, im_names, apply_scale=False):
        # try:
        #     from skimage import io as sio
        #     from skimage import transform as strans
        #     im = sio.imread(im_names) / 255.
        #     h, w, c = im.shape
        #     scale = self.scale if apply_scale else 1
        #     if h != self.cfg.data.eval_in_size[0]*scale or w != self.cfg.data.eval_in_size[1]*scale:
        #         im = strans.resize(im, (self.cfg.data.eval_in_size[0]*scale, self.cfg.data.eval_in_size[1]*scale))
        # except ImportError:
        im = imageio.imread(im_names) / 255.
        return im

    def evaluate(self, sess_cfg):
        self.build()

        sess = tf.Session(config=sess_cfg)

        self.load(sess)

        set_file = os.path.join(self.data_dir, 'sets', self.set_file)
        with open(set_file, 'r') as fid:
            meta = json.load(fid)

        mse_acc = None
        ave_time = 0
        for vid in meta['videos']:
            print('Evaluate {}'.format(vid['name']))
            if meta['prefix']:
                in_path = os.path.join(self.data_dir, 'images', meta['x{}_folder'.format(self.scale)], vid['name'])
                gt_path = os.path.join(self.data_dir, 'images', meta['gt_folder'], vid['name'])
            else:
                in_path = os.path.join(self.data_dir, 'images', vid['name'], meta['x{}_folder'.format(self.scale)])
                gt_path = os.path.join(self.data_dir, 'images', vid['name'], meta['gt_folder'])
            assert os.path.exists(in_path)
            assert os.path.exists(gt_path)
            lrImgs = sorted(glob.glob(os.path.join(in_path, '*.png')))
            lrImgs = np.array([self.process_input_image(i) for i in lrImgs]).astype(np.float32)
            gtImgs = sorted(glob.glob(os.path.join(gt_path, '*.png')))
            gtImgs = np.array([self.process_input_image(i, True) for i in gtImgs]).astype(np.float32)

            lr_list = []
            max_frame = lrImgs.shape[0]
            for i in range(max_frame):
                index = np.array([k for k in range(i - self.num_frames // 2, i + self.num_frames // 2 + 1)])
                index = np.clip(index, 0, max_frame - 1).tolist()
                lr_list.append(np.array([lrImgs[k] for k in index]))
            lr_list = np.array(lr_list)

            mse_vid = None
            ave_time = 0
            for i in range(max_frame):
                st_time = time.time()
                sr = sess.run(self.SR, feed_dict={self.LR: lr_list[i][None]})
                onece_time = time.time() - st_time
                # ave_time = ave_time * 0.9 + onece_time * 0.1
                if i > 0:
                    ave_time += onece_time
                # print('Done frame {}: runtime = {:.0f} ms/frame ({:.1f} FPS)'.format(i, ave_time * 1000, 1 / ave_time))
                sr = np.clip(sr * 255., 0, 255).squeeze()
                gt = gtImgs[i] * 255.
                mse_val = np.mean((gt - sr) ** 2)[None]
                if mse_acc is None:
                    mse_acc = mse_val
                else:
                    mse_acc = np.concatenate([mse_acc, mse_val], axis=0)
                if mse_vid is None:
                    mse_vid = mse_val
                else:
                    mse_vid = np.concatenate([mse_vid, mse_val], axis=0)
            print('Video {} PSNR = {}'.format(vid['name'], np.mean(20. * np.log10(255. / np.sqrt(mse_vid)), axis=0)))
            print(f'\tInference time: {(ave_time / (max_frame - 1))*1000:.2f}')
        psnr_acc = 20. * np.log10(255. / np.sqrt(mse_acc))
        psnr_avg = np.mean(psnr_acc, axis=0)
        print('PSNR = {}'.format(psnr_avg))

    def inference(self, sess_cfg):
        self.build()

        sess = tf.Session(config=sess_cfg)

        self.load(sess)

        set_file = os.path.join(self.data_dir, 'sets', self.set_file)
        with open(set_file, 'r') as fid:
            meta = json.load(fid)
           
        output_dir = os.path.join(self.output_dir, 'test')
        
        ave_time = 0
        for vid in meta['videos']:
            print('Inference {}'.format(vid['name']))
            
            try:
                os.makedirs(os.path.join(output_dir, vid['name']), exist_ok=True)
            except Exception:
                pass
            
            if meta['prefix']:
                in_path = os.path.join(self.data_dir, 'images', meta['x{}_folder'.format(self.scale)], vid['name'])
            else:
                in_path = os.path.join(self.data_dir, 'images', vid['name'], meta['x{}_folder'.format(self.scale)])
            assert os.path.exists(in_path)
            lr_img_names = sorted(glob.glob(os.path.join(in_path, '*.png')))
            lrImgs = np.array([self.process_input_image(i) for i in lr_img_names]).astype(np.float32)

            lr_list = []
            max_frame = lrImgs.shape[0]
            for i in range(max_frame):
                index = np.array([k for k in range(i - self.num_frames // 2, i + self.num_frames // 2 + 1)])
                index = np.clip(index, 0, max_frame - 1).tolist()
                lr_list.append(np.array([lrImgs[k] for k in index]))
            lr_list = np.array(lr_list)

            mse_vid = None
            ave_time = 0
            for i in range(max_frame):
                st_time = time.time()
                if self.cfg.data.eval_in_patch:
                    sr = self._inference_stitching(sess, lr_list[i])
                else:
                    sr = self._inference_whole(sess, lr_list[i][None])
                onece_time = time.time() - st_time
                # ave_time = ave_time * 0.9 + onece_time * 0.1
                if i > 0:
                    ave_time += onece_time
                # print('Done frame {}: runtime = {:.0f} ms/frame ({:.1f} FPS)'.format(i, ave_time * 1000, 1 / ave_time))
                
                im_name = os.path.split(lr_img_names[i])[-1]
                output_img_path = os.path.join(output_dir, vid['name'], im_name)
                imageio.imwrite(output_img_path, sr)
            print(f'Inference time: {(ave_time / (max_frame-1))*1000:.2f}')
    
    def _inference_whole(self, sess, img):
        sr = sess.run(self.SR, feed_dict={self.LR: img})
        sr = np.clip(sr * 255., 0, 255).squeeze()
        sr = np.round(sr).astype(np.uint8)
        return sr

    def _inference_stitching(self, sess, img, patch_per_step=1, patch_size=(180, 320), pad=32):
        from itertools import product

        _, h, w, _ = img.shape
        ph, pw = patch_size
        
        # image padding
        image_pad_right = int(float(h)/ph + 1) * ph - h
        image_pad_bottom = int(float(w)/pw + 1) * pw - w
        image_pad_right = 0 if image_pad_right == ph else image_pad_right
        image_pad_bottom = 0 if image_pad_bottom == pw else image_pad_bottom

        # patch padding
        patch_pad_top = patch_pad_bottom = patch_pad_left = patch_pad_right = pad

        # pad image
        pad_t = patch_pad_top
        pad_b = patch_pad_bottom + image_pad_bottom
        pad_l = patch_pad_left
        pad_r = patch_pad_right + image_pad_right

        img_paded = np.pad(img, ((0, 0),
                                 (pad_t, pad_b), 
                                 (pad_l, pad_r),
                                 (0, 0)), constant_values=0.)

        # number of patches
        num_split_y = h // ph
        num_split_x = w // pw

        sr_all = np.zeros((h*self.scale, w*self.scale, 3), dtype=np.float32)
        img_patches = []
        for split_j, split_i in product(range(num_split_y), range(num_split_x)):
            patch_start_y = split_j * ph
            patch_end_y = patch_start_y + ph + patch_pad_top + patch_pad_bottom
            patch_start_x = split_i * pw
            patch_end_x = patch_start_x + pw + patch_pad_left + patch_pad_right
            img_patches.append(img_paded[:, patch_start_y:patch_end_y, patch_start_x:patch_end_x, :])

        # img_patches [num_patches, ph, pw, channel]
        img_patches = np.array(img_patches)
        num_patches = img_patches.shape[0]
        batch_pad = (num_patches // patch_per_step + 1) * patch_per_step - num_patches
        batch_pad = 0 if batch_pad == patch_per_step else batch_pad
        num_step = (num_patches + batch_pad) // patch_per_step

        if batch_pad > 0:
            img_patches_padded = np.concatenate([
                np.zeros([batch_pad, *img_patches.shape[1:]], dtype=np.float32),
                img_patches
                ], axis=0)
        else:
            img_patches_padded = img_patches

        patch_sr = []
        for i in range(num_step):
            batch_data = img_patches[i*patch_per_step:(i+1):patch_per_step]
            if patch_per_step == 1 and batch_data.shape[0] != 1:
                batch_data = batch_data[None, ...]
            
            _patch_sr = sess.run(self.SR, feed_dict={self.LR: batch_data})
            patch_sr.extend(_patch_sr)
        
        patch_sr = np.array(patch_sr)
        patch_s_y = patch_pad_top*self.scale
        patch_e_y = (patch_pad_top+ph)*self.scale
        patch_s_x = patch_pad_left*self.scale
        patch_e_x = (patch_pad_left+pw)*self.scale
        
        patch_id = 0
        for split_j, split_i in product(range(num_split_y), range(num_split_x)):
            im_s_y = split_j * ph * self.scale
            im_e_y = im_s_y + ph * self.scale
            im_s_x = split_i * pw * self.scale
            im_e_x = im_s_x + pw * self.scale

            sr_all[im_s_y:im_e_y, im_s_x:im_e_x] = patch_sr[patch_id, patch_s_y:patch_e_y, patch_s_x:patch_e_x]
            patch_id += 1

        sr_all = np.clip(sr_all * 255., 0, 255).squeeze()
        sr_all = np.round(sr_all).astype(np.uint8)

        return sr_all

    def freeze(self, sess_cfg):
        raise NotImplementedError
    
    def offline_inference(self, sess_cfg):
        raise NotImplementedError
